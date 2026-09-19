//! Bipartite normal-equation Schur elimination in diagonally scaled coordinates.
use crate::{
    EvaluationError, Real, SolverError,
    linear_model::{
        LinearizedModel, StepWorkspace, block_factors::BlockFactors, relative, relative_norm,
    },
    optimization::{DampedLeastSquaresBackend, DampedStep, LeastSquaresBackend, norm_inf},
};
use faer::{
    MatMut, MatRef, Par,
    dyn_stack::{MemBuffer, MemStack, StackReq},
    mat::AsMatMut,
    matrix_free::{InitialGuessStatus, LinOp, Precond, conjugate_gradient as cg},
};
use faer_ext::nalgebra::{DMatrixView, Dyn};
use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

/// Work performed since preparation of the Schur backend.
#[derive(Debug, Default, Clone, Copy)]
pub struct SchurStatistics {
    /// Linear solve attempts, including failures.
    pub solves: usize,
    /// Completed reduced-system CG iterations, including failed solves.
    pub iterations: usize,
}

/// Termination of the last reduced linear solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchurTermination {
    /// The reduced residual met CG's relative stopping criterion.
    Converged,
    /// CG exhausted its iteration budget.
    IterationLimit,
    /// A local Cholesky factorization or CG positive-definiteness check failed.
    NotPositiveDefinite,
    /// A recovered step or residual was nonfinite.
    NonFinite,
}

/// Last-attempt diagnostics; nonlinear convergence is checked separately by LM.
#[derive(Debug, Clone, Copy)]
pub struct SchurDiagnostics<R: Real = f64> {
    /// None before a numerical attempt.
    pub termination: Option<SchurTermination>,
    /// Completed CG iterations in this attempt.
    pub iterations: usize,
    /// Damping used for this attempt.
    pub damping: R,
    /// CG's reduced-system relative residual estimate.
    pub estimated_relative_residual: Option<R>,
    /// Independently recomputed reduced-system relative residual.
    pub reduced_relative_residual: Option<R>,
    /// Full damped normal residual relative to the gradient, in D-scaled coordinates.
    pub scaled_relative_residual: Option<R>,
    /// Full damped normal residual relative to the original gradient.
    pub original_relative_residual: Option<R>,
}
impl<R: Real> Default for SchurDiagnostics<R> {
    fn default() -> Self {
        Self {
            termination: None,
            iterations: 0,
            damping: R::zero(),
            estimated_relative_residual: None,
            reduced_relative_residual: None,
            scaled_relative_residual: None,
            original_relative_residual: None,
        }
    }
}

/// Opt-in cumulative wall-clock measurements. Products/preconditioning are subsets
/// of `linear_solve`, not additional costs.
#[derive(Debug, Default, Clone, Copy)]
pub struct SchurTimings {
    /// Column norms, local Gram blocks and cross-block assembly.
    pub preparation: Duration,
    /// Local factorization and reduced right-hand-side construction.
    pub factorization: Duration,
    /// Complete reduced CG solves.
    pub linear_solve: Duration,
    /// Implicit Schur products, including eliminated-block triangular solves.
    pub products: Duration,
    /// Retained-block preconditioner applications.
    pub preconditioner: Duration,
    /// Back-substitution, prediction, and explicit residual checks.
    pub verification: Duration,
}

#[derive(Debug)]
struct Edge {
    retained: usize,
    eliminated: usize,
    retained_jacobian: usize,
    eliminated_jacobian: usize,
}

/// Iterative Schur backend for bipartite least squares, including bundle adjustment.
///
/// Coordinates must be ordered as a retained prefix followed by independent
/// eliminated blocks. Each partition has a fixed variable-block width. A residual
/// may touch at most one complete block in each partition; unary factors are allowed.
/// Other couplings return [`EvaluationError::UnsupportedStructure`]. This includes
/// general marginal priors that couple multiple eliminated or retained blocks.
///
/// Forms small normalized Gram/cross blocks, factors B and C with Cholesky, and
/// solves `(B - E C^-1 E^T) z = b` using faer's CG with B-block preconditioning.
/// Cross blocks stay sparse; no global normal or reduced matrix is assembled.
/// This uses normal equations and can lose accuracy on ill-conditioned systems.
/// Failed factorizations/CG solves return errors rather than approximate success.
/// Positive LM damping regularizes gauge freedoms without fixing the gauge.
/// Workspace and the undamped linearization are retained across retries/calls.
pub struct Schur<R: Real = f64> {
    /// Maximum CG iterations per attempt. Default: 1000.
    pub max_iterations: usize,
    /// Relative reduced-system residual tolerance, in [0,1).
    /// Default matches LSMR's precision-aware tolerance; the stopping norm differs.
    pub relative_tolerance: R,
    /// Enable component timings. Default: false.
    pub collect_timings: bool,
    /// Observe every numerically attempted solve, including failures.
    pub on_solve: Option<fn(SchurDiagnostics<R>)>,
    retained: usize,
    retained_width: usize,
    eliminated_width: usize,
    model: LinearizedModel<R>,
    blocks: BlockFactors<R>,
    work: StepWorkspace<R>,
    edges: Vec<Edge>,
    cross: Vec<R>,
    reduced_rhs: Vec<R>,
    reduced_step: Vec<R>,
    scaled_work: Vec<R>,
    reduced_residual: Vec<R>,
    cg_scratch: Option<MemBuffer>,
    ready: bool,
    statistics: SchurStatistics,
    diagnostics: SchurDiagnostics<R>,
    timings: Mutex<SchurTimings>,
}

impl<R: Real> Schur<R> {
    /// Select a retained prefix and fixed block widths. Validated by backend
    /// preparation against the graph's coordinate dimension. Schema/pool ordering
    /// defines the prefix; reconstruct this backend if the partition changes.
    pub fn new(
        retained_dimension: usize,
        retained_block_width: usize,
        eliminated_block_width: usize,
    ) -> Self {
        Self {
            max_iterations: 1000,
            relative_tolerance: R::from_f64_impl(1e-10)
                .max(R::epsilon_impl() * R::from_f64_impl(128.)),
            collect_timings: false,
            on_solve: None,
            retained: retained_dimension,
            retained_width: retained_block_width,
            eliminated_width: eliminated_block_width,
            model: LinearizedModel::default(),
            blocks: BlockFactors::default(),
            work: StepWorkspace::default(),
            edges: Vec::new(),
            cross: Vec::new(),
            reduced_rhs: Vec::new(),
            reduced_step: Vec::new(),
            scaled_work: Vec::new(),
            reduced_residual: Vec::new(),
            cg_scratch: None,
            ready: false,
            statistics: SchurStatistics::default(),
            diagnostics: SchurDiagnostics::default(),
            timings: Mutex::new(SchurTimings::default()),
        }
    }
    /// Cumulative work since the last preparation.
    pub fn statistics(&self) -> SchurStatistics {
        self.statistics
    }
    /// Last linear attempt, including on errors.
    pub fn diagnostics(&self) -> SchurDiagnostics<R> {
        self.diagnostics
    }
    /// Cumulative component timings; disabled by default.
    pub fn timings(&self) -> SchurTimings {
        *self.timings.lock().unwrap()
    }

    fn attempt(&mut self, lambda: R) -> Result<R, SolverError> {
        self.factor_and_reduce_rhs(lambda)?;
        self.solve_reduced(lambda);
        let start = self.collect_timings.then(Instant::now);
        self.recover_step(lambda);
        let verified = self.verify_step(lambda);
        if let Some(start) = start {
            self.timings.get_mut().unwrap().verification += start.elapsed();
        }
        verified
    }

    fn factor_and_reduce_rhs(&mut self, lambda: R) -> Result<(), SolverError> {
        let n = self.model.jacobian.cols;
        let start = self.collect_timings.then(Instant::now);
        // Exact elimination cannot use the diagonal fallback appropriate for a
        // preconditioner: if any local factor fails, reject this linear attempt.
        if self.blocks.factor(lambda) != 0 {
            if let Some(start) = start {
                self.timings.get_mut().unwrap().factorization += start.elapsed();
            }
            self.diagnostics.termination = Some(SchurTermination::NotPositiveDefinite);
            return Err(SolverError::LinearSolveFailed);
        }
        for j in 0..n {
            self.scaled_work[j] = -self.model.gradient[j] / self.model.scales[j];
        }
        self.blocks.solve_normal(
            MatMut::from_column_major_slice_mut(
                &mut self.scaled_work[self.retained..],
                n - self.retained,
                1,
            ),
            self.retained,
            lambda,
        );
        self.reduced_rhs
            .copy_from_slice(&self.scaled_work[..self.retained]);
        cross_apply(
            &self.edges,
            &self.cross,
            (self.retained_width, self.eliminated_width),
            MatMut::from_column_major_slice_mut(&mut self.reduced_rhs, self.retained, 1),
            MatRef::from_column_major_slice(
                &self.scaled_work[self.retained..],
                n - self.retained,
                1,
            ),
            false,
            -R::one(),
        );
        if let Some(start) = start {
            self.timings.get_mut().unwrap().factorization += start.elapsed();
        }
        Ok(())
    }

    fn solve_reduced(&mut self, lambda: R) {
        let n = self.model.jacobian.cols;
        let operator = Reduced {
            retained: self.retained,
            eliminated: n - self.retained,
            retained_width: self.retained_width,
            eliminated_width: self.eliminated_width,
            blocks: &self.blocks,
            edges: &self.edges,
            cross: &self.cross,
            lambda,
            timings: self.collect_timings.then_some(&self.timings),
        };
        let start = self.collect_timings.then(Instant::now);
        self.reduced_step.fill(R::zero());
        let req = cg::conjugate_gradient_scratch::<R>(
            RetainedPreconditioner(&operator),
            &operator,
            1,
            Par::Seq,
        );
        if !self
            .cg_scratch
            .as_mut()
            .is_some_and(|s| MemStack::new(s).can_hold(req))
        {
            self.cg_scratch = Some(MemBuffer::new(req));
        }
        // faer 0.24.4's CG Zero path leaves its initial residual uninitialized.
        // With our explicit zero step, MaybeNonZero computes the correct b-A*0.
        let params = cg::CgParams {
            initial_guess: InitialGuessStatus::MaybeNonZero,
            rel_tolerance: self.relative_tolerance,
            max_iters: self.max_iterations,
            ..Default::default()
        };
        let result = cg::conjugate_gradient(
            MatMut::from_column_major_slice_mut(&mut self.reduced_step, self.retained, 1),
            RetainedPreconditioner(&operator),
            &operator,
            MatRef::from_column_major_slice(&self.reduced_rhs, self.retained, 1),
            params,
            |_| {
                self.statistics.iterations += 1;
                self.diagnostics.iterations += 1;
            },
            Par::Seq,
            MemStack::new(self.cg_scratch.as_mut().unwrap()),
        );
        let (termination, relative) = match result {
            Ok(info) => (SchurTermination::Converged, Some(info.rel_residual)),
            // faer 0.24.4 returns its initial (shadowed) residual on exhaustion;
            // use the explicit reduced residual below instead of that stale value.
            Err(cg::CgError::NoConvergence { .. }) => (SchurTermination::IterationLimit, None),
            Err(_) => (SchurTermination::NotPositiveDefinite, None),
        };
        self.diagnostics.termination = Some(termination);
        self.diagnostics.estimated_relative_residual = relative;
        if let Some(start) = start {
            self.timings.lock().unwrap().linear_solve += start.elapsed();
        }
    }

    fn recover_step(&mut self, lambda: R) {
        let n = self.model.jacobian.cols;
        // z_p = C^-1 (-g_p - E^T z_c); delta = D^-1 z.
        self.scaled_work[..self.retained].copy_from_slice(&self.reduced_step);
        for j in self.retained..n {
            self.scaled_work[j] = -self.model.gradient[j] / self.model.scales[j];
        }
        cross_apply(
            &self.edges,
            &self.cross,
            (self.retained_width, self.eliminated_width),
            MatMut::from_column_major_slice_mut(
                &mut self.scaled_work[self.retained..],
                n - self.retained,
                1,
            ),
            MatRef::from_column_major_slice(&self.reduced_step, self.retained, 1),
            true,
            -R::one(),
        );
        self.blocks.solve_normal(
            MatMut::from_column_major_slice_mut(
                &mut self.scaled_work[self.retained..],
                n - self.retained,
                1,
            ),
            self.retained,
            lambda,
        );
        for j in 0..n {
            self.work.step[j] = self.scaled_work[j] / self.model.scales[j];
        }
    }

    fn verify_step(&mut self, lambda: R) -> Result<R, SolverError> {
        // The reduced residual and the original-coordinate full model provide
        // independent checks of CG and elimination/back-substitution respectively.
        let operator = Reduced {
            retained: self.retained,
            eliminated: self.model.jacobian.cols - self.retained,
            retained_width: self.retained_width,
            eliminated_width: self.eliminated_width,
            blocks: &self.blocks,
            edges: &self.edges,
            cross: &self.cross,
            lambda,
            timings: None,
        };
        operator.apply(
            MatMut::from_column_major_slice_mut(&mut self.reduced_residual, self.retained, 1),
            MatRef::from_column_major_slice(&self.reduced_step, self.retained, 1),
            Par::Seq,
            MemStack::new(self.cg_scratch.as_mut().unwrap()),
        );
        for (r, &b) in self.reduced_residual.iter_mut().zip(&self.reduced_rhs) {
            *r -= b;
        }
        self.diagnostics.reduced_relative_residual =
            Some(relative_norm(&self.reduced_residual, &self.reduced_rhs));
        let verified = self.model.verify_step(&mut self.work, lambda);
        self.diagnostics.original_relative_residual =
            Some(relative(verified.residual_norm, verified.reference_norm));
        self.work.scale_residual(&self.model.scales);
        self.diagnostics.scaled_relative_residual = Some(relative_norm(
            &self.work.normal_residual,
            &self.work.reference,
        ));
        if !verified.finite_step
            || !verified.predicted_reduction.is_finite()
            || [
                self.diagnostics.reduced_relative_residual,
                self.diagnostics.scaled_relative_residual,
                self.diagnostics.original_relative_residual,
            ]
            .into_iter()
            .flatten()
            .any(|r| !r.is_finite())
        {
            self.diagnostics.termination = Some(SchurTermination::NonFinite);
            return Err(SolverError::LinearSolveFailed);
        }
        if self.diagnostics.termination != Some(SchurTermination::Converged) {
            return Err(SolverError::LinearSolveFailed);
        }
        Ok(verified.predicted_reduction)
    }
}

impl<R: Real> LeastSquaresBackend for Schur<R> {
    type Scalar = R;
    fn prepare(&mut self, dimension: usize) -> Result<(), SolverError> {
        self.ready = false;
        self.statistics = SchurStatistics::default();
        self.diagnostics = SchurDiagnostics::default();
        *self.timings.get_mut().unwrap() = SchurTimings::default();
        if self.retained_width == 0
            || self.eliminated_width == 0
            || self.retained > dimension
            || !self.retained.is_multiple_of(self.retained_width)
            || !(dimension - self.retained).is_multiple_of(self.eliminated_width)
            || self.max_iterations == 0
            || !(R::zero()..R::one()).contains(&self.relative_tolerance)
        {
            return Err(SolverError::InvalidOptions);
        }
        self.model.prepare(dimension);
        self.work.step.resize(dimension, R::zero());
        self.reduced_rhs.resize(self.retained, R::zero());
        self.reduced_step.resize(self.retained, R::zero());
        self.reduced_residual.resize(self.retained, R::zero());
        self.scaled_work.resize(dimension, R::zero());
        Ok(())
    }
    fn clear(&mut self) {
        self.ready = false;
        self.edges.clear();
        self.cross.clear();
        self.model.clear();
    }
    fn accumulate<'a>(
        &mut self,
        residual: &[R],
        jacobians: impl Iterator<Item = (usize, DMatrixView<'a, R, Dyn, Dyn>)> + Clone,
    ) -> Result<(), EvaluationError> {
        self.ready = false;
        let (mut left, mut right) = (None, None);
        let start = self.model.jacobian.blocks.len();
        for (index, (col, j)) in jacobians.clone().enumerate() {
            let end = col
                .checked_add(j.ncols())
                .ok_or(EvaluationError::DimensionMismatch)?;
            if end > self.model.jacobian.cols || j.nrows() != residual.len() {
                return Err(EvaluationError::DimensionMismatch);
            }
            if col < self.retained {
                if !col.is_multiple_of(self.retained_width)
                    || j.ncols() != self.retained_width
                    || end > self.retained
                    || left.is_some()
                {
                    return Err(EvaluationError::UnsupportedStructure);
                }
                left = Some((col, start + index));
            } else {
                if !(col - self.retained).is_multiple_of(self.eliminated_width)
                    || j.ncols() != self.eliminated_width
                    || right.is_some()
                {
                    return Err(EvaluationError::UnsupportedStructure);
                }
                right = Some((col - self.retained, start + index));
            }
        }
        self.model.accumulate(residual, jacobians)?;
        if let (Some((retained, left)), Some((eliminated, right))) = (left, right) {
            self.edges.push(Edge {
                retained,
                eliminated,
                retained_jacobian: left,
                eliminated_jacobian: right,
            });
        }
        Ok(())
    }
    fn gradient_norm(&self) -> Result<R, SolverError> {
        norm_inf(&self.model.gradient)
    }
    fn solve(&mut self) -> Result<&[R], SolverError> {
        self.prepare_damping(R::from_f64_impl(1e-6))?;
        self.solve_damped(R::zero()).map(|s| s.delta)
    }
}

impl<R: Real> DampedLeastSquaresBackend for Schur<R> {
    fn prepare_damping(&mut self, min_column_norm: R) -> Result<(), SolverError> {
        self.ready = false;
        let start = self.collect_timings.then(Instant::now);
        self.model.prepare_scales(min_column_norm)?;
        self.work
            .prepare_verification(self.model.jacobian.rows, self.model.jacobian.cols);
        self.blocks
            .prepare(&self.model.jacobian, &self.model.scales)?;
        let size = self
            .retained_width
            .checked_mul(self.eliminated_width)
            .and_then(|n| n.checked_mul(self.edges.len()))
            .ok_or(EvaluationError::DimensionMismatch)?;
        self.cross.resize(size, R::zero());
        for (e, edge) in self.edges.iter().enumerate() {
            let left = &self.model.jacobian.blocks[edge.retained_jacobian];
            let right = &self.model.jacobian.blocks[edge.eliminated_jacobian];
            for p in 0..self.eliminated_width {
                for c in 0..self.retained_width {
                    let mut sum = R::zero();
                    for row in 0..left.rows {
                        sum += (self.model.jacobian.values[left.start + c * left.rows + row]
                            / self.model.scales[left.col + c])
                            * (self.model.jacobian.values[right.start + p * right.rows + row]
                                / self.model.scales[right.col + p]);
                    }
                    self.cross[(e * self.eliminated_width + p) * self.retained_width + c] = sum;
                }
            }
        }
        norm_inf(&self.cross)?;
        self.ready = true;
        if let Some(start) = start {
            self.timings.get_mut().unwrap().preparation += start.elapsed();
        }
        Ok(())
    }
    fn solve_damped(&mut self, lambda: R) -> Result<DampedStep<'_, R>, SolverError> {
        self.diagnostics = SchurDiagnostics {
            damping: lambda,
            ..Default::default()
        };
        if !self.ready || !lambda.is_finite() || lambda < R::zero() {
            return Err(SolverError::InvalidOptions);
        }
        self.statistics.solves += 1;
        let result = self.attempt(lambda);
        if let Some(observe) = self.on_solve {
            observe(self.diagnostics);
        }
        let predicted_reduction = result?;
        Ok(DampedStep {
            delta: &self.work.step,
            predicted_reduction,
        })
    }
}

fn cross_apply<R: Real>(
    edges: &[Edge],
    values: &[R],
    shape: (usize, usize),
    mut out: MatMut<'_, R>,
    rhs: MatRef<'_, R>,
    transpose: bool,
    alpha: R,
) {
    let (cw, pw) = shape;
    for (e, edge) in edges.iter().enumerate() {
        for k in 0..rhs.ncols() {
            for p in 0..pw {
                for c in 0..cw {
                    let value = alpha * values[(e * pw + p) * cw + c];
                    if transpose {
                        out[(edge.eliminated + p, k)] += value * rhs[(edge.retained + c, k)];
                    } else {
                        out[(edge.retained + c, k)] += value * rhs[(edge.eliminated + p, k)];
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
struct Reduced<'a, R> {
    retained: usize,
    eliminated: usize,
    retained_width: usize,
    eliminated_width: usize,
    blocks: &'a BlockFactors<R>,
    edges: &'a [Edge],
    cross: &'a [R],
    lambda: R,
    timings: Option<&'a Mutex<SchurTimings>>,
}
impl<R: Real> LinOp<R> for Reduced<'_, R> {
    fn nrows(&self) -> usize {
        self.retained
    }
    fn ncols(&self) -> usize {
        self.retained
    }
    fn apply_scratch(&self, cols: usize, _: Par) -> StackReq {
        faer::linalg::temp_mat_scratch::<R>(self.eliminated, cols)
    }
    fn apply(&self, mut out: MatMut<'_, R>, rhs: MatRef<'_, R>, _: Par, stack: &mut MemStack) {
        let start = self.timings.map(|_| Instant::now());
        for k in 0..rhs.ncols() {
            for i in 0..self.retained {
                out[(i, k)] = self.lambda * rhs[(i, k)];
            }
        }
        for (offset, gram) in self.blocks.gram_blocks() {
            if offset >= self.retained {
                continue;
            }
            for k in 0..rhs.ncols() {
                for j in 0..gram.ncols() {
                    for i in 0..gram.nrows() {
                        let value = gram[(i.max(j), i.min(j))];
                        out[(offset + i, k)] += value * rhs[(offset + j, k)];
                    }
                }
            }
        }
        let (mut tmp, _) =
            faer::linalg::temp_mat_zeroed::<R, _, _>(self.eliminated, rhs.ncols(), stack);
        let mut tmp = tmp.as_mat_mut();
        cross_apply(
            self.edges,
            self.cross,
            (self.retained_width, self.eliminated_width),
            tmp.as_mut(),
            rhs,
            true,
            R::one(),
        );
        self.blocks
            .solve_normal(tmp.as_mut(), self.retained, self.lambda);
        cross_apply(
            self.edges,
            self.cross,
            (self.retained_width, self.eliminated_width),
            out,
            tmp.as_ref(),
            false,
            -R::one(),
        );
        if let (Some(t), Some(start)) = (self.timings, start) {
            t.lock().unwrap().products += start.elapsed();
        }
    }
    fn conj_apply(&self, out: MatMut<'_, R>, rhs: MatRef<'_, R>, par: Par, stack: &mut MemStack) {
        self.apply(out, rhs, par, stack);
    }
}

#[derive(Debug)]
struct RetainedPreconditioner<'a, R>(&'a Reduced<'a, R>);
impl<R: Real> LinOp<R> for RetainedPreconditioner<'_, R> {
    fn nrows(&self) -> usize {
        self.0.retained
    }
    fn ncols(&self) -> usize {
        self.0.retained
    }
    fn apply_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn apply(&self, mut out: MatMut<'_, R>, rhs: MatRef<'_, R>, _: Par, _: &mut MemStack) {
        let start = self.0.timings.map(|_| Instant::now());
        out.copy_from(rhs);
        self.0.blocks.solve_normal(out, 0, self.0.lambda);
        if let (Some(t), Some(start)) = (self.0.timings, start) {
            t.lock().unwrap().preconditioner += start.elapsed();
        }
    }
    fn conj_apply(&self, out: MatMut<'_, R>, rhs: MatRef<'_, R>, par: Par, stack: &mut MemStack) {
        self.apply(out, rhs, par, stack);
    }
}
impl<R: Real> Precond<R> for RetainedPreconditioner<'_, R> {}

#[cfg(test)]
mod tests {
    use super::*;
    use faer_ext::nalgebra::{DMatrix, DVector, SMatrix, SVector};

    #[test]
    fn schur_matches_augmented_svd_with_retries_unaries_and_unused_blocks() {
        fn check<R: Real + Into<f64>>(tolerance: f64) {
            let c = R::from_f64_impl;
            let mut backend = Schur::<R>::new(6, 2, 2);
            backend.collect_timings = true;
            backend.prepare(12).unwrap();
            backend.clear();
            let observations = [(0, 6), (2, 6), (0, 8), (2, 8), (0, 6)];
            let mut j = DMatrix::zeros(12, 12);
            let mut r = DVector::zeros(12);
            for (i, (cam, point)) in observations.into_iter().enumerate() {
                let a = SMatrix::<R, 2, 2>::new(c(0.01), c(2.), c(-0.03), c(1. + i as f64));
                let b = SMatrix::<R, 2, 2>::new(c(10.), c(-0.3), c(20. + i as f64), c(1.));
                let residual = SVector::<R, 2>::new(c(i as f64 - 2.), c(1.));
                // Reverse emission order; repeated camera-point pairs are legal.
                backend
                    .accumulate(
                        residual.as_slice(),
                        [(point, b.as_view()), (cam, a.as_view())].into_iter(),
                    )
                    .unwrap();
                j.view_mut((2 * i, cam), (2, 2))
                    .copy_from(&a.map(Into::into));
                j.view_mut((2 * i, point), (2, 2))
                    .copy_from(&b.map(Into::into));
                r.rows_mut(2 * i, 2).copy_from(&residual.map(Into::into));
            }
            for (i, col) in [0, 8].into_iter().enumerate() {
                let a = SMatrix::<R, 1, 2>::new(c(0.5), c(-1.));
                backend
                    .accumulate(&[c(0.7)], std::iter::once((col, a.as_view())))
                    .unwrap();
                j.view_mut((10 + i, col), (1, 2))
                    .copy_from(&a.map(Into::into));
                r[10 + i] = 0.7;
            }
            backend.prepare_damping(c(0.1)).unwrap();
            for lambda in [0.001_f64, 1., 100., 0.01] {
                let mut augmented = DMatrix::zeros(24, 12);
                augmented.view_mut((0, 0), (12, 12)).copy_from(&j);
                for k in 0..12 {
                    augmented[(12 + k, k)] = lambda.sqrt() * backend.model.scales[k].into();
                }
                let mut rhs = DVector::zeros(24);
                rhs.rows_mut(0, 12).copy_from(&(-&r));
                let expected = augmented.svd(true, true).solve(&rhs, 1e-13).unwrap();
                let step = backend.solve_damped(c(lambda)).unwrap();
                let actual = DVector::from_iterator(12, step.delta.iter().copied().map(Into::into));
                assert!(
                    (&actual - &expected).norm() < tolerance * expected.norm().max(1.),
                    "{} lambda={lambda} actual={actual:?} expected={expected:?}",
                    std::any::type_name::<R>()
                );
                assert!(
                    (step.predicted_reduction.into()
                        - 0.5 * (r.norm_squared() - (&j * &actual + &r).norm_squared()))
                    .abs()
                        < tolerance
                );
                for k in [4, 5, 10, 11] {
                    assert_eq!(actual[k], 0.);
                }
                let d = backend.diagnostics();
                assert_eq!(d.termination, Some(SchurTermination::Converged));
                assert!(d.reduced_relative_residual.unwrap().into() < tolerance);
                assert!(d.scaled_relative_residual.unwrap().into() < tolerance);
            }
            backend.max_iterations = 1;
            assert!(matches!(
                backend.solve_damped(c(0.001)),
                Err(SolverError::LinearSolveFailed)
            ));
            assert_eq!(
                backend.diagnostics().termination,
                Some(SchurTermination::IterationLimit)
            );
            assert_eq!(backend.diagnostics().iterations, 1);
            assert!(backend.diagnostics().estimated_relative_residual.is_none());
            assert!(
                backend
                    .diagnostics()
                    .reduced_relative_residual
                    .unwrap()
                    .is_finite()
            );
            assert!(
                backend.diagnostics().reduced_relative_residual.unwrap()
                    > backend.relative_tolerance
            );
            assert!(backend.timings().linear_solve > Duration::ZERO);
            backend.prepare(12).unwrap();
            assert_eq!(backend.statistics().solves, 0);
            assert_eq!(backend.timings().linear_solve, Duration::ZERO);
        }
        check::<f64>(1e-7);
        check::<f32>(3e-3);
    }

    #[test]
    fn partition_structure_and_singular_elimination_fail_explicitly() {
        let j = SMatrix::<f64, 1, 2>::new(1., 1.);
        let mut backend = Schur::new(2, 2, 2);
        assert!(matches!(
            backend.prepare(5),
            Err(SolverError::InvalidOptions)
        ));
        backend.prepare(6).unwrap();
        backend.clear();
        assert!(matches!(
            backend.accumulate(&[1.], [(2, j.as_view()), (4, j.as_view())].into_iter()),
            Err(EvaluationError::UnsupportedStructure)
        ));
        assert!(matches!(
            backend.accumulate(&[1.], std::iter::once((1, j.as_view()))),
            Err(EvaluationError::UnsupportedStructure)
        ));
        assert!(backend.solve_damped(0.1).is_err());
        backend
            .accumulate(&[1.], [(0, j.as_view()), (2, j.as_view())].into_iter())
            .unwrap();
        backend.prepare_damping(0.1).unwrap();
        assert!(backend.solve_damped(0.).is_err());
        assert_eq!(
            backend.diagnostics().termination,
            Some(SchurTermination::NotPositiveDefinite)
        );
        let result = backend.solve_damped(0.1).map(|s| s.delta.to_vec());
        assert!(result.is_ok(), "{result:?} {:?}", backend.diagnostics());
        assert!(backend.solve_damped(f64::NAN).is_err());
        let mut retained_coupling = Schur::new(4, 2, 2);
        retained_coupling.prepare(6).unwrap();
        assert!(matches!(
            retained_coupling.accumulate(&[1.], [(0, j.as_view()), (2, j.as_view())].into_iter()),
            Err(EvaluationError::UnsupportedStructure)
        ));
        let identity = SMatrix::<f64, 2, 2>::identity();
        let mut undamped = Schur::new(2, 2, 2);
        undamped.prepare(4).unwrap();
        for col in [0, 2] {
            undamped
                .accumulate(&[1., 2.], std::iter::once((col, identity.as_view())))
                .unwrap();
        }
        for _ in 0..2 {
            for (&actual, expected) in undamped.solve().unwrap().iter().zip([-1., -2., -1., -2.]) {
                assert!((actual - expected).abs() < 1e-10);
            }
        }
        for (retained, n) in [(0, 0), (0, 2), (2, 2)] {
            let mut b = Schur::new(retained, 2, 2);
            b.prepare(n).unwrap();
            b.clear();
            if n != 0 {
                b.accumulate(&[1.], std::iter::once((0, j.as_view())))
                    .unwrap();
            }
            b.prepare_damping(0.1).unwrap();
            assert!(b.solve_damped(0.1).is_ok());
        }
    }
}
