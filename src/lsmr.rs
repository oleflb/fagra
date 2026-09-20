use faer::{
    MatMut, MatRef, Par,
    dyn_stack::{MemBuffer, MemStack, StackReq},
    mat::AsMatMut,
    matrix_free::{BiLinOp, BiPrecond, IdentityPrecond, InitialGuessStatus, LinOp, lsmr},
};
use faer_ext::nalgebra::{DMatrixView, Dyn};
use std::{
    sync::Mutex,
    time::{Duration, Instant},
};
mod preconditioner;
use preconditioner::FactorPreconditioner;

use crate::{
    EvaluationError, Real, SolverError,
    linear_model::{
        Jacobian, LinearizedModel, StepWorkspace, block_factors::BlockFactors, norm_l2, relative,
    },
    optimization::{DampedLeastSquaresBackend, DampedStep, LeastSquaresBackend, norm_inf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Preconditioning {
    Identity,
    Diagonal,
    BlockJacobi,
}

/// Work performed since the most recent optimization/backend preparation.
#[derive(Debug, Default, Clone, Copy)]
pub struct LsmrStatistics {
    /// Nonempty linear solve attempts, including unsuccessful attempts.
    pub solves: usize,
    /// Completed inner iterations, including unsuccessful attempts.
    pub iterations: usize,
}

/// Optional cumulative wall-clock measurements since backend preparation.
/// Forward/transpose/preconditioner times are subsets of `linear_solve`.
#[derive(Debug, Default, Clone, Copy)]
pub struct LsmrTimings {
    /// Column norms and block Gram assembly across linearizations.
    pub preparation: Duration,
    /// Updating/factoring the small blocks across damping attempts.
    pub factorization: Duration,
    /// Complete iterative solves, including scratch preparation.
    pub linear_solve: Duration,
    /// Augmented forward products, including implicit diagonal scaling.
    pub forward: Duration,
    /// Augmented transpose products, including implicit diagonal scaling.
    pub transpose: Duration,
    /// Block triangular solves inside the iterative solver.
    pub preconditioner: Duration,
    /// Step recovery, prediction, and explicit residual verification.
    pub residual_checks: Duration,
}

/// Termination of the most recent linear solve (not nonlinear convergence).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LsmrTermination {
    /// Faer's relative stopping test was satisfied.
    Converged,
    /// The iteration budget was exhausted.
    IterationLimit,
    /// A step or residual diagnostic was nonfinite.
    NonFinite,
}

/// Diagnostics for the last attempt. Residuals are Euclidean normal-residual norms.
/// Explicit residuals are recomputed for damped solves, including failed attempts.
#[derive(Debug, Clone, Copy)]
pub struct LsmrDiagnostics<R: Real = f64> {
    /// None until a solve runs, or if its input controls are invalid.
    pub termination: Option<LsmrTermination>,
    /// Completed iterations in this attempt.
    pub iterations: usize,
    /// Whether the returned original-coordinate step is finite.
    pub finite_step: bool,
    /// Faer's recurrence estimate in solver coordinates.
    pub estimated_absolute_residual: Option<R>,
    /// Recurrence estimate relative to the initial solver-coordinate normal residual.
    pub estimated_relative_residual: Option<R>,
    /// ||Jᵀ(Jδ+r) + λD²δ||₂ in original coordinates.
    pub original_absolute_residual: Option<R>,
    /// Original residual divided by ||Jᵀr||₂.
    pub original_relative_residual: Option<R>,
    /// Explicit residual in solver coordinates: Pᵀ times the original normal residual.
    /// P is D⁻¹ for diagonal scaling, or D⁻¹L⁻ᵀ for block scaling.
    pub scaled_absolute_residual: Option<R>,
    /// Explicit solver-coordinate residual divided by its initial norm.
    /// When an initial norm is zero, relative fields report the absolute norm.
    pub scaled_relative_residual: Option<R>,
    /// Floored column norms used for damping and optional preconditioning.
    pub min_scale: Option<R>,
    /// Largest floored column norm for this linearization.
    pub max_scale: Option<R>,
    /// Damping in this attempt; None for an undamped solve.
    pub damping: Option<R>,
    /// Number of blocks falling back to diagonal scaling in this attempt.
    pub block_fallbacks: usize,
}

impl<R: Real> Default for LsmrDiagnostics<R> {
    fn default() -> Self {
        Self {
            termination: None,
            iterations: 0,
            finite_step: false,
            estimated_absolute_residual: None,
            estimated_relative_residual: None,
            original_absolute_residual: None,
            original_relative_residual: None,
            scaled_absolute_residual: None,
            scaled_relative_residual: None,
            min_scale: None,
            max_scale: None,
            damping: None,
            block_fallbacks: 0,
        }
    }
}

// The lower diagonal block is implicit; every operator application adds O(n)
// work and no allocations. The original Jacobian remains untouched on retries.
#[derive(Debug)]
struct Damped<'a, R> {
    jacobian: &'a Jacobian<R>,
    diagonal: &'a [R],
    scales: Option<&'a [R]>,
    timings: Option<&'a Mutex<LsmrTimings>>,
}
impl<R: Real> LinOp<R> for Damped<'_, R> {
    fn nrows(&self) -> usize {
        self.jacobian.rows + self.jacobian.cols
    }
    fn ncols(&self) -> usize {
        self.jacobian.cols
    }
    fn apply_scratch(&self, rhs_ncols: usize, _: Par) -> StackReq {
        if self.scales.is_some() {
            faer::linalg::temp_mat_scratch::<R>(self.jacobian.cols, rhs_ncols)
        } else {
            StackReq::EMPTY
        }
    }
    fn apply(&self, out: MatMut<'_, R>, rhs: MatRef<'_, R>, par: Par, stack: &mut MemStack) {
        let start = self.timings.map(|_| Instant::now());
        let (top, mut bottom) = out.split_at_row_mut(self.jacobian.rows);
        if let Some(scales) = self.scales {
            // Scale each input coordinate once, rather than once per incident
            // Jacobian block. Scratch is retained by the surrounding LSMR solve.
            let (mut scaled, stack) =
                faer::linalg::temp_mat_zeroed::<R, _, _>(self.jacobian.cols, rhs.ncols(), stack);
            let mut scaled = scaled.as_mat_mut();
            for k in 0..rhs.ncols() {
                for j in 0..self.jacobian.cols {
                    scaled[(j, k)] = rhs[(j, k)] / scales[j];
                }
            }
            self.jacobian.apply(top, scaled.as_ref(), par, stack);
        } else {
            self.jacobian.apply(top, rhs, par, stack);
        }
        for k in 0..rhs.ncols() {
            for j in 0..self.jacobian.cols {
                bottom[(j, k)] = self.diagonal[j] * rhs[(j, k)];
            }
        }
        if let (Some(clock), Some(start)) = (self.timings, start) {
            let mut t = clock.lock().unwrap();
            t.forward += start.elapsed();
        }
    }
    fn conj_apply(&self, out: MatMut<'_, R>, rhs: MatRef<'_, R>, par: Par, stack: &mut MemStack) {
        self.apply(out, rhs, par, stack);
    }
}
impl<R: Real> BiLinOp<R> for Damped<'_, R> {
    fn transpose_apply_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn transpose_apply(
        &self,
        mut out: MatMut<'_, R>,
        rhs: MatRef<'_, R>,
        par: Par,
        stack: &mut MemStack,
    ) {
        let start = self.timings.map(|_| Instant::now());
        let (top, bottom) = rhs.split_at_row(self.jacobian.rows);
        self.jacobian.transpose_apply(out.as_mut(), top, par, stack);
        for k in 0..rhs.ncols() {
            for j in 0..self.jacobian.cols {
                if let Some(scales) = self.scales {
                    out[(j, k)] /= scales[j];
                }
                out[(j, k)] += self.diagonal[j] * bottom[(j, k)];
            }
        }
        if let (Some(clock), Some(start)) = (self.timings, start) {
            let mut t = clock.lock().unwrap();
            t.transpose += start.elapsed();
        }
    }
    fn adjoint_apply(
        &self,
        out: MatMut<'_, R>,
        rhs: MatRef<'_, R>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.transpose_apply(out, rhs, par, stack);
    }
}

// Borrow disjoint workspace fields explicitly; the operator and preconditioner
// remain immutable while faer writes the step, scratch, and diagnostics.
#[allow(clippy::too_many_arguments)]
fn solve<R: Real>(
    operator: &impl BiLinOp<R>,
    precond: impl BiPrecond<R>,
    rhs: &[R],
    step: &mut [R],
    params: lsmr::LsmrParams<R>,
    scratch: &mut Option<MemBuffer>,
    stats: &mut LsmrStatistics,
    diagnostics: &mut LsmrDiagnostics<R>,
) -> Result<(), SolverError> {
    let (m, n) = (operator.nrows(), operator.ncols());
    step.fill(R::zero());
    if m == 0 || n == 0 {
        diagnostics.termination = Some(LsmrTermination::Converged);
        diagnostics.finite_step = true;
        return Ok(());
    }
    stats.solves += 1;
    // faer 0.24.4 omits wbar and vold from lsmr_scratch; both are n-by-1.
    let vector = faer::linalg::temp_mat_scratch::<R>(n, 1);
    let req = lsmr::lsmr_scratch::<R>(&precond, operator, 1, Par::Seq)
        .and(vector)
        .and(vector);
    if !scratch
        .as_mut()
        .is_some_and(|s| MemStack::new(s).can_hold(req))
    {
        *scratch = Some(MemBuffer::new(req));
    }
    let result = lsmr::lsmr(
        MatMut::from_column_major_slice_mut(step, n, 1),
        precond,
        operator,
        MatRef::from_column_major_slice(rhs, m, 1),
        params,
        |_| {
            stats.iterations += 1;
            diagnostics.iterations += 1;
        },
        Par::Seq,
        MemStack::new(scratch.as_mut().unwrap()),
    );
    let (absolute, relative, termination) = match result {
        Ok(info) => (
            info.abs_residual,
            info.rel_residual,
            LsmrTermination::Converged,
        ),
        Err(lsmr::LsmrError::NoConvergence {
            abs_residual,
            rel_residual,
        }) => (abs_residual, rel_residual, LsmrTermination::IterationLimit),
    };
    diagnostics.estimated_absolute_residual = Some(absolute);
    diagnostics.estimated_relative_residual = Some(relative);
    diagnostics.finite_step = norm_inf(step).is_ok();
    diagnostics.termination = Some(
        if diagnostics.finite_step && absolute.is_finite() && relative.is_finite() {
            termination
        } else {
            LsmrTermination::NonFinite
        },
    );
    if diagnostics.termination == Some(LsmrTermination::Converged) {
        Ok(())
    } else {
        Err(SolverError::LinearSolveFailed)
    }
}

/// Least squares solved by faer's LSMR over cached local Jacobian blocks.
///
/// Select with `GaussNewton::new(Lsmr::default())`. No global Jacobian or normal
/// matrix is assembled. Borrowed blocks are copied once per linearization;
/// operator applications never reevaluate factors. Storage is O(nnz + m + n),
/// where nnz counts all coefficients in the emitted blocks, including zeros.
/// Buffers and scratch are retained across solves and optimization calls.
///
/// Products run sequentially from a zero initial step. Damped solves use diagonal
/// right-preconditioning by default; undamped GN retains identity preconditioning.
/// Iteration exhaustion or a nonfinite step returns `LinearSolveFailed`.
pub struct Lsmr<R: Real = f64> {
    /// Maximum inner LSMR iterations per step; must be positive. Default: 1000.
    pub max_iterations: usize,
    /// Relative linearized normal-residual tolerance, in [0, 1).
    /// Defaults to `max(1e-10, 128ε)`, where `ε` is the scalar's machine epsilon.
    pub relative_tolerance: R,
    /// Scale damped solver coordinates by Jacobian column norms. Default: true.
    /// Does not change the damped objective; changes the inner stopping norm.
    /// Mode changes require a new damping preparation before solving again.
    pub diagonal_preconditioning: bool,
    /// Use variable-block Jacobi for damped solves. Default: false.
    /// Implies diagonal scaling, regardless of `diagonal_preconditioning`.
    /// Gram blocks are cached per linearization; failed Cholesky blocks fall back
    /// to diagonal scaling. Emissions must use consistent, nonoverlapping variable ranges.
    pub block_preconditioning: bool,
    /// Enable damped-solve component wall-clock measurements. Default: false.
    pub collect_timings: bool,
    /// Optional allocation-free observer after each numerically attempted solve.
    /// Called on success and exhaustion, outside faer's inner loop. Default: None.
    pub on_solve: Option<fn(LsmrDiagnostics<R>)>,
    model: LinearizedModel<R>,
    work: StepWorkspace<R>,
    scratch: Option<MemBuffer>,
    diagonal: Vec<R>,
    augmented_rhs: Vec<R>,
    blocks: BlockFactors<R>,
    timings: Mutex<LsmrTimings>,
    prepared_mode: Option<Preconditioning>,
    statistics: LsmrStatistics,
    diagnostics: LsmrDiagnostics<R>,
}

impl<R: Real> Default for Lsmr<R> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            relative_tolerance: R::from_f64_impl(1e-10)
                .max(R::epsilon_impl() * R::from_f64_impl(128.0)),
            diagonal_preconditioning: true,
            block_preconditioning: false,
            collect_timings: false,
            on_solve: None,
            model: LinearizedModel::default(),
            work: StepWorkspace::default(),
            scratch: None,
            diagonal: Vec::new(),
            augmented_rhs: Vec::new(),
            blocks: BlockFactors::default(),
            timings: Mutex::new(LsmrTimings::default()),
            prepared_mode: None,
            statistics: LsmrStatistics::default(),
            diagnostics: LsmrDiagnostics::default(),
        }
    }
}

impl<R: Real> Lsmr<R> {
    /// Inspect work performed since the last optimizer call prepared this backend.
    pub fn statistics(&self) -> LsmrStatistics {
        self.statistics
    }

    /// Last solve diagnostics, retained on both success and failure.
    pub fn diagnostics(&self) -> LsmrDiagnostics<R> {
        self.diagnostics
    }

    /// Cumulative damped-solve measurements; zero when timing is disabled.
    pub fn timings(&self) -> LsmrTimings {
        *self.timings.lock().unwrap()
    }

    fn params(&self) -> lsmr::LsmrParams<R> {
        lsmr::LsmrParams {
            initial_guess: InitialGuessStatus::Zero,
            rel_tolerance: self.relative_tolerance,
            max_iters: self.max_iterations,
            ..Default::default()
        }
    }

    fn requested_mode(&self) -> Preconditioning {
        if self.block_preconditioning {
            Preconditioning::BlockJacobi
        } else if self.diagonal_preconditioning {
            Preconditioning::Diagonal
        } else {
            Preconditioning::Identity
        }
    }

    fn factor_preconditioner(&mut self, lambda: R, mode: Preconditioning) {
        let start = self.collect_timings.then(Instant::now);
        if mode == Preconditioning::BlockJacobi {
            self.diagnostics.block_fallbacks = self.blocks.factor(lambda);
        }
        if let Some(start) = start {
            self.timings.get_mut().unwrap().factorization += start.elapsed();
        }
    }

    fn solve_damped_system(&mut self, mode: Preconditioning) -> Result<(), SolverError> {
        let params = self.params();
        let start = self.collect_timings.then(Instant::now);
        let result = solve(
            &Damped {
                jacobian: &self.model.jacobian,
                diagonal: &self.diagonal,
                scales: (mode != Preconditioning::Identity).then_some(&self.model.scales),
                timings: self.collect_timings.then_some(&self.timings),
            },
            FactorPreconditioner {
                dimension: self.model.jacobian.cols,
                factors: (mode == Preconditioning::BlockJacobi).then_some(&self.blocks),
                timings: self.collect_timings.then_some(&self.timings),
            },
            &self.augmented_rhs,
            &mut self.work.step,
            params,
            &mut self.scratch,
            &mut self.statistics,
            &mut self.diagnostics,
        );
        if let Some(start) = start {
            self.timings.get_mut().unwrap().linear_solve += start.elapsed();
        }
        result
    }

    fn recover_and_verify(&mut self, lambda: R, mode: Preconditioning) -> Result<R, SolverError> {
        let start = self.collect_timings.then(Instant::now);
        // faer has already applied the block preconditioner to its solution;
        // only diagonal scaling remains to recover original coordinates.
        if mode != Preconditioning::Identity {
            for (d, &s) in self.work.step.iter_mut().zip(&self.model.scales) {
                *d /= s;
            }
        }
        let verified = self.model.verify_step(&mut self.work, lambda);
        self.diagnostics.finite_step = verified.finite_step;
        if mode != Preconditioning::Identity {
            self.work.scale_residual(&self.model.scales);
        }
        if mode == Preconditioning::BlockJacobi {
            let n = self.model.jacobian.cols;
            self.blocks.apply_factor_inverse(
                MatMut::from_column_major_slice_mut(&mut self.work.normal_residual, n, 1),
                true,
            );
            self.blocks.apply_factor_inverse(
                MatMut::from_column_major_slice_mut(&mut self.work.reference, n, 1),
                true,
            );
        }
        let scaled_norm = norm_l2(&self.work.normal_residual);
        let scaled_reference = norm_l2(&self.work.reference);
        if let Some(start) = start {
            self.timings.get_mut().unwrap().residual_checks += start.elapsed();
        }
        self.diagnostics.original_absolute_residual = Some(verified.residual_norm);
        self.diagnostics.original_relative_residual =
            Some(relative(verified.residual_norm, verified.reference_norm));
        self.diagnostics.scaled_absolute_residual = Some(scaled_norm);
        self.diagnostics.scaled_relative_residual = Some(relative(scaled_norm, scaled_reference));
        if !verified.finite_step || !verified.residual_norm.is_finite() || !scaled_norm.is_finite()
        {
            self.diagnostics.termination = Some(LsmrTermination::NonFinite);
            return Err(SolverError::LinearSolveFailed);
        }
        Ok(verified.predicted_reduction)
    }
}

impl<R: Real> LeastSquaresBackend for Lsmr<R> {
    type Scalar = R;

    fn prepare(&mut self, dimension: usize) -> Result<(), SolverError> {
        self.statistics = LsmrStatistics::default();
        *self.timings.get_mut().unwrap() = LsmrTimings::default();
        self.diagnostics = LsmrDiagnostics::default();
        self.prepared_mode = None;
        if self.max_iterations == 0 || !(R::zero()..R::one()).contains(&self.relative_tolerance) {
            return Err(SolverError::InvalidOptions);
        }
        self.model.prepare(dimension);
        self.work.step.resize(dimension, R::zero());
        Ok(())
    }

    fn clear(&mut self) {
        self.prepared_mode = None;
        self.model.clear();
    }

    fn accumulate<'a>(
        &mut self,
        residual: &[R],
        jacobians: impl Iterator<Item = (usize, DMatrixView<'a, R, Dyn, Dyn>)> + Clone,
    ) -> Result<(), EvaluationError> {
        self.prepared_mode = None;
        self.model.accumulate(residual, jacobians)
    }

    fn gradient_norm(&self) -> Result<R, SolverError> {
        norm_inf(&self.model.gradient)
    }

    fn solve(&mut self) -> Result<&[R], SolverError> {
        self.diagnostics = LsmrDiagnostics::default();
        let params = self.params();
        let result = solve(
            &self.model.jacobian,
            IdentityPrecond {
                dim: self.model.jacobian.cols,
            },
            &self.model.negative_residual,
            &mut self.work.step,
            params,
            &mut self.scratch,
            &mut self.statistics,
            &mut self.diagnostics,
        );
        if let Some(observe) = self.on_solve {
            observe(self.diagnostics);
        }
        result?;
        Ok(&self.work.step)
    }
}

impl<R: Real> crate::covariance::CovarianceBackend for Lsmr<R> {
    fn covariance(
        &mut self,
        normal: &mut crate::DenseNormalCholesky<R>,
        selected: &mut crate::covariance::SelectedCovariance<R>,
    ) -> Result<(), SolverError> {
        self.model.covariance(normal, selected)
    }
}

impl<R: Real> DampedLeastSquaresBackend for Lsmr<R> {
    fn prepare_damping(&mut self, min_column_norm: R) -> Result<(), SolverError> {
        let start = self.collect_timings.then(Instant::now);
        self.prepared_mode = None;
        self.model.prepare_scales(min_column_norm)?;
        let mode = self.requested_mode();
        let (m, n) = (self.model.jacobian.rows, self.model.jacobian.cols);
        self.diagonal.resize(n, R::zero());
        self.work.prepare_verification(m, n);
        if mode == Preconditioning::BlockJacobi {
            self.blocks
                .prepare(&self.model.jacobian, &self.model.scales)?;
        }
        self.augmented_rhs.resize(
            m.checked_add(n).ok_or(EvaluationError::DimensionMismatch)?,
            R::zero(),
        );
        self.augmented_rhs[..m].copy_from_slice(&self.model.negative_residual);
        self.augmented_rhs[m..].fill(R::zero());
        self.prepared_mode = Some(mode);
        if let Some(start) = start {
            let t = self.timings.get_mut().unwrap();
            t.preparation += start.elapsed();
        }
        Ok(())
    }

    fn solve_damped(&mut self, lambda: R) -> Result<DampedStep<'_, R>, SolverError> {
        self.diagnostics = LsmrDiagnostics::default();
        let mode = self.prepared_mode.ok_or(SolverError::InvalidOptions)?;
        if mode != self.requested_mode() || !lambda.is_finite() || lambda < R::zero() {
            return Err(SolverError::InvalidOptions);
        }
        let root = lambda.sqrt();
        let scaled = mode != Preconditioning::Identity;
        self.diagnostics.damping = Some(lambda);
        self.diagnostics.min_scale = self.model.scales.iter().copied().reduce(|a, b| a.min(b));
        self.diagnostics.max_scale = self.model.scales.iter().copied().reduce(|a, b| a.max(b));
        for (d, &s) in self.diagonal.iter_mut().zip(&self.model.scales) {
            *d = if scaled { root } else { root * s };
        }
        if norm_inf(&self.diagonal).is_err() {
            self.diagnostics.termination = Some(LsmrTermination::NonFinite);
            return Err(SolverError::LinearSolveFailed);
        }
        self.factor_preconditioner(lambda, mode);
        let result = self.solve_damped_system(mode);
        // Even exhausted solves get independent residual diagnostics. Verification
        // failures take precedence over the iterative termination status.
        let verified = self.recover_and_verify(lambda, mode);
        if let Some(observe) = self.on_solve {
            observe(self.diagnostics);
        }
        let predicted_reduction = verified?;
        result?;
        Ok(DampedStep {
            delta: &self.work.step,
            predicted_reduction,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DenseNormalCholesky, JacobianBlock, Variable, storage::StatePool};
    use faer::Mat;
    use faer_ext::nalgebra::{SMatrix, SVector};

    type Pair = crate::variable::test_support::Vector<2>;

    #[test]
    fn damped_products_match_dense_with_shared_columns_and_multiple_rhs() {
        use faer_ext::nalgebra::DMatrix;
        fn check<R: Real + Into<f64>>(tolerance: f64) {
            let c = R::from_f64_impl;
            let j = SMatrix::<R, 4, 3>::from_row_slice(
                &[
                    0.001, 2., 3., 0.002, -1., 4., -0.001, 3., -2., 0.003, 1., 2.,
                ]
                .map(c),
            );
            let mut backend = Lsmr::<R>::default();
            backend.prepare(3).unwrap();
            for row in [0, 2] {
                backend
                    .accumulate(
                        &[c(0.), c(0.)],
                        std::iter::once((0, j.fixed_rows::<2>(row).as_view())),
                    )
                    .unwrap();
            }
            backend.prepare_damping(c(0.1)).unwrap();
            let x = Mat::from_fn(3, 2, |i, k| c((i + k) as f64 - 1.25));
            let y = Mat::from_fn(7, 2, |i, k| c((2 * i + k) as f64 - 2.5));
            for scaled in [false, true] {
                for lambda in [0.0_f64, 0.01, 1e8] {
                    let diagonal: Vec<_> = backend
                        .model
                        .scales
                        .iter()
                        .map(|&s| {
                            if scaled {
                                c(lambda.sqrt())
                            } else {
                                c(lambda.sqrt()) * s
                            }
                        })
                        .collect();
                    let operator = Damped {
                        jacobian: &backend.model.jacobian,
                        diagonal: &diagonal,
                        scales: scaled.then_some(&backend.model.scales),
                        timings: None,
                    };
                    let dense = DMatrix::from_fn(7, 3, |i, k| {
                        if i < 4 {
                            j[(i, k)].into()
                                / if scaled {
                                    backend.model.scales[k].into()
                                } else {
                                    1.
                                }
                        } else if i - 4 == k {
                            diagonal[k].into()
                        } else {
                            0.
                        }
                    });
                    let expected = &dense * DMatrix::from_fn(3, 2, |i, k| x[(i, k)].into());
                    let expected_transpose =
                        dense.transpose() * DMatrix::from_fn(7, 2, |i, k| y[(i, k)].into());
                    let mut forward = Mat::zeros(7, 2);
                    let mut transpose = Mat::zeros(3, 2);
                    let mut scratch = MemBuffer::new(operator.apply_scratch(2, Par::Seq));
                    for conjugate in [false, true] {
                        if conjugate {
                            operator.conj_apply(
                                forward.as_mut(),
                                x.as_ref(),
                                Par::Seq,
                                MemStack::new(&mut scratch),
                            );
                            operator.adjoint_apply(
                                transpose.as_mut(),
                                y.as_ref(),
                                Par::Seq,
                                MemStack::new(&mut scratch),
                            );
                        } else {
                            operator.apply(
                                forward.as_mut(),
                                x.as_ref(),
                                Par::Seq,
                                MemStack::new(&mut scratch),
                            );
                            operator.transpose_apply(
                                transpose.as_mut(),
                                y.as_ref(),
                                Par::Seq,
                                MemStack::new(&mut scratch),
                            );
                        }
                        for k in 0..2 {
                            for i in 0..7 {
                                assert!(
                                    (forward[(i, k)].into() - expected[(i, k)]).abs()
                                        < tolerance * expected[(i, k)].abs().max(1.)
                                );
                            }
                            for i in 0..3 {
                                assert!(
                                    (transpose[(i, k)].into() - expected_transpose[(i, k)]).abs()
                                        < tolerance * expected_transpose[(i, k)].abs().max(1.)
                                );
                            }
                        }
                    }
                }
            }
        }
        check::<f64>(1e-12);
        check::<f32>(1e-5);
    }

    #[test]
    fn prepared_modes_reject_stale_configuration_and_observe_attempts_once() {
        std::thread_local! { static OBSERVED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
        OBSERVED.with(|n| n.set(0));
        let mut backend = Lsmr::default();
        backend.block_preconditioning = true;
        backend.on_solve = Some(|_| OBSERVED.with(|n| n.set(n.get() + 1)));
        backend.prepare(2).unwrap();
        let j = SMatrix::<f64, 2, 2>::new(1., 0., 0., 3.);
        backend
            .accumulate(&[-1., -1.], std::iter::once((0, j.as_view())))
            .unwrap();
        backend.prepare_damping(0.1).unwrap();
        backend.block_preconditioning = false;
        assert!(matches!(
            backend.solve_damped(1.),
            Err(SolverError::InvalidOptions)
        ));
        assert_eq!(OBSERVED.with(|n| n.get()), 0);
        backend.prepare_damping(0.1).unwrap();
        let step = backend.solve_damped(1.).unwrap().delta;
        assert!((step[0] - 0.5).abs() < 1e-12 && (step[1] - 1. / 6.).abs() < 1e-12);
        assert_eq!(OBSERVED.with(|n| n.get()), 1);
        backend.diagonal_preconditioning = false;
        assert!(matches!(
            backend.solve_damped(1.),
            Err(SolverError::InvalidOptions)
        ));
        backend.prepare_damping(0.1).unwrap();
        backend.max_iterations = 1;
        assert!(matches!(
            backend.solve_damped(1.),
            Err(SolverError::LinearSolveFailed)
        ));
        assert_eq!(
            backend.diagnostics().termination,
            Some(LsmrTermination::IterationLimit)
        );
        assert_eq!(OBSERVED.with(|n| n.get()), 2);
        backend.clear();
        assert!(matches!(
            backend.solve_damped(1.),
            Err(SolverError::InvalidOptions)
        ));
    }

    #[test]
    fn mixed_scale_damping_matches_svd_and_exposes_exhaustion() {
        use faer_ext::nalgebra::{DMatrix, DVector};
        fn check<R: Real + Into<f64>>(tolerance: f64, block: bool) {
            let c = R::from_f64_impl;
            let j = SMatrix::<R, 3, 4>::from_row_slice(
                &[1e-4, 2., 0., 0., 2e-4, -1., 3e4, 0., 0., 1., -1e4, 0.].map(c),
            );
            let r = SVector::<R, 3>::from_row_slice(&[1., -2., 3.].map(c));
            let mut backend = Lsmr::<R>::default();
            backend.block_preconditioning = block;
            backend.prepare(4).unwrap();
            backend
                .accumulate(r.as_slice(), std::iter::once((0, j.as_view())))
                .unwrap();
            backend.prepare_damping(c(1e-3)).unwrap();
            for lambda in [0.1, 10., 1e-3, 1e6] {
                let mut augmented = DMatrix::<f64>::zeros(7, 4);
                augmented
                    .view_mut((0, 0), (3, 4))
                    .copy_from(&j.map(Into::into));
                for k in 0..4 {
                    augmented[(3 + k, k)] = f64::sqrt(lambda) * backend.model.scales[k].into();
                }
                let mut rhs = DVector::zeros(7);
                rhs.rows_mut(0, 3).copy_from(&(-r.map(Into::into)));
                let expected = augmented.svd(true, true).solve(&rhs, 1e-14).unwrap();
                let step = backend.solve_damped(c(lambda)).unwrap();
                let actual = DVector::from_iterator(4, step.delta.iter().copied().map(Into::into));
                for k in 0..4 {
                    assert!(
                        (actual[k] - expected[k]).abs() < tolerance * expected[k].abs().max(1.),
                        "lambda={lambda}, column={k}"
                    );
                }
                assert_eq!(actual[3], 0.);
                let prediction = 0.5
                    * (r.map(Into::into).norm_squared()
                        - (j.map(Into::into) * actual + r.map(Into::into)).norm_squared());
                assert!((step.predicted_reduction.into() - prediction).abs() < tolerance);
                let d = backend.diagnostics();
                assert_eq!(d.termination, Some(LsmrTermination::Converged));
                assert!(d.finite_step);
                assert!(d.scaled_relative_residual.unwrap().into() < tolerance);
                assert!(
                    (d.scaled_relative_residual.unwrap().into()
                        - d.estimated_relative_residual.unwrap().into())
                    .abs()
                        < tolerance
                );
                assert!(d.original_relative_residual.unwrap().into() < tolerance);
            }
            backend.block_preconditioning = false;
            backend.prepare_damping(c(1e-3)).unwrap();
            backend.max_iterations = 1;
            assert!(matches!(
                backend.solve_damped(c(1e-3)),
                Err(SolverError::LinearSolveFailed)
            ));
            let d = backend.diagnostics();
            assert_eq!(d.termination, Some(LsmrTermination::IterationLimit));
            assert_eq!(d.iterations, 1);
            assert!(d.finite_step && d.original_relative_residual.unwrap().is_finite());
            assert!(d.scaled_relative_residual.unwrap() > backend.relative_tolerance);
            backend.prepare(4).unwrap();
            assert!(backend.diagnostics().termination.is_none());
        }
        for block in [false, true] {
            check::<f64>(1e-7, block);
            check::<f32>(2e-3, block);
        }
    }

    #[test]
    fn damped_retries_match_augmented_svd_and_preserve_the_model() {
        use faer_ext::nalgebra::{DMatrix, DVector};
        fn check<R: Real + Into<f64>>(tolerance: f64, precondition: bool, block: bool) {
            let c = R::from_f64_impl;
            let j =
                SMatrix::<R, 3, 3>::from_row_slice(&[1., 2., 0., 3., -1., 0., 0., 1., 0.].map(c));
            let residual = SVector::<R, 3>::from_column_slice(&[-3., 2., 1.].map(c));
            let mut backend = Lsmr::<R>::default();
            backend.diagonal_preconditioning = precondition;
            backend.block_preconditioning = block;
            backend.prepare(3).unwrap();
            backend
                .accumulate(residual.as_slice(), std::iter::once((0, j.as_view())))
                .unwrap();
            backend.prepare_damping(c(1.0)).unwrap();
            let plain = backend.solve().unwrap().to_vec();
            for lambda in [1e-8_f64, 0.01, 1.0, 100.0, 1e-4, 0.0] {
                let j = j.map(Into::<f64>::into);
                let r = residual.map(Into::<f64>::into);
                let mut augmented = DMatrix::zeros(6, 3);
                augmented.view_mut((0, 0), (3, 3)).copy_from(&j);
                for col in 0..3 {
                    augmented[(3 + col, col)] = lambda.sqrt() * j.column(col).norm().max(1.0);
                }
                let mut rhs = DVector::zeros(6);
                rhs.rows_mut(0, 3).copy_from(&(-r));
                let expected = augmented.svd(true, true).solve(&rhs, 1e-12).unwrap();
                let step = backend.solve_damped(c(lambda)).unwrap();
                let actual = SVector::<f64, 3>::from_iterator(step.delta.iter().map(|&x| x.into()));
                assert!(
                    (actual - &expected).norm() < tolerance,
                    "{} lambda={lambda} actual={actual:?} expected={expected:?}",
                    std::any::type_name::<R>()
                );
                let prediction = 0.5 * (r.norm_squared() - (j * actual + r).norm_squared());
                assert!((step.predicted_reduction.into() - prediction).abs() < tolerance);
            }
            for (&a, &b) in backend.solve().unwrap().iter().zip(&plain) {
                assert!((a - b).abs() < c(tolerance));
            }
            assert_eq!(backend.statistics().solves, 8);
            assert!(backend.statistics().iterations > 0);
        }
        for (precondition, block) in [(false, false), (true, false), (true, true)] {
            check::<f64>(1e-9, precondition, block);
            check::<f32>(3e-4, precondition, block);
        }
    }

    #[test]
    fn block_preconditioning_handles_coupling_gaps_and_adjoint_products() {
        use faer_ext::nalgebra::{DMatrix, DVector};
        let j = SMatrix::<f64, 4, 5>::from_row_slice(&[
            1., 2., 0., 4., 2., 3., 1., 0., 2., 1., 2., 1., 0., 1., 3., 1., 1., 0., 3., 2.,
        ]);
        let r = SVector::<f64, 4>::new(1., -2., 3., -1.);
        let mut backend = Lsmr::default();
        backend.block_preconditioning = true;
        backend.collect_timings = true;
        backend.prepare(5).unwrap();
        backend
            .accumulate(
                r.as_slice(),
                [
                    (3, j.fixed_columns::<2>(3).as_view()),
                    (0, j.fixed_columns::<2>(0).as_view()),
                ]
                .into_iter(),
            )
            .unwrap();
        backend.prepare_damping(0.1).unwrap();
        for lambda in [0.01, 1., 100.] {
            let mut augmented = DMatrix::zeros(9, 5);
            augmented.view_mut((0, 0), (4, 5)).copy_from(&j);
            for k in 0..5 {
                augmented[(4 + k, k)] = f64::sqrt(lambda) * backend.model.scales[k];
            }
            let mut rhs = DVector::zeros(9);
            rhs.rows_mut(0, 4).copy_from(&(-r));
            let expected = augmented.svd(true, true).solve(&rhs, 1e-12).unwrap();
            let actual = backend.solve_damped(lambda).unwrap().delta;
            assert!((DVector::from_column_slice(actual) - expected).norm() < 1e-8);
            assert_eq!(actual[2], 0.);
            assert_eq!(backend.diagnostics().block_fallbacks, 0);
            assert!(backend.diagnostics().scaled_relative_residual.unwrap() < 1e-8);
            let x = Mat::from_fn(5, 2, |i, k| (i + 2 * k) as f64 - 2.);
            let y = Mat::from_fn(5, 2, |i, k| (2 * i + k) as f64 + 1.);
            let mut px = Mat::zeros(5, 2);
            let mut pty = Mat::zeros(5, 2);
            let preconditioner = FactorPreconditioner {
                dimension: 5,
                factors: Some(&backend.blocks),
                timings: None,
            };
            preconditioner.apply(px.as_mut(), x.as_ref(), Par::Seq, MemStack::new(&mut []));
            preconditioner.transpose_apply(
                pty.as_mut(),
                y.as_ref(),
                Par::Seq,
                MemStack::new(&mut []),
            );
            for k in 0..2 {
                let lhs: f64 = (0..5).map(|i| px[(i, k)] * y[(i, k)]).sum();
                let rhs: f64 = (0..5).map(|i| x[(i, k)] * pty[(i, k)]).sum();
                assert!((lhs - rhs).abs() < 1e-10);
            }
        }
        assert!(backend.timings().linear_solve > Duration::ZERO);
        assert!(backend.timings().preconditioner > Duration::ZERO);
        backend.prepare(0).unwrap();
        assert_eq!(backend.timings().linear_solve, Duration::ZERO);
    }

    #[test]
    fn workspace_handles_large_coordinate_vectors() {
        let mut solver = Lsmr::default();
        let mut states = StatePool::default();
        let jacobian = SMatrix::<f64, 2, 2>::identity();
        for n in [300, 1024] {
            solver.prepare(n).unwrap();
            solver.clear();
            for col in (0..n).step_by(2) {
                let _key = states.insert(Pair::identity());
                let residual = SVector::<f64, 2>::new(-(col as f64), -(col as f64 + 1.0));
                solver
                    .accumulate(
                        residual.as_slice(),
                        std::iter::once((col, jacobian.as_view())),
                    )
                    .unwrap();
            }
            for (i, &value) in solver.solve().unwrap().iter().enumerate() {
                assert!((value - i as f64).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn cached_products_and_rectangular_solve_match_dense() {
        let mut states = StatePool::default();
        let a = states.insert(Pair::identity());
        let b = states.insert(Pair::identity());
        let mut solver = Lsmr::default();
        let mut dense = DenseNormalCholesky::default();
        solver.prepare(0).unwrap();
        assert!(solver.solve().unwrap().is_empty());
        solver.prepare(4).unwrap();
        solver.clear();
        assert_eq!(solver.solve().unwrap(), &[0.0; 4]);
        dense.prepare(4).unwrap();
        for _ in 0..2 {
            solver.clear();
            dense.clear();
            // Stack-local, noncontiguous input views must be copied correctly.
            {
                let storage = SMatrix::<f64, 5, 4>::from_fn(|i, j| (i * 4 + j) as f64);
                let ja = storage.fixed_view::<3, 2>(1, 1);
                let jb = SMatrix::<f64, 3, 2>::from_row_slice(&[1., 0., 0., 2., 3., -1.]);
                let blocks = [JacobianBlock::new(a, &ja), JacobianBlock::new(b, &jb)];
                let r = SVector::<f64, 3>::new(1., -2., 3.);
                solver
                    .accumulate(
                        r.as_slice(),
                        blocks.iter().zip([2, 0]).map(|(b, c)| (c, b.jacobian())),
                    )
                    .unwrap();
                dense
                    .accumulate(
                        r.as_slice(),
                        blocks.iter().zip([2, 0]).map(|(b, c)| (c, b.jacobian())),
                    )
                    .unwrap();
                let jb = SMatrix::<f64, 2, 2>::new(2., 0., 0., 1.);
                let r = SVector::<f64, 2>::new(-4., 5.);
                let blocks = [JacobianBlock::new(b, &jb)];
                solver
                    .accumulate(r.as_slice(), blocks.iter().map(|b| (0, b.jacobian())))
                    .unwrap();
                dense
                    .accumulate(r.as_slice(), blocks.iter().map(|b| (0, b.jacobian())))
                    .unwrap();
            }
            let j = faer::mat![
                [1., 0., 5., 6.],
                [0., 2., 9., 10.],
                [3., -1., 13., 14.],
                [2., 0., 0., 0.],
                [0., 1., 0., 0.],
            ];
            let x = Mat::from_fn(4, 2, |i, k| (i + 2 * k) as f64);
            let y = Mat::from_fn(5, 2, |i, k| (2 * i + k) as f64);
            let mut forward = Mat::zeros(5, 2);
            let mut transpose = Mat::zeros(4, 2);
            let mut memory = MemBuffer::new(StackReq::EMPTY);
            let stack = MemStack::new(&mut memory);
            for conjugate in [false, true] {
                forward.fill(f64::NAN);
                transpose.fill(f64::NAN);
                if conjugate {
                    solver
                        .model
                        .jacobian
                        .conj_apply(forward.as_mut(), x.as_ref(), Par::Seq, stack);
                    solver.model.jacobian.adjoint_apply(
                        transpose.as_mut(),
                        y.as_ref(),
                        Par::Seq,
                        stack,
                    );
                } else {
                    solver
                        .model
                        .jacobian
                        .apply(forward.as_mut(), x.as_ref(), Par::Seq, stack);
                    solver.model.jacobian.transpose_apply(
                        transpose.as_mut(),
                        y.as_ref(),
                        Par::Seq,
                        stack,
                    );
                }
                for k in 0..2 {
                    for i in 0..5 {
                        assert_eq!(
                            forward[(i, k)],
                            (0..4).map(|c| j[(i, c)] * x[(c, k)]).sum::<f64>()
                        );
                    }
                    for c in 0..4 {
                        assert_eq!(
                            transpose[(c, k)],
                            (0..5).map(|i| j[(i, c)] * y[(i, k)]).sum::<f64>()
                        );
                    }
                }
            }
            assert_eq!(
                solver.gradient_norm().unwrap(),
                dense.gradient_norm().unwrap()
            );
            solver.max_iterations = 1;
            assert!(matches!(
                solver.solve(),
                Err(SolverError::LinearSolveFailed)
            ));
            solver.max_iterations = 1000;
            let expected = dense.solve().unwrap();
            for (&actual, &expected) in solver.solve().unwrap().iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
            }
            let scratch = solver.scratch.as_ref().unwrap().as_ptr();
            // Repeated solves use the same frozen model and retained scratch.
            for (&actual, &expected) in solver.solve().unwrap().iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-8);
            }
            assert_eq!(scratch, solver.scratch.as_ref().unwrap().as_ptr());
        }
        for tolerance in [f64::NAN, f64::INFINITY, -1.0, 1.0] {
            solver.relative_tolerance = tolerance;
            assert!(matches!(
                solver.prepare(4),
                Err(SolverError::InvalidOptions)
            ));
        }
    }
}
