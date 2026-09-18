use faer::{
    MatMut, MatRef, Par,
    dyn_stack::{MemBuffer, MemStack, StackReq},
    matrix_free::{BiLinOp, IdentityPrecond, InitialGuessStatus, LinOp, lsmr},
};
use faer_ext::nalgebra::{DMatrixView, Dyn};

use crate::{
    EvaluationError, Real, SolverError,
    optimization::{DampedLeastSquaresBackend, DampedStep, LeastSquaresBackend, norm_inf},
};

#[derive(Debug)]
struct Block {
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
    start: usize,
}

#[derive(Debug)]
struct Jacobian<R> {
    rows: usize,
    cols: usize,
    blocks: Vec<Block>,
    values: Vec<R>,
}

impl<R: Real> LinOp<R> for Jacobian<R> {
    fn nrows(&self) -> usize {
        self.rows
    }
    fn ncols(&self) -> usize {
        self.cols
    }
    fn apply_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn apply(&self, mut out: MatMut<'_, R>, rhs: MatRef<'_, R>, _: Par, _: &mut MemStack) {
        assert_eq!(
            (out.nrows(), rhs.nrows(), out.ncols()),
            (self.rows, self.cols, rhs.ncols())
        );
        out.fill(R::zero());
        for b in &self.blocks {
            for k in 0..rhs.ncols() {
                for j in 0..b.cols {
                    let x = rhs[(b.col + j, k)];
                    for i in 0..b.rows {
                        out[(b.row + i, k)] += self.values[b.start + j * b.rows + i] * x;
                    }
                }
            }
        }
    }
    fn conj_apply(&self, out: MatMut<'_, R>, rhs: MatRef<'_, R>, par: Par, stack: &mut MemStack) {
        self.apply(out, rhs, par, stack);
    }
}

impl<R: Real> BiLinOp<R> for Jacobian<R> {
    fn transpose_apply_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn transpose_apply(
        &self,
        mut out: MatMut<'_, R>,
        rhs: MatRef<'_, R>,
        _: Par,
        _: &mut MemStack,
    ) {
        assert_eq!(
            (out.nrows(), rhs.nrows(), out.ncols()),
            (self.cols, self.rows, rhs.ncols())
        );
        out.fill(R::zero());
        for b in &self.blocks {
            for k in 0..rhs.ncols() {
                for j in 0..b.cols {
                    let mut sum = R::zero();
                    for i in 0..b.rows {
                        sum += self.values[b.start + j * b.rows + i] * rhs[(b.row + i, k)];
                    }
                    out[(b.col + j, k)] += sum;
                }
            }
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

/// Work performed since the most recent optimization/backend preparation.
#[derive(Debug, Default, Clone, Copy)]
pub struct LsmrStatistics {
    /// Nonempty linear solve attempts, including unsuccessful attempts.
    pub solves: usize,
    /// Completed inner iterations, including unsuccessful attempts.
    pub iterations: usize,
}

// The lower diagonal block is implicit; every operator application adds O(n)
// work and no allocations. The original Jacobian remains untouched on retries.
#[derive(Debug)]
struct Damped<'a, R> {
    jacobian: &'a Jacobian<R>,
    diagonal: &'a [R],
}
impl<R: Real> LinOp<R> for Damped<'_, R> {
    fn nrows(&self) -> usize {
        self.jacobian.rows + self.jacobian.cols
    }
    fn ncols(&self) -> usize {
        self.jacobian.cols
    }
    fn apply_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn apply(&self, out: MatMut<'_, R>, rhs: MatRef<'_, R>, par: Par, stack: &mut MemStack) {
        let (top, mut bottom) = out.split_at_row_mut(self.jacobian.rows);
        self.jacobian.apply(top, rhs, par, stack);
        for k in 0..rhs.ncols() {
            for j in 0..self.jacobian.cols {
                bottom[(j, k)] = self.diagonal[j] * rhs[(j, k)];
            }
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
        let (top, bottom) = rhs.split_at_row(self.jacobian.rows);
        self.jacobian.transpose_apply(out.as_mut(), top, par, stack);
        for k in 0..rhs.ncols() {
            for j in 0..self.jacobian.cols {
                out[(j, k)] += self.diagonal[j] * bottom[(j, k)];
            }
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

fn solve<R: Real>(
    operator: &impl BiLinOp<R>,
    rhs: &[R],
    step: &mut [R],
    params: lsmr::LsmrParams<R>,
    scratch: &mut Option<MemBuffer>,
    stats: &mut LsmrStatistics,
) -> Result<(), SolverError> {
    let (m, n) = (operator.nrows(), operator.ncols());
    step.fill(R::zero());
    if m == 0 || n == 0 {
        return Ok(());
    }
    stats.solves += 1;
    let precond = IdentityPrecond { dim: n };
    // faer 0.24.4 omits wbar and vold from lsmr_scratch; both are n-by-1.
    let vector = faer::linalg::temp_mat_scratch::<R>(n, 1);
    let req = lsmr::lsmr_scratch::<R>(precond, operator, 1, Par::Seq)
        .and(vector)
        .and(vector);
    if !scratch
        .as_mut()
        .is_some_and(|s| MemStack::new(s).can_hold(req))
    {
        *scratch = Some(MemBuffer::new(req));
    }
    lsmr::lsmr(
        MatMut::from_column_major_slice_mut(step, n, 1),
        precond,
        operator,
        MatRef::from_column_major_slice(rhs, m, 1),
        params,
        |_| stats.iterations += 1,
        Par::Seq,
        MemStack::new(scratch.as_mut().unwrap()),
    )
    .map_err(|_| SolverError::LinearSolveFailed)?;
    norm_inf(step).map_err(|_| SolverError::LinearSolveFailed)?;
    Ok(())
}

/// Least squares solved by faer's LSMR over cached local Jacobian blocks.
///
/// Select with `GaussNewton::new(Lsmr::default())`. No global Jacobian or normal
/// matrix is assembled. Borrowed blocks are copied once per linearization;
/// operator applications never reevaluate factors. Storage is O(nnz + m + n),
/// where nnz counts all coefficients in the emitted blocks, including zeros.
/// Buffers and scratch are retained across solves and optimization calls.
///
/// Products run sequentially, with an identity preconditioner and a zero initial
/// step. Iteration exhaustion or a nonfinite step returns `LinearSolveFailed`.
pub struct Lsmr<R: Real = f64> {
    /// Maximum inner LSMR iterations per step; must be positive. Default: 1000.
    pub max_iterations: usize,
    /// Relative linearized normal-residual tolerance, in [0, 1).
    /// Defaults to `max(1e-10, 128ε)`, where `ε` is the scalar's machine epsilon.
    pub relative_tolerance: R,
    jacobian: Jacobian<R>,
    rhs: Vec<R>,
    gradient: Vec<R>,
    step: Vec<R>,
    scratch: Option<MemBuffer>,
    scales: Vec<R>,
    diagonal: Vec<R>,
    augmented_rhs: Vec<R>,
    product: Vec<R>,
    damping_ready: bool,
    statistics: LsmrStatistics,
}

impl<R: Real> Default for Lsmr<R> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            relative_tolerance: R::from_f64_impl(1e-10)
                .max(R::epsilon_impl() * R::from_f64_impl(128.0)),
            jacobian: Jacobian {
                rows: 0,
                cols: 0,
                blocks: Vec::new(),
                values: Vec::new(),
            },
            rhs: Vec::new(),
            gradient: Vec::new(),
            step: Vec::new(),
            scratch: None,
            scales: Vec::new(),
            diagonal: Vec::new(),
            augmented_rhs: Vec::new(),
            product: Vec::new(),
            damping_ready: false,
            statistics: LsmrStatistics::default(),
        }
    }
}

impl<R: Real> Lsmr<R> {
    /// Inspect work performed since the last optimizer call prepared this backend.
    pub fn statistics(&self) -> LsmrStatistics {
        self.statistics
    }

    fn params(&self) -> lsmr::LsmrParams<R> {
        lsmr::LsmrParams {
            initial_guess: InitialGuessStatus::Zero,
            rel_tolerance: self.relative_tolerance,
            max_iters: self.max_iterations,
            ..Default::default()
        }
    }
}

impl<R: Real> LeastSquaresBackend for Lsmr<R> {
    type Scalar = R;

    fn prepare(&mut self, dimension: usize) -> Result<(), SolverError> {
        self.statistics = LsmrStatistics::default();
        self.damping_ready = false;
        if self.max_iterations == 0 || !(R::zero()..R::one()).contains(&self.relative_tolerance) {
            return Err(SolverError::InvalidOptions);
        }
        self.jacobian.cols = dimension;
        self.gradient.resize(dimension, R::zero());
        self.step.resize(dimension, R::zero());
        Ok(())
    }

    fn clear(&mut self) {
        self.damping_ready = false;
        self.jacobian.rows = 0;
        self.jacobian.blocks.clear();
        self.jacobian.values.clear();
        self.rhs.clear();
        self.gradient.fill(R::zero());
    }

    fn accumulate<'a>(
        &mut self,
        residual: &[R],
        jacobians: impl Iterator<Item = (usize, DMatrixView<'a, R, Dyn, Dyn>)> + Clone,
    ) -> Result<(), EvaluationError> {
        self.damping_ready = false;
        let row = self.jacobian.rows;
        self.jacobian.rows = row
            .checked_add(residual.len())
            .ok_or(EvaluationError::DimensionMismatch)?;
        self.rhs.extend(residual.iter().map(|&r| -r));
        for (col, matrix) in jacobians {
            self.jacobian.blocks.push(Block {
                row,
                col,
                rows: residual.len(),
                cols: matrix.ncols(),
                start: self.jacobian.values.len(),
            });
            // Pack explicitly: emitted views may have arbitrary strides.
            for j in 0..matrix.ncols() {
                let mut gradient = R::zero();
                for i in 0..residual.len() {
                    let value = matrix[(i, j)];
                    self.jacobian.values.push(value);
                    gradient += value * residual[i];
                }
                self.gradient[col + j] += gradient;
            }
        }
        Ok(())
    }

    fn gradient_norm(&self) -> Result<R, SolverError> {
        norm_inf(&self.gradient)
    }

    fn solve(&mut self) -> Result<&[R], SolverError> {
        let params = self.params();
        solve(
            &self.jacobian,
            &self.rhs,
            &mut self.step,
            params,
            &mut self.scratch,
            &mut self.statistics,
        )?;
        Ok(&self.step)
    }
}

impl<R: Real> DampedLeastSquaresBackend for Lsmr<R> {
    fn prepare_damping(&mut self, min_column_norm: R) -> Result<(), SolverError> {
        self.damping_ready = false;
        if !min_column_norm.is_finite() || min_column_norm <= R::zero() {
            return Err(SolverError::InvalidOptions);
        }
        let (m, n) = (self.jacobian.rows, self.jacobian.cols);
        self.scales.resize(n, R::zero());
        self.scales.fill(R::zero());
        // Stable column norms, computed only once per accepted-state linearization.
        for b in &self.jacobian.blocks {
            for col in 0..b.cols {
                let norm = &mut self.scales[b.col + col];
                for row in 0..b.rows {
                    *norm = norm.hypot(self.jacobian.values[b.start + col * b.rows + row]);
                }
            }
        }
        norm_inf(&self.scales)?;
        for scale in &mut self.scales {
            *scale = scale.max(min_column_norm);
        }
        self.diagonal.resize(n, R::zero());
        self.product.resize(m, R::zero());
        self.augmented_rhs.resize(
            m.checked_add(n).ok_or(EvaluationError::DimensionMismatch)?,
            R::zero(),
        );
        self.augmented_rhs[..m].copy_from_slice(&self.rhs);
        self.augmented_rhs[m..].fill(R::zero());
        self.damping_ready = true;
        Ok(())
    }

    fn solve_damped(&mut self, lambda: R) -> Result<DampedStep<'_, R>, SolverError> {
        if !self.damping_ready || !lambda.is_finite() || lambda < R::zero() {
            return Err(SolverError::InvalidOptions);
        }
        let root = lambda.sqrt();
        for (d, &s) in self.diagonal.iter_mut().zip(&self.scales) {
            *d = root * s;
        }
        norm_inf(&self.diagonal).map_err(|_| SolverError::LinearSolveFailed)?;
        let params = self.params();
        solve(
            &Damped {
                jacobian: &self.jacobian,
                diagonal: &self.diagonal,
            },
            &self.augmented_rhs,
            &mut self.step,
            params,
            &mut self.scratch,
            &mut self.statistics,
        )?;
        let (m, n) = (self.jacobian.rows, self.jacobian.cols);
        self.jacobian.apply(
            MatMut::from_column_major_slice_mut(&mut self.product, m, 1),
            MatRef::from_column_major_slice(&self.step, n, 1),
            Par::Seq,
            MemStack::new(&mut []),
        );
        let half = R::from_f64_impl(0.5);
        let mut predicted_reduction = R::zero();
        for (&g, &d) in self.gradient.iter().zip(&self.step) {
            predicted_reduction -= g * d;
        }
        for &value in &self.product {
            predicted_reduction -= (half * value) * value;
        }
        Ok(DampedStep {
            delta: &self.step,
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
    fn damped_retries_match_augmented_svd_and_preserve_the_model() {
        use faer_ext::nalgebra::{DMatrix, DVector};
        fn check<R: Real + Into<f64>>(tolerance: f64) {
            let c = R::from_f64_impl;
            let j =
                SMatrix::<R, 3, 3>::from_row_slice(&[1., 2., 0., 3., -1., 0., 0., 1., 0.].map(c));
            let residual = SVector::<R, 3>::from_column_slice(&[-3., 2., 1.].map(c));
            let mut backend = Lsmr::<R>::default();
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
        check::<f64>(1e-9);
        check::<f32>(3e-4);
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
                        .jacobian
                        .conj_apply(forward.as_mut(), x.as_ref(), Par::Seq, stack);
                    solver
                        .jacobian
                        .adjoint_apply(transpose.as_mut(), y.as_ref(), Par::Seq, stack);
                } else {
                    solver
                        .jacobian
                        .apply(forward.as_mut(), x.as_ref(), Par::Seq, stack);
                    solver.jacobian.transpose_apply(
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
