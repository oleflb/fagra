use faer::{
    MatMut, MatRef, Par,
    dyn_stack::{MemBuffer, MemStack, StackReq},
    matrix_free::{BiLinOp, IdentityPrecond, InitialGuessStatus, LinOp, lsmr},
};
use faer_ext::nalgebra::{DimName, Matrix, Storage, U1, storage::IsContiguous};

use crate::{
    EvaluationError, JacobianBlock, Real, SolverError,
    optimization::{LeastSquaresBackend, norm_inf},
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
        }
    }
}

impl<R: Real> LeastSquaresBackend for Lsmr<R> {
    type Scalar = R;

    fn prepare(&mut self, dimension: usize) -> Result<(), SolverError> {
        if self.max_iterations == 0 || !(R::zero()..R::one()).contains(&self.relative_tolerance) {
            return Err(SolverError::InvalidOptions);
        }
        self.jacobian.cols = dimension;
        self.gradient.resize(dimension, R::zero());
        self.step.resize(dimension, R::zero());
        Ok(())
    }

    fn clear(&mut self) {
        self.jacobian.rows = 0;
        self.jacobian.blocks.clear();
        self.jacobian.values.clear();
        self.rhs.clear();
        self.gradient.fill(R::zero());
    }

    fn accumulate<Rows: DimName, S>(
        &mut self,
        residual: &Matrix<R, Rows, U1, S>,
        jacobians: &[JacobianBlock<'_, R>],
        columns: &[usize],
    ) -> Result<(), EvaluationError>
    where
        S: Storage<R, Rows, U1> + IsContiguous,
    {
        let row = self.jacobian.rows;
        self.jacobian.rows = row
            .checked_add(Rows::DIM)
            .ok_or(EvaluationError::DimensionMismatch)?;
        self.rhs.extend(residual.iter().map(|&r| -r));
        for (block, &col) in jacobians.iter().zip(columns) {
            let matrix = block.jacobian();
            self.jacobian.blocks.push(Block {
                row,
                col,
                rows: Rows::DIM,
                cols: matrix.ncols(),
                start: self.jacobian.values.len(),
            });
            // Pack explicitly: emitted views may have arbitrary strides.
            for j in 0..matrix.ncols() {
                let mut gradient = R::zero();
                for i in 0..Rows::DIM {
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
        let (m, n) = (self.jacobian.rows, self.jacobian.cols);
        self.step.fill(R::zero());
        if m == 0 || n == 0 {
            return Ok(&self.step);
        }
        let precond = IdentityPrecond { dim: n };
        // faer 0.24.4 omits wbar and vold from lsmr_scratch; both are n-by-1.
        // Remove this extra space when upstream's scratch calculation is fixed.
        let vector = faer::linalg::temp_mat_scratch::<R>(n, 1);
        let req = lsmr::lsmr_scratch::<R>(precond, &self.jacobian, 1, Par::Seq)
            .and(vector)
            .and(vector);
        if !self
            .scratch
            .as_mut()
            .is_some_and(|buffer| MemStack::new(buffer).can_hold(req))
        {
            self.scratch = Some(MemBuffer::new(req));
        }
        lsmr::lsmr(
            MatMut::from_column_major_slice_mut(&mut self.step, n, 1),
            precond,
            &self.jacobian,
            MatRef::from_column_major_slice(&self.rhs, m, 1),
            lsmr::LsmrParams {
                initial_guess: InitialGuessStatus::Zero,
                rel_tolerance: self.relative_tolerance,
                max_iters: self.max_iterations,
                ..Default::default()
            },
            |_| {},
            Par::Seq,
            MemStack::new(self.scratch.as_mut().unwrap()),
        )
        .map_err(|_| SolverError::LinearSolveFailed)?;
        norm_inf(&self.step).map_err(|_| SolverError::LinearSolveFailed)?;
        Ok(&self.step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DenseNormalCholesky, Variable, storage::StatePool};
    use faer::Mat;
    use faer_ext::nalgebra::{SMatrix, SVector};

    type Pair = crate::variable::test_support::Vector<2>;

    #[test]
    fn workspace_handles_large_coordinate_vectors() {
        let mut solver = Lsmr::default();
        let mut states = StatePool::default();
        let jacobian = SMatrix::<f64, 2, 2>::identity();
        for n in [300, 1024] {
            solver.prepare(n).unwrap();
            solver.clear();
            for col in (0..n).step_by(2) {
                let key = states.insert(Pair::identity());
                let residual = SVector::<f64, 2>::new(-(col as f64), -(col as f64 + 1.0));
                solver
                    .accumulate(&residual, &[JacobianBlock::new(key, &jacobian)], &[col])
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
                solver.accumulate(&r, &blocks, &[2, 0]).unwrap();
                dense.accumulate(&r, &blocks, &[2, 0]).unwrap();
                let jb = SMatrix::<f64, 2, 2>::new(2., 0., 0., 1.);
                let r = SVector::<f64, 2>::new(-4., 5.);
                let blocks = [JacobianBlock::new(b, &jb)];
                solver.accumulate(&r, &blocks, &[0]).unwrap();
                dense.accumulate(&r, &blocks, &[0]).unwrap();
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
