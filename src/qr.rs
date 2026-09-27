use faer::{
    Conj, Par,
    dyn_stack::{MemBuffer, MemStack},
    linalg::{householder, qr::col_pivoting::factor},
};
use faer_ext::nalgebra::{DMatrixView, Dim, Dyn, VectorView};

use crate::{
    EvaluationError, Real, SolverError,
    linear_model::LinearizedModel,
    marginalization::MatrixBuffer,
    optimization::{DampedStep, LeastSquaresBackend, norm_inf},
};

/// Dense column-pivoted QR on Jacobian rows, without normal equations.
///
/// Supports Gauss–Newton and LM. LM solves in coordinates `z = D delta`,
/// factoring `[J D⁻¹; sqrt(lambda) I]` on each retry and recovering `delta`,
/// preserving the original model for prediction and covariance. Numerical rank
/// deficiency returns `LinearSolveFailed`; no pseudoinverse is used.
/// Storage is O((m+n)n) for LM and factorization is O((m+n)n²).
/// Matrices, cached blocks, and scratch are retained across calls. Covariance
/// extraction uses the shared dense normal-equation path, not the QR factor.
pub struct DenseQr<R: Real = f64> {
    /// Reject pivots at most this fraction of the largest absolute QR pivot.
    /// LM applies this cutoff after column scaling by D⁻¹.
    /// Default: 64ε. Must be finite and in [0, 1). Zero rejects only zero pivots.
    pub relative_rank_tolerance: R,
    model: LinearizedModel<R>,
    matrix: MatrixBuffer<R>,
    coefficients: MatrixBuffer<R>,
    permutation: Vec<usize>,
    inverse: Vec<usize>,
    scratch: Option<MemBuffer>,
    step: Vec<R>,
    product: Vec<R>,
    damping_ready: bool,
}

impl<R: Real> Default for DenseQr<R> {
    fn default() -> Self {
        Self {
            relative_rank_tolerance: R::from_f64_impl(64.0) * R::epsilon_impl(),
            model: LinearizedModel::default(),
            matrix: MatrixBuffer::default(),
            coefficients: MatrixBuffer::default(),
            permutation: Vec::new(),
            inverse: Vec::new(),
            scratch: None,
            step: Vec::new(),
            product: Vec::new(),
            damping_ready: false,
        }
    }
}

impl<R: Real> DenseQr<R> {
    fn validate(&self) -> Result<(), SolverError> {
        if !(R::zero()..R::one()).contains(&self.relative_rank_tolerance) {
            return Err(SolverError::InvalidOptions);
        }
        Ok(())
    }

    fn factor_solve(&mut self, damping: Option<R>) -> Result<(), SolverError> {
        self.validate()?;
        let n = self.model.jacobian.cols;
        if n == 0 {
            return Ok(());
        }
        let m = self.model.jacobian.rows;
        let height = m
            .checked_add(if damping.is_some() { n } else { 0 })
            .ok_or(EvaluationError::DimensionMismatch)?;
        if height < n {
            return Err(SolverError::LinearSolveFailed);
        }
        let width = n.checked_add(1).ok_or(EvaluationError::DimensionMismatch)?;
        height
            .checked_mul(width)
            .and_then(|v| v.checked_mul(size_of::<R>()))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(EvaluationError::DimensionMismatch)?;
        self.matrix.resize(height, width);
        self.matrix.fill(R::zero());
        for b in &self.model.jacobian.blocks {
            for col in 0..b.cols {
                let scale = if damping.is_some() {
                    self.model.scales[b.col + col]
                } else {
                    R::one()
                };
                for row in 0..b.rows {
                    self.matrix[(b.row + row, b.col + col)] =
                        self.model.jacobian.values[b.start + col * b.rows + row] / scale;
                }
            }
        }
        for row in 0..m {
            self.matrix[(row, n)] = self.model.negative_residual[row];
        }
        if let Some(lambda) = damping {
            for col in 0..n {
                self.matrix[(m + col, col)] = lambda.sqrt();
            }
        }
        let mut largest = R::zero();
        for col in 0..width {
            for row in 0..height {
                let value = self.matrix[(row, col)];
                if !value.is_finite() {
                    return Err(SolverError::LinearSolveFailed);
                }
                if col < n {
                    largest = largest.max(value.abs());
                }
            }
        }
        // faer's pivoted QR divides by the largest column norm, even if zero.
        if largest == R::zero() {
            return Err(SolverError::LinearSolveFailed);
        }
        let block = 32.min(n);
        self.coefficients.resize(block, n);
        self.permutation.resize(n, 0);
        self.inverse.resize(n, 0);
        let req = factor::qr_in_place_scratch::<usize, R>(height, n, block, Par::Seq, Default::default())
            .or(householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<R>(height, block, 1));
        if !self
            .scratch
            .as_mut()
            .is_some_and(|s| MemStack::new(s).can_hold(req))
        {
            self.scratch = Some(MemBuffer::new(req));
        }
        let stack = MemStack::new(self.scratch.as_mut().unwrap());
        let (mut a, rhs) = self.matrix.as_mut().split_at_col_mut(n);
        factor::qr_in_place(
            a.as_mut(),
            self.coefficients.as_mut(),
            &mut self.permutation,
            &mut self.inverse,
            Par::Seq,
            stack,
            Default::default(),
        );
        householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
            a.as_ref(),
            self.coefficients.as_ref(),
            Conj::No,
            rhs,
            Par::Seq,
            stack,
        );
        let mut largest = R::zero();
        for col in 0..n {
            let pivot = self.matrix[(col, col)].abs();
            if !pivot.is_finite() {
                return Err(SolverError::LinearSolveFailed);
            }
            largest = largest.max(pivot);
        }
        for col in (0..n).rev() {
            let pivot = self.matrix[(col, col)];
            if pivot.abs() <= self.relative_rank_tolerance * largest {
                return Err(SolverError::LinearSolveFailed);
            }
            let mut value = self.matrix[(col, n)];
            for j in col + 1..n {
                value -= self.matrix[(col, j)] * self.step[self.permutation[j]];
            }
            self.step[self.permutation[col]] = value / pivot;
        }
        if damping.is_some() {
            for (d, &scale) in self.step.iter_mut().zip(&self.model.scales) {
                *d /= scale;
            }
        }
        norm_inf(&self.step).map_err(|_| SolverError::LinearSolveFailed)?;
        Ok(())
    }
}

impl<R: Real> LeastSquaresBackend for DenseQr<R> {
    type Scalar = R;

    fn prepare(&mut self, dimension: usize) -> Result<(), SolverError> {
        self.damping_ready = false;
        self.validate()?;
        self.model.prepare(dimension);
        self.step.resize(dimension, R::zero());
        Ok(())
    }

    fn clear(&mut self) {
        self.damping_ready = false;
        self.model.clear();
    }

    fn accumulate<'a, Rows: Dim>(
        &mut self,
        residual: VectorView<'_, R, Rows>,
        jacobians: impl Iterator<Item = (usize, DMatrixView<'a, R, Dyn, Dyn>)> + Clone,
    ) -> Result<(), EvaluationError> {
        self.damping_ready = false;
        self.model.accumulate(residual.as_slice(), jacobians)
    }

    fn gradient_norm(&self) -> Result<R, SolverError> {
        norm_inf(&self.model.gradient)
    }

    fn solve(&mut self) -> Result<&[R], SolverError> {
        self.factor_solve(None)?;
        Ok(&self.step)
    }

    fn prepare_damping(&mut self, min_column_norm: R) -> Result<(), SolverError> {
        self.damping_ready = false;
        self.model.prepare_scales(min_column_norm)?;
        self.product.resize(self.model.jacobian.rows, R::zero());
        self.damping_ready = true;
        Ok(())
    }

    fn solve_damped(&mut self, lambda: R) -> Result<DampedStep<'_, R>, SolverError> {
        if !self.damping_ready || !lambda.is_finite() || lambda < R::zero() {
            return Err(SolverError::InvalidOptions);
        }
        self.factor_solve(Some(lambda))?;
        let predicted_reduction = self
            .model
            .predicted_reduction(&self.step, &mut self.product);
        if !predicted_reduction.is_finite() {
            return Err(SolverError::LinearSolveFailed);
        }
        Ok(DampedStep {
            delta: &self.step,
            predicted_reduction,
        })
    }
}

impl<R: Real> crate::covariance::CovarianceBackend for DenseQr<R> {
    fn covariance(
        &mut self,
        normal: &mut crate::DenseNormalCholesky<R>,
        selected: &mut crate::covariance::SelectedCovariance<R>,
    ) -> Result<(), SolverError> {
        self.model.covariance(normal, selected)
    }
}
