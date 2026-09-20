use faer::{
    Accum, Mat, MatMut, MatRef, Par,
    dyn_stack::{MemBuffer, MemStack},
    linalg::{
        cholesky::llt,
        matmul::{
            matmul,
            triangular::{self, BlockStructure},
        },
    },
};
use faer_ext::{
    IntoFaer,
    nalgebra::{DMatrixView, Dyn},
};

use crate::{
    EvaluationError, Real, SolverError,
    optimization::{LeastSquaresBackend, norm_inf},
};

/// Dense normal equations solved by in-place faer Cholesky.
///
/// Stores one dense n-by-n matrix, an RHS/step vector, and the original diagonal
/// for covariance rank checks. Only the lower triangle is assembled and read.
/// Factorization overwrites that triangle; solving overwrites the RHS with delta.
/// Buffers and scratch are retained.
///
/// Uses sequential dense kernels and no damping, pivot repair, or inverse.
/// A step requires a positive-definite normal matrix (full column rank of J).
/// Normal equations square J's condition number; use a future QR backend for
/// ill-conditioned problems. Matrix storage is O(n²), factorization O(n³).
pub struct DenseNormalCholesky<R: Real = f64> {
    normal: Mat<R>,
    rhs: Vec<R>,
    scratch: Option<MemBuffer>,
    scratch_dimension: usize,
    diagonal: Vec<R>,
    factorized: bool,
}

impl<R: Real> Default for DenseNormalCholesky<R> {
    fn default() -> Self {
        Self {
            normal: Mat::new(),
            rhs: Vec::new(),
            scratch: None,
            scratch_dimension: 0,
            diagonal: Vec::new(),
            factorized: false,
        }
    }
}

impl<R: Real> LeastSquaresBackend for DenseNormalCholesky<R> {
    type Scalar = R;

    fn prepare(&mut self, dimension: usize) -> Result<(), SolverError> {
        dimension
            .checked_mul(dimension)
            .and_then(|n| n.checked_mul(size_of::<R>()))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(EvaluationError::DimensionMismatch)?;
        self.normal
            .resize_with(dimension, dimension, |_, _| R::zero());
        self.rhs.resize(dimension, R::zero());
        self.diagonal.resize(dimension, R::zero());
        self.factorized = false;
        if dimension > self.scratch_dimension {
            self.scratch = Some(MemBuffer::new(llt::factor::cholesky_in_place_scratch::<R>(
                dimension,
                Par::Seq,
                Default::default(),
            )));
            self.scratch_dimension = dimension;
        }
        Ok(())
    }

    fn clear(&mut self) {
        self.factorized = false;
        let n = self.rhs.len();
        for col in 0..n {
            self.normal
                .as_mut()
                .col_mut(col)
                .subrows_mut(col, n - col)
                .fill(R::zero());
        }
        self.rhs.fill(R::zero());
    }

    fn accumulate<'a>(
        &mut self,
        residual: &[R],
        jacobians: impl Iterator<Item = (usize, DMatrixView<'a, R, Dyn, Dyn>)> + Clone,
    ) -> Result<(), EvaluationError> {
        let residual = MatRef::from_column_major_slice(residual, residual.len(), 1);
        for (i, (offset, block)) in jacobians.clone().enumerate() {
            let left: MatRef<'_, R> = block.into_faer();
            let width = left.ncols();
            if width == 0 {
                continue;
            }
            matmul(
                MatMut::from_column_major_slice_mut(
                    &mut self.rhs[offset..offset + width],
                    width,
                    1,
                ),
                Accum::Add,
                left.transpose(),
                residual,
                -R::one(),
                Par::Seq,
            );
            triangular::matmul(
                self.normal
                    .as_mut()
                    .submatrix_mut(offset, offset, width, width),
                BlockStructure::TriangularLower,
                Accum::Add,
                left.transpose(),
                BlockStructure::Rectangular,
                left,
                BlockStructure::Rectangular,
                R::one(),
                Par::Seq,
            );
            for (other_offset, block) in jacobians.clone().take(i) {
                let right: MatRef<'_, R> = block.into_faer();
                if right.ncols() == 0 {
                    continue;
                }
                // Emission order need not match numerical ordering. Always
                // write the lower block directly, without a mirrored product.
                let (row, col, lhs, rhs) = if offset > other_offset {
                    (offset, other_offset, left, right)
                } else {
                    (other_offset, offset, right, left)
                };
                matmul(
                    self.normal
                        .as_mut()
                        .submatrix_mut(row, col, lhs.ncols(), rhs.ncols()),
                    Accum::Add,
                    lhs.transpose(),
                    rhs,
                    R::one(),
                    Par::Seq,
                );
            }
        }
        Ok(())
    }

    fn gradient_norm(&self) -> Result<R, SolverError> {
        norm_inf(&self.rhs)
    }

    fn solve(&mut self) -> Result<&[R], SolverError> {
        let n = self.rhs.len();
        if n == 0 {
            return Ok(&self.rhs);
        }
        self.factorize()?;
        let stack = MemStack::new(self.scratch.as_mut().expect("prepared Cholesky workspace"));
        llt::solve::solve_in_place(
            self.normal.as_ref(),
            MatMut::from_column_major_slice_mut(&mut self.rhs, n, 1),
            Par::Seq,
            stack,
        );
        Ok(&self.rhs)
    }
}

impl<R: Real> DenseNormalCholesky<R> {
    pub(crate) fn factorize(&mut self) -> Result<(), SolverError> {
        if self.factorized {
            return Ok(());
        }
        let n = self.rhs.len();
        if n == 0 {
            self.factorized = true;
            return Ok(());
        }
        for col in 0..n {
            self.diagonal[col] = self.normal[(col, col)];
            for row in col..n {
                if !self.normal[(row, col)].is_finite() {
                    return Err(EvaluationError::InvalidEvaluation.into());
                }
            }
        }
        let stack = MemStack::new(self.scratch.as_mut().expect("prepared Cholesky workspace"));
        llt::factor::cholesky_in_place(
            self.normal.as_mut(),
            Default::default(),
            Par::Seq,
            stack,
            Default::default(),
        )
        .map_err(|_| SolverError::LinearSolveFailed)?;
        self.factorized = true;
        Ok(())
    }

    pub(crate) fn information_factor(
        &mut self,
        tolerance: R,
    ) -> Result<MatRef<'_, R>, SolverError> {
        self.factorize().map_err(|error| match error {
            SolverError::LinearSolveFailed => SolverError::SingularInformation,
            _ => SolverError::InvalidInformation,
        })?;
        for (i, &diagonal) in self.diagonal.iter().enumerate() {
            let pivot = self.normal[(i, i)];
            if !pivot.is_finite() || !diagonal.is_finite() {
                return Err(SolverError::InvalidInformation);
            }
            // Dimensionless squared pivot after implicit column normalization.
            let ratio = pivot / diagonal.sqrt();
            if diagonal <= R::zero() || pivot <= R::zero() || ratio * ratio <= tolerance {
                return Err(SolverError::SingularInformation);
            }
        }
        Ok(self.normal.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer_ext::nalgebra::{SMatrix, SVector};

    #[test]
    fn covariance_rejects_nonfinite_information_and_scaled_small_pivots() {
        let mut backend = DenseNormalCholesky::<f64>::default();
        for scale in [1e-6, 1., 1e6] {
            backend.prepare(2).unwrap();
            backend.clear();
            let jacobian = SMatrix::<f64, 2, 2>::new(scale, 1., 0., 1e-5);
            backend
                .accumulate(&[0., 0.], std::iter::once((0, jacobian.as_view())))
                .unwrap();
            assert!(matches!(
                backend.information_factor(1e-8),
                Err(SolverError::SingularInformation)
            ));
            // Checking rank did not damage the reusable factorization.
            backend.information_factor(1e-12).unwrap();
        }
        backend.prepare(1).unwrap();
        backend.clear();
        let huge = SMatrix::<f64, 1, 1>::new(1e200);
        backend
            .accumulate(&[0.], std::iter::once((0, huge.as_view())))
            .unwrap();
        assert!(matches!(
            backend.information_factor(0.),
            Err(SolverError::InvalidInformation)
        ));
    }

    #[test]
    fn lower_triangle_and_rhs_are_assembled_and_solved_in_place() {
        let jacobian = SMatrix::<f64, 2, 2>::from_row_slice(&[1.0, 2.0, 3.0, 4.0]);
        let residual = SVector::<f64, 2>::new(-5.0, -11.0);
        let mut backend = DenseNormalCholesky::default();
        backend.prepare(2).unwrap();
        let matrix_pointer = backend.normal.as_ref().as_ptr();
        let rhs_pointer = backend.rhs.as_ptr();
        let scratch_pointer = backend.scratch.as_ref().unwrap().as_ptr();
        for _ in 0..2 {
            backend.prepare(2).unwrap();
            backend.clear();
            backend.normal[(0, 1)] = f64::NAN; // The upper triangle must never be used.
            backend
                .accumulate(
                    residual.as_slice(),
                    std::iter::once((0, jacobian.as_view())),
                )
                .unwrap();
            assert_eq!(backend.normal[(0, 0)], 10.0);
            assert_eq!(backend.normal[(1, 0)], 14.0);
            assert_eq!(backend.normal[(1, 1)], 20.0);
            assert!(backend.normal[(0, 1)].is_nan());
            assert_eq!(backend.rhs, [38.0, 54.0]);
            assert_eq!(backend.gradient_norm().unwrap(), 54.0);
            let step = backend.solve().unwrap();
            assert_eq!(step.as_ptr(), rhs_pointer);
            assert!((step[0] - 1.0).abs() < 1e-12);
            assert!((step[1] - 2.0).abs() < 1e-12);
            assert_eq!(backend.normal.as_ref().as_ptr(), matrix_pointer);
            assert_eq!(backend.scratch.as_ref().unwrap().as_ptr(), scratch_pointer);
        }
    }
}
