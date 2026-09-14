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
use faer_ext::{IntoFaer, nalgebra::SVector};

use crate::{
    EvaluationError, JacobianBlock, Real, SolverError,
    optimization::{LeastSquaresBackend, norm_inf},
};

/// Dense normal equations solved by in-place faer Cholesky.
///
/// Stores one dense n-by-n matrix and one RHS/step vector. Only the lower
/// triangle is assembled and read. Factorization overwrites that triangle;
/// solving overwrites the RHS with delta. Buffers and scratch are retained.
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
}

impl<R: Real> Default for DenseNormalCholesky<R> {
    fn default() -> Self {
        Self {
            normal: Mat::new(),
            rhs: Vec::new(),
            scratch: None,
            scratch_dimension: 0,
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

    fn accumulate<const ROWS: usize>(
        &mut self,
        residual: &SVector<R, ROWS>,
        jacobians: &[JacobianBlock<'_, R>],
        columns: &[usize],
    ) -> Result<(), EvaluationError> {
        let residual = MatRef::from_column_major_slice(residual.as_slice(), ROWS, 1);
        for (i, block) in jacobians.iter().enumerate() {
            let left: MatRef<'_, R> = block.jacobian().into_faer();
            let width = left.ncols();
            if width == 0 {
                continue;
            }
            let offset = columns[i];
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
            for j in 0..i {
                let right: MatRef<'_, R> = jacobians[j].jacobian().into_faer();
                if right.ncols() == 0 {
                    continue;
                }
                // Emission order need not match numerical ordering. Always
                // write the lower block directly, without a mirrored product.
                let (row, col, lhs, rhs) = if offset > columns[j] {
                    (offset, columns[j], left, right)
                } else {
                    (columns[j], offset, right, left)
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
        for col in 0..n {
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
        llt::solve::solve_in_place(
            self.normal.as_ref(),
            MatMut::from_column_major_slice_mut(&mut self.rhs, n, 1),
            Par::Seq,
            stack,
        );
        Ok(&self.rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Variable, storage::StatePool};
    use faer_ext::nalgebra::SMatrix;

    struct Pair;
    impl Variable for Pair {
        type Scalar = f64;
        type Tangent = [f64; 2];
        const DOF: usize = 2;
        fn tangent_from_slice(delta: &[f64]) -> Self::Tangent {
            [delta[0], delta[1]]
        }
        fn retract(&self, _: &Self::Tangent) -> Self {
            Self
        }
    }

    #[test]
    fn lower_triangle_and_rhs_are_assembled_and_solved_in_place() {
        let mut states = StatePool::default();
        let key = states.insert(Pair);
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
                .accumulate(&residual, &[JacobianBlock::new(key, &jacobian)], &[0])
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
