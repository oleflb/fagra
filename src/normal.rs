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
    nalgebra::{Const, DMatrixView, Dim, Dyn, VectorView},
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

    fn accumulate<'a, Rows: Dim>(
        &mut self,
        residual: VectorView<'_, R, Rows>,
        jacobians: impl Iterator<Item = (usize, DMatrixView<'a, R, Dyn, Dyn>)> + Clone,
    ) -> Result<(), EvaluationError> {
        // Static dimensions select their kernel at monomorphization. Only Dyn
        // needs to recover a static loop bound for the small-row kernel.
        match (Rows::try_to_usize(), residual.nrows()) {
            (Some(1..=4), _) => self.accumulate_small(residual, jacobians),
            (None, 1) => self.accumulate_small(
                VectorView::<_, Const<1>>::from_slice(residual.as_slice()),
                jacobians,
            ),
            (None, 2) => self.accumulate_small(
                VectorView::<_, Const<2>>::from_slice(residual.as_slice()),
                jacobians,
            ),
            (None, 3) => self.accumulate_small(
                VectorView::<_, Const<3>>::from_slice(residual.as_slice()),
                jacobians,
            ),
            (None, 4) => self.accumulate_small(
                VectorView::<_, Const<4>>::from_slice(residual.as_slice()),
                jacobians,
            ),
            _ => self.accumulate_matmul(residual.as_slice(), jacobians),
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
    // Tiny residuals do too little arithmetic to amortize matrix-product dispatch.
    // Keep the row count static so the compiler can unroll these short dot products.
    // The four-row cutoff is measured in docs/normal-performance.md.
    fn accumulate_small<'a, Rows: Dim>(
        &mut self,
        residual: VectorView<'_, R, Rows>,
        jacobians: impl Iterator<Item = (usize, DMatrixView<'a, R, Dyn, Dyn>)> + Clone,
    ) {
        for (i, (offset, left)) in jacobians.clone().enumerate() {
            for col in 0..left.ncols() {
                let mut gradient = R::zero();
                for k in 0..residual.nrows() {
                    gradient += left[(k, col)] * residual[k];
                }
                self.rhs[offset + col] -= gradient;
                for row in col..left.ncols() {
                    let mut product = R::zero();
                    for k in 0..residual.nrows() {
                        product += left[(k, row)] * left[(k, col)];
                    }
                    self.normal[(offset + row, offset + col)] += product;
                }
            }
            for (other_offset, right) in jacobians.clone().take(i) {
                let (row_offset, col_offset, lhs, rhs) = if offset > other_offset {
                    (offset, other_offset, left, right)
                } else {
                    (other_offset, offset, right, left)
                };
                for col in 0..rhs.ncols() {
                    for row in 0..lhs.ncols() {
                        let mut product = R::zero();
                        for k in 0..residual.nrows() {
                            product += lhs[(k, row)] * rhs[(k, col)];
                        }
                        self.normal[(row_offset + row, col_offset + col)] += product;
                    }
                }
            }
        }
    }

    fn accumulate_matmul<'a>(
        &mut self,
        residual: &[R],
        jacobians: impl Iterator<Item = (usize, DMatrixView<'a, R, Dyn, Dyn>)> + Clone,
    ) {
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
    }

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
    use faer::traits::ext::ComplexFieldExt;
    use faer_ext::nalgebra::{DMatrix, DVectorView, SMatrix, SVector};

    #[test]
    fn accumulation_matches_dense_reference() {
        fn check<R: Real, const ROWS: usize>() {
            let c = R::from_f64_impl;
            let mut actual = DenseNormalCholesky::<R>::default();
            let mut fixed = DenseNormalCholesky::<R>::default();
            let mut direct = DenseNormalCholesky::<R>::default();
            let mut matmul = DenseNormalCholesky::<R>::default();
            for backend in [&mut actual, &mut fixed, &mut direct, &mut matmul] {
                backend.prepare(31).unwrap();
                backend.clear();
                for col in 0..31 {
                    for row in 0..col {
                        backend.normal[(row, col)] = c(f64::NAN);
                    }
                }
            }
            let mut normal = DMatrix::<R>::zeros(31, 31);
            let mut rhs = DMatrix::<R>::zeros(31, 1);
            // Separate observations with distinct frozen square-root weights.
            for weight in [1., 0.13, 0.] {
                let storage = DMatrix::from_fn(2 * ROWS, 31, |row, col| {
                    c(weight * (((row * 7 + col * 11) % 23) as f64 - 11.) / 13.)
                });
                let j = storage.view_with_steps((0, 0), (ROWS, 31), (1, 0));
                let residual = SVector::<R, ROWS>::from_fn(|row, _| c(weight * (row as f64 - 1.3)));
                // Both strides, shuffled emission order, and an empty block.
                let blocks = [(24, 3), (6, 6), (31, 0), (18, 6), (0, 6), (27, 4), (12, 6)].map(
                    |(offset, width)| {
                        (
                            offset,
                            storage.view_with_steps((0, offset), (ROWS, width), (1, 0)),
                        )
                    },
                );
                actual
                    .accumulate(
                        DVectorView::from_slice(residual.as_slice(), ROWS),
                        blocks.into_iter(),
                    )
                    .unwrap();
                fixed
                    .accumulate(residual.column(0), blocks.into_iter())
                    .unwrap();
                direct.accumulate_small(residual.column(0), blocks.into_iter());
                matmul.accumulate_matmul(residual.as_slice(), blocks.into_iter());
                normal += j.transpose() * j;
                rhs -= j.transpose() * residual;
            }
            let tolerance = c(256.) * R::epsilon_impl();
            for backend in [&actual, &fixed, &direct, &matmul] {
                for col in 0..31 {
                    assert!(
                        (backend.rhs[col] - rhs[(col, 0)]).abs()
                            <= tolerance * (R::one() + rhs[(col, 0)].abs())
                    );
                    for row in 0..31 {
                        if row < col {
                            assert!(backend.normal[(row, col)].is_nan());
                        } else {
                            assert!(
                                (backend.normal[(row, col)] - normal[(row, col)]).abs()
                                    <= tolerance * (R::one() + normal[(row, col)].abs())
                            );
                        }
                    }
                }
            }
        }
        macro_rules! rows {
            ($($rows:literal),*) => {$(check::<f32, $rows>(); check::<f64, $rows>();)*};
        }
        rows!(0, 1, 2, 3, 4, 5, 8, 16, 32);
    }

    #[test]
    #[ignore = "release assembly microbenchmark; use --release --ignored --nocapture --test-threads=1"]
    fn small_row_timings() {
        use std::{hint::black_box, time::Instant};

        fn measure<R: Real, const ROWS: usize>() {
            let observations: Vec<_> = (0..100)
                .map(|observation| {
                    let weight = 1. / (1. + observation as f64 * 0.1);
                    let j = DMatrix::from_fn(ROWS, 31, |row, col| {
                        R::from_f64_impl(
                            weight * (((row * 7 + col * 11 + observation) % 23) as f64 - 11.) / 13.,
                        )
                    });
                    let residual: Vec<_> = (0..ROWS)
                        .map(|row| R::from_f64_impl(weight * (row as f64 - 1.3)))
                        .collect();
                    (j, residual)
                })
                .collect();
            let mut backend = DenseNormalCholesky::<R>::default();
            backend.prepare(31).unwrap();
            let mut pass = |kind| {
                backend.clear();
                for (j, residual) in black_box(&observations) {
                    let blocks = [(0, 6), (6, 6), (12, 6), (18, 6), (24, 3), (27, 4)]
                        .into_iter()
                        .map(|(offset, width)| {
                            (
                                offset,
                                j.view_with_steps((0, offset), (ROWS, width), (0, 0)),
                            )
                        });
                    match kind {
                        0 => backend.accumulate_matmul(residual, blocks),
                        1 => backend.accumulate_small(
                            VectorView::<_, Const<ROWS>>::from_slice(residual),
                            blocks,
                        ),
                        2 => backend
                            .accumulate(VectorView::<_, Const<ROWS>>::from_slice(residual), blocks)
                            .unwrap(),
                        _ => backend
                            .accumulate(DVectorView::from_slice(residual, black_box(ROWS)), blocks)
                            .unwrap(),
                    }
                }
                black_box((&backend.normal, &backend.rhs));
            };
            for _ in 0..20 {
                for kind in 0..4 {
                    pass(kind);
                }
            }
            let mut timings = [[0.; 4]; 7];
            for (sample, timings) in timings.iter_mut().enumerate() {
                // Rotate order to reduce thermal/frequency bias.
                for index in 0..4 {
                    let kind = (sample + index) % 4;
                    let start = Instant::now();
                    for _ in 0..100 {
                        pass(kind);
                    }
                    timings[kind] = start.elapsed().as_secs_f64() * 1e6 / 100.;
                }
            }
            let medians: [f64; 4] = std::array::from_fn(|kind| {
                let mut samples = timings.map(|sample| sample[kind]);
                samples.sort_by(f64::total_cmp);
                samples[3]
            });
            println!(
                "{}, rows={ROWS}, faer={:.2} us, direct={:.2} us, static={:.2} us, dynamic={:.2} us",
                std::any::type_name::<R>(),
                medians[0],
                medians[1],
                medians[2],
                medians[3]
            );
        }
        macro_rules! rows {
            ($($rows:literal),*) => {$(measure::<f32, $rows>(); measure::<f64, $rows>();)*};
        }
        rows!(1, 2, 3, 4, 8, 16, 32);
    }

    #[test]
    fn covariance_rejects_nonfinite_information_and_scaled_small_pivots() {
        let mut backend = DenseNormalCholesky::<f64>::default();
        for scale in [1e-6, 1., 1e6] {
            backend.prepare(2).unwrap();
            backend.clear();
            let jacobian = SMatrix::<f64, 2, 2>::new(scale, 1., 0., 1e-5);
            backend
                .accumulate(
                    SVector::from([0., 0.]).column(0),
                    std::iter::once((0, jacobian.as_view())),
                )
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
            .accumulate(
                SVector::from([0.]).column(0),
                std::iter::once((0, huge.as_view())),
            )
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
                .accumulate(residual.column(0), std::iter::once((0, jacobian.as_view())))
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
