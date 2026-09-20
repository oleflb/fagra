//! Selected inverse information, with retained dense numerical storage.
// ponytail: dense O(n²) storage/O(n³) factorization; use a sparse direct backend
// when bounded-window covariance factorization becomes the measured bottleneck.

use faer::{
    Accum, Mat, MatRef, Par,
    linalg::{matmul::matmul, triangular_solve},
};

use crate::{
    BlockId, DenseNormalCholesky, Real, SolverError,
    factors::FactorSchema,
    marginalization::{MatrixBuffer, Priors},
    optimization::{Layout, OptimizeOptions, OptimizeReport, Optimizer, Workspace},
    states::StateSchema,
};

/// Numerical controls for selected, undamped inverse-information covariance.
#[derive(Debug, Clone, Copy)]
pub struct CovarianceOptions<R: Real = f64> {
    /// Reject a Cholesky pivot when `L[i,i]² / H[i,i]` is at most this value.
    /// This dimensionless, ordering-dependent rank screen is not a condition-number
    /// estimate. Default: `64 * epsilon`. Must be finite and in `[0, 1)`.
    /// Zero still rejects non-positive pivots. No regularization is applied.
    pub relative_pivot_tolerance: R,
}

impl<R: Real> Default for CovarianceOptions<R> {
    fn default() -> Self {
        Self {
            relative_pivot_tolerance: R::from_f64_impl(64.0) * R::epsilon_impl(),
        }
    }
}

impl<R: Real> CovarianceOptions<R> {
    pub(crate) fn validate(&self) -> Result<(), SolverError> {
        let t = self.relative_pivot_tolerance;
        if !t.is_finite() || t < R::zero() || t >= R::one() {
            return Err(SolverError::InvalidRankTolerance);
        }
        Ok(())
    }
}

/// Internal reusable selected-solve buffers; public only for backend bounds.
pub struct SelectedCovariance<R: Real> {
    columns: Vec<usize>,
    rhs: MatrixBuffer<R>,
    result: Mat<R>,
    pub(crate) options: CovarianceOptions<R>,
}

impl<R: Real> Default for SelectedCovariance<R> {
    fn default() -> Self {
        Self {
            columns: Vec::new(),
            rhs: MatrixBuffer::default(),
            result: Mat::new(),
            options: CovarianceOptions::default(),
        }
    }
}

impl<R: Real> SelectedCovariance<R> {
    pub(crate) fn prepare(
        &mut self,
        layout: &Layout,
        blocks: &[BlockId],
    ) -> Result<(), SolverError> {
        self.columns.clear();
        for (i, id) in blocks.iter().enumerate() {
            if blocks[..i].contains(id) {
                return Err(SolverError::DuplicateCovarianceBlock);
            }
            let block =
                &layout.blocks[*layout.block_index.get(id).ok_or(crate::KeyError::Unknown)?];
            self.columns
                .extend(block.offset..block.offset + block.width);
        }
        let k = self.columns.len();
        layout
            .dimension
            .checked_mul(k)
            .and_then(|n| n.checked_mul(size_of::<R>()))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(SolverError::InvalidInformation)?;
        self.rhs.resize(layout.dimension, k);
        self.result.resize_with(k, k, |_, _| R::zero());
        Ok(())
    }

    pub(crate) fn compute(
        &mut self,
        normal: &mut DenseNormalCholesky<R>,
    ) -> Result<(), SolverError> {
        let lower = normal.information_factor(self.options.relative_pivot_tolerance)?;
        self.rhs.as_mut().fill(R::zero());
        for (j, &i) in self.columns.iter().enumerate() {
            self.rhs[(i, j)] = R::one();
        }
        if !self.columns.is_empty() {
            triangular_solve::solve_lower_triangular_in_place(lower, self.rhs.as_mut(), Par::Seq);
            matmul(
                self.result.as_mut(),
                Accum::Replace,
                self.rhs.as_ref().transpose(),
                self.rhs.as_ref(),
                R::one(),
                Par::Seq,
            );
        }
        for col in 0..self.result.ncols() {
            for row in 0..self.result.nrows() {
                if !self.result[(row, col)].is_finite() {
                    return Err(SolverError::InvalidInformation);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn matrix(&self) -> MatRef<'_, R> {
        self.result.as_ref()
    }
}

/// Internal backend capability for extracting covariance from a current model.
pub trait CovarianceBackend: crate::optimization::LeastSquaresBackend {
    /// Reuse undamped information or assemble it from cached Jacobians.
    fn covariance(
        &mut self,
        normal: &mut DenseNormalCholesky<Self::Scalar>,
        selected: &mut SelectedCovariance<Self::Scalar>,
    ) -> Result<(), SolverError>;
}

impl<R: Real> CovarianceBackend for DenseNormalCholesky<R> {
    fn covariance(
        &mut self,
        _: &mut DenseNormalCholesky<R>,
        selected: &mut SelectedCovariance<R>,
    ) -> Result<(), SolverError> {
        selected.compute(self)
    }
}

/// Internal workspace retained by the solver; no cross-call numerical cache.
pub struct CovarianceWorkspace<R: Real> {
    pub(crate) work: Workspace<DenseNormalCholesky<R>>,
    pub(crate) selected: SelectedCovariance<R>,
}

impl<R: Real> Default for CovarianceWorkspace<R> {
    fn default() -> Self {
        Self {
            work: Workspace::new(DenseNormalCholesky::default()),
            selected: SelectedCovariance::default(),
        }
    }
}

impl<R: Real> CovarianceWorkspace<R> {
    pub(crate) fn evaluate<S: StateSchema<Scalar = R>, F: FactorSchema<S, Scalar = R>>(
        &mut self,
        states: &mut S,
        factors: &mut F,
        priors: &mut Priors<R>,
        blocks: &[BlockId],
    ) -> Result<(), SolverError> {
        self.work.prepare(states, factors)?;
        self.selected.prepare(&self.work.layout, blocks)?;
        self.work.linearize(states, factors, priors)?;
        self.selected.compute(&mut self.work.backend)
    }
}

/// Internal scoped optimizer capability. Extraction runs before returning to callers.
pub trait CovarianceOptimizer<S: StateSchema, F>: Optimizer<S, F> {
    /// Optimize and extract current-state covariance; accepted estimates survive errors.
    fn optimize_covariance(
        &mut self,
        states: &mut S,
        factors: &mut F,
        priors: &mut Priors<S::Scalar>,
        options: &OptimizeOptions<S::Scalar>,
        blocks: &[BlockId],
        covariance: &mut CovarianceWorkspace<S::Scalar>,
    ) -> Result<OptimizeReport<S::Scalar>, SolverError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EvaluationError, Factor, JacobianBlock, LinearizationSink, Solver, StateKey, StateStore,
        Variable, variable::test_support::Vector,
    };
    use faer_ext::nalgebra::{SMatrix, SVector};
    crate::states! { States { knots: Vector<9>, alignments: Vector<3> } }
    crate::factors! { Factors { measurements: Measurement } }
    struct Measurement {
        knots: [StateKey<Vector<9>>; 2],
        alignment: StateKey<Vector<3>>,
        nuisance: StateKey<Vector<3>>,
        j: SMatrix<f64, 24, 24>,
    }
    impl Measurement {
        fn residual(&self, states: &States) -> SVector<f64, 24> {
            let mut x = SVector::<f64, 24>::zeros();
            for (i, &key) in self.knots.iter().enumerate() {
                x.fixed_rows_mut::<9>(9 * i)
                    .copy_from(&states.get(key).unwrap().log());
            }
            x.fixed_rows_mut::<3>(18)
                .copy_from(&states.get(self.alignment).unwrap().log());
            x.fixed_rows_mut::<3>(21)
                .copy_from(&states.get(self.nuisance).unwrap().log());
            self.j * x
        }
    }
    impl Factor<States> for Measurement {
        type Scalar = f64;
        fn visit_variables(&self, mut v: impl FnMut(BlockId)) {
            for key in self.knots {
                v(key.block_id());
            }
            v(self.alignment.block_id());
            v(self.nuisance.block_id());
        }
        fn cost(&self, states: &States) -> Result<f64, EvaluationError> {
            Ok(0.5 * self.residual(states).norm_squared())
        }
        fn linearize<L: LinearizationSink<Scalar = f64>>(
            &self,
            states: &States,
            out: &mut L,
        ) -> Result<(), EvaluationError> {
            let start = self.j.fixed_columns::<9>(0);
            let end = self.j.fixed_columns::<9>(9);
            let alignment = self.j.fixed_columns::<3>(18);
            let nuisance = self.j.fixed_columns::<3>(21);
            out.residual(
                &self.residual(states),
                &[
                    JacobianBlock::new(self.knots[0], &start),
                    JacobianBlock::new(self.knots[1], &end),
                    JacobianBlock::new(self.alignment, &alignment),
                    JacobianBlock::new(self.nuisance, &nuisance),
                ],
            )
        }
    }
    #[test]
    fn twenty_one_coordinates_preserve_order_cross_terms_and_history() {
        let mut graph = Solver::<States, Factors>::new();
        let start = graph.add(Vector::<9>::identity());
        let end = graph.add(Vector::<9>::identity());
        let alignment = graph.add(Vector::<3>::identity());
        let nuisance = graph.add(Vector::<3>::identity());
        let j = SMatrix::<f64, 24, 24>::from_fn(|i, k| {
            if i == k {
                4.
            } else {
                ((i * 7 + k * 11) % 13) as f64 / 30.
            }
        });
        let reference = (j.transpose() * j).try_inverse().unwrap();
        graph
            .add_factor(Measurement {
                knots: [start, end],
                alignment,
                nuisance,
                j,
            })
            .unwrap();
        let selected = [end.block_id(), start.block_id(), alignment.block_id()];
        let indices: Vec<_> = (9..18).chain(0..9).chain(18..21).collect();
        for marginalized in [false, true] {
            if marginalized {
                graph.marginalize(&[nuisance.block_id()]).unwrap();
            }
            let mut method = crate::LevenbergMarquardt::default();
            let (_, cov) = graph
                .optimize_with_covariance(
                    &mut method,
                    &Default::default(),
                    &selected,
                    &Default::default(),
                )
                .unwrap();
            assert_eq!(cov.shape(), (21, 21));
            for (i, &row) in indices.iter().enumerate() {
                for (k, &col) in indices.iter().enumerate() {
                    assert!((cov[(i, k)] - reference[(row, col)]).abs() < 1e-12);
                }
            }
        }
    }
}
