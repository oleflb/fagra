//! Direct Schur elimination for bounded retained systems and independent landmarks.
use std::ops::Range;

use faer::{
    Accum, MatMut, MatRef, Par,
    dyn_stack::{MemBuffer, MemStack},
    linalg::{
        cholesky::llt,
        matmul::{
            matmul,
            triangular::{self, BlockStructure},
        },
        triangular_solve,
    },
};
use faer_ext::nalgebra::{DMatrixView, Dim, Dyn, VectorView};

use crate::{
    EvaluationError, Real, SolverError,
    covariance::{CovarianceBackend, SelectedCovariance},
    linear_model::{LinearizedModel, StepWorkspace},
    marginalization::MatrixBuffer,
    optimization::{DampedStep, LeastSquaresBackend, norm_inf},
};

/// CPU direct Schur backend with a dense retained system and heterogeneous landmarks.
///
/// Coordinates are a retained prefix followed by independent eliminated blocks of
/// the supplied widths, in schema/pool order. Retained Jacobian blocks may have any
/// widths and arbitrary couplings. Each residual emission may touch at most one
/// complete eliminated block. Unary and retained-only factors/priors are supported;
/// couplings between landmarks are rejected, never silently approximated.
///
/// Retire connected landmarks before/with old trajectory states: marginalizing a
/// trajectory state while retaining its landmarks can create a prior coupling those
/// landmarks, which this backend cannot accept. No retirement or FEJ policy is imposed.
/// Reconstruct the backend when partition dimensions/order change.
///
/// Uses scaled normal equations, exact local Cholesky elimination, and dense reduced
/// Cholesky. LM damping applies to both partitions. No pivot repair/pseudoinverse is
/// used. Covariance supports retained selections only, using the undamped reduced
/// system; all landmarks and the retained system must pass rank checks even for an
/// empty selection. Existing `Problem::joint_covariance` remains a separate dense query.
///
/// Storage is O(r² + r*l + sum(w²) + cached Jacobians), not O((r+l)²).
/// Buffers are reused within warmed capacities. Dense cross blocks trade sparsity
/// for simplicity; use this for bounded windows, not persistent maps.
pub struct DenseSchur<R: Real = f64> {
    retained: usize,
    widths: Vec<usize>,
    landmarks: Vec<Range<usize>>,
    owners: Vec<usize>,
    starts: Vec<usize>,
    gram: Vec<R>,
    factors: Vec<R>,
    retained_gram: MatrixBuffer<R>,
    cross: MatrixBuffer<R>,
    solved: MatrixBuffer<R>,
    reduced: MatrixBuffer<R>,
    rhs: Vec<R>,
    model: LinearizedModel<R>,
    work: StepWorkspace<R>,
    scratch: Option<MemBuffer>,
    ready: bool,
}

impl<R: Real> DenseSchur<R> {
    /// Choose the retained prefix dimension and ordered, positive landmark widths.
    /// Validated against the graph dimension on preparation. An empty landmark list
    /// supports a general retained-only graph, including dense marginal priors.
    pub fn new(
        retained_dimension: usize,
        eliminated_widths: impl IntoIterator<Item = usize>,
    ) -> Self {
        Self {
            retained: retained_dimension,
            widths: eliminated_widths.into_iter().collect(),
            landmarks: Vec::new(),
            owners: Vec::new(),
            starts: Vec::new(),
            gram: Vec::new(),
            factors: Vec::new(),
            retained_gram: MatrixBuffer::default(),
            cross: MatrixBuffer::default(),
            solved: MatrixBuffer::default(),
            reduced: MatrixBuffer::default(),
            rhs: Vec::new(),
            model: LinearizedModel::default(),
            work: StepWorkspace::default(),
            scratch: None,
            ready: false,
        }
    }

    fn assemble(&mut self) {
        self.gram.fill(R::zero());
        self.retained_gram.as_mut().fill(R::zero());
        self.cross.as_mut().fill(R::zero());
        let j = &self.model.jacobian;
        for (index, left) in j.blocks.iter().enumerate() {
            if left.rows == 0 {
                continue;
            }
            for (other, right) in j.blocks[..=index].iter().enumerate().rev() {
                if right.row != left.row || right.rows != left.rows {
                    break;
                }
                for col in 0..right.cols {
                    for row in 0..left.cols {
                        if other == index && row < col {
                            continue;
                        }
                        let a = left.col + row;
                        let b = right.col + col;
                        let mut value = R::zero();
                        for k in 0..left.rows {
                            value += (j.values[left.start + row * left.rows + k]
                                / self.model.scales[a])
                                * (j.values[right.start + col * right.rows + k]
                                    / self.model.scales[b]);
                        }
                        if a < self.retained && b < self.retained {
                            self.retained_gram[(a.max(b), a.min(b))] += value;
                        } else if a >= self.retained && b >= self.retained {
                            let id = self.owners[a - self.retained];
                            let range = &self.landmarks[id];
                            let (a, b) = (a - range.start, b - range.start);
                            self.gram[self.starts[id] + a.min(b) * range.len() + a.max(b)] += value;
                        } else {
                            let (landmark, retained) =
                                if a >= self.retained { (a, b) } else { (b, a) };
                            self.cross[(landmark - self.retained, retained)] += value;
                        }
                    }
                }
            }
        }
    }

    fn solve_landmarks(&mut self, lambda: R, tolerance: Option<R>) -> Result<(), SolverError> {
        let r = self.retained;
        self.factors.copy_from_slice(&self.gram);
        for (id, range) in self.landmarks.iter().enumerate() {
            let w = range.len();
            let start = self.starts[id];
            let factor = &mut self.factors[start..start + w * w];
            for i in 0..w {
                factor[i * w + i] += lambda;
            }
            factorize(
                MatMut::from_column_major_slice_mut(factor, w, w),
                &mut self.scratch,
            )?;
            if let Some(tolerance) = tolerance {
                check_rank(
                    MatRef::from_column_major_slice(factor, w, w),
                    (0..w).map(|i| self.gram[start + i * w + i]),
                    tolerance,
                )?;
            }
            let offset = range.start - r;
            let mut solved = self.solved.as_mut().subrows_mut(offset, w);
            for row in 0..w {
                for col in 0..r {
                    solved[(row, col)] = self.cross[(offset + row, col)];
                }
                solved[(row, r)] =
                    -self.model.gradient[range.start + row] / self.model.scales[range.start + row];
            }
            solve(MatRef::from_column_major_slice(factor, w, w), solved);
        }
        Ok(())
    }

    fn reduce(&mut self, lambda: R, tolerance: Option<R>) -> Result<(), SolverError> {
        self.solve_landmarks(lambda, tolerance)?;
        let r = self.retained;
        for col in 0..r {
            self.rhs[col] = -self.model.gradient[col] / self.model.scales[col];
            for row in col..r {
                self.reduced[(row, col)] = self.retained_gram[(row, col)];
            }
            self.reduced[(col, col)] += lambda;
        }
        // One product over all local solves lets faer's dense kernels amortize
        // dispatch across small, heterogeneous landmarks.
        triangular::matmul(
            self.reduced.as_mut(),
            BlockStructure::TriangularLower,
            Accum::Add,
            self.cross.as_ref().transpose(),
            BlockStructure::Rectangular,
            self.solved.as_ref().subcols(0, r),
            BlockStructure::Rectangular,
            -R::one(),
            Par::Seq,
        );
        matmul(
            MatMut::from_column_major_slice_mut(&mut self.rhs, r, 1),
            Accum::Add,
            self.cross.as_ref().transpose(),
            self.solved.as_ref().subcols(r, 1),
            -R::one(),
            Par::Seq,
        );
        factorize(self.reduced.as_mut(), &mut self.scratch)?;
        if let Some(tolerance) = tolerance {
            // Compare with the original retained diagonal, not a cancellation-small
            // Schur diagonal: eliminating a free landmark must not hide a gauge.
            check_rank(
                self.reduced.as_ref(),
                (0..r).map(|i| self.retained_gram[(i, i)]),
                tolerance,
            )?;
        }
        Ok(())
    }
}

impl<R: Real> LeastSquaresBackend for DenseSchur<R> {
    type Scalar = R;

    fn prepare(&mut self, dimension: usize) -> Result<(), SolverError> {
        self.ready = false;
        let eliminated = dimension
            .checked_sub(self.retained)
            .ok_or(SolverError::InvalidOptions)?;
        self.landmarks.clear();
        self.starts.clear();
        let mut offset = self.retained;
        let mut size = 0usize;
        for &width in &self.widths {
            let end = offset
                .checked_add(width)
                .filter(|&end| width > 0 && end <= dimension)
                .ok_or(SolverError::InvalidOptions)?;
            self.landmarks.push(offset..end);
            self.starts.push(size);
            size = width
                .checked_mul(width)
                .and_then(|n| size.checked_add(n))
                .ok_or(SolverError::InvalidOptions)?;
            offset = end;
        }
        if offset != dimension {
            return Err(SolverError::InvalidOptions);
        }
        let columns = self
            .retained
            .checked_add(1)
            .ok_or(SolverError::InvalidOptions)?;
        for (rows, cols) in [
            (self.retained, self.retained),
            (eliminated, columns),
            (size, 1),
        ] {
            rows.checked_mul(cols)
                .and_then(|n| n.checked_mul(size_of::<R>()))
                .filter(|&bytes| bytes <= isize::MAX as usize)
                .ok_or(SolverError::InvalidOptions)?;
        }
        eliminated
            .checked_mul(size_of::<usize>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(SolverError::InvalidOptions)?;
        self.owners.resize(eliminated, 0);
        for (id, range) in self.landmarks.iter().enumerate() {
            self.owners[range.start - self.retained..range.end - self.retained].fill(id);
        }
        self.gram.resize(size, R::zero());
        self.factors.resize(size, R::zero());
        self.retained_gram.resize(self.retained, self.retained);
        self.reduced.resize(self.retained, self.retained);
        self.cross.resize(eliminated, self.retained);
        self.solved.resize(eliminated, columns);
        self.rhs.resize(self.retained, R::zero());
        self.model.prepare(dimension);
        self.work.step.resize(dimension, R::zero());
        Ok(())
    }

    fn clear(&mut self) {
        self.ready = false;
        self.model.clear();
    }

    fn accumulate<'a, Rows: Dim>(
        &mut self,
        residual: VectorView<'_, R, Rows>,
        jacobians: impl Iterator<Item = (usize, DMatrixView<'a, R, Dyn, Dyn>)> + Clone,
    ) -> Result<(), EvaluationError> {
        self.ready = false;
        let mut landmark = None;
        for (index, (col, j)) in jacobians.clone().enumerate() {
            let end = col
                .checked_add(j.ncols())
                .filter(|&end| end <= self.model.jacobian.cols)
                .ok_or(EvaluationError::DimensionMismatch)?;
            if j.nrows() != residual.len() {
                return Err(EvaluationError::DimensionMismatch);
            }
            if j.ncols() == 0 {
                continue;
            }
            if col >= self.retained {
                let id = self.owners[col - self.retained];
                if self.landmarks[id] != (col..end) || landmark.replace(id).is_some() {
                    return Err(EvaluationError::UnsupportedStructure);
                }
            } else if end > self.retained {
                return Err(EvaluationError::UnsupportedStructure);
            }
            for (other, block) in jacobians.clone().take(index) {
                if col < other + block.ncols() && other < end {
                    return Err(EvaluationError::DimensionMismatch);
                }
            }
        }
        self.model.accumulate(residual.as_slice(), jacobians)
    }

    fn gradient_norm(&self) -> Result<R, SolverError> {
        norm_inf(&self.model.gradient)
    }

    fn solve(&mut self) -> Result<&[R], SolverError> {
        self.prepare_damping(R::from_f64_impl(1e-6))?;
        self.solve_damped(R::zero()).map(|step| step.delta)
    }

    fn prepare_damping(&mut self, min_column_norm: R) -> Result<(), SolverError> {
        self.ready = false;
        self.model.prepare_scales(min_column_norm)?;
        norm_inf(&self.model.gradient)?;
        self.work
            .prepare_verification(self.model.jacobian.rows, self.model.jacobian.cols);
        self.assemble();
        self.ready = true;
        Ok(())
    }

    fn solve_damped(&mut self, lambda: R) -> Result<DampedStep<'_, R>, SolverError> {
        if !self.ready || !lambda.is_finite() || lambda < R::zero() {
            return Err(SolverError::InvalidOptions);
        }
        self.reduce(lambda, None)?;
        solve(
            self.reduced.as_ref(),
            MatMut::from_column_major_slice_mut(&mut self.rhs, self.retained, 1),
        );
        for i in 0..self.retained {
            self.work.step[i] = self.rhs[i] / self.model.scales[i];
        }
        for range in &self.landmarks {
            for i in range.clone() {
                let row = i - self.retained;
                let mut value = self.solved[(row, self.retained)];
                for col in 0..self.retained {
                    value -= self.solved[(row, col)] * self.rhs[col];
                }
                self.work.step[i] = value / self.model.scales[i];
            }
        }
        let verified = self.model.verify_step(&mut self.work, lambda);
        if !verified.finite_step || !verified.predicted_reduction.is_finite() {
            return Err(SolverError::LinearSolveFailed);
        }
        Ok(DampedStep {
            delta: &self.work.step,
            predicted_reduction: verified.predicted_reduction,
        })
    }
}

impl<R: Real> CovarianceBackend for DenseSchur<R> {
    fn covariance(
        &mut self,
        _: &mut crate::DenseNormalCholesky<R>,
        selected: &mut SelectedCovariance<R>,
    ) -> Result<(), SolverError> {
        selected.retained_only(self.retained)?;
        self.prepare_damping(R::from_f64_impl(1e-6))
            .and_then(|()| self.reduce(R::zero(), Some(selected.options.relative_pivot_tolerance)))
            .map_err(|error| match error {
                SolverError::LinearSolveFailed => SolverError::SingularInformation,
                SolverError::Evaluation(EvaluationError::InvalidEvaluation) => {
                    SolverError::InvalidInformation
                }
                other => other,
            })?;
        selected.compute_factor(self.reduced.as_ref(), &self.model.scales[..self.retained])
    }
}

fn factorize<R: Real>(
    matrix: MatMut<'_, R>,
    scratch: &mut Option<MemBuffer>,
) -> Result<(), SolverError> {
    let n = matrix.nrows();
    for col in 0..n {
        for row in col..n {
            if !matrix[(row, col)].is_finite() {
                return Err(EvaluationError::InvalidEvaluation.into());
            }
        }
    }
    if n == 0 {
        return Ok(());
    }
    let req = llt::factor::cholesky_in_place_scratch::<R>(n, Par::Seq, Default::default());
    if !scratch
        .as_mut()
        .is_some_and(|s| MemStack::new(s).can_hold(req))
    {
        *scratch = Some(MemBuffer::new(req));
    }
    let Some(scratch) = scratch else {
        return Err(SolverError::LinearSolveFailed);
    };
    llt::factor::cholesky_in_place(
        matrix,
        Default::default(),
        Par::Seq,
        MemStack::new(scratch),
        Default::default(),
    )
    .map_err(|_| SolverError::LinearSolveFailed)?;
    Ok(())
}

fn solve<R: Real>(lower: MatRef<'_, R>, mut rhs: MatMut<'_, R>) {
    triangular_solve::solve_lower_triangular_in_place(lower, rhs.as_mut(), Par::Seq);
    triangular_solve::solve_upper_triangular_in_place(lower.transpose(), rhs, Par::Seq);
}

fn check_rank<R: Real>(
    lower: MatRef<'_, R>,
    diagonal: impl Iterator<Item = R>,
    tolerance: R,
) -> Result<(), SolverError> {
    for (i, diagonal) in diagonal.enumerate() {
        let pivot = lower[(i, i)];
        if !pivot.is_finite() || !diagonal.is_finite() {
            return Err(SolverError::InvalidInformation);
        }
        if diagonal <= R::zero()
            || pivot <= R::zero()
            || (pivot / diagonal.sqrt()).powi(2) <= tolerance
        {
            return Err(SolverError::SingularInformation);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
