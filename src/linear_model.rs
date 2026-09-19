//! Cached linearization and original-coordinate verification shared by the backends.
use crate::{EvaluationError, Real, SolverError, optimization::norm_inf};
use faer::{
    MatMut, MatRef, Par,
    dyn_stack::{MemStack, StackReq},
    matrix_free::{BiLinOp, LinOp},
};
use faer_ext::nalgebra::{DMatrixView, Dyn};

pub(crate) mod block_factors;

#[derive(Debug)]
pub(crate) struct Block {
    pub row: usize,
    pub col: usize,
    pub rows: usize,
    pub cols: usize,
    pub start: usize,
}

#[derive(Debug)]
pub(crate) struct Jacobian<R> {
    pub rows: usize,
    pub cols: usize,
    pub blocks: Vec<Block>,
    pub values: Vec<R>,
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

pub(crate) struct LinearizedModel<R> {
    pub jacobian: Jacobian<R>,
    /// -r, the right-hand side of the linearized least-squares problem.
    pub negative_residual: Vec<R>,
    pub gradient: Vec<R>,
    pub scales: Vec<R>,
}
impl<R: Real> Default for LinearizedModel<R> {
    fn default() -> Self {
        Self {
            jacobian: Jacobian {
                rows: 0,
                cols: 0,
                blocks: Vec::new(),
                values: Vec::new(),
            },
            negative_residual: Vec::new(),
            gradient: Vec::new(),
            scales: Vec::new(),
        }
    }
}
impl<R: Real> LinearizedModel<R> {
    pub fn prepare(&mut self, dimension: usize) {
        self.jacobian.cols = dimension;
        self.gradient.resize(dimension, R::zero());
    }
    pub fn clear(&mut self) {
        self.jacobian.rows = 0;
        self.jacobian.blocks.clear();
        self.jacobian.values.clear();
        self.negative_residual.clear();
        self.gradient.fill(R::zero());
    }
    pub fn accumulate<'a>(
        &mut self,
        residual: &[R],
        jacobians: impl Iterator<Item = (usize, DMatrixView<'a, R, Dyn, Dyn>)>,
    ) -> Result<(), EvaluationError> {
        let row = self.jacobian.rows;
        self.jacobian.rows = row
            .checked_add(residual.len())
            .ok_or(EvaluationError::DimensionMismatch)?;
        self.negative_residual.extend(residual.iter().map(|&r| -r));
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
    pub fn prepare_scales(&mut self, floor: R) -> Result<(), SolverError> {
        if !floor.is_finite() || floor <= R::zero() {
            return Err(SolverError::InvalidOptions);
        }
        self.scales.resize(self.jacobian.cols, R::zero());
        self.scales.fill(R::zero());
        for b in &self.jacobian.blocks {
            for col in 0..b.cols {
                let norm = &mut self.scales[b.col + col];
                for row in 0..b.rows {
                    *norm = norm.hypot(self.jacobian.values[b.start + col * b.rows + row]);
                }
            }
        }
        norm_inf(&self.scales)?;
        for s in &mut self.scales {
            *s = s.max(floor);
        }
        Ok(())
    }

    /// Leaves the full damped normal residual and its initial gradient in original
    /// coordinates. Each backend subsequently applies its own diagnostic transform.
    pub fn verify_step(&self, work: &mut StepWorkspace<R>, lambda: R) -> StepVerification<R> {
        let (m, n) = (self.jacobian.rows, self.jacobian.cols);
        let finite_step = norm_inf(&work.step).is_ok();
        self.jacobian.apply(
            MatMut::from_column_major_slice_mut(&mut work.product, m, 1),
            MatRef::from_column_major_slice(&work.step, n, 1),
            Par::Seq,
            MemStack::new(&mut []),
        );
        let half = R::from_f64_impl(0.5);
        let mut predicted_reduction = R::zero();
        for (&g, &d) in self.gradient.iter().zip(&work.step) {
            predicted_reduction -= g * d;
        }
        for &v in &work.product {
            predicted_reduction -= (half * v) * v;
        }
        for (v, &rhs) in work.product.iter_mut().zip(&self.negative_residual) {
            *v -= rhs;
        }
        self.jacobian.transpose_apply(
            MatMut::from_column_major_slice_mut(&mut work.normal_residual, n, 1),
            MatRef::from_column_major_slice(&work.product, m, 1),
            Par::Seq,
            MemStack::new(&mut []),
        );
        let (mut residual_norm, mut reference_norm) = (R::zero(), R::zero());
        for j in 0..n {
            let s = self.scales[j];
            work.normal_residual[j] += (lambda * (s * work.step[j])) * s;
            residual_norm = residual_norm.hypot(work.normal_residual[j]);
            reference_norm = reference_norm.hypot(self.gradient[j]);
        }
        work.reference.copy_from_slice(&self.gradient);
        StepVerification {
            predicted_reduction,
            finite_step,
            residual_norm,
            reference_norm,
        }
    }
}

/// Each backend owns its verification scratch; the cached model stays immutable.
pub(crate) struct StepWorkspace<R> {
    pub step: Vec<R>,
    product: Vec<R>,
    pub normal_residual: Vec<R>,
    pub reference: Vec<R>,
}
impl<R: Real> Default for StepWorkspace<R> {
    fn default() -> Self {
        Self {
            step: Vec::new(),
            product: Vec::new(),
            normal_residual: Vec::new(),
            reference: Vec::new(),
        }
    }
}
impl<R: Real> StepWorkspace<R> {
    pub fn prepare_verification(&mut self, rows: usize, cols: usize) {
        self.product.resize(rows, R::zero());
        self.normal_residual.resize(cols, R::zero());
        self.reference.resize(cols, R::zero());
    }
    pub fn scale_residual(&mut self, scales: &[R]) {
        for ((r, g), &s) in self
            .normal_residual
            .iter_mut()
            .zip(&mut self.reference)
            .zip(scales)
        {
            *r /= s;
            *g /= s;
        }
    }
}
pub(crate) struct StepVerification<R> {
    pub predicted_reduction: R,
    pub finite_step: bool,
    pub residual_norm: R,
    pub reference_norm: R,
}
pub(crate) fn norm_l2<R: Real>(values: &[R]) -> R {
    values.iter().fold(R::zero(), |n, &x| n.hypot(x))
}
pub(crate) fn relative<R: Real>(norm: R, reference: R) -> R {
    if reference == R::zero() {
        norm
    } else {
        norm / reference
    }
}
pub(crate) fn relative_norm<R: Real>(values: &[R], reference: &[R]) -> R {
    relative(norm_l2(values), norm_l2(reference))
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer_ext::nalgebra::SMatrix;

    #[test]
    fn verification_reports_an_arbitrary_uphill_step_and_unobserved_coordinate() {
        let mut model = LinearizedModel::default();
        model.prepare(3);
        let j = SMatrix::<f64, 2, 3>::from_row_slice(&[1., 2., 0., 3., -1., 0.]);
        model
            .accumulate(&[-2., 5.], std::iter::once((0, j.as_view())))
            .unwrap();
        model.prepare_scales(0.5).unwrap();
        let mut work = StepWorkspace::default();
        work.step = vec![0.2, -0.3, 0.4];
        work.prepare_verification(2, 3);
        let verified = model.verify_step(&mut work, 0.7);
        // Original cost 14.5; J*delta+r = [-2.4,5.9], cost 20.285.
        assert!((verified.predicted_reduction + 5.785).abs() < 1e-12);
        // The unused third column still contributes its floored damping term.
        for (&actual, expected) in work.normal_residual.iter().zip([16.7, -11.75, 0.07]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(work.reference, vec![13., -9., 0.]);
        assert!(verified.finite_step && verified.residual_norm > 20.);
        assert_eq!(model.gradient, work.reference);
        assert_eq!(model.negative_residual, vec![2., -5.]);
    }
}
