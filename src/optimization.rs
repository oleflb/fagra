use std::{collections::HashMap, convert::Infallible, ops::Range};

use faer_ext::nalgebra::{DimName, Matrix, Storage, U1, storage::IsContiguous};

use crate::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorId, JacobianBlock, LinearizationSink,
    Real, SolverError, Variable,
    factors::{FactorSchema, FactorVisitor},
    states::{StateSchema, StateVisitor},
    storage::{BatchPool, FactorPool, StatePool, checked_cost},
};

mod gauss_newton;

pub use gauss_newton::GaussNewton;

/// Stopping controls shared by nonlinear optimizers.
///
/// Tolerances must be finite and nonnegative; zero requires an exact match.
/// Defaults are `max(1e-8, 8ε)` for the gradient, `max(1e-10, ε)` for the step,
/// and `max(1e-12, ε)` for cost, where `ε` is the scalar's machine epsilon.
/// Absolute gradient and step tolerances may need tuning for the model's units.
#[derive(Debug, Clone, Copy)]
pub struct OptimizeOptions<R: Real = f64> {
    /// Maximum number of accepted full steps; must be positive.
    pub max_iterations: usize,
    /// Stop when the infinity norm of Jᵀr is at most this absolute tolerance.
    pub gradient_tolerance: R,
    /// Stop before applying a step whose infinity norm is at most this value.
    /// Units are the variables' tangent coordinates, not an ambient state norm.
    pub step_tolerance: R,
    /// Stop after a nonnegative cost decrease no larger than this value times
    /// `max(1, previous_cost)`. An exactly zero cost also terminates.
    pub cost_tolerance: R,
}

impl<R: Real> Default for OptimizeOptions<R> {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            gradient_tolerance: R::from_f64_impl(1e-8)
                .max(R::epsilon_impl() * R::from_f64_impl(8.0)),
            step_tolerance: R::from_f64_impl(1e-10).max(R::epsilon_impl()),
            cost_tolerance: R::from_f64_impl(1e-12).max(R::epsilon_impl()),
        }
    }
}

impl<R: Real> OptimizeOptions<R> {
    fn validate(&self) -> Result<(), SolverError> {
        if self.max_iterations == 0
            || [
                self.gradient_tolerance,
                self.step_tolerance,
                self.cost_tolerance,
            ]
            .iter()
            .any(|x| !x.is_finite() || *x < R::zero())
        {
            return Err(SolverError::InvalidOptions);
        }
        Ok(())
    }
}

/// The stopping criterion met by an optimization call, not a claim of global optimality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TerminationReason {
    /// No scalar optimization coordinates were present.
    NoVariables,
    /// The linearized gradient is sufficiently small.
    GradientTolerance,
    /// The proposed tangent step is sufficiently small; it was not applied.
    StepTolerance,
    /// The accepted cost is zero or its nonnegative decrease is sufficiently small.
    CostTolerance,
}

/// Summary of a successful optimization call.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct OptimizeReport<R: Real = f64> {
    /// Number of committed steps, excluding convergence-only linearizations.
    pub iterations: usize,
    /// Nonlinear objective before the first step.
    pub initial_cost: R,
    /// Nonlinear objective at the returned estimates.
    pub final_cost: R,
    /// Criterion responsible for stopping.
    pub termination: TerminationReason,
}

/// Internal extension point for nonlinear methods, independent of graph ownership.
pub trait Optimizer<S: StateSchema, F> {
    /// Optimize the supplied schemas. Ordinary errors must leave accepted estimates visible.
    fn optimize(
        &mut self,
        states: &mut S,
        factors: &mut F,
        options: &OptimizeOptions<S::Scalar>,
    ) -> Result<OptimizeReport<S::Scalar>, SolverError>;
}

/// Internal extension point for assembling and solving a linearized least-squares model.
///
/// The shared sink validates emissions and resolves column offsets. Backends may
/// accumulate normal equations, retain Jacobian rows for QR, or build an operator.
/// No Hessian representation is imposed on implementations. Calls are statically dispatched.
pub trait LeastSquaresBackend {
    /// Scalar shared by the model, gradient, and step buffers.
    type Scalar: Real;

    /// Prepare reusable storage for this number of scalar coordinates.
    fn prepare(&mut self, dimension: usize) -> Result<(), SolverError>;
    /// Start a fresh linearization, discarding any previous factorization.
    fn clear(&mut self);
    /// Consume validated residuals and Jacobians. Columns correspond one-to-one
    /// with the Jacobian descriptors; repeated variable identities have been rejected.
    fn accumulate<Rows: DimName, S>(
        &mut self,
        residual: &Matrix<Self::Scalar, Rows, U1, S>,
        jacobians: &[JacobianBlock<'_, Self::Scalar>],
        columns: &[usize],
    ) -> Result<(), EvaluationError>
    where
        S: Storage<Self::Scalar, Rows, U1> + IsContiguous;
    /// Infinity norm of the unmodified linearized gradient, rejecting nonfinite values.
    /// Read before `solve`, which may overwrite gradient storage.
    fn gradient_norm(&self) -> Result<Self::Scalar, SolverError>;
    /// Solve the local model and borrow its step buffer. The caller checks the
    /// step's finiteness before retraction; no separate output copy is required.
    /// May destroy the model and RHS: call once per completed linearization.
    fn solve(&mut self) -> Result<&[Self::Scalar], SolverError>;
}

struct Block {
    id: BlockId,
    offset: usize,
    width: usize,
}
struct FactorPlan {
    id: FactorId,
    dependencies: Range<usize>,
}

#[derive(Default)]
struct Layout {
    dimension: usize,
    blocks: Vec<Block>,
    block_index: HashMap<BlockId, usize>,
    factors: Vec<FactorPlan>,
    factor_index: HashMap<FactorId, usize>,
    dependencies: Vec<usize>,
}

impl Layout {
    fn clear(&mut self) {
        self.dimension = 0;
        self.blocks.clear();
        self.block_index.clear();
        self.factors.clear();
        self.factor_index.clear();
        self.dependencies.clear();
    }

    fn factor(
        &mut self,
        id: FactorId,
        visit: impl FnOnce(&mut dyn FnMut(BlockId)),
    ) -> Result<(), SolverError> {
        let start = self.dependencies.len();
        let mut invalid = false;
        visit(&mut |id| match self.block_index.get(&id).copied() {
            Some(index) => {
                if !self.dependencies[start..].contains(&index) {
                    self.dependencies.push(index);
                }
            }
            None => invalid = true,
        });
        if invalid || self.factor_index.insert(id, self.factors.len()).is_some() {
            return Err(EvaluationError::InvalidEmission.into());
        }
        self.factors.push(FactorPlan {
            id,
            dependencies: start..self.dependencies.len(),
        });
        Ok(())
    }
}

impl<R: Real> StateVisitor<R> for Layout {
    type Error = SolverError;
    fn pool<T: Variable<Scalar = R>>(
        &mut self,
        pool: &mut StatePool<T>,
    ) -> Result<(), SolverError> {
        pool.prepare_trial();
        for (id, _) in pool.iter() {
            if self.block_index.insert(id, self.blocks.len()).is_some() {
                return Err(EvaluationError::InvalidEmission.into());
            }
            self.blocks.push(Block {
                id,
                offset: self.dimension,
                width: T::Dim::DIM,
            });
            self.dimension = self
                .dimension
                .checked_add(T::Dim::DIM)
                .ok_or(EvaluationError::DimensionMismatch)?;
        }
        Ok(())
    }
}

impl<S, R: Real> FactorVisitor<S, R> for Layout {
    type Error = SolverError;
    fn standalone<T: Factor<S, Scalar = R>>(
        &mut self,
        pool: &mut FactorPool<T>,
    ) -> Result<(), SolverError> {
        for (id, factor) in pool.iter() {
            self.factor(id, |visit| factor.visit_variables(visit))?;
        }
        Ok(())
    }
    fn batch<B: FactorBatch<S, Scalar = R>>(
        &mut self,
        pool: &mut BatchPool<B, B::Factor>,
    ) -> Result<(), SolverError> {
        for (_, model, factors) in pool.iter() {
            for (id, factor) in factors {
                self.factor(id, |visit| model.visit_variables(factor, visit))?;
            }
        }
        Ok(())
    }
}

struct Cost<'a, S, R> {
    states: &'a S,
    total: R,
}
impl<S, R: Real> FactorVisitor<S, R> for Cost<'_, S, R> {
    type Error = SolverError;
    fn standalone<T: Factor<S, Scalar = R>>(
        &mut self,
        pool: &mut FactorPool<T>,
    ) -> Result<(), SolverError> {
        for (_, factor) in pool.iter() {
            self.total += checked_cost(factor.cost(self.states))?;
        }
        Ok(())
    }
    fn batch<B: FactorBatch<S, Scalar = R>>(
        &mut self,
        pool: &mut BatchPool<B, B::Factor>,
    ) -> Result<(), SolverError> {
        for (_, model, factors) in pool.iter() {
            if !factors.is_empty() {
                self.total += checked_cost(model.cost(self.states, factors))?;
            }
        }
        Ok(())
    }
}

fn cost<S, F: FactorSchema<S>>(states: &S, factors: &mut F) -> Result<F::Scalar, SolverError> {
    let mut visitor = Cost {
        states,
        total: faer::traits::math_utils::zero::<F::Scalar>(),
    };
    factors.visit(&mut visitor)?;
    checked_cost(Ok(visitor.total))
}

struct Stage<'a, R> {
    delta: &'a [R],
    position: usize,
}
impl<R: Real> StateVisitor<R> for Stage<'_, R> {
    type Error = EvaluationError;
    fn pool<T: Variable<Scalar = R>>(
        &mut self,
        pool: &mut StatePool<T>,
    ) -> Result<(), EvaluationError> {
        let count = pool
            .iter()
            .len()
            .checked_mul(T::Dim::DIM)
            .ok_or(EvaluationError::DimensionMismatch)?;
        let end = self
            .position
            .checked_add(count)
            .ok_or(EvaluationError::DimensionMismatch)?;
        let delta = self
            .delta
            .get(self.position..end)
            .ok_or(EvaluationError::DimensionMismatch)?;
        pool.stage(delta)?;
        self.position = end;
        Ok(())
    }
}

struct Accept;
struct Reject;
impl<R: Real> StateVisitor<R> for Accept {
    type Error = Infallible;
    fn pool<T: Variable<Scalar = R>>(&mut self, pool: &mut StatePool<T>) -> Result<(), Infallible> {
        pool.accept_trial();
        Ok(())
    }
}
impl<R: Real> StateVisitor<R> for Reject {
    type Error = Infallible;
    fn pool<T: Variable<Scalar = R>>(&mut self, pool: &mut StatePool<T>) -> Result<(), Infallible> {
        pool.reject_trial();
        Ok(())
    }
}

// Rejection also runs on a retraction or evaluator panic, including partially
// staged pools. Acceptance swaps all pools before dropping the old estimates.
struct Trial<'a, S: StateSchema> {
    states: &'a mut S,
}
impl<S: StateSchema> Drop for Trial<'_, S> {
    fn drop(&mut self) {
        self.states.visit(&mut Reject).unwrap();
    }
}

struct CheckedSink<'a, B> {
    backend: &'a mut B,
    layout: &'a Layout,
    seen: &'a mut [bool],
    block_marks: &'a mut [usize],
    columns: &'a mut Vec<usize>,
    emission: usize,
    allowed: Range<usize>,
    next_expected: usize,
    active: Option<usize>,
    failed: bool,
}

impl<R: Real, B: LeastSquaresBackend<Scalar = R>> LinearizationSink for CheckedSink<'_, B> {
    type Scalar = R;

    fn factor(
        &mut self,
        id: FactorId,
        emit: impl FnOnce(&mut Self) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        let result = (|| {
            if self.active.is_some() {
                return Err(EvaluationError::InvalidEmission);
            }
            // Most evaluators emit in selection order: avoid hashing on that path.
            let index = if self.next_expected < self.allowed.end
                && self.layout.factors[self.next_expected].id == id
            {
                let index = self.next_expected;
                self.next_expected += 1;
                index
            } else {
                *self
                    .layout
                    .factor_index
                    .get(&id)
                    .ok_or(EvaluationError::InvalidEmission)?
            };
            if !self.allowed.contains(&index) || self.seen[index] {
                return Err(EvaluationError::InvalidEmission);
            }
            self.seen[index] = true;
            self.active = Some(index);
            let result = emit(self);
            self.active = None;
            result
        })();
        self.failed |= result.is_err();
        result
    }

    fn residual<Rows: DimName, S>(
        &mut self,
        residual: &Matrix<Self::Scalar, Rows, U1, S>,
        jacobians: &[JacobianBlock<'_, Self::Scalar>],
    ) -> Result<(), EvaluationError>
    where
        S: Storage<Self::Scalar, Rows, U1> + IsContiguous,
    {
        let result = (|| {
            let index = self.active.ok_or(EvaluationError::InvalidEmission)?;
            let dependencies =
                &self.layout.dependencies[self.layout.factors[index].dependencies.clone()];
            if jacobians.len() > dependencies.len() {
                return Err(EvaluationError::InvalidEmission);
            }
            if residual.iter().any(|x| !x.is_finite()) {
                return Err(EvaluationError::InvalidEvaluation);
            }
            self.emission = match self.emission.checked_add(1) {
                Some(value) => value,
                None => {
                    self.block_marks.fill(0);
                    1
                }
            };
            self.columns.clear();
            for (position, jacobian) in jacobians.iter().enumerate() {
                let id = jacobian.variable();
                let block_index = match dependencies.get(position).copied() {
                    Some(i) if self.layout.blocks[i].id == id => i,
                    _ => {
                        let i = *self
                            .layout
                            .block_index
                            .get(&id)
                            .ok_or(EvaluationError::InvalidEmission)?;
                        if !dependencies.contains(&i) {
                            return Err(EvaluationError::InvalidEmission);
                        }
                        i
                    }
                };
                if self.block_marks[block_index] == self.emission {
                    return Err(EvaluationError::InvalidEmission);
                }
                self.block_marks[block_index] = self.emission;
                let block = &self.layout.blocks[block_index];
                let matrix = jacobian.jacobian();
                let (rs, cs) = matrix.strides();
                if matrix.nrows() != Rows::DIM
                    || matrix.ncols() != block.width
                    || rs > isize::MAX as usize
                    || cs > isize::MAX as usize
                {
                    return Err(EvaluationError::DimensionMismatch);
                }
                if matrix.iter().any(|x| !x.is_finite()) {
                    return Err(EvaluationError::InvalidEvaluation);
                }
                self.columns.push(block.offset);
            }
            self.backend.accumulate(residual, jacobians, self.columns)
        })();
        self.failed |= result.is_err();
        result
    }
}

struct Linearize<'a, 'b, S, B> {
    states: &'a S,
    sink: &'a mut CheckedSink<'b, B>,
    position: usize,
}
impl<S, R: Real, B: LeastSquaresBackend<Scalar = R>> FactorVisitor<S, R>
    for Linearize<'_, '_, S, B>
{
    type Error = SolverError;
    fn standalone<T: Factor<S, Scalar = B::Scalar>>(
        &mut self,
        pool: &mut FactorPool<T>,
    ) -> Result<(), SolverError> {
        for (id, factor) in pool.iter() {
            self.sink.allowed = self.position..self.position + 1;
            self.sink.next_expected = self.position;
            self.position += 1;
            self.sink
                .factor(id, |sink| factor.linearize(self.states, sink))?;
        }
        Ok(())
    }
    fn batch<M: FactorBatch<S, Scalar = B::Scalar>>(
        &mut self,
        pool: &mut BatchPool<M, M::Factor>,
    ) -> Result<(), SolverError> {
        for (_, model, factors) in pool.iter() {
            self.sink.allowed = self.position..self.position + factors.len();
            self.sink.next_expected = self.position;
            self.position += factors.len();
            if !factors.is_empty() {
                model.linearize(self.states, factors, self.sink)?;
            }
        }
        Ok(())
    }
}

pub(crate) fn norm_inf<R: Real>(values: &[R]) -> Result<R, SolverError> {
    let mut norm = R::zero();
    for &value in values {
        if !value.is_finite() {
            return Err(EvaluationError::InvalidEvaluation.into());
        }
        norm = norm.max(value.abs());
    }
    Ok(norm)
}
