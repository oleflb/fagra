//! One scalar variable constrained by an ordinary prior factor.
//!
//! Run with `cargo run --example scalar_prior` to optimize in both precisions.

use faer_ext::nalgebra::{Const, DefaultAllocator, SMatrix, SVector};
use fagra::{
    BlockId, EvaluationError, Factor, Jacobian, JacobianBlock, LinearizationSink, Real, Solver,
    SolverError, StateKey, StateStore, Tangent, Variable,
};

/// The additive real line, with one tangent coordinate in the stored value's units.
pub struct Scalar<R: Real = f64>(pub R);

impl<R: Real> Variable for Scalar<R> {
    type Scalar = R;
    type Dim = Const<1>;
    type Allocator = DefaultAllocator;

    fn identity() -> Self {
        Self(R::zero())
    }

    fn compose(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }

    fn inverse(&self) -> Self {
        Self(-self.0)
    }

    fn exp(delta: &Tangent<Self>) -> Self {
        Self(delta[0])
    }

    fn log(&self) -> Tangent<Self> {
        SVector::<R, 1>::new(self.0)
    }

    fn adjoint(&self) -> Jacobian<Self> {
        SMatrix::identity()
    }

    fn right_jacobian(_: &Tangent<Self>) -> Jacobian<Self> {
        SMatrix::identity()
    }

    fn right_jacobian_inverse(_: &Tangent<Self>) -> Jacobian<Self> {
        SMatrix::identity()
    }
}

pub struct Prior<R: Real = f64> {
    pub variable: StateKey<Scalar<R>>,
    pub measurement: R,
}

impl<R: Real, S: StateStore<Scalar<R>>> Factor<S> for Prior<R> {
    type Scalar = R;

    fn visit_variables(&self, mut visitor: impl FnMut(BlockId)) {
        visitor(self.variable.block_id());
    }

    fn cost(&self, states: &S) -> Result<R, EvaluationError> {
        let residual = states.get(self.variable)?.0 - self.measurement;
        let cost = R::from_f64_impl(0.5) * residual * residual;
        if !cost.is_finite() {
            return Err(EvaluationError::InvalidEvaluation);
        }
        Ok(cost)
    }

    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let residual = SVector::<R, 1>::new(states.get(self.variable)?.0 - self.measurement);
        let jacobian = SMatrix::<R, 1, 1>::new(R::one());
        // Ordinary factors receive an already-scoped sink. It checks finiteness.
        sink.residual(&residual, &[JacobianBlock::new(self.variable, &jacobian)])
    }
}

fagra::states! { pub States<R> { scalars: Scalar<R> } }
fagra::factors! { pub Factors<R> { priors: Prior<R> } }

fn run<R: Real>() -> Result<(), SolverError> {
    let mut solver = Solver::<States<R>, Factors<R>>::new();
    let x = solver.add(Scalar(R::zero()));
    let prior = solver.add_factor(Prior {
        variable: x,
        measurement: R::from_f64_impl(3.0),
    })?;

    let report = solver.optimize()?;
    let estimate = solver.get(x)?;
    let cost = solver.factor_cost(prior)?;
    println!(
        "estimate = {}, cost = {cost}, steps = {}",
        estimate.0, report.iterations
    );
    Ok(())
}

fn main() -> Result<(), SolverError> {
    run::<f32>()?;
    run::<f64>()
}
