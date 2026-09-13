//! One scalar variable constrained by an ordinary prior factor.
//!
//! Compile with `cargo check --example scalar_prior`. The model is complete,
//! and storage/cost operations work. The optimization call is still an API stub.

use faer_ext::nalgebra::{SMatrix, SVector};
use fagra::{
    BlockId, EvaluationError, Factor, JacobianBlock, LinearizationSink, Solver, SolverError,
    StateKey, StateStore, Variable,
};

pub struct Scalar(pub f64);

impl Variable for Scalar {
    type Tangent = f64;
    const DOF: usize = 1;

    fn tangent_from_slice(delta: &[f64]) -> f64 {
        delta[0]
    }

    fn retract(&self, delta: &f64) -> Self {
        Self(self.0 + *delta)
    }
}

pub struct Prior {
    pub variable: StateKey<Scalar>,
    pub measurement: f64,
}

impl<S: StateStore<Scalar>> Factor<S> for Prior {
    fn visit_variables(&self, mut visitor: impl FnMut(BlockId)) {
        visitor(self.variable.block_id());
    }

    fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
        let residual = states.get(self.variable)?.0 - self.measurement;
        let cost = 0.5 * residual * residual;
        if !cost.is_finite() {
            return Err(EvaluationError::InvalidEvaluation);
        }
        Ok(cost)
    }

    fn linearize<L: LinearizationSink>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let residual = SVector::<f64, 1>::new(states.get(self.variable)?.0 - self.measurement);
        let jacobian = SMatrix::<f64, 1, 1>::new(1.0);
        // Ordinary factors receive an already-scoped sink. It checks finiteness.
        sink.residual(&residual, &[JacobianBlock::new(self.variable, &jacobian)])
    }
}

fagra::states! { pub States { scalars: Scalar } }
fagra::factors! { pub Factors { priors: Prior } }

fn main() -> Result<(), SolverError> {
    let mut solver = Solver::<States, Factors>::new();
    let x = solver.add(Scalar(0.0));
    let prior = solver.add_factor(Prior {
        variable: x,
        measurement: 3.0,
    })?;

    solver.optimize()?;
    let estimate = solver.get(x)?;
    let cost = solver.factor_cost(prior)?;
    println!("estimate = {}, cost = {cost}", estimate.0);
    Ok(())
}
