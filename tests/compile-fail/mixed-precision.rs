//@ revisions: valid state jacobian factor batch backend options schemas
//@ edition: 2024
//@[valid] check-pass

#![allow(dead_code, unused_imports, unused_variables, unused_mut)]

use faer_ext::nalgebra::SMatrix;
use fagra::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorSelection, GaussNewton,
    JacobianBlock, LinearizationSink, Lsmr, OptimizeOptions, Real, Solver,
};

#[path = "../../examples/scalar_prior.rs"]
mod scalar;
use scalar::{Factors, Scalar, States};

fagra::states! { EmptyStates<R> {} }
fagra::factors! { EmptyFactors<R> {} }

#[cfg(state)]
fagra::states! { MixedStates<R> { values: Scalar<f64> } }
//~[state]^ ERROR: type mismatch resolving

struct Constant<R: Real>(R);
impl<S, R: Real> Factor<S> for Constant<R> {
    type Scalar = R;
    fn visit_variables(&self, _: impl FnMut(BlockId)) {}
    fn cost(&self, _: &S) -> Result<R, EvaluationError> { Ok(self.0) }
    fn linearize<L: LinearizationSink<Scalar = R>>(&self, _: &S, _: &mut L) -> Result<(), EvaluationError> { Ok(()) }
}

fagra::factors! { WrongFactors<R> { constants: Constant<f64> } }

struct Batch64;
impl<S> FactorBatch<S> for Batch64 {
    type Scalar = f64;
    type Factor = ();
    fn visit_variables(&self, _: &(), _: impl FnMut(BlockId)) {}
    fn cost(&self, _: &S, _: FactorSelection<'_, ()>) -> Result<f64, EvaluationError> { Ok(0.0) }
    fn linearize<L: LinearizationSink<Scalar = f64>>(&self, _: &S, _: FactorSelection<'_, ()>, _: &mut L) -> Result<(), EvaluationError> { Ok(()) }
}

fagra::factors! { WrongBatches<R> { batches: Batch<Batch64, ()> } }

fn main() {
    let mut graph = Solver::<States<f32>, Factors<f32>>::new();
    let key = graph.add(Scalar(0.0_f32));

    #[cfg(valid)]
    {
        JacobianBlock::new(key, &SMatrix::<f32, 1, 1>::identity());
        graph.add_factor(scalar::Prior { variable: key, measurement: 3.0_f32 }).unwrap();
        graph.optimize_with(&mut GaussNewton::new(Lsmr::<f32>::default()), &OptimizeOptions::default()).unwrap();
        Solver::<EmptyStates<f64>, WrongFactors<f64>>::new();
        Solver::<EmptyStates<f64>, WrongBatches<f64>>::new();
    }

    #[cfg(jacobian)]
    JacobianBlock::new(key, &SMatrix::<f64, 1, 1>::identity());
    //~[jacobian]^ ERROR: mismatched types

    #[cfg(factor)]
    Solver::<EmptyStates<f32>, WrongFactors<f32>>::new();
    //~[factor]^ ERROR: trait bounds were not satisfied

    #[cfg(batch)]
    Solver::<EmptyStates<f32>, WrongBatches<f32>>::new();
    //~[batch]^ ERROR: trait bounds were not satisfied

    #[cfg(backend)]
    graph.optimize_with(&mut GaussNewton::new(Lsmr::<f64>::default()), &OptimizeOptions::default());
    //~[backend]^ ERROR: type mismatch resolving

    #[cfg(options)]
    graph.optimize_with(&mut GaussNewton::new(Lsmr::<f32>::default()), &OptimizeOptions::<f64>::default());
    //~[options]^ ERROR: mismatched types

    #[cfg(schemas)]
    Solver::<EmptyStates<f32>, EmptyFactors<f64>>::new();
    //~[schemas]^ ERROR: trait bounds were not satisfied
}
