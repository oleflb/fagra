//@ revisions: valid ordinary factor model payload generic
//@ edition: 2024
//@[valid] check-pass
//@[ordinary] check-pass

#![allow(dead_code, unused_variables, unused_mut)]

use fagra::{BlockId, EvaluationError, Factor, FactorBatch, FactorSelection, LinearizationSink, Real};

#[cfg_attr(not(any(ordinary, factor)), derive(Clone))]
struct Plain;
impl<S> Factor<S> for Plain {
    type Scalar = f64;
    fn visit_variables(&self, _: impl FnMut(BlockId)) {}
    fn cost(&self, _: &S) -> Result<f64, EvaluationError> { Ok(0.) }
    fn linearize<L: LinearizationSink<Scalar = f64>>(&self, _: &S, _: &mut L) -> Result<(), EvaluationError> { Ok(()) }
}

#[cfg_attr(not(any(ordinary, model)), derive(Clone))]
struct Model;
#[cfg_attr(not(any(ordinary, payload)), derive(Clone))]
struct Payload;
impl<S> FactorBatch<S> for Model {
    type Scalar = f64;
    type Factor = Payload;
    fn visit_variables(&self, _: &Payload, _: impl FnMut(BlockId)) {}
    fn cost(&self, _: &S, _: FactorSelection<'_, Payload>) -> Result<f64, EvaluationError> { Ok(0.) }
    fn linearize<L: LinearizationSink<Scalar = f64>>(&self, _: &S, _: FactorSelection<'_, Payload>, _: &mut L) -> Result<(), EvaluationError> { Ok(()) }
}

struct GenericFactor<R: Real>(R);
impl Clone for GenericFactor<f32> { fn clone(&self) -> Self { Self(self.0) } }
impl<S, R: Real> Factor<S> for GenericFactor<R> {
    type Scalar = R;
    fn visit_variables(&self, _: impl FnMut(BlockId)) {}
    fn cost(&self, _: &S) -> Result<R, EvaluationError> { Ok(self.0) }
    fn linearize<L: LinearizationSink<Scalar = R>>(&self, _: &S, _: &mut L) -> Result<(), EvaluationError> { Ok(()) }
}

fagra::states! { pub States<R> {} }
fagra::factors! { pub Factors { plain: Plain, batches: Batch<Model, Payload> } }
fagra::factors! { pub Generic<R> { factors: GenericFactor<R> } }

fn main() {
    let mut graph = fagra::Problem::<States<f64>, Factors>::new();
    let factor = graph.add_factor(Plain).unwrap();
    let batch = graph.add_batch(Model);
    let payload = graph.add_factor_to(batch, Payload).unwrap();
    graph.factor_cost(factor).unwrap();
    graph.remove_factor(payload).unwrap();
    graph.remove_batch(batch).unwrap();
    graph.restore_states(graph.checkpoint_states()).unwrap();
    let mut generic = fagra::Problem::<States<f64>, Generic<f64>>::new();
    generic.add_factor(GenericFactor(0_f64)).unwrap();

    #[cfg(valid)] {
        let saved = graph.checkpoint();
        graph.remove_factor(factor).unwrap();
        graph.restore(saved).unwrap();
        let mut cloneable = fagra::Problem::<States<f32>, Generic<f32>>::new();
        cloneable.add_factor(GenericFactor(0_f32)).unwrap();
        let saved = cloneable.checkpoint();
        cloneable.restore(saved).unwrap();
    }
    #[cfg(factor)]
    graph.checkpoint();
    //~[factor]^ ERROR: trait bound
    #[cfg(model)]
    graph.checkpoint();
    //~[model]^ ERROR: trait bound
    #[cfg(payload)]
    graph.checkpoint();
    //~[payload]^ ERROR: trait bound
    #[cfg(generic)]
    generic.checkpoint();
    //~[generic]^ ERROR: trait bound
}
