#[allow(dead_code)]
#[path = "../examples/marginalization.rs"]
mod example;

use example::{Difference, Prior, Scalar};
use fagra::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorSelection, GaussNewton,
    GraphCheckpointError, LinearizationSink, Problem, ProblemChange, Real, StateKey, StateStore,
    TrackedProblem,
};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

impl<R: Real> Clone for Scalar<R> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<R: Real> Clone for Prior<R> {
    fn clone(&self) -> Self {
        Self {
            variable: self.variable,
            measurement: self.measurement,
        }
    }
}
impl<R: Real> Clone for Difference<R> {
    fn clone(&self) -> Self {
        Self {
            x: self.x,
            y: self.y,
            measurement: self.measurement,
        }
    }
}

// Custom Clone intentionally detaches mutable external inputs.
struct Input(Arc<AtomicU64>);
impl Input {
    fn new(value: f64) -> Self {
        Self(Arc::new(AtomicU64::new(value.to_bits())))
    }
    fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
    fn set(&self, value: f64) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }
}
impl Clone for Input {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

#[derive(Clone)]
struct MutablePrior {
    variable: StateKey<Scalar>,
    target: Input,
}
impl<S: StateStore<Scalar>> Factor<S> for MutablePrior {
    type Scalar = f64;
    fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
        visit(self.variable.block_id());
    }
    fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
        Prior {
            variable: self.variable,
            measurement: self.target.get(),
        }
        .cost(states)
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        Prior {
            variable: self.variable,
            measurement: self.target.get(),
        }
        .linearize(states, sink)
    }
}

#[derive(Clone)]
struct Observation(MutablePrior);
#[derive(Clone)]
struct Model {
    offset: Input,
}
impl<S: StateStore<Scalar>> FactorBatch<S> for Model {
    type Scalar = f64;
    type Factor = Observation;
    fn visit_variables(&self, factor: &Observation, mut visit: impl FnMut(BlockId)) {
        visit(factor.0.variable.block_id());
    }
    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, Observation>,
    ) -> Result<f64, EvaluationError> {
        factors
            .map(|(_, factor)| {
                Prior {
                    variable: factor.0.variable,
                    measurement: factor.0.target.get() + self.offset.get(),
                }
                .cost(states)
            })
            .sum()
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        states: &S,
        factors: FactorSelection<'_, Observation>,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        for (id, factor) in factors {
            sink.factor(id, |sink| {
                Prior {
                    variable: factor.0.variable,
                    measurement: factor.0.target.get() + self.offset.get(),
                }
                .linearize(states, sink)
            })?;
        }
        Ok(())
    }
}

fagra::states! { pub States<R> { values: Scalar<R> } }
fagra::factors! { pub Factors<R> { priors: Prior<R>, differences: Difference<R> } }
fagra::factors! { pub Batches { priors: MutablePrior, batches: Batch<Model, Observation> } }
fagra::states! { pub EmptyStates<R> {} }
fagra::factors! { pub EmptyFactors<R> {} }

#[test]
fn restores_marginal_priors_anchors_constant_and_solver_behavior() {
    let mut graph = Problem::<States<f64>, Factors<f64>>::new();
    let x = graph.add(Scalar(0.));
    let y = graph.add(Scalar(0.));
    let z = graph.add(Scalar(0.));
    for measurement in [0., 2.] {
        graph
            .add_factor(Prior {
                variable: z,
                measurement,
            })
            .unwrap();
    }
    graph.marginalize(&[z.block_id()]).unwrap(); // Retained irreducible constant = 1.
    let prior = graph
        .add_factor(Prior {
            variable: x,
            measurement: 1.,
        })
        .unwrap();
    let difference = graph
        .add_factor(Difference {
            x,
            y,
            measurement: 2.,
        })
        .unwrap();
    graph
        .add_factor(Prior {
            variable: y,
            measurement: 6.,
        })
        .unwrap();
    let outer = graph.checkpoint();
    graph.marginalize(&[x.block_id()]).unwrap();
    graph.set(y, Scalar(7.)).unwrap();
    let inner = graph.checkpoint();
    graph.marginalize(&[y.block_id()]).unwrap();
    let discarded = graph.add(Scalar(99.));

    graph.restore(inner).unwrap();
    assert!(graph.get(x).is_err());
    assert!(graph.get(discarded).is_err());
    assert_eq!(graph.get(y).unwrap().0, 7.);
    assert!((graph.joint_covariance(&[y.block_id()]).unwrap()[(0, 0)] - 2. / 3.).abs() < 1e-12);
    let mut method = GaussNewton::default();
    let report = method.solve_batch(&mut graph, &Default::default()).unwrap();
    assert!((report.initial_cost - 5.5).abs() < 1e-12);
    assert!((report.final_cost - 2.5).abs() < 1e-12);
    assert!((graph.get(y).unwrap().0 - 5.).abs() < 1e-12);

    graph.restore(outer).unwrap();
    assert_eq!(graph.get(x).unwrap().0, 0.);
    assert_eq!(graph.factor_cost(prior).unwrap(), 0.5);
    assert_eq!(graph.factor_cost(difference).unwrap(), 2.);
    let report = method.solve_batch(&mut graph, &Default::default()).unwrap();
    assert!((report.final_cost - 2.5).abs() < 1e-12);
    assert!((graph.get(x).unwrap().0 - 2.).abs() < 1e-12);
    graph.marginalize(&[x.block_id()]).unwrap();
    assert!((graph.joint_covariance(&[y.block_id()]).unwrap()[(0, 0)] - 2. / 3.).abs() < 1e-12);
    for _ in 0..8 {
        let added = graph.add(Scalar(20.));
        assert_ne!(added, discarded);
        graph.marginalize(&[added.block_id()]).unwrap();
    }
}

#[test]
fn restores_batch_models_payloads_compaction_and_independent_mutable_inputs() {
    let mut graph = Problem::<States<f64>, Batches>::new();
    let x = graph.add(Scalar(10.));
    let shared = Input::new(1.);
    let external = Arc::clone(&shared.0);
    let ordinary = graph
        .add_factor(MutablePrior {
            variable: x,
            target: shared,
        })
        .unwrap();
    let a = graph.add_batch(Model {
        offset: Input::new(2.),
    });
    let b = graph.add_batch(Model {
        offset: Input::new(3.),
    });
    let empty = graph.add_batch(Model {
        offset: Input::new(4.),
    });
    let observation = || {
        Observation(MutablePrior {
            variable: x,
            target: Input::new(1.),
        })
    };
    let a0 = graph.add_factor_to(a, observation()).unwrap();
    let b0 = graph.add_factor_to(b, observation()).unwrap();
    let b1 = graph.add_factor_to(b, observation()).unwrap();
    let checkpoint = graph.checkpoint();
    external.store(50_f64.to_bits(), Ordering::Relaxed);
    graph
        .with_batch_factor(b1, |model, payload, _| {
            model.offset.set(60.);
            payload.0.target.set(70.);
        })
        .unwrap();
    graph.remove_factor(ordinary).unwrap();
    graph.remove_factor(a0).unwrap();
    graph.remove_batch(a).unwrap();
    graph.remove_batch(empty).unwrap();
    graph.remove_factor(b0).unwrap();
    let late_batch = graph.add_batch(Model {
        offset: Input::new(8.),
    });
    let late = graph.add_factor_to(late_batch, observation()).unwrap();
    let late_ordinary = graph
        .add_factor(MutablePrior {
            variable: x,
            target: Input::new(9.),
        })
        .unwrap();
    graph.restore(checkpoint).unwrap();
    assert_eq!(f64::from_bits(external.load(Ordering::Relaxed)), 50.);
    assert_eq!(graph.factor_cost(ordinary).unwrap(), 40.5);
    assert_eq!(graph.factor_cost(a0).unwrap(), 24.5);
    for key in [b0, b1] {
        assert_eq!(graph.factor_cost(key).unwrap(), 18.);
    }
    assert!(graph.factor_cost(late).is_err());
    assert!(graph.factor_cost(late_ordinary).is_err());
    assert!(graph.add_factor_to(late_batch, observation()).is_err());
    graph.remove_batch(empty).unwrap();
    for _ in 0..8 {
        let batch = graph.add_batch(Model {
            offset: Input::new(0.),
        });
        assert_ne!(batch, late_batch);
        let payload = graph.add_factor_to(batch, observation()).unwrap();
        assert_ne!(payload, late);
        graph.remove_factor(payload).unwrap();
        graph.remove_batch(batch).unwrap();
        let key = graph
            .add_factor(MutablePrior {
                variable: x,
                target: Input::new(0.),
            })
            .unwrap();
        assert_ne!(key, late_ordinary);
        graph.remove_factor(key).unwrap();
    }
    // Restored batched factors remain usable by both numerical paths.
    graph.joint_covariance(&[x.block_id()]).unwrap();
    GaussNewton::default()
        .solve_batch(&mut graph, &Default::default())
        .unwrap();
    graph.marginalize(&[x.block_id()]).unwrap();
}

#[test]
fn abandoned_descendants_and_foreign_graphs_reject_without_mutation() {
    let mut graph = TrackedProblem::new(Problem::<States<f32>, Factors<f32>>::new());
    let x = graph.add(Scalar(1.));
    let outer = graph.checkpoint();
    let discarded = graph.add(Scalar(2.));
    let descendant = graph.checkpoint();
    graph.restore(outer).unwrap();
    assert_eq!(graph.changes(), &[ProblemChange::Rebuild]);
    graph.set(x, Scalar(9.)).unwrap();
    let journal = graph.changes().to_vec();
    assert_eq!(
        graph.restore(descendant),
        Err(GraphCheckpointError::Invalidated)
    );
    assert_eq!(graph.changes(), journal);
    assert_eq!(graph.get(x).unwrap().0, 9.);
    assert!(graph.get(discarded).is_err());
    let mut foreign = Problem::<States<f32>, Factors<f32>>::new();
    foreign.add(Scalar(50.));
    assert_eq!(
        graph.restore(foreign.checkpoint()),
        Err(GraphCheckpointError::ForeignGraph)
    );
    assert_eq!(graph.changes(), journal);
    assert_eq!(graph.get(x).unwrap().0, 9.);
    let current = graph.checkpoint();
    graph.restore(current).unwrap();
}

#[test]
fn empty_generic_schemas_and_send_sync_survive_graph_moves() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Problem<States<f64>, Batches>>();
    send_sync::<fagra::GraphCheckpoint<States<f64>, Batches>>();
    let mut graph = Problem::<EmptyStates<f32>, EmptyFactors<f32>>::new();
    let saved = graph.checkpoint();
    let mut moved = graph;
    let mut other = Problem::<EmptyStates<f32>, EmptyFactors<f32>>::new();
    assert_eq!(
        moved.restore(other.checkpoint()),
        Err(GraphCheckpointError::ForeignGraph)
    );
    moved.restore(saved).unwrap();
}

#[test]
fn state_and_graph_checkpoints_share_ownership_in_either_capture_order() {
    for state_first in [true, false] {
        let mut graph = Problem::<States<f64>, Factors<f64>>::new();
        let x = graph.add(Scalar(1.));
        let (state, full) = if state_first {
            (graph.checkpoint_states(), graph.checkpoint())
        } else {
            let full = graph.checkpoint();
            (graph.checkpoint_states(), full)
        };
        let mut foreign = Problem::<States<f64>, Factors<f64>>::new();
        foreign.add(Scalar(20.));
        assert_eq!(
            graph.restore_states(foreign.checkpoint_states()),
            Err(fagra::StateCheckpointError::ForeignGraph)
        );
        assert_eq!(
            graph.restore(foreign.checkpoint()),
            Err(GraphCheckpointError::ForeignGraph)
        );
        graph.marginalize(&[x.block_id()]).unwrap();
        graph.add(Scalar(9.));
        let mut moved = graph;
        moved.restore(full).unwrap();
        moved.set(x, Scalar(8.)).unwrap();
        moved.restore_states(state).unwrap();
        assert_eq!(moved.get(x).unwrap().0, 1.);
        let full = moved.checkpoint();
        moved.set(x, Scalar(7.)).unwrap();
        let state = moved.checkpoint_states();
        moved.restore_states(state).unwrap();
        moved.restore(full).unwrap();
        assert_eq!(moved.get(x).unwrap().0, 1.);
    }
}

#[derive(Clone)]
struct Constant<const N: usize>(f64);
impl<S, const N: usize> Factor<S> for Constant<N> {
    type Scalar = f64;
    fn visit_variables(&self, _: impl FnMut(BlockId)) {}
    fn cost(&self, _: &S) -> Result<f64, EvaluationError> {
        Ok(self.0)
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        _: &S,
        _: &mut L,
    ) -> Result<(), EvaluationError> {
        Ok(())
    }
}

#[test]
fn conditional_graph_capability_supports_many_private_families_and_name_collisions() {
    fagra::factors! {
        #[allow(non_snake_case)]
        pub Many {
            __Scalar: Constant<0>, __Pools: Constant<1>, pools: Constant<2>,
            pools_mut: Constant<3>, snapshot: Constant<4>, values: Constant<5>,
            a: Constant<6>, b: Constant<7>, c: Constant<8>, d: Constant<9>,
            e: Constant<10>, f: Constant<11>, g: Constant<12>, h: Constant<13>,
            i: Constant<14>, j: Constant<15>,
        }
    }
    let mut graph = Problem::<States<f64>, Many>::new();
    let first = graph.add_factor(Constant::<0>(2.)).unwrap();
    let last = graph.add_factor(Constant::<15>(9.)).unwrap();
    let checkpoint = graph.checkpoint();
    graph.remove_factor(first).unwrap();
    graph.remove_factor(last).unwrap();
    graph.restore(checkpoint).unwrap();
    assert_eq!(graph.factor_cost(first).unwrap(), 2.);
    assert_eq!(graph.factor_cost(last).unwrap(), 9.);

    fagra::factors! {
        #[allow(non_snake_case)]
        pub Generic<__Pools> { __Pools: Prior<__Pools>, pools: Difference<__Pools> }
    }
    let mut graph = Problem::<States<f32>, Generic<f32>>::new();
    let x = graph.add(Scalar(2.));
    let key = graph
        .add_factor(Prior {
            variable: x,
            measurement: 1.,
        })
        .unwrap();
    let checkpoint = graph.checkpoint();
    graph.remove_factor(key).unwrap();
    graph.restore(checkpoint).unwrap();
    assert_eq!(graph.factor_cost(key).unwrap(), 0.5);
}

#[test]
fn factor_preflight_failure_does_not_restore_states_or_change_journal() {
    // A manual schema simulates exhaustion in a later pool after state preflight.
    #[derive(Default)]
    struct ExhaustedFactors;
    impl<S> fagra::__private::FactorSchema<S> for ExhaustedFactors {
        type Scalar = f64;
        fn visit<V: fagra::__private::FactorVisitor<S>>(
            &mut self,
            _: &mut V,
        ) -> Result<(), V::Error> {
            Ok(())
        }
    }
    impl fagra::SnapshotSchema for ExhaustedFactors {
        type Snapshot = ();
        fn snapshot_graph(&self) {}
        fn preflight_restore(&self) -> Result<(), GraphCheckpointError> {
            Err(GraphCheckpointError::GenerationExhausted)
        }
        fn restore_graph(&mut self, _: ()) {
            panic!("must preflight all pools before mutation");
        }
    }
    let mut graph = TrackedProblem::new(Problem::<States<f64>, ExhaustedFactors>::new());
    let x = graph.add(Scalar(1.));
    let checkpoint = graph.checkpoint();
    graph.set(x, Scalar(8.)).unwrap();
    let added = graph.add(Scalar(9.));
    let journal = graph.changes().to_vec();
    assert_eq!(
        graph.restore(checkpoint),
        Err(GraphCheckpointError::GenerationExhausted)
    );
    assert_eq!(graph.get(x).unwrap().0, 8.);
    assert_eq!(graph.get(added).unwrap().0, 9.);
    assert_eq!(graph.changes(), journal);
}
