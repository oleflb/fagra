//! Runtime storage and identity contracts; optimization is not invoked.

use faer_ext::{IntoFaer, nalgebra::SMatrix};
use fagra::{
    __private::{BatchPool, FactorPool, PoolAccess, StatePool},
    BatchKey, BlockId, EvaluationError, Factor, FactorBatch, FactorId, FactorKey, FactorSelection,
    JacobianBlock, KeyError, LinearizationSink, Solver, SolverError, StateKey, StateStore,
};

#[allow(dead_code)]
#[path = "../examples/scalar_prior.rs"]
mod scalar;
use scalar::{Prior, Scalar};

struct Model {
    shared: StateKey<Scalar>,
}
struct Observation {
    variable: StateKey<Scalar>,
    offset: f64,
}

impl<S: StateStore<Scalar>> FactorBatch<S> for Model {
    type Scalar = f64;
    type Factor = Observation;

    fn visit_variables(&self, factor: &Observation, mut visitor: impl FnMut(BlockId)) {
        visitor(self.shared.block_id());
        visitor(factor.variable.block_id());
    }

    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, Observation>,
    ) -> Result<f64, EvaluationError> {
        if factors.is_empty() {
            return Ok(0.0);
        }
        let shared = states.get(self.shared)?.0;
        factors
            .map(|(_, factor)| {
                let residual = states.get(factor.variable)?.0 - shared - factor.offset;
                Ok(0.5 * residual * residual)
            })
            .sum()
    }

    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        _: &S,
        _: FactorSelection<'_, Observation>,
        _: &mut L,
    ) -> Result<(), EvaluationError> {
        unreachable!("these tests evaluate costs only")
    }
}

struct ReportedCost(f64);
impl<S> Factor<S> for ReportedCost {
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
        unreachable!("these tests evaluate costs only")
    }
}

fagra::factors! {
    Factors {
        priors: Prior,
        observations: Batch<Model, Observation>,
        reported: ReportedCost,
    }
}
type Graph = Solver<scalar::States<f64>, Factors>;

#[test]
fn state_keys_survive_growth_and_jacobians_borrow_real_storage() {
    let mut graph = Graph::default();
    let key = graph.add(Scalar(7.0));
    let id = key.block_id();
    let mut keys = std::collections::HashSet::from([key]);
    for value in 0..1000 {
        assert!(keys.insert(graph.add(Scalar(value as f64))));
    }
    assert_eq!(graph.get(key).unwrap().0, 7.0);
    assert_eq!(key.block_id(), id);
    assert!(matches!(
        Graph::new().get(key),
        Err(KeyError::ForeignSolver)
    ));

    // These bounds must not require the payloads to implement any of them.
    fn traits<T: Copy + Clone + Eq + std::hash::Hash + std::fmt::Debug>() {}
    traits::<StateKey<Scalar>>();
    traits::<FactorKey<Prior>>();
    traits::<BatchKey<Model>>();
    traits::<BlockId>();
    traits::<FactorId>();

    let workspace = SMatrix::<f64, 5, 3>::from_fn(|row, col| (row + 10 * col) as f64);
    let view = workspace.fixed_view_with_steps::<2, 1>((1, 1), (1, 1));
    let block = JacobianBlock::new(key, &view);
    assert_eq!(block.variable(), id);
    let converted: faer::MatRef<'_, f64> = block.jacobian().into_faer();
    assert_eq!(converted.as_ptr(), view.as_ptr());
    assert_eq!((converted.row_stride(), converted.col_stride()), (2, 10));
    assert_eq!(converted[(0, 0)], 11.0);
    assert_eq!(converted[(1, 0)], 13.0);
}

#[test]
fn state_pool_removal_reuses_slots_without_reusing_identities() {
    let mut pool = StatePool::<Scalar>::default();
    pool.reserve(3);
    let a = pool.insert(Scalar(1.0));
    let b = pool.insert(Scalar(2.0));
    let c = pool.insert(Scalar(3.0));
    assert_eq!(pool.remove(b).unwrap().0, 2.0);
    assert_eq!(
        pool.iter().map(|(_, value)| value.0).collect::<Vec<_>>(),
        [1.0, 3.0]
    );
    assert_eq!(pool.get(c).unwrap().0, 3.0);
    let replacement = pool.insert(Scalar(4.0));
    assert_ne!(replacement, b);
    assert!(matches!(pool.get(b), Err(KeyError::Stale)));
    assert!(matches!(pool.remove(b), Err(KeyError::Stale)));
    assert_eq!(pool.get(a).unwrap().0, 1.0);
}

#[test]
fn ordinary_factors_validate_dependencies_and_preserve_siblings() {
    let mut graph = Graph::new();
    let variable = graph.add(Scalar(0.0));
    let a = graph
        .add_factor(Prior {
            variable,
            measurement: 1.0,
        })
        .unwrap();
    let b = graph
        .add_factor(Prior {
            variable,
            measurement: 2.0,
        })
        .unwrap();
    let c = graph
        .add_factor(Prior {
            variable,
            measurement: 3.0,
        })
        .unwrap();
    let foreign = Graph::new().add(Scalar(4.0));
    assert!(matches!(
        graph.add_factor(Prior {
            variable: foreign,
            measurement: 0.0
        }),
        Err(SolverError::Key(KeyError::ForeignSolver)),
    ));
    assert_eq!(graph.factor_cost(a).unwrap(), 0.5);
    graph.remove_factor(b).unwrap();
    assert_eq!(graph.factor_cost(c).unwrap(), 4.5);
    let replacement = graph
        .add_factor(Prior {
            variable,
            measurement: 4.0,
        })
        .unwrap();
    assert_ne!(replacement, b);
    assert!(matches!(
        graph.factor_cost(b),
        Err(SolverError::Key(KeyError::Stale))
    ));
    assert!(matches!(
        graph.remove_factor(b),
        Err(SolverError::Key(KeyError::Stale))
    ));
    assert!(matches!(
        Graph::new().factor_cost(a),
        Err(SolverError::Key(KeyError::ForeignSolver))
    ));
    for invalid in [f64::NAN, f64::INFINITY, -1.0] {
        let factor = graph.add_factor(ReportedCost(invalid)).unwrap();
        assert!(matches!(
            graph.factor_cost(factor),
            Err(SolverError::Evaluation(EvaluationError::InvalidEvaluation))
        ));
    }
}

#[test]
fn batches_route_one_factor_and_keep_empty_models_reusable() {
    let mut graph = Graph::new();
    let variable = graph.add(Scalar(0.0));
    let shared_a = graph.add(Scalar(1.0));
    let shared_b = graph.add(Scalar(5.0));
    let a = graph.add_batch(Model { shared: shared_a });
    let b = graph.add_batch(Model { shared: shared_b });
    let a0 = graph
        .add_factor_to(
            a,
            Observation {
                variable,
                offset: 0.0,
            },
        )
        .unwrap();
    let a1 = graph
        .add_factor_to(
            a,
            Observation {
                variable,
                offset: 2.0,
            },
        )
        .unwrap();
    let b0 = graph
        .add_factor_to(
            b,
            Observation {
                variable,
                offset: 0.0,
            },
        )
        .unwrap();
    assert_eq!(graph.factor_cost(a0).unwrap(), 0.5);
    assert_eq!(graph.factor_cost(a1).unwrap(), 4.5);
    assert_eq!(graph.factor_cost(b0).unwrap(), 12.5);
    let ordinary = graph
        .add_factor(Prior {
            variable,
            measurement: 3.0,
        })
        .unwrap();
    assert_eq!(
        graph.factor_cost(ordinary).unwrap(),
        graph.factor_cost(a1).unwrap()
    );

    graph.remove_factor(a0).unwrap();
    assert_eq!(graph.factor_cost(a1).unwrap(), 4.5);
    assert_eq!(graph.factor_cost(b0).unwrap(), 12.5);
    graph.remove_factor(a1).unwrap();
    let replacement = graph
        .add_factor_to(
            a,
            Observation {
                variable,
                offset: 3.0,
            },
        )
        .unwrap();
    assert_ne!(replacement, a0);
    assert_ne!(replacement, a1);
    assert_eq!(graph.factor_cost(replacement).unwrap(), 8.0);
    assert!(matches!(
        graph.factor_cost(a0),
        Err(SolverError::Key(KeyError::Stale))
    ));

    let foreign = Graph::new().add(Scalar(9.0));
    assert!(matches!(
        graph.add_factor_to(
            a,
            Observation {
                variable: foreign,
                offset: 0.0
            }
        ),
        Err(SolverError::Key(KeyError::ForeignSolver))
    ));
    let bad_model = graph.add_batch(Model { shared: foreign });
    assert!(matches!(
        graph.add_factor_to(
            bad_model,
            Observation {
                variable,
                offset: 0.0
            }
        ),
        Err(SolverError::Key(KeyError::ForeignSolver))
    ));
    let foreign_batch = Graph::new().add_batch(Model { shared: variable });
    assert!(matches!(
        graph.add_factor_to(
            foreign_batch,
            Observation {
                variable,
                offset: 0.0
            }
        ),
        Err(SolverError::Key(KeyError::ForeignSolver))
    ));
    assert_eq!(graph.factor_cost(replacement).unwrap(), 8.0);
}

#[test]
fn batch_selections_preserve_identity_order_and_remaining_slices() {
    let mut pool = BatchPool::<(), usize>::default();
    pool.reserve(2);
    let batch = pool.insert_batch(());
    pool.reserve_factors(batch, 6).unwrap();
    let keys: Vec<_> = (0..6)
        .map(|value| pool.insert_into(batch, value).unwrap())
        .collect();
    let original: Vec<_> = pool.iter().next().unwrap().2.collect();
    let original_ids: Vec<_> = original.iter().map(|(id, _)| *id).collect();
    drop(original);
    pool.remove_factor(keys[1]).unwrap();
    let (_, _, all) = pool.iter().next().unwrap();
    assert_eq!(all.as_slice(), [0, 5, 2, 3, 4]);
    assert_eq!(
        all.map(|(id, _)| id).collect::<Vec<_>>(),
        [
            original_ids[0],
            original_ids[5],
            original_ids[2],
            original_ids[3],
            original_ids[4]
        ]
    );

    let mut selected = pool.iter().next().unwrap().2;
    let expected = [0, 5, 2, 3, 4];
    for (position, &value) in expected.iter().enumerate() {
        let remaining = expected.len() - position;
        assert_eq!(selected.len(), remaining);
        assert_eq!(ExactSizeIterator::len(&selected), remaining);
        assert_eq!(selected.size_hint(), (remaining, Some(remaining)));
        assert_eq!(selected.as_slice(), &expected[position..]);
        assert_eq!(selected.next(), Some((original_ids[value], &value)));
    }
    assert!(selected.is_empty());
    assert_eq!(selected.as_slice(), []);
    assert_eq!(selected.size_hint(), (0, Some(0)));
    assert_eq!(selected.next(), None);
    assert_eq!(selected.next(), None);

    let empty = pool.insert_batch(());
    let mut selected = pool.iter().find(|(key, _, _)| *key == empty).unwrap().2;
    assert!(selected.is_empty());
    assert!(selected.as_slice().is_empty());
    assert_eq!(selected.next(), None);
}

#[test]
#[ignore = "run in release mode with --ignored --nocapture for storage timings"]
fn storage_workload_timing() {
    use std::{hint::black_box, time::Instant};
    const N: usize = 16_384;
    let mut states = StatePool::<Scalar>::default();
    let mut factors = FactorPool::<usize>::default();
    states.reserve(N);
    factors.reserve(N);
    let keys: Vec<_> = (0..N).map(|i| states.insert(Scalar(i as f64))).collect();
    let mut factors_keys: Vec<_> = (0..N).map(|i| factors.insert(i)).collect();
    let start = Instant::now();
    for i in 0..1_000_000_usize {
        let key = black_box(keys[i.wrapping_mul(7919) & (N - 1)]);
        black_box(states.get(key).unwrap().0);
    }
    println!("1M checked state lookups: {:?}", start.elapsed());
    let start = Instant::now();
    for _ in 0..100 {
        for (_, value) in factors.iter() {
            black_box(*value);
        }
    }
    println!("{} dense factor visits: {:?}", N * 100, start.elapsed());
    let start = Instant::now();
    for i in 0..100_000 {
        let index = i & (N - 1);
        factors.remove_factor(factors_keys[index]).unwrap();
        factors_keys[index] = factors.insert(black_box(i));
    }
    println!("100K reserved remove/insert pairs: {:?}", start.elapsed());
}

#[test]
fn failed_insertions_do_not_publish_payloads() {
    // A drop-counting payload detects accidentally retaining a rejected factor.
    use std::{cell::Cell, rc::Rc};
    struct Watched {
        variable: StateKey<Scalar>,
        drops: Rc<Cell<usize>>,
    }
    impl Drop for Watched {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }
    impl<S> Factor<S> for Watched {
        type Scalar = f64;
        fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
            visit(self.variable.block_id());
        }
        fn cost(&self, _: &S) -> Result<f64, EvaluationError> {
            Ok(0.0)
        }
        fn linearize<L: LinearizationSink<Scalar = f64>>(
            &self,
            _: &S,
            _: &mut L,
        ) -> Result<(), EvaluationError> {
            Ok(())
        }
    }
    fagra::factors! { WatchedFactors { watched: Watched } }
    let drops = Rc::new(Cell::new(0));
    let mut graph = Solver::<scalar::States<f64>, WatchedFactors>::new();
    let foreign = Graph::new().add(Scalar(0.0));
    assert!(
        graph
            .add_factor(Watched {
                variable: foreign,
                drops: drops.clone()
            })
            .is_err()
    );
    assert_eq!(drops.get(), 1);
    let valid = graph.add(Scalar(0.0));
    graph
        .add_factor(Watched {
            variable: valid,
            drops: drops.clone(),
        })
        .unwrap();
    drop(graph);
    assert_eq!(drops.get(), 2);
}

#[test]
fn blanket_state_access_preserves_stale_key_errors() {
    fagra::states! { States { scalars: Scalar } }
    let mut states = States::default();
    let key = states.pool_mut().insert(Scalar(0.0));
    states.pool_mut().remove(key).unwrap();
    assert!(matches!(states.get(key), Err(KeyError::Stale)));
}
