#[allow(dead_code)]
#[path = "../examples/marginalization.rs"]
mod example;

use example::{Difference, Factors, Prior, Scalar, States};
use fagra::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorSelection, GaussNewton, KeyError,
    LinearizationSink, Lsmr, MarginalizationOptions, OptimizeOptions, Real, Solver, SolverError,
    StateKey, StateStore,
};
use std::{cell::Cell, rc::Rc};

#[allow(dead_code)]
#[path = "../examples/slam.rs"]
mod slam;

fagra::states! { MixedStates { scalars: Scalar, poses: slam::Pose } }
fagra::factors! { MixedFactors { scalars: Prior, poses: slam::PosePrior } }

#[test]
fn heterogeneous_keys_and_unaffected_priors_remain_valid() {
    use fagra::Variable;
    let mut graph = Solver::<MixedStates, MixedFactors>::new();
    let scalar = graph.add(Scalar(0.0));
    let pose = graph.add(slam::Pose::identity());
    let survivor = graph.add(Scalar(0.0));
    graph
        .add_factor(Prior {
            variable: scalar,
            measurement: 2.0,
        })
        .unwrap();
    graph
        .add_factor(slam::PosePrior {
            pose,
            measurement: slam::Pose::identity(),
        })
        .unwrap();
    let kept = graph
        .add_factor(Prior {
            variable: survivor,
            measurement: 3.0,
        })
        .unwrap();
    let report = graph
        .marginalize(&[scalar.block_id(), pose.block_id()])
        .unwrap();
    assert_eq!(report.removed_dof, 7);
    assert_eq!(report.eliminated_rank, 7);
    assert_eq!(report.separator_dof, 0);
    assert!(matches!(graph.get(pose), Err(KeyError::Stale)));
    assert!(matches!(graph.get(scalar), Err(KeyError::Stale)));
    assert_eq!(graph.factor_cost(kept).unwrap(), 4.5);
    graph.optimize().unwrap();
    assert!((graph.get(survivor).unwrap().0 - 3.0).abs() < 1e-10);
}

#[test]
fn advancing_a_scalar_window_reuses_prior_information() {
    let mut graph = Solver::<States<f64>, Factors<f64>>::new();
    let mut oldest = graph.add(Scalar(0.0));
    graph
        .add_factor(Prior {
            variable: oldest,
            measurement: 0.0,
        })
        .unwrap();
    let mut method = GaussNewton::new(Lsmr::default());
    for time in 1..=100 {
        let next = graph.add(Scalar(time as f64 - 0.2));
        graph
            .add_factor(Difference {
                x: oldest,
                y: next,
                measurement: 1.0,
            })
            .unwrap();
        graph
            .optimize_with(&mut method, &OptimizeOptions::default())
            .unwrap();
        let report = graph.marginalize(&[oldest.block_id()]).unwrap();
        assert_eq!(report.separator_dof, 1);
        assert_eq!(report.prior_rows, 1);
        assert_eq!(report.absorbed_priors, usize::from(time > 1));
        assert_eq!(report.absorbed_factors, if time == 1 { 2 } else { 1 });
        assert!((graph.get(next).unwrap().0 - time as f64).abs() < 1e-8);
        oldest = next;
    }
    // One initial prior and 100 differences give the final state variance 101.
    graph
        .add_factor(Prior {
            variable: oldest,
            measurement: 102.0,
        })
        .unwrap();
    let report = graph
        .optimize_with(&mut method, &OptimizeOptions::default())
        .unwrap();
    assert!((graph.get(oldest).unwrap().0 - (102.0 - 2.0 / 102.0)).abs() < 1e-9);
    assert!((report.final_cost - 2.0 / 102.0).abs() < 1e-9);
}

#[test]
fn scalar_example_matches_the_analytical_solution() {
    example::run().unwrap();
}

#[test]
fn repeated_and_joint_elimination_agree_in_both_precisions_and_backends() {
    fn check<R: Real>() {
        let c = R::from_f64_impl;
        let tolerance = R::epsilon_impl().max(c(1e-12)) * c(512.0);
        for sequential in [false, true] {
            for iterative in [false, true] {
                let mut graph = Solver::<States<R>, Factors<R>>::new();
                let x = graph.add(Scalar(c(0.0)));
                let y = graph.add(Scalar(c(0.0)));
                let z = graph.add(Scalar(c(0.0)));
                graph
                    .add_factor(Prior {
                        variable: x,
                        measurement: c(1.0),
                    })
                    .unwrap();
                graph
                    .add_factor(Difference {
                        x,
                        y,
                        measurement: c(2.0),
                    })
                    .unwrap();
                graph
                    .add_factor(Difference {
                        x: y,
                        y: z,
                        measurement: c(3.0),
                    })
                    .unwrap();
                graph
                    .add_factor(Prior {
                        variable: z,
                        measurement: c(8.0),
                    })
                    .unwrap();
                let mut lsmr = GaussNewton::new(Lsmr::<R>::default());
                if sequential {
                    graph.marginalize(&[x.block_id()]).unwrap();
                    // Move away from the first prior's reference before absorbing it.
                    graph
                        .optimize_with(&mut lsmr, &OptimizeOptions::default())
                        .unwrap();
                    let report = graph.marginalize(&[y.block_id()]).unwrap();
                    assert_eq!(report.absorbed_priors, 1);
                } else {
                    let report = graph
                        .marginalize(&[x.block_id(), y.block_id(), x.block_id()])
                        .unwrap();
                    assert_eq!(report.removed_states, 2);
                    assert_eq!(report.eliminated_rank, 2);
                }
                // New data must still update the reduced graph, not just recover a
                // previously optimized answer stored in the prior.
                graph
                    .add_factor(Prior {
                        variable: z,
                        measurement: c(10.0),
                    })
                    .unwrap();
                let report = if iterative {
                    graph
                        .optimize_with(&mut lsmr, &OptimizeOptions::default())
                        .unwrap()
                } else {
                    graph.optimize().unwrap()
                };
                let expected = c(60.0 / 7.0);
                let energy = c(0.5)
                    * ((expected - c(6.0)).powi(2) / c(3.0)
                        + (expected - c(8.0)).powi(2)
                        + (expected - c(10.0)).powi(2));
                assert!((graph.get(z).unwrap().0 - expected).abs() < tolerance);
                assert!((report.final_cost - energy).abs() < tolerance);
                assert!(matches!(graph.get(x), Err(KeyError::Stale)));
                graph.marginalize(&[z.block_id()]).unwrap();
                // With no states left, irreducible cost must survive as a constant.
                assert!((graph.optimize().unwrap().final_cost - energy).abs() < tolerance);
                let fresh = graph.add(Scalar(c(0.0)));
                graph
                    .add_factor(Prior {
                        variable: fresh,
                        measurement: c(4.0),
                    })
                    .unwrap();
                graph.optimize().unwrap();
                assert!((graph.get(fresh).unwrap().0 - c(4.0)).abs() < tolerance);
            }
        }
    }
    check::<f64>();
    check::<f32>();
}

#[test]
fn invalid_selection_or_evaluation_leaves_the_graph_unchanged() {
    let mut graph = Solver::<States<f64>, Factors<f64>>::new();
    let x = graph.add(Scalar(0.0));
    let prior = graph
        .add_factor(Prior {
            variable: x,
            measurement: 1.0,
        })
        .unwrap();
    let foreign = Solver::<States<f64>, Factors<f64>>::new().add(Scalar(0.0));
    assert!(matches!(
        graph.marginalize(&[x.block_id(), foreign.block_id()]),
        Err(SolverError::Key(KeyError::ForeignSolver))
    ));
    assert_eq!(graph.marginalize(&[]).unwrap().removed_states, 0);
    for tolerance in [f64::NAN, f64::INFINITY, -1.0, 1.0] {
        assert!(matches!(
            graph.marginalize_with(
                &[x.block_id()],
                &MarginalizationOptions {
                    relative_rank_tolerance: Some(tolerance),
                }
            ),
            Err(SolverError::InvalidRankTolerance)
        ));
    }
    let bad = graph
        .add_factor(Prior {
            variable: x,
            measurement: f64::NAN,
        })
        .unwrap();
    assert!(graph.marginalize(&[x.block_id()]).is_err());
    assert_eq!(graph.get(x).unwrap().0, 0.0);
    assert_eq!(graph.factor_cost(prior).unwrap(), 0.5);
    graph.remove_factor(bad).unwrap();
    graph.marginalize(&[x.block_id()]).unwrap();
    let replacement = graph.add(Scalar(2.0));
    assert!(matches!(
        graph.marginalize(&[replacement.block_id(), x.block_id()]),
        Err(SolverError::Key(KeyError::Stale))
    ));
    assert_eq!(graph.get(replacement).unwrap().0, 2.0);
    // An isolated state has no rows and requires no artificial prior.
    let report = graph.marginalize(&[replacement.block_id()]).unwrap();
    assert_eq!(report.prior_rows, 0);
    assert_eq!(report.eliminated_rank, 0);
}

struct Shared {
    offset: StateKey<Scalar>,
    fault: Rc<Cell<u8>>,
    calls: Rc<Cell<usize>>,
}
struct Observation {
    value: StateKey<Scalar>,
    measurement: f64,
}

impl<S: StateStore<Scalar>> FactorBatch<S> for Shared {
    type Scalar = f64;
    type Factor = Observation;
    fn visit_variables(&self, factor: &Observation, mut visit: impl FnMut(BlockId)) {
        visit(self.offset.block_id());
        visit(factor.value.block_id());
    }
    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, Observation>,
    ) -> Result<f64, EvaluationError> {
        factors
            .map(|(_, f)| {
                Difference {
                    x: self.offset,
                    y: f.value,
                    measurement: f.measurement,
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
        self.calls.set(self.calls.get() + 1);
        for (id, f) in factors {
            sink.factor(id, |out| {
                Difference {
                    x: self.offset,
                    y: f.value,
                    measurement: f.measurement,
                }
                .linearize(states, out)
            })?;
            match self.fault.get() {
                1 => return Err(EvaluationError::InvalidEvaluation),
                2 => panic!("deliberate evaluator panic"),
                _ => {}
            }
        }
        Ok(())
    }
}

fagra::factors! { Batched { priors: Prior, frames: Batch<Shared, Observation> } }

#[test]
fn partial_batches_restore_on_failure_and_preserve_sibling_handles() {
    let mut graph = Solver::<States<f64>, Batched>::new();
    let offset = graph.add(Scalar(0.0));
    graph
        .add_factor(Prior {
            variable: offset,
            measurement: 1.0,
        })
        .unwrap();
    let fault = Rc::new(Cell::new(0));
    let calls = Rc::new(Cell::new(0));
    let batch = graph.add_batch(Shared {
        offset,
        fault: fault.clone(),
        calls: calls.clone(),
    });
    let values: [_; 5] = std::array::from_fn(|_| graph.add(Scalar(0.0)));
    let observations: [_; 5] = std::array::from_fn(|i| {
        graph
            .add_factor_to(
                batch,
                Observation {
                    value: values[i],
                    measurement: i as f64,
                },
            )
            .unwrap()
    });
    let selected = [
        values[0].block_id(),
        values[2].block_id(),
        values[4].block_id(),
    ];
    assert!(matches!(
        graph.remove_batch(batch),
        Err(SolverError::BatchNotEmpty)
    ));
    for mode in [1, 2] {
        fault.set(mode);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            graph.marginalize(&selected)
        }));
        if mode == 1 {
            assert!(result.unwrap().is_err());
        } else {
            assert!(result.is_err());
        }
        for (i, &key) in observations.iter().enumerate() {
            assert_eq!(graph.factor_cost(key).unwrap(), 0.5 * (i * i) as f64);
            assert_eq!(graph.get(values[i]).unwrap().0, 0.0);
        }
    }
    fault.set(0);
    calls.set(0);
    let report = graph.marginalize(&selected).unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(report.absorbed_factors, 3);
    assert_eq!(report.eliminated_rank, 3);
    for i in [0, 2, 4] {
        assert!(graph.factor_cost(observations[i]).is_err());
    }
    for i in [1, 3] {
        assert_eq!(
            graph.factor_cost(observations[i]).unwrap(),
            0.5 * (i * i) as f64
        );
    }
    graph.marginalize(&[offset.block_id()]).unwrap();
    graph.remove_batch(batch).unwrap();
    graph.optimize().unwrap();
    assert!((graph.get(values[1]).unwrap().0 - 2.0).abs() < 1e-10);
    assert!((graph.get(values[3]).unwrap().0 - 4.0).abs() < 1e-10);
    assert!(matches!(
        graph.add_factor_to(
            batch,
            Observation {
                value: values[1],
                measurement: 1.0
            }
        ),
        Err(SolverError::Key(KeyError::Stale))
    ));
}
