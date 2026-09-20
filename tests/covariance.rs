use faer_ext::nalgebra::{SMatrix, SVector};
use fagra::{
    BlockId, CovarianceOptions, EvaluationError, Factor, FactorBatch, FactorSelection, GaussNewton,
    JacobianBlock, KeyError, LinearizationSink, Real, Solver, SolverError, StateStore,
};

#[allow(dead_code)]
#[path = "../examples/marginalization.rs"]
mod example;
use example::{Difference, Prior, Scalar, States};

struct Edge<R: Real>(Difference<R>);
impl<R: Real, S: StateStore<Scalar<R>>> FactorBatch<S> for Batch<R> {
    type Scalar = R;
    type Factor = Edge<R>;
    fn visit_variables(&self, edge: &Edge<R>, visit: impl FnMut(BlockId)) {
        <Difference<R> as Factor<S>>::visit_variables(&edge.0, visit);
    }
    fn cost(
        &self,
        states: &S,
        mut factors: FactorSelection<'_, Edge<R>>,
    ) -> Result<R, EvaluationError> {
        factors.try_fold(R::zero(), |sum, (_, f)| Ok(sum + f.0.cost(states)?))
    }
    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        factors: FactorSelection<'_, Edge<R>>,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        for (id, f) in factors {
            sink.factor(id, |out| f.0.linearize(states, out))?;
        }
        Ok(())
    }
}
struct Batch<R>(std::marker::PhantomData<R>);
fagra::factors! { Factors<R> { priors: Prior<R>, edges: Difference<R>, batches: Batch<Batch<R>, Edge<R>> } }

#[test]
fn selected_inverse_matches_reference_and_marginalized_history() {
    fn check<R: Real>() {
        let c = R::from_f64_impl;
        let tolerance = c(1024.) * R::epsilon_impl();
        let mut graph = Solver::<States<R>, Factors<R>>::new();
        let x = graph.add(Scalar(c(0.)));
        let y = graph.add(Scalar(c(0.)));
        let z = graph.add(Scalar(c(0.)));
        let q = graph.add(Scalar(c(0.)));
        let prior = graph
            .add_factor(Prior {
                variable: x,
                measurement: c(1.),
            })
            .unwrap();
        graph
            .add_factor(Difference {
                x,
                y,
                measurement: c(2.),
            })
            .unwrap();
        let batch = graph.add_batch(Batch(std::marker::PhantomData));
        graph
            .add_factor_to(
                batch,
                Edge(Difference {
                    x: y,
                    y: z,
                    measurement: c(3.),
                }),
            )
            .unwrap();
        graph
            .add_factor_to(
                batch,
                Edge(Difference {
                    x: z,
                    y: q,
                    measurement: c(1.),
                }),
            )
            .unwrap();
        graph
            .add_factor(Prior {
                variable: q,
                measurement: c(8.),
            })
            .unwrap();
        // Independently assembled full Jacobian; request a non-storage ordering.
        let j = SMatrix::<R, 5, 4>::from_row_slice(&[
            c(1.),
            c(0.),
            c(0.),
            c(0.),
            c(-1.),
            c(1.),
            c(0.),
            c(0.),
            c(0.),
            c(-1.),
            c(1.),
            c(0.),
            c(0.),
            c(0.),
            c(-1.),
            c(1.),
            c(0.),
            c(0.),
            c(0.),
            c(1.),
        ]);
        let inverse = (j.transpose() * j).try_inverse().unwrap();
        for stage in 0..3 {
            let cov = graph
                .joint_covariance(&[q.block_id(), z.block_id()])
                .unwrap();
            for (i, row) in [3, 2].into_iter().enumerate() {
                for (k, col) in [3, 2].into_iter().enumerate() {
                    assert!((cov[(i, k)] - inverse[(row, col)]).abs() < tolerance);
                }
            }
            if stage == 0 {
                graph.marginalize(&[x.block_id()]).unwrap();
                assert!(matches!(graph.set(x, Scalar(c(99.))), Err(KeyError::Stale)));
                assert!(matches!(
                    graph.factor_cost(prior),
                    Err(SolverError::Key(KeyError::Stale))
                ));
                graph.set(y, Scalar(c(12.))).unwrap();
            } else if stage == 1 {
                graph.marginalize(&[y.block_id()]).unwrap();
            }
        }
        // A replacement changes the estimate, not the historical prior's anchor.
        graph.set(z, Scalar(c(-10.))).unwrap();
        graph.optimize().unwrap();
        assert!((graph.get(z).unwrap().0 - c(6.6)).abs() < tolerance);
        assert!((graph.get(q).unwrap().0 - c(7.8)).abs() < tolerance);
    }
    check::<f64>();
    check::<f32>();
}

#[test]
fn checked_selection_and_singular_information_do_not_change_estimates() {
    let mut graph = Solver::<States<f64>, Factors<f64>>::new();
    let x = graph.add(Scalar(3.));
    let foreign = Solver::<States<f64>, Factors<f64>>::new().add(Scalar(0.));
    assert!(matches!(
        graph.set(foreign, Scalar(7.)),
        Err(KeyError::ForeignSolver)
    ));
    assert!(matches!(
        graph.joint_covariance(&[foreign.block_id()]),
        Err(SolverError::Key(KeyError::ForeignSolver))
    ));
    assert!(matches!(
        graph.joint_covariance(&[x.block_id(), x.block_id()]),
        Err(SolverError::DuplicateCovarianceBlock)
    ));
    assert!(matches!(
        graph.joint_covariance(&[x.block_id()]),
        Err(SolverError::SingularInformation)
    ));
    assert!(matches!(
        graph.joint_covariance(&[]),
        Err(SolverError::SingularInformation)
    ));
    let p = graph
        .add_factor(Prior {
            variable: x,
            measurement: 2.,
        })
        .unwrap();
    graph.set(x, Scalar(5.)).unwrap();
    assert_eq!(graph.factor_cost(p).unwrap(), 4.5);
    assert_eq!(graph.joint_covariance(&[x.block_id()]).unwrap()[(0, 0)], 1.);
    for relative_pivot_tolerance in [-1., 1., f64::NAN, f64::INFINITY] {
        assert!(matches!(
            graph.joint_covariance_with(
                &[x.block_id()],
                &CovarianceOptions {
                    relative_pivot_tolerance
                }
            ),
            Err(SolverError::InvalidRankTolerance)
        ));
    }
    graph.marginalize(&[x.block_id()]).unwrap();
    assert!(matches!(
        graph.joint_covariance(&[x.block_id()]),
        Err(SolverError::Key(KeyError::Stale))
    ));
    assert_eq!(graph.joint_covariance(&[]).unwrap().shape(), (0, 0));
}

struct Square {
    key: fagra::StateKey<Scalar>,
    calls: std::rc::Rc<std::cell::Cell<usize>>,
}
impl<S: StateStore<Scalar>> Factor<S> for Square {
    type Scalar = f64;
    fn visit_variables(&self, mut v: impl FnMut(BlockId)) {
        v(self.key.block_id());
    }
    fn cost(&self, s: &S) -> Result<f64, EvaluationError> {
        Ok(0.5 * (s.get(self.key)?.0.powi(2) - 4.).powi(2))
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        s: &S,
        out: &mut L,
    ) -> Result<(), EvaluationError> {
        self.calls.set(self.calls.get() + 1);
        let x = s.get(self.key)?.0;
        out.residual(
            &SVector::<f64, 1>::new(x * x - 4.),
            &[JacobianBlock::new(
                self.key,
                &SMatrix::<f64, 1, 1>::new(2. * x),
            )],
        )
    }
}
fagra::factors! { Curves { squares: Square } }

#[test]
fn gn_refreshes_only_when_termination_leaves_a_stale_model() {
    use fagra::{OptimizeOptions, TerminationReason};
    for (initial, gradient, step, cost, reason, calls_expected) in [
        (2., 1e-8, 0., 0., TerminationReason::GradientTolerance, 1),
        (3., 0., 10., 0., TerminationReason::StepTolerance, 1),
        (3., 0., 0., 1., TerminationReason::CostTolerance, 2),
    ] {
        let mut graph = Solver::<States<f64>, Curves>::new();
        let key = graph.add(Scalar(initial));
        let calls = std::rc::Rc::new(std::cell::Cell::new(0));
        graph
            .add_factor(Square {
                key,
                calls: calls.clone(),
            })
            .unwrap();
        let options = OptimizeOptions {
            gradient_tolerance: gradient,
            step_tolerance: step,
            cost_tolerance: cost,
            ..Default::default()
        };
        let (report, cov) = graph
            .optimize_with_covariance(
                &mut GaussNewton::default(),
                &options,
                &[key.block_id()],
                &Default::default(),
            )
            .unwrap();
        assert_eq!(report.termination, reason);
        let variance = cov[(0, 0)];
        let x = graph.get(key).unwrap().0;
        assert!((variance - 1. / (4. * x * x)).abs() < 1e-12);
        assert_eq!(calls.get(), calls_expected);
        graph.set(key, Scalar(4.)).unwrap();
        assert!(
            (graph.joint_covariance(&[key.block_id()]).unwrap()[(0, 0)] - 1. / 64.).abs() < 1e-12
        );
    }
}

#[test]
fn covariance_failure_preserves_successfully_optimized_state() {
    let mut graph = Solver::<States<f64>, Factors<f64>>::new();
    let x = graph.add(Scalar(3.));
    graph
        .add_factor(Prior {
            variable: x,
            measurement: 1.,
        })
        .unwrap();
    graph.add(Scalar(0.)); // Unobservable, even though not selected.
    let mut method = fagra::LevenbergMarquardt::default();
    assert!(matches!(
        graph.optimize_with_covariance(
            &mut method,
            &Default::default(),
            &[x.block_id()],
            &Default::default()
        ),
        Err(SolverError::SingularInformation)
    ));
    assert!((graph.get(x).unwrap().0 - 1.).abs() < 1e-7);
}
