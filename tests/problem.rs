#[allow(dead_code)]
#[path = "../examples/scalar_prior.rs"]
mod example;

use example::{Factors, Prior, Scalar, States};
use fagra::{GaussNewton, LevenbergMarquardt, Problem, ProblemChange, TrackedProblem};

#[test]
fn batch_solvers_share_a_problem_and_tracking_records_only_successful_edits() {
    let mut problem = Problem::<States<f64>, Factors<f64>>::new();
    let x = problem.add(Scalar(0.));
    let prior = problem
        .add_factor(Prior {
            variable: x,
            measurement: 3.,
        })
        .unwrap();
    let mut gn = GaussNewton::default();
    gn.solve_batch(&mut problem, &Default::default()).unwrap();
    assert!((problem.get(x).unwrap().0 - 3.).abs() < 1e-12);

    let mut tracked = TrackedProblem::new(problem);
    tracked.set(x, Scalar(8.)).unwrap();
    assert_eq!(
        tracked.changes(),
        &[
            ProblemChange::Rebuild,
            ProblemChange::StateSet(x.block_id())
        ]
    );
    let foreign = Problem::<States<f64>, Factors<f64>>::new().add(Scalar(0.));
    let before = tracked.changes().to_vec();
    assert!(tracked.set(foreign, Scalar(1.)).is_err());
    assert!(
        tracked
            .add_factor(Prior {
                variable: foreign,
                measurement: 0.
            })
            .is_err()
    );
    assert_eq!(tracked.changes(), before);

    LevenbergMarquardt::default()
        .solve_batch(&mut tracked, &Default::default())
        .unwrap();
    assert!((tracked.get(x).unwrap().0 - 3.).abs() < 1e-7);
    assert_eq!(tracked.changes(), &[ProblemChange::Rebuild]);
    tracked.set(x, Scalar(7.)).unwrap();
    let (_, cov) = gn
        .solve_batch_with_covariance(
            &mut tracked,
            &Default::default(),
            &[x.block_id()],
            &Default::default(),
        )
        .unwrap();
    assert_eq!(cov[(0, 0)], 1.);
    assert_eq!(tracked.changes(), &[ProblemChange::Rebuild]);
    assert_eq!(
        tracked.joint_covariance(&[x.block_id()]).unwrap()[(0, 0)],
        1.
    );

    tracked.remove_factor(prior).unwrap();
    assert_eq!(
        tracked.changes().last(),
        Some(&ProblemChange::FactorRemoved(prior.factor_id()))
    );
    let before = tracked.changes().to_vec();
    assert!(tracked.remove_factor(prior).is_err());
    assert_eq!(tracked.changes(), before);
    let y = tracked.add(Scalar(2.));
    let added = tracked
        .add_factor(Prior {
            variable: y,
            measurement: 2.,
        })
        .unwrap();
    assert_eq!(
        &tracked.changes()[2..],
        &[
            ProblemChange::StateAdded(y.block_id()),
            ProblemChange::FactorAdded(added.factor_id())
        ]
    );
    assert!(tracked.marginalize(&[foreign.block_id()]).is_err());
    tracked.marginalize(&[y.block_id()]).unwrap();
    assert_eq!(tracked.changes(), &[ProblemChange::Rebuild]);
    let problem = tracked.into_inner();
    assert!(problem.get(y).is_err());
    assert!((problem.get(x).unwrap().0 - 3.).abs() < 1e-7);
}

struct Batch;
impl<S: fagra::StateStore<Scalar>> fagra::FactorBatch<S> for Batch {
    type Scalar = f64;
    type Factor = Prior;
    fn visit_variables(&self, p: &Prior, mut visit: impl FnMut(fagra::BlockId)) {
        visit(p.variable.block_id());
    }
    fn cost(
        &self,
        s: &S,
        mut factors: fagra::FactorSelection<'_, Prior>,
    ) -> Result<f64, fagra::EvaluationError> {
        use fagra::Factor;
        factors.try_fold(0., |sum, (_, p)| Ok(sum + p.cost(s)?))
    }
    fn linearize<L: fagra::LinearizationSink<Scalar = f64>>(
        &self,
        s: &S,
        factors: fagra::FactorSelection<'_, Prior>,
        out: &mut L,
    ) -> Result<(), fagra::EvaluationError> {
        use fagra::Factor;
        for (id, p) in factors {
            out.factor(id, |out| p.linearize(s, out))?;
        }
        Ok(())
    }
}
fagra::factors! { Batched { batches: Batch<Batch, Prior> } }

#[test]
fn tracked_batch_payloads_have_independent_identities() {
    let mut problem = TrackedProblem::new(Problem::<States<f64>, Batched>::new());
    let x = problem.add(Scalar(0.));
    let batch = problem.add_batch(Batch);
    let key = problem
        .add_factor_to(
            batch,
            Prior {
                variable: x,
                measurement: 3.,
            },
        )
        .unwrap();
    let before = problem.changes().to_vec();
    assert_eq!(
        before.last(),
        Some(&ProblemChange::FactorAdded(key.factor_id()))
    );
    assert!(problem.remove_batch(batch).is_err());
    assert_eq!(problem.changes(), before);
    let (_, cov) = LevenbergMarquardt::default()
        .solve_batch_with_covariance(
            &mut problem,
            &Default::default(),
            &[x.block_id()],
            &Default::default(),
        )
        .unwrap();
    assert_eq!(cov[(0, 0)], 1.);
    assert!((problem.get(x).unwrap().0 - 3.).abs() < 1e-7);
    problem.remove_factor(key).unwrap();
    problem.remove_batch(batch).unwrap();
    let before = problem.changes().to_vec();
    assert!(
        problem
            .add_factor_to(
                batch,
                Prior {
                    variable: x,
                    measurement: 0.
                }
            )
            .is_err()
    );
    assert_eq!(problem.changes(), before);
}

#[test]
fn failed_batch_solve_invalidates_tracking_before_evaluation() {
    let mut tracked = TrackedProblem::new(Problem::<States<f64>, Factors<f64>>::new());
    let x = tracked.add(Scalar(2.));
    tracked
        .add_factor(Prior {
            variable: x,
            measurement: f64::NAN,
        })
        .unwrap();
    assert!(
        GaussNewton::default()
            .solve_batch(&mut tracked, &Default::default())
            .is_err()
    );
    assert_eq!(tracked.changes(), &[ProblemChange::Rebuild]);
    assert_eq!(tracked.get(x).unwrap().0, 2.);
}
