use super::*;
use fagra::{LmWork, Problem};

#[test]
fn accumulated_work_saturates_on_overflow() {
    let mut total = LmWork {
        linearizations: usize::MAX,
        attempts: usize::MAX,
        rejected_steps: usize::MAX,
        accepted_steps: usize::MAX,
        cost_evaluations: usize::MAX,
        linear_solve: Some(fagra::LinearSolveWork {
            solves: usize::MAX,
            iterations: usize::MAX,
        }),
    };
    let maximum = total;
    total += maximum;
    assert_eq!(total, maximum);
}

fn problem() -> (Problem<States<f64>, Factors<f64>>, Rc<Calls>) {
    let mut graph = Problem::new();
    let key = graph.add(Scalar(2.0));
    let calls = Rc::new(Calls::default());
    graph
        .add_factor(Curve {
            key,
            target: 0.0,
            kind: Kind::Atan,
            reset: Rc::new(Cell::new(false)),
            offset: 0.0,
            calls: calls.clone(),
            malformed: false,
        })
        .unwrap();
    (graph, calls)
}

#[test]
fn captures_success_and_backend_work_without_extra_evaluations() {
    let (mut graph, calls) = problem();
    let mut method = LevenbergMarquardt::default();
    let outcome = method.solve_batch_with_outcome(&mut graph, &Default::default());
    let report = outcome.result.unwrap();
    assert_eq!(outcome.statistics.cost, Some(report.final_cost));
    assert_eq!(outcome.work.accepted_steps, report.iterations);
    assert_eq!(outcome.work.cost_evaluations, calls.cost.get());
    assert_eq!(outcome.work.linearizations, calls.jacobian.get());
    assert!(outcome.work.rejected_steps > 0);
    let linear = outcome.work.linear_solve.unwrap();
    assert_eq!(linear.solves, method.backend().statistics().solves);
    assert_eq!(linear.iterations, method.backend().statistics().iterations);
    assert!(linear.iterations > method.backend().diagnostics().iterations);
}

#[test]
fn rejected_attempts_are_captured_on_failure() {
    let (mut graph, _) = problem();
    let mut method = LevenbergMarquardt::default();
    method.options.max_trials = 1;
    let outcome = method.solve_batch_with_outcome(&mut graph, &Default::default());
    assert!(matches!(outcome.result, Err(SolverError::NoProgress)));
    assert_eq!(outcome.work.attempts, 1);
    assert_eq!(outcome.work.rejected_steps, 1);
    assert_eq!(outcome.work.accepted_steps, 0);
    let linear = outcome.work.linear_solve.unwrap();
    assert_eq!(linear.solves, 1);
    assert!(linear.iterations > 0);
    assert_eq!(outcome.statistics.cost, Some(0.5 * 2.0_f64.atan().powi(2)));
}

#[test]
fn retry_totals_add_work_but_keep_latest_observations() {
    let (mut graph, _) = problem();
    let mut method = LevenbergMarquardt::default();
    let options = OptimizeOptions {
        max_iterations: 1,
        ..Default::default()
    };
    let failed = method.solve_batch_with_outcome(&mut graph, &options);
    assert!(matches!(failed.result, Err(SolverError::NoConvergence)));
    assert!(failed.work.accepted_steps > 0);
    let latest = method.solve_batch_with_outcome(&mut graph, &Default::default());
    let report = latest.result.unwrap();
    let mut total = LmWork::default();
    total += failed.work;
    total += latest.work;
    assert_eq!(total.attempts, failed.work.attempts + latest.work.attempts);
    assert_eq!(total.accepted_steps, 1 + report.iterations);
    assert_eq!(
        total.linearizations,
        failed.work.linearizations + latest.work.linearizations
    );
    assert_eq!(
        total.cost_evaluations,
        failed.work.cost_evaluations + latest.work.cost_evaluations
    );
    assert_eq!(
        total.rejected_steps,
        failed.work.rejected_steps + latest.work.rejected_steps
    );
    let linear = total.linear_solve.unwrap();
    assert_eq!(
        linear.solves,
        failed.work.linear_solve.unwrap().solves + latest.work.linear_solve.unwrap().solves
    );
    assert_eq!(
        linear.iterations,
        failed.work.linear_solve.unwrap().iterations + latest.work.linear_solve.unwrap().iterations
    );
    assert_eq!(latest.statistics.cost, Some(report.final_cost));
    assert_ne!(failed.statistics.cost, latest.statistics.cost);
    assert_eq!(
        latest.statistics.gradient_norm,
        method.statistics().gradient_norm
    );
    assert_eq!(latest.statistics.damping, method.statistics().damping);
}

#[test]
fn early_failure_does_not_capture_prior_backend_work() {
    let (mut graph, _) = problem();
    let mut method = LevenbergMarquardt::default();
    let previous = method.solve_batch_with_outcome(&mut graph, &Default::default());
    assert!(previous.result.is_ok());
    assert!(previous.work.linear_solve.unwrap().iterations > 0);
    let invalid = OptimizeOptions {
        max_iterations: 0,
        ..Default::default()
    };
    let outcome = method.solve_batch_with_outcome(&mut graph, &invalid);
    assert!(matches!(outcome.result, Err(SolverError::InvalidOptions)));
    assert_eq!(outcome.work, LmWork::default());
    assert!(outcome.statistics.cost.is_none());
    assert!(outcome.statistics.gradient_norm.is_none());
    // The backend was not prepared; its old counters must not leak into the outcome.
    assert!(method.backend().statistics().iterations > 0);
}

#[test]
fn unavailable_backend_work_remains_unknown_when_accumulated() {
    let (mut graph, _) = problem();
    let mut method = LevenbergMarquardt::new(fagra::DenseQr::<f64>::default());
    let outcome = method.solve_batch_with_outcome(&mut graph, &Default::default());
    assert!(outcome.result.is_ok());
    assert_eq!(outcome.work.linear_solve, None);
    let mut total = LmWork::default();
    total += outcome.work;
    total += LmWork::default();
    assert_eq!(total.linear_solve, None);
}
