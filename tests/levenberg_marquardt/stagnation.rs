use super::*;

struct Coupled {
    x: StateKey<Scalar>,
    y: StateKey<Scalar>,
}

impl<S: StateStore<Scalar>> Factor<S> for Coupled {
    type Scalar = f64;

    fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
        visit(self.x.block_id());
        visit(self.y.block_id());
    }

    fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
        let x = states.get(self.x)?.0;
        let y = states.get(self.y)?.0;
        Ok(0.5 * ((1e8 * (x + y)).powi(2) + (100. * (x - y - 1e-4)).powi(2)))
    }

    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let x = states.get(self.x)?.0;
        let y = states.get(self.y)?.0;
        sink.residual(
            &SVector::<f64, 2>::new(1e8 * (x + y), 100. * (x - y - 1e-4)),
            &[
                JacobianBlock::new(self.x, &SMatrix::<f64, 2, 1>::new(1e8, 100.)),
                JacobianBlock::new(self.y, &SMatrix::<f64, 2, 1>::new(1e8, -100.)),
            ],
        )
    }
}

fagra::factors! { CoupledFactors { objective: Coupled } }

fn imu_tolerances(max_iterations: usize) -> OptimizeOptions {
    OptimizeOptions {
        max_iterations,
        gradient_tolerance: 0.001,
        step_tolerance: 1e-5,
        cost_tolerance: 1e-8,
    }
}

#[test]
fn full_rank_weak_direction_recovers_at_unchanged_tolerances() {
    fn check<B: fagra::__private::CovarianceBackend<Scalar = f64>>(backend: B) {
        // Full rank, cond(J)=1e6: column normalization leaves weak curvature ~2e-12.
        let mut graph = fagra::Problem::<States<f64>, CoupledFactors>::new();
        let x = graph.add(Scalar(0.));
        let y = graph.add(Scalar(0.));
        graph.add_factor(Coupled { x, y }).unwrap();
        let mut lm = LevenbergMarquardt::new(backend);
        lm.options.max_trials = 1;
        let options = imu_tolerances(20);
        for _ in 0..2 {
            graph.set(x, Scalar(0.)).unwrap();
            graph.set(y, Scalar(0.)).unwrap();
            let (report, covariance) = lm
                .solve_batch_with_covariance(
                    &mut graph,
                    &options,
                    &[x.block_id(), y.block_id()],
                    &Default::default(),
                )
                .unwrap();
            assert_eq!(
                report.termination,
                fagra::TerminationReason::GradientTolerance
            );
            assert_eq!(report.iterations, 2);
            assert!(report.final_cost < report.initial_cost);
            assert!(covariance[(0, 0)] > 0.);
            assert!(covariance[(1, 1)] > 0.);
            assert!(lm.statistics().gradient_norm.unwrap() <= options.gradient_tolerance);
            assert_eq!(lm.statistics().damping, 0.);
            assert_eq!(lm.statistics().attempts, 2);
            assert!((graph.get(x).unwrap().0 - 5e-5).abs() < 1e-7);
            assert!((graph.get(y).unwrap().0 + 5e-5).abs() < 1e-7);
        }
    }
    check(fagra::DenseQr::default());
    check(fagra::DenseNormalCholesky::default());
    check(fagra::DenseSchur::new(2, []));
}

#[test]
fn stagnation_trial_does_not_extend_the_accepted_step_budget() {
    let mut graph = fagra::Problem::<States<f64>, CoupledFactors>::new();
    let x = graph.add(Scalar(0.));
    let y = graph.add(Scalar(0.));
    graph.add_factor(Coupled { x, y }).unwrap();
    let mut lm = LevenbergMarquardt::new(fagra::DenseQr::default());
    assert!(matches!(
        lm.solve_batch(&mut graph, &imu_tolerances(1)),
        Err(SolverError::NoProgress)
    ));
    assert_eq!(lm.statistics().accepted_steps, 1);
    assert_eq!(lm.statistics().attempts, 1);
    assert!(lm.statistics().damping > 0.);
    assert!(lm.statistics().gradient_norm.unwrap() > 0.001);
}

#[test]
fn stagnation_trial_rejects_uphill_or_poor_agreement_and_rolls_back() {
    for (initial, threshold) in [(0.1, 1e-4), (0.5, 0.9)] {
        let mut graph = Solver::<States<f64>, Factors<f64>>::new();
        let x = graph.add(Scalar(initial));
        let calls = Rc::new(Calls::default());
        graph
            .add_factor(Curve {
                key: x,
                target: 1.,
                kind: Kind::Square,
                reset: Rc::new(Cell::new(false)),
                offset: 0.,
                calls: calls.clone(),
                malformed: false,
            })
            .unwrap();
        let mut lm = LevenbergMarquardt::new(fagra::DenseQr::default());
        lm.options.initial_damping = 1e8;
        lm.options.acceptance_threshold = threshold;
        assert!(matches!(
            graph.optimize_with(&mut lm, &imu_tolerances(20)),
            Err(SolverError::NoProgress)
        ));
        let accepted = initial + (1. - initial * initial) / (2. * initial * (1. + 1e8));
        assert!((graph.get(x).unwrap().0 - accepted).abs() < 1e-15);
        assert_eq!(lm.statistics().accepted_steps, 1);
        assert_eq!(lm.statistics().attempts, 2);
        assert_eq!(lm.statistics().rejected_steps, 1);
        assert_eq!(
            lm.statistics().last_rejection,
            Some(TrialFailure::PoorAgreement)
        );
        assert_eq!(lm.statistics().damping, 0.);
        assert_eq!(calls.jacobian.get(), 2);
        assert_eq!(calls.cost.get(), 3);
        assert!(
            (lm.statistics().cost.unwrap() - 0.5 * (accepted * accepted - 1.).powi(2)).abs()
                < 1e-15
        );
    }
}

#[test]
fn singular_undamped_stagnation_trial_remains_a_bounded_failure() {
    let mut graph = fagra::Problem::<States<f64>, Factors<f64>>::new();
    let x = graph.add(Scalar(0.));
    let unused = graph.add(Scalar(7.));
    graph
        .add_factor(Curve {
            key: x,
            target: 1.,
            kind: Kind::Linear,
            reset: Rc::new(Cell::new(false)),
            offset: 0.,
            calls: Rc::new(Calls::default()),
            malformed: false,
        })
        .unwrap();
    let mut lm = LevenbergMarquardt::new(fagra::DenseNormalCholesky::default());
    lm.options.initial_damping = 1e9;
    assert!(matches!(
        lm.solve_batch(&mut graph, &imu_tolerances(20)),
        Err(SolverError::NoProgress)
    ));
    assert_eq!(lm.statistics().accepted_steps, 1);
    assert_eq!(lm.statistics().attempts, 2);
    assert_eq!(lm.statistics().rejected_steps, 1);
    assert_eq!(
        lm.statistics().last_rejection,
        Some(TrialFailure::LinearSolve)
    );
    assert_eq!(lm.statistics().cost_evaluations, 2);
    assert!((graph.get(x).unwrap().0 - 1. / (1. + 1e9)).abs() < 1e-20);
    assert_eq!(graph.get(unused).unwrap().0, 7.);
}

#[test]
fn an_accepted_undamped_trial_does_not_restart_stagnation_recovery() {
    let mut graph = Solver::<States<f64>, Factors<f64>>::new();
    let x = graph.add(Scalar(0.8));
    graph
        .add_factor(Curve {
            key: x,
            target: 1.,
            kind: Kind::Square,
            reset: Rc::new(Cell::new(false)),
            offset: 0.,
            calls: Rc::new(Calls::default()),
            malformed: false,
        })
        .unwrap();
    let mut lm = LevenbergMarquardt::new(fagra::DenseQr::default());
    lm.options.initial_damping = 1e9;
    assert!(matches!(
        graph.optimize_with(&mut lm, &imu_tolerances(20)),
        Err(SolverError::NoProgress)
    ));
    // Small damped step, useful undamped step, then small positive-damping step.
    assert_eq!(lm.statistics().accepted_steps, 3);
    assert_eq!(lm.statistics().attempts, 3);
    assert_eq!(lm.statistics().rejected_steps, 0);
    assert!(lm.statistics().damping > 0.);
    assert!(lm.statistics().gradient_norm.unwrap() > 0.001);
    assert!(graph.get(x).unwrap().0 > 1.);
}

#[test]
fn stagnation_recovery_does_not_regularize_covariance_or_hide_a_gauge() {
    let mut graph = fagra::Problem::<States<f64>, Factors<f64>>::new();
    let x = graph.add(Scalar(0.));
    let unused = graph.add(Scalar(7.));
    graph
        .add_factor(Curve {
            key: x,
            target: 1.,
            kind: Kind::Linear,
            reset: Rc::new(Cell::new(false)),
            offset: 0.,
            calls: Rc::new(Calls::default()),
            malformed: false,
        })
        .unwrap();
    let mut lm = LevenbergMarquardt::new(Lsmr::default());
    lm.options.initial_damping = 1e9;
    assert!(matches!(
        lm.solve_batch_with_covariance(
            &mut graph,
            &imu_tolerances(20),
            &[x.block_id()],
            &Default::default(),
        ),
        Err(SolverError::SingularInformation)
    ));
    assert_eq!(lm.statistics().accepted_steps, 2);
    assert!(lm.statistics().gradient_norm.unwrap() <= 0.001);
    assert_eq!(lm.statistics().damping, 0.);
    assert!((graph.get(x).unwrap().0 - 1.).abs() < 1e-10);
    assert_eq!(graph.get(unused).unwrap().0, 7.);
}
