use faer_ext::nalgebra::{SMatrix, SVector};
use fagra::{
    BlockId, EvaluationError, Factor, GaussNewton, JacobianBlock, LevenbergMarquardt,
    LinearizationSink, Lsmr, OptimizeOptions, Real, Solver, SolverError, StateKey, StateStore,
    TrialFailure,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    rc::Rc,
    time::Instant,
};

#[allow(dead_code)]
#[path = "../examples/scalar_prior.rs"]
mod scalar;
use scalar::Scalar;

struct Counter;
thread_local! { static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) }; }
fn record() {
    let _ = ALLOCATIONS.try_with(|c| {
        if let Some(n) = c.get() {
            c.set(Some(n + 1));
        }
    });
}
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;
fn measured<T>(f: impl FnOnce() -> T) -> (T, usize, f64) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ALLOCATIONS.with(|c| c.set(None));
        }
    }
    ALLOCATIONS.with(|c| {
        assert!(c.get().is_none());
        c.set(Some(0));
    });
    let reset = Reset;
    let start = Instant::now();
    let result = f();
    let us = start.elapsed().as_secs_f64() * 1e6;
    let count = ALLOCATIONS.with(|c| c.get().unwrap());
    drop(reset);
    (result, count, us)
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Linear,
    Square,
    Atan,
    Log,
}
#[derive(Default)]
struct Calls {
    cost: Cell<usize>,
    jacobian: Cell<usize>,
    first: Cell<Option<BlockId>>,
}
impl Calls {
    fn clear(&self) {
        self.cost.set(0);
        self.jacobian.set(0);
    }
}
struct Curve<R: Real> {
    key: StateKey<Scalar<R>>,
    target: R,
    kind: Kind,
    reset: Rc<Cell<bool>>,
    offset: R,
    calls: Rc<Calls>,
    malformed: bool,
}
impl<R: Real> Curve<R> {
    fn model(&self, x: R) -> Result<(R, R), EvaluationError> {
        if self.reset.get() {
            return Ok((x - self.target - self.offset, R::one()));
        }
        Ok(match self.kind {
            Kind::Linear => (x - self.target, R::one()),
            Kind::Square => (x * x - self.target * self.target, x + x),
            Kind::Atan => {
                let v = x - self.target;
                (v.atan(), R::one() / (R::one() + v * v))
            }
            Kind::Log => {
                if x <= R::zero() {
                    return Err(EvaluationError::InvalidEvaluation);
                }
                ((x / self.target).ln(), R::one() / x)
            }
        })
    }
}
impl<R: Real, S: StateStore<Scalar<R>>> Factor<S> for Curve<R> {
    type Scalar = R;
    fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
        visit(self.key.block_id());
    }
    fn cost(&self, states: &S) -> Result<R, EvaluationError> {
        let first = self.calls.first.get().unwrap_or_else(|| {
            self.calls.first.set(Some(self.key.block_id()));
            self.key.block_id()
        });
        if self.key.block_id() == first {
            self.calls.cost.set(self.calls.cost.get() + 1);
        }
        let (r, _) = self.model(states.get(self.key)?.0)?;
        Ok(R::from_f64_impl(0.5) * r * r)
    }
    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        out: &mut L,
    ) -> Result<(), EvaluationError> {
        self.calls.jacobian.set(self.calls.jacobian.get() + 1);
        let (r, j) = self.model(states.get(self.key)?.0)?;
        let r = SVector::<R, 1>::new(r);
        let j = SMatrix::<R, 1, 1>::new(j);
        let block = JacobianBlock::new(self.key, &j);
        if self.malformed {
            out.residual(&r, &[block, JacobianBlock::new(self.key, &j)])
        } else {
            out.residual(&r, &[block])
        }
    }
}
struct Link<R: Real> {
    a: StateKey<Scalar<R>>,
    b: StateKey<Scalar<R>>,
    difference: R,
    reset: Rc<Cell<bool>>,
}
impl<R: Real, S: StateStore<Scalar<R>>> Factor<S> for Link<R> {
    type Scalar = R;
    fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
        visit(self.a.block_id());
        visit(self.b.block_id());
    }
    fn cost(&self, states: &S) -> Result<R, EvaluationError> {
        if self.reset.get() {
            return Ok(R::zero());
        }
        let r = R::from_f64_impl(0.1)
            * (states.get(self.b)?.0 - states.get(self.a)?.0 - self.difference);
        Ok(R::from_f64_impl(0.5) * r * r)
    }
    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        out: &mut L,
    ) -> Result<(), EvaluationError> {
        let w = if self.reset.get() {
            R::zero()
        } else {
            R::from_f64_impl(0.1)
        };
        let r = SVector::<R, 1>::new(
            w * (states.get(self.b)?.0 - states.get(self.a)?.0 - self.difference),
        );
        let a = SMatrix::<R, 1, 1>::new(-w);
        let b = -a;
        out.residual(
            &r,
            &[
                JacobianBlock::new(self.a, &a),
                JacobianBlock::new(self.b, &b),
            ],
        )
    }
}
fagra::states! { States<R> { values: Scalar<R> } }
fagra::factors! { Factors<R> { curves: Curve<R>, links: Link<R> } }

#[test]
fn retries_cache_jacobians_and_handle_invalid_trials_in_both_precisions() {
    fn check<R: Real>() {
        let c = R::from_f64_impl;
        for (kind, initial, target) in [
            (Kind::Atan, 2., 0.),
            (Kind::Log, 10., 1.),
            (Kind::Square, 0.1, 1.),
        ] {
            let mut graph = Solver::<States<R>, Factors<R>>::new();
            let x = graph.add(Scalar(c(initial)));
            let calls = Rc::new(Calls::default());
            graph
                .add_factor(Curve {
                    key: x,
                    target: c(target),
                    kind,
                    reset: Rc::new(Cell::new(false)),
                    offset: c(0.),
                    calls: calls.clone(),
                    malformed: false,
                })
                .unwrap();
            let mut method = LevenbergMarquardt::new(Lsmr::<R>::default());
            let options = OptimizeOptions {
                max_iterations: 100,
                ..Default::default()
            };
            let (report, covariance) = graph
                .optimize_with_covariance(
                    &mut method,
                    &options,
                    &[x.block_id()],
                    &Default::default(),
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "{} {kind:?}: {e}, {:?}",
                        std::any::type_name::<R>(),
                        method.statistics()
                    )
                });
            let expected_variance = match kind {
                Kind::Square => c(0.25),
                _ => c(1.0),
            };
            assert!((covariance[(0, 0)] - expected_variance).abs() < c(1e-4));
            let stats = method.statistics();
            assert!((graph.get(x).unwrap().0 - c(target)).abs() < c(2e-5));
            assert!(report.final_cost < report.initial_cost);
            assert!(stats.rejected_steps > 0, "{kind:?}");
            assert_eq!(stats.attempts, stats.accepted_steps + stats.rejected_steps);
            assert_eq!(calls.jacobian.get(), stats.linearizations);
            assert_eq!(stats.linearizations, stats.accepted_steps + 1);
            assert_eq!(calls.cost.get(), stats.cost_evaluations);
            assert!(stats.gradient_norm.unwrap() <= options.gradient_tolerance);
        }
    }
    check::<f64>();
    check::<f32>();
}

#[test]
fn rejection_limits_and_overdamping_do_not_report_convergence() {
    for overdamped in [false, true] {
        let mut graph = Solver::<States<f64>, Factors<f64>>::new();
        let initial = if overdamped { 0.0 } else { 10.0 };
        let x = graph.add(Scalar(initial));
        let calls = Rc::new(Calls::default());
        graph
            .add_factor(Curve {
                key: x,
                target: 1.,
                kind: if overdamped { Kind::Linear } else { Kind::Log },
                reset: Rc::new(Cell::new(false)),
                offset: 0.,
                calls: calls.clone(),
                malformed: false,
            })
            .unwrap();
        let mut method = LevenbergMarquardt::default();
        method.options.max_trials = 1;
        if overdamped {
            method.options.initial_damping = 1e20;
            method.options.max_damping = 1e20;
        }
        assert!(matches!(
            graph.optimize_with(&mut method, &Default::default()),
            Err(SolverError::NoProgress)
        ));
        assert_eq!(graph.get(x).unwrap().0, initial);
        assert_eq!(method.statistics().accepted_steps, 0);
        assert_eq!(method.statistics().attempts, 1);
        assert_eq!(calls.jacobian.get(), 1);
        assert_eq!(
            method.statistics().last_rejection,
            Some(if overdamped {
                TrialFailure::PoorAgreement
            } else {
                TrialFailure::InvalidEvaluation
            })
        );
    }
}

#[test]
fn invalid_controls_and_emissions_are_not_retried() {
    let mut graph = Solver::<States<f64>, Factors<f64>>::new();
    let x = graph.add(Scalar(0.0));
    let calls = Rc::new(Calls::default());
    graph
        .add_factor(Curve {
            key: x,
            target: 1.,
            kind: Kind::Linear,
            reset: Rc::new(Cell::new(false)),
            offset: 0.,
            calls: calls.clone(),
            malformed: true,
        })
        .unwrap();
    let mut method = LevenbergMarquardt::default();
    for damping in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        method.options.initial_damping = damping;
        assert!(matches!(
            graph.optimize_with(&mut method, &Default::default()),
            Err(SolverError::InvalidOptions)
        ));
        assert_eq!(calls.cost.get(), 0);
    }
    method.options = Default::default();
    assert!(matches!(
        graph.optimize_with(&mut method, &Default::default()),
        Err(SolverError::Evaluation(EvaluationError::InvalidEmission))
    ));
    assert_eq!(method.statistics().attempts, 0);
    assert_eq!(graph.get(x).unwrap().0, 0.0);
}

#[test]
fn linear_solve_failures_are_bounded_and_accepted_progress_survives_limits() {
    let mut graph = Solver::<States<f64>, Factors<f64>>::new();
    let reset = Rc::new(Cell::new(false));
    let calls = Rc::new(Calls::default());
    let a = graph.add(Scalar(0.5));
    let b = graph.add(Scalar(3.0));
    for (key, target) in [(a, 1.0), (b, 2.0)] {
        graph
            .add_factor(Curve {
                key,
                target,
                kind: Kind::Square,
                reset: reset.clone(),
                offset: 0.0,
                calls: calls.clone(),
                malformed: false,
            })
            .unwrap();
    }
    let mut backend = Lsmr::default();
    backend.max_iterations = 1;
    // Identity preconditioning keeps this deliberately unequal diagonal system
    // from becoming a one-iteration solve under column normalization.
    backend.diagonal_preconditioning = false;
    let mut lm = LevenbergMarquardt::new(backend);
    lm.options.max_trials = 2;
    assert!(matches!(
        graph.optimize_with(&mut lm, &Default::default()),
        Err(SolverError::NoProgress)
    ));
    assert_eq!(
        lm.statistics().last_rejection,
        Some(TrialFailure::LinearSolve)
    );
    assert_eq!(lm.statistics().attempts, 2);
    assert_eq!(lm.statistics().cost_evaluations, 1);
    assert_eq!(graph.get(a).unwrap().0, 0.5);
    assert_eq!(graph.get(b).unwrap().0, 3.0);

    let mut lm = LevenbergMarquardt::default();
    let options = OptimizeOptions {
        max_iterations: 1,
        ..Default::default()
    };
    assert!(matches!(
        graph.optimize_with(&mut lm, &options),
        Err(SolverError::NoConvergence)
    ));
    assert_eq!(lm.statistics().accepted_steps, 1);
    assert!(graph.get(a).unwrap().0 > 0.5);
    assert!(graph.get(b).unwrap().0 < 3.0);
    assert!(lm.statistics().cost.unwrap() < 0.5 * (0.75_f64.powi(2) + 5.0_f64.powi(2)));
    graph.optimize_with(&mut lm, &Default::default()).unwrap();
    assert!((graph.get(a).unwrap().0 - 1.0).abs() < 1e-8);
    assert!((graph.get(b).unwrap().0 - 2.0).abs() < 1e-8);
}

#[test]
fn lm_evaluates_marginal_priors_and_does_not_marginalize_damping() {
    let mut graph = Solver::<States<f64>, Factors<f64>>::new();
    let x = graph.add(Scalar(0.0));
    let y = graph.add(Scalar(0.0));
    let reset = Rc::new(Cell::new(false));
    for (key, target) in [(x, 1.0), (y, 6.0)] {
        graph
            .add_factor(Curve {
                key,
                target,
                kind: Kind::Linear,
                reset: reset.clone(),
                offset: 0.,
                calls: Rc::new(Calls::default()),
                malformed: false,
            })
            .unwrap();
    }
    graph
        .add_factor(Link {
            a: x,
            b: y,
            difference: 2.,
            reset,
        })
        .unwrap();
    let mut lm = LevenbergMarquardt::default();
    let first = graph.optimize_with(&mut lm, &Default::default()).unwrap();
    assert!(first.final_cost > 0.0);
    graph.marginalize(&[x.block_id()]).unwrap();
    let prior = graph
        .add_factor(Curve {
            key: y,
            target: 8.,
            kind: Kind::Linear,
            reset: Rc::new(Cell::new(false)),
            offset: 0.,
            calls: Rc::new(Calls::default()),
            malformed: false,
        })
        .unwrap();
    let result = graph.optimize_with(&mut lm, &Default::default()).unwrap();
    // Marginalized x contributes (y-3)^2/(2*101), plus y's two unit priors.
    let expected = (3.0 / 101.0 + 14.0) / (1.0 / 101.0 + 2.0);
    assert!((graph.get(y).unwrap().0 - expected).abs() < 1e-8);
    assert!(
        (result.final_cost
            - (0.5 * (expected - 3.0).powi(2) / 101.0
                + 0.5 * (expected - 6.0).powi(2)
                + 0.5 * (expected - 8.0).powi(2)))
        .abs()
            < 1e-10
    );
    graph.remove_factor(prior).unwrap();
    let reference = graph.optimize().unwrap();
    assert!((reference.final_cost - first.final_cost).abs() < 1e-10);
}

#[derive(Debug)]
struct Measurement {
    us: f64,
    allocations: usize,
    solved: bool,
    returned_error: bool,
    linearizations: usize,
    costs: usize,
    solves: usize,
    inner: usize,
    rejected: usize,
    objective: f64,
    error: f64,
    gradient: f64,
}

fn compare(n: usize, kind: Kind, use_lm: bool, samples: usize, block: bool) -> Vec<Measurement> {
    let options = OptimizeOptions {
        max_iterations: 100,
        gradient_tolerance: 1e-8,
        step_tolerance: 0.0,
        cost_tolerance: 0.0,
    };
    let mut gn = GaussNewton::new(Lsmr::default());
    let mut backend = Lsmr::default();
    backend.block_preconditioning = block;
    let mut lm = LevenbergMarquardt::new(backend);
    let mut resetter = GaussNewton::new(Lsmr::default());
    let mut measurements = Vec::new();
    for sample in 0..samples + 3 {
        let mut graph = Solver::<States<f64>, Factors<f64>>::new();
        let reset = Rc::new(Cell::new(true));
        let calls = Rc::new(Calls::default());
        let targets: Vec<_> = (0..n).map(|i| 0.8 + (i % 17) as f64 * 0.025).collect();
        let mut keys = Vec::new();
        for (i, &target) in targets.iter().enumerate() {
            let offset = match kind {
                Kind::Square => 0.05 * target,
                Kind::Atan => 2.0 + 0.1 * (i as f64).sin(),
                Kind::Log => 9.0 * target,
                Kind::Linear => 0.1,
            };
            let key = graph.add(Scalar(target + offset));
            graph
                .add_factor(Curve {
                    key,
                    target,
                    kind,
                    offset,
                    calls: calls.clone(),
                    reset: reset.clone(),
                    malformed: false,
                })
                .unwrap();
            if let Some(&previous) = keys.last() {
                graph
                    .add_factor(Link {
                        a: previous,
                        b: key,
                        difference: target - targets[i - 1],
                        reset: reset.clone(),
                    })
                    .unwrap();
            }
            keys.push(key);
        }
        // Prepare graph-owned trial capacity outside timing, with an identical
        // linear reset objective. Each sample starts from the same estimates.
        graph.optimize_with(&mut resetter, &options).unwrap();
        reset.set(false);
        calls.clear();
        let (result, allocations, us) = measured(|| {
            if use_lm {
                graph.optimize_with(&mut lm, &options)
            } else {
                graph.optimize_with(&mut gn, &options)
            }
        });
        let (linearizations, costs) = (calls.jacobian.get() / n, calls.cost.get());
        let stats = if use_lm {
            lm.backend().statistics()
        } else {
            gn.backend().statistics()
        };
        let rejected = if use_lm {
            lm.statistics().rejected_steps
        } else {
            0
        };
        // Independently check accuracy, cost, and gradient. A saturated atan
        // residual can have tiny gradient far from its solution: API success
        // alone must not make a divergent GN run look competitive.
        let values: Vec<_> = keys.iter().map(|&key| graph.get(key).unwrap().0).collect();
        let mut gradient = vec![0.0; n];
        let mut objective = 0.0;
        let mut error = 0.0_f64;
        for i in 0..n {
            let x = values[i];
            let v = x - targets[i];
            let (r, j) = match kind {
                Kind::Square => (x * x - targets[i] * targets[i], 2.0 * x),
                Kind::Atan => (v.atan(), 1.0 / (1.0 + v * v)),
                Kind::Log => ((x / targets[i]).ln(), 1.0 / x),
                Kind::Linear => (v, 1.0),
            };
            objective += 0.5 * r * r;
            gradient[i] += r * j;
            error = error.max(v.abs());
            if i > 0 {
                let r = 0.1 * (values[i] - values[i - 1] - targets[i] + targets[i - 1]);
                objective += 0.5 * r * r;
                gradient[i] += 0.1 * r;
                gradient[i - 1] -= 0.1 * r;
            }
        }
        let gradient = gradient.iter().fold(0.0_f64, |m, &g| {
            if g.is_finite() {
                m.max(g.abs())
            } else {
                f64::INFINITY
            }
        });
        let solved = result.is_ok() && error < 1e-6 && gradient <= 1.01e-8 && objective.is_finite();
        if sample >= 3 {
            measurements.push(Measurement {
                us,
                allocations,
                solved,
                returned_error: result.is_err(),
                linearizations,
                costs,
                solves: stats.solves,
                inner: stats.iterations,
                rejected,
                objective,
                error,
                gradient,
            });
        }
    }
    measurements
}

#[test]
fn warmed_lm_retries_and_graph_switching_allocate_nothing() {
    for block in [false, true] {
        for kind in [Kind::Square, Kind::Atan, Kind::Log] {
            for m in compare(16, kind, true, 3, block) {
                assert!(m.solved, "{kind:?}: {m:?}");
                assert_eq!(m.allocations, 0, "{kind:?}: {m:?}");
                if matches!(kind, Kind::Atan | Kind::Log) {
                    assert!(m.rejected > 0);
                }
            }
        }
    }
}

#[test]
fn schur_reuses_warm_workspace_and_converges_on_fresh_graphs() {
    let mut lm = LevenbergMarquardt::new(fagra::Schur::new(1, 1, 1));
    for sample in 0..4 {
        let mut graph = Solver::<States<f64>, Factors<f64>>::new();
        let a = graph.add(Scalar(0.));
        let b = graph.add(Scalar(0.));
        let reset = Rc::new(Cell::new(true));
        for (key, target, offset) in [(a, 1., -0.5), (b, 2., 1.)] {
            graph
                .add_factor(Curve {
                    key,
                    target,
                    offset,
                    kind: Kind::Square,
                    reset: reset.clone(),
                    calls: Rc::new(Calls::default()),
                    malformed: false,
                })
                .unwrap();
        }
        graph
            .add_factor(Link {
                a,
                b,
                difference: 1.,
                reset: reset.clone(),
            })
            .unwrap();
        // Prepare graph-owned trial capacity and reset to identical initial values.
        graph.optimize().unwrap();
        reset.set(false);
        let (result, allocations, _) =
            measured(|| graph.optimize_with(&mut lm, &Default::default()));
        let report = result.unwrap();
        assert!(report.final_cost < 1e-12);
        assert!((graph.get(a).unwrap().0 - 1.).abs() < 1e-8);
        assert!((graph.get(b).unwrap().0 - 2.).abs() < 1e-8);
        if sample > 0 {
            assert_eq!(allocations, 0);
        }
    }
}

#[test]
fn scoped_covariance_reuses_hot_workspace_for_all_backends() {
    fn check<O: fagra::__private::CovarianceOptimizer<States<f64>, Factors<f64>>>(mut method: O) {
        let mut graph = Solver::<States<f64>, Factors<f64>>::new();
        let a = graph.add(Scalar(0.5));
        let b = graph.add(Scalar(1.));
        let reset = Rc::new(Cell::new(false));
        for (key, target) in [(a, 1.), (b, 2.)] {
            graph
                .add_factor(Curve {
                    key,
                    target,
                    offset: 0.,
                    kind: Kind::Square,
                    reset: reset.clone(),
                    calls: Rc::new(Calls::default()),
                    malformed: false,
                })
                .unwrap();
        }
        graph
            .add_factor(Link {
                a,
                b,
                difference: 1.,
                reset,
            })
            .unwrap();
        let blocks = [b.block_id(), a.block_id()];
        for sample in 0..6 {
            let (_, allocations, _) = measured(|| {
                graph.set(a, Scalar(0.5)).unwrap();
                graph.set(b, Scalar(1.)).unwrap();
                let (_, cov) = graph
                    .optimize_with_covariance(
                        &mut method,
                        &Default::default(),
                        &blocks[..if sample % 2 == 0 { 2 } else { 1 }],
                        &Default::default(),
                    )
                    .unwrap();
                assert!((cov[(0, 0)] - 4.01 / 64.2).abs() < 1e-8);
                if cov.ncols() == 2 {
                    assert!((cov[(0, 1)] - 0.01 / 64.2).abs() < 1e-8);
                    assert!((cov[(1, 1)] - 16.01 / 64.2).abs() < 1e-8);
                }
            });
            if sample > 0 {
                assert_eq!(allocations, 0, "sample {sample}");
            }
        }
    }
    check(GaussNewton::default());
    check(GaussNewton::new(Lsmr::default()));
    check(LevenbergMarquardt::default());
    check(LevenbergMarquardt::new(fagra::Schur::new(1, 1, 1)));
}

#[test]
#[ignore = "release GN/LM comparison; run with --ignored --nocapture --test-threads=1"]
fn benchmark() {
    println!(
        "problem,dof,method,median_us,p95_us,successes,api_errors,jacobians,costs,solves,inner_iterations,rejections,max_allocations,max_cost,max_error,max_gradient"
    );
    for kind in [Kind::Square, Kind::Atan, Kind::Log] {
        for n in [32, 256] {
            for use_lm in [false, true] {
                let mut runs = compare(n, kind, use_lm, 31, false);
                runs.sort_by(|a, b| a.us.total_cmp(&b.us));
                let median = &runs[runs.len() / 2];
                let successes = runs.iter().filter(|m| m.solved).count();
                let errors = runs.iter().filter(|m| m.returned_error).count();
                let allocations = runs.iter().map(|m| m.allocations).max().unwrap();
                let max = |f: fn(&Measurement) -> f64| runs.iter().map(f).fold(0.0_f64, f64::max);
                println!(
                    "{kind:?},{n},{},{:.3},{:.3},{successes}/31,{errors},{},{},{},{},{},{allocations},{:.3e},{:.3e},{:.3e}",
                    if use_lm { "LM" } else { "GN" },
                    median.us,
                    runs[(runs.len() * 95).div_ceil(100) - 1].us,
                    median.linearizations,
                    median.costs,
                    median.solves,
                    median.inner,
                    median.rejected,
                    max(|m| m.objective),
                    max(|m| m.error),
                    max(|m| m.gradient)
                );
                if use_lm || matches!(kind, Kind::Square) {
                    assert_eq!(successes, 31);
                }
                assert_eq!(allocations, 0);
            }
        }
    }
}
