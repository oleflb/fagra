use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    rc::Rc,
};

use faer_ext::nalgebra::{SMatrix, SVector};
use fagra::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorId, FactorSelection, GaussNewton,
    JacobianBlock, LinearizationSink, Lsmr, OptimizeOptions, Solver, SolverError, StateKey,
    StateStore, TerminationReason, Variable,
};

#[path = "optimizer/precision.rs"]
mod precision;
#[allow(dead_code)]
#[path = "../examples/scalar_prior.rs"]
mod scalar;

use precision::Square;
use scalar::Scalar;

// Count only allocations on the thread currently exercising an optimization call.
// The unsafe forwarding is confined to the test allocator; production uses safe APIs.
struct CountingAllocator;
thread_local! { static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) }; }
fn record_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(value) = count.get() {
            count.set(Some(value + 1));
        }
    });
}
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            System.dealloc(ptr, layout);
        }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations<T>(f: impl FnOnce() -> T) -> (T, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ALLOCATIONS.with(|count| count.set(None));
        }
    }
    ALLOCATIONS.with(|count| {
        assert!(count.get().is_none());
        count.set(Some(0));
    });
    let reset = Reset;
    let result = f();
    let count = ALLOCATIONS.with(|count| count.get().unwrap());
    drop(reset);
    (result, count)
}

struct Prior {
    key: StateKey<Scalar>,
    target: Rc<Cell<f64>>,
}
impl<S: StateStore<Scalar>> Factor<S> for Prior {
    type Scalar = f64;
    fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
        visit(self.key.block_id());
    }
    fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
        Ok(0.5 * (states.get(self.key)?.0 - self.target.get()).powi(2))
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        sink.residual(
            &SVector::<f64, 1>::new(states.get(self.key)?.0 - self.target.get()),
            &[JacobianBlock::new(
                self.key,
                &SMatrix::<f64, 1, 1>::new(1.0),
            )],
        )
    }
}

struct Difference {
    left: StateKey<Scalar>,
    right: StateKey<Scalar>,
}
impl<S: StateStore<Scalar>> Factor<S> for Difference {
    type Scalar = f64;
    fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
        visit(self.left.block_id());
        visit(self.right.block_id());
    }
    fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
        Ok(0.5 * (states.get(self.left)?.0 - states.get(self.right)?.0).powi(2))
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        sink.residual(
            &SVector::<f64, 1>::new(states.get(self.left)?.0 - states.get(self.right)?.0),
            &[
                JacobianBlock::new(self.left, &SMatrix::<f64, 1, 1>::new(1.0)),
                JacobianBlock::new(self.right, &SMatrix::<f64, 1, 1>::new(-1.0)),
            ],
        )
    }
}
struct Pair([f64; 2]);
impl Variable for Pair {
    type Scalar = f64;
    type Tangent = [f64; 2];
    const DOF: usize = 2;
    fn tangent_from_slice(delta: &[f64]) -> [f64; 2] {
        [delta[0], delta[1]]
    }
    fn retract(&self, delta: &[f64; 2]) -> Self {
        Self([self.0[0] + delta[0], self.0[1] + delta[1]])
    }
}

struct Coupled {
    x: StateKey<Scalar>,
    y: StateKey<Pair>,
}
impl<S: StateStore<Scalar> + StateStore<Pair>> Factor<S> for Coupled {
    type Scalar = f64;
    fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
        visit(self.x.block_id());
        visit(self.y.block_id());
    }
    fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
        let x = states.get(self.x)?.0;
        let [y, z] = states.get(self.y)?.0;
        Ok(0.5 * ((x + y - 3.0).powi(2) + (y + z - 5.0).powi(2) + (x + 2.0 * z - 7.0).powi(2)))
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let x = states.get(self.x)?.0;
        let [y, z] = states.get(self.y)?.0;
        let jx = SMatrix::<f64, 3, 1>::new(1.0, 0.0, 1.0);
        let jy = SMatrix::<f64, 3, 2>::from_row_slice(&[1.0, 0.0, 1.0, 1.0, 0.0, 2.0]);
        // Reverse the declared block order to exercise offset resolution/orientation.
        sink.residual(
            &SVector::<f64, 3>::new(x + y - 3.0, y + z - 5.0, x + 2.0 * z - 7.0),
            &[
                JacobianBlock::new(self.y, &jy),
                JacobianBlock::new(self.x, &jx),
            ],
        )
    }
}

struct Domain {
    key: StateKey<Scalar>,
    target: Rc<Cell<f64>>,
}
impl<S: StateStore<Scalar>> Factor<S> for Domain {
    type Scalar = f64;
    fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
        visit(self.key.block_id());
    }
    fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
        let x = states.get(self.key)?.0;
        if x > 1.0 {
            return Err(EvaluationError::InvalidEvaluation);
        }
        Ok(0.5 * (x - self.target.get()).powi(2))
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        sink.residual(
            &SVector::<f64, 1>::new(states.get(self.key)?.0 - self.target.get()),
            &[JacobianBlock::new(
                self.key,
                &SMatrix::<f64, 1, 1>::new(1.0),
            )],
        )
    }
}

struct Measurement {
    key: StateKey<Scalar>,
    target: f64,
}
struct BatchModel {
    omit: Rc<Cell<bool>>,
}
impl<S: StateStore<Scalar>> FactorBatch<S> for BatchModel {
    type Scalar = f64;
    type Factor = Measurement;
    fn visit_variables(&self, factor: &Measurement, mut visit: impl FnMut(BlockId)) {
        visit(factor.key.block_id());
    }
    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, Measurement>,
    ) -> Result<f64, EvaluationError> {
        factors
            .map(|(_, f)| Ok(0.5 * (states.get(f.key)?.0 - f.target).powi(2)))
            .sum()
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        states: &S,
        factors: FactorSelection<'_, Measurement>,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        if self.omit.get() {
            return Ok(());
        }
        // Rotate emissions to exercise out-of-order factor scopes without allocation.
        let mut factors = factors;
        let first = factors.next();
        for (id, factor) in factors.chain(first) {
            sink.factor(id, |sink| {
                sink.residual(
                    &SVector::<f64, 1>::new(states.get(factor.key)?.0 - factor.target),
                    &[JacobianBlock::new(
                        factor.key,
                        &SMatrix::<f64, 1, 1>::new(1.0),
                    )],
                )
            })?;
        }
        Ok(())
    }
}

fagra::states! { States { scalars: Scalar, pairs: Pair } }
fagra::factors! {
    Factors {
        priors: Prior, squares: Square<f64>, coupled: Coupled, differences: Difference,
        domains: Domain, batch: Batch<BatchModel, Measurement>,
    }
}
type Graph = Solver<States, Factors>;

#[test]
fn lsmr_optimizes_ordinary_and_batched_factors_and_reuses_workspace() {
    let options = OptimizeOptions::default();
    let mut method = GaussNewton::new(Lsmr::default());
    let mut graph = Graph::new();
    let x = graph.add(Scalar(0.0));
    let y = graph.add(Pair([0.0, 0.0]));
    graph.add_factor(Coupled { x, y }).unwrap();
    graph.optimize_with(&mut method, &options).unwrap();
    assert!((graph.get(x).unwrap().0 - 1.0).abs() < 1e-8);
    assert!((graph.get(y).unwrap().0[0] - 2.0).abs() < 1e-8);
    assert!((graph.get(y).unwrap().0[1] - 3.0).abs() < 1e-8);

    let mut graph = Graph::new();
    let x = graph.add(Scalar(2.0));
    graph.add_factor(Square(x)).unwrap();
    graph.optimize_with(&mut method, &options).unwrap();
    assert!((graph.get(x).unwrap().0 - 1.0).abs() < 1e-8);

    let mut graph = Graph::new();
    let x = graph.add(Scalar(0.0));
    let unused = graph.add(Pair([7.0, 8.0]));
    let target = Rc::new(Cell::new(3.0));
    graph
        .add_factor(Prior {
            key: x,
            target: target.clone(),
        })
        .unwrap();
    graph.optimize_with(&mut method, &options).unwrap();
    assert!((graph.get(x).unwrap().0 - 3.0).abs() < 1e-8);
    assert_eq!(graph.get(unused).unwrap().0, [7.0, 8.0]);
    // Model edits between calls invalidate the previous linearization.
    target.set(4.0);
    let (result, count) = allocations(|| graph.optimize_with(&mut method, &options));
    result.unwrap();
    assert_eq!(count, 0, "warmed LSMR optimization allocated");
    assert!((graph.get(x).unwrap().0 - 4.0).abs() < 1e-8);

    let batch = graph.add_batch(BatchModel {
        omit: Rc::new(Cell::new(false)),
    });
    graph
        .add_factor_to(
            batch,
            Measurement {
                key: x,
                target: 2.0,
            },
        )
        .unwrap();
    graph.optimize_with(&mut method, &options).unwrap();
    assert!((graph.get(x).unwrap().0 - 3.0).abs() < 1e-8);
}

#[test]
fn scalar_nonlinear_and_coupled_problems_converge() {
    let mut graph = Graph::new();
    let x = graph.add(Scalar(0.0));
    graph
        .add_factor(Prior {
            key: x,
            target: Rc::new(Cell::new(3.0)),
        })
        .unwrap();
    let report = graph.optimize().unwrap();
    assert_eq!(report.iterations, 1);
    assert_eq!(report.initial_cost, 4.5);
    assert_eq!(report.final_cost, 0.0);
    assert_eq!(graph.get(x).unwrap().0, 3.0);
    assert_eq!(graph.optimize().unwrap().iterations, 0);

    let mut graph = Graph::new();
    let x = graph.add(Scalar(2.0));
    graph.add_factor(Square(x)).unwrap();
    let report = graph.optimize().unwrap();
    assert!(report.iterations > 1);
    assert!((graph.get(x).unwrap().0 - 1.0).abs() < 1e-8);
    assert!(report.final_cost < 1e-16);

    let mut graph = Graph::new();
    let x = graph.add(Scalar(0.0));
    let y = graph.add(Pair([0.0, 0.0]));
    graph.add_factor(Coupled { x, y }).unwrap();
    graph.optimize().unwrap();
    assert!((graph.get(x).unwrap().0 - 1.0).abs() < 1e-12);
    assert!((graph.get(y).unwrap().0[0] - 2.0).abs() < 1e-12);
    assert!((graph.get(y).unwrap().0[1] - 3.0).abs() < 1e-12);
}

#[test]
fn pure_gn_accepts_uphill_steps_and_keeps_accepted_progress_on_limit() {
    let mut graph = Graph::new();
    let x = graph.add(Scalar(0.1));
    let factor = graph.add_factor(Square(x)).unwrap();
    let initial = graph.factor_cost(factor).unwrap();
    let options = OptimizeOptions {
        max_iterations: 1,
        ..Default::default()
    };
    assert!(matches!(
        graph.optimize_with(&mut GaussNewton::default(), &options),
        Err(SolverError::NoConvergence)
    ));
    assert!((graph.get(x).unwrap().0 - 5.05).abs() < 1e-12);
    assert!(graph.factor_cost(factor).unwrap() > initial);
    graph.optimize().unwrap();
    assert!((graph.get(x).unwrap().0 - 1.0).abs() < 1e-8);
}

#[test]
fn failed_trial_restores_estimates_and_workspace_remains_reusable() {
    let mut graph = Graph::new();
    let x = graph.add(Scalar(0.0));
    let target = Rc::new(Cell::new(2.0));
    let factor = graph
        .add_factor(Domain {
            key: x,
            target: target.clone(),
        })
        .unwrap();
    let mut method = GaussNewton::default();
    assert!(matches!(
        graph.optimize_with(&mut method, &OptimizeOptions::default()),
        Err(SolverError::Evaluation(EvaluationError::InvalidEvaluation))
    ));
    assert_eq!(graph.get(x).unwrap().0, 0.0);
    assert_eq!(graph.factor_cost(factor).unwrap(), 2.0);
    target.set(0.5);
    graph
        .optimize_with(&mut method, &OptimizeOptions::default())
        .unwrap();
    assert_eq!(graph.get(x).unwrap().0, 0.5);
}

#[test]
fn options_empty_graph_and_small_step_have_defined_results() {
    let mut graph = Graph::new();
    let report = graph.optimize().unwrap();
    assert_eq!(report.termination, TerminationReason::NoVariables);
    assert_eq!(report.final_cost, 0.0);
    let x = graph.add(Scalar(0.0));
    graph
        .add_factor(Prior {
            key: x,
            target: Rc::new(Cell::new(1.0)),
        })
        .unwrap();
    for options in [
        OptimizeOptions {
            max_iterations: 0,
            ..Default::default()
        },
        OptimizeOptions {
            gradient_tolerance: f64::NAN,
            ..Default::default()
        },
        OptimizeOptions {
            step_tolerance: -1.0,
            ..Default::default()
        },
        OptimizeOptions {
            cost_tolerance: f64::INFINITY,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            graph.optimize_with(&mut GaussNewton::default(), &options),
            Err(SolverError::InvalidOptions)
        ));
        assert_eq!(graph.get(x).unwrap().0, 0.0);
    }
    let options = OptimizeOptions {
        step_tolerance: 2.0,
        ..Default::default()
    };
    let report = graph
        .optimize_with(&mut GaussNewton::default(), &options)
        .unwrap();
    assert_eq!(report.termination, TerminationReason::StepTolerance);
    assert_eq!(report.iterations, 0);
    assert_eq!(graph.get(x).unwrap().0, 0.0);
}

#[test]
fn batch_equivalence_missing_scopes_and_edits_are_handled() {
    let mut ordinary = Graph::new();
    let x = ordinary.add(Scalar(0.0));
    for target in [1.0, 3.0] {
        ordinary
            .add_factor(Prior {
                key: x,
                target: Rc::new(Cell::new(target)),
            })
            .unwrap();
    }
    let mut batched = Graph::new();
    let y = batched.add(Scalar(0.0));
    let omit = Rc::new(Cell::new(true));
    let batch = batched.add_batch(BatchModel { omit: omit.clone() });
    let first = batched
        .add_factor_to(
            batch,
            Measurement {
                key: y,
                target: 1.0,
            },
        )
        .unwrap();
    batched
        .add_factor_to(
            batch,
            Measurement {
                key: y,
                target: 3.0,
            },
        )
        .unwrap();
    assert!(matches!(
        batched.optimize(),
        Err(SolverError::Evaluation(EvaluationError::InvalidEmission))
    ));
    assert_eq!(batched.get(y).unwrap().0, 0.0);
    omit.set(false);
    let a = ordinary.optimize().unwrap();
    let b = batched.optimize().unwrap();
    assert!((a.final_cost - b.final_cost).abs() < 1e-12);
    assert!((ordinary.get(x).unwrap().0 - batched.get(y).unwrap().0).abs() < 1e-12);
    batched.remove_factor(first).unwrap();
    batched.optimize().unwrap();
    assert!((batched.get(y).unwrap().0 - 3.0).abs() < 1e-12);

    // One method may switch to an unrelated graph, even with matching dimensions.
    let mut method = GaussNewton::default();
    ordinary
        .optimize_with(&mut method, &OptimizeOptions::default())
        .unwrap();
    batched
        .optimize_with(&mut method, &OptimizeOptions::default())
        .unwrap();
    let z = batched.add(Scalar(0.0));
    batched
        .add_factor(Prior {
            key: z,
            target: Rc::new(Cell::new(7.0)),
        })
        .unwrap();
    batched
        .optimize_with(&mut method, &OptimizeOptions::default())
        .unwrap();
    assert_eq!(batched.get(z).unwrap().0, 7.0);
    // Shrink the system again and force a step, retaining the larger leading
    // dimension and scratch allocation in the backend.
    ordinary
        .add_factor(Prior {
            key: x,
            target: Rc::new(Cell::new(5.0)),
        })
        .unwrap();
    ordinary
        .optimize_with(&mut method, &OptimizeOptions::default())
        .unwrap();
    assert!((ordinary.get(x).unwrap().0 - 3.0).abs() < 1e-12);
}

#[derive(Clone, Copy)]
enum BadEmission {
    Rows,
    Duplicate,
    Nonfinite,
    Overflow,
    Foreign,
    Nested,
    Swallowed,
}
struct Bad {
    key: StateKey<Scalar>,
    extra: StateKey<Scalar>,
    foreign: StateKey<Scalar>,
    factor_id: FactorId,
    mode: BadEmission,
}
impl<S> Factor<S> for Bad {
    type Scalar = f64;
    fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
        visit(self.key.block_id());
        visit(self.extra.block_id());
    }
    fn cost(&self, _: &S) -> Result<f64, EvaluationError> {
        Ok(0.5)
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        _: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let jacobian = SMatrix::<f64, 1, 1>::new(1.0);
        let residual = SVector::<f64, 1>::new(1.0);
        match self.mode {
            BadEmission::Rows => sink.residual(
                &residual,
                &[JacobianBlock::new(
                    self.key,
                    &SMatrix::<f64, 2, 1>::new(1.0, 1.0),
                )],
            ),
            BadEmission::Duplicate => sink.residual(
                &residual,
                &[
                    JacobianBlock::new(self.key, &jacobian),
                    JacobianBlock::new(self.key, &jacobian),
                ],
            ),
            BadEmission::Nonfinite => sink.residual(
                &residual,
                &[JacobianBlock::new(
                    self.key,
                    &SMatrix::<f64, 1, 1>::new(f64::NAN),
                )],
            ),
            BadEmission::Overflow => sink.residual(
                &residual,
                &[JacobianBlock::new(
                    self.key,
                    &SMatrix::<f64, 1, 1>::new(1e200),
                )],
            ),
            BadEmission::Foreign => {
                sink.residual(&residual, &[JacobianBlock::new(self.foreign, &jacobian)])
            }
            BadEmission::Nested => sink.factor(self.factor_id, |_| Ok(())),
            BadEmission::Swallowed => {
                let _ = sink.factor(self.factor_id, |_| Ok(()));
                Ok(())
            }
        }
    }
}

#[test]
fn invalid_emissions_and_singular_systems_never_change_states() {
    fagra::factors! { BadFactors { bad: Bad } }
    let mut id_pool = fagra::__private::FactorPool::default();
    id_pool.insert(());
    let factor_id = id_pool.iter().next().unwrap().0;
    let foreign = Graph::new().add(Scalar(0.0));
    for mode in [
        BadEmission::Rows,
        BadEmission::Duplicate,
        BadEmission::Nonfinite,
        BadEmission::Overflow,
        BadEmission::Foreign,
        BadEmission::Nested,
        BadEmission::Swallowed,
    ] {
        let mut graph = Solver::<States, BadFactors>::new();
        let key = graph.add(Scalar(0.0));
        let extra = graph.add(Scalar(0.0));
        graph
            .add_factor(Bad {
                key,
                extra,
                foreign,
                factor_id,
                mode,
            })
            .unwrap();
        assert!(matches!(graph.optimize(), Err(SolverError::Evaluation(_))));
        assert_eq!(graph.get(key).unwrap().0, 0.0);
    }
    let mut graph = Graph::new();
    let key = graph.add(Scalar(0.0));
    let unconstrained = graph.add(Scalar(9.0));
    graph
        .add_factor(Prior {
            key,
            target: Rc::new(Cell::new(1.0)),
        })
        .unwrap();
    assert!(matches!(
        graph.optimize(),
        Err(SolverError::LinearSolveFailed)
    ));
    assert_eq!(graph.get(key).unwrap().0, 0.0);
    assert_eq!(graph.get(unconstrained).unwrap().0, 9.0);
}

#[test]
fn one_factor_can_emit_multiple_residual_blocks_for_the_same_variable() {
    struct TwoMeasurements(StateKey<Scalar>);
    impl<S: StateStore<Scalar>> Factor<S> for TwoMeasurements {
        type Scalar = f64;
        fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
            visit(self.0.block_id());
        }
        fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
            let x = states.get(self.0)?.0;
            Ok(0.5 * ((x - 1.0).powi(2) + (x - 3.0).powi(2)))
        }
        fn linearize<L: LinearizationSink<Scalar = f64>>(
            &self,
            states: &S,
            sink: &mut L,
        ) -> Result<(), EvaluationError> {
            let x = states.get(self.0)?.0;
            let matrix = SMatrix::<f64, 1, 1>::new(1.0);
            let blocks = [JacobianBlock::new(self.0, &matrix)];
            sink.residual(&SVector::<f64, 1>::new(x - 1.0), &blocks)?;
            sink.residual(&SVector::<f64, 1>::new(x - 3.0), &blocks)
        }
    }
    fagra::factors! { Multiple { measurements: TwoMeasurements } }
    let mut graph = Solver::<States, Multiple>::new();
    let x = graph.add(Scalar(0.0));
    graph.add_factor(TwoMeasurements(x)).unwrap();
    let report = graph.optimize().unwrap();
    assert!((graph.get(x).unwrap().0 - 2.0).abs() < 1e-12);
    assert!((report.final_cost - 1.0).abs() < 1e-12);
}

#[test]
fn retraction_panic_discards_partially_staged_pools() {
    struct Fragile {
        value: f64,
        panic: Rc<Cell<bool>>,
    }
    impl Variable for Fragile {
        type Scalar = f64;
        type Tangent = f64;
        const DOF: usize = 1;
        fn tangent_from_slice(delta: &[f64]) -> f64 {
            delta[0]
        }
        fn retract(&self, delta: &f64) -> Self {
            assert!(!self.panic.get(), "test retraction panic");
            Self {
                value: self.value + delta,
                panic: self.panic.clone(),
            }
        }
    }
    struct FragilePrior(StateKey<Fragile>);
    impl<S: StateStore<Fragile>> Factor<S> for FragilePrior {
        type Scalar = f64;
        fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
            visit(self.0.block_id());
        }
        fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
            Ok(0.5 * (states.get(self.0)?.value - 1.0).powi(2))
        }
        fn linearize<L: LinearizationSink<Scalar = f64>>(
            &self,
            states: &S,
            sink: &mut L,
        ) -> Result<(), EvaluationError> {
            sink.residual(
                &SVector::<f64, 1>::new(states.get(self.0)?.value - 1.0),
                &[JacobianBlock::new(self.0, &SMatrix::<f64, 1, 1>::new(1.0))],
            )
        }
    }
    fagra::states! { MixedStates { first: Scalar, second: Fragile } }
    fagra::factors! { MixedFactors { first: Prior, second: FragilePrior } }
    let mut graph = Solver::<MixedStates, MixedFactors>::new();
    let x = graph.add(Scalar(0.0));
    let panic = Rc::new(Cell::new(true));
    let y = graph.add(Fragile {
        value: 0.0,
        panic: panic.clone(),
    });
    graph
        .add_factor(Prior {
            key: x,
            target: Rc::new(Cell::new(1.0)),
        })
        .unwrap();
    graph.add_factor(FragilePrior(y)).unwrap();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| graph.optimize())).is_err());
    assert_eq!(graph.get(x).unwrap().0, 0.0);
    assert_eq!(graph.get(y).unwrap().value, 0.0);
    panic.set(false);
    graph.optimize().unwrap();
    assert_eq!(graph.get(x).unwrap().0, 1.0);
    assert_eq!(graph.get(y).unwrap().value, 1.0);
}

#[test]
fn warmed_optimization_reuses_all_workspace() {
    let mut graph = Graph::new();
    let x = graph.add(Scalar(0.0));
    let y = graph.add(Pair([0.0, 0.0]));
    graph.add_factor(Coupled { x, y }).unwrap();
    let target = Rc::new(Cell::new(4.0));
    graph
        .add_factor(Prior {
            key: x,
            target: target.clone(),
        })
        .unwrap();
    let mut method = GaussNewton::default();
    graph
        .optimize_with(&mut method, &OptimizeOptions::default())
        .unwrap();
    target.set(8.0);
    let (result, count) =
        allocations(|| graph.optimize_with(&mut method, &OptimizeOptions::default()));
    assert!(result.unwrap().iterations > 0);
    assert_eq!(count, 0, "prepared optimization allocated");
    // Check the retained default method too, while executing a real step again.
    graph.optimize().unwrap();
    target.set(10.0);
    let (result, count) = allocations(|| graph.optimize());
    assert!(result.unwrap().iterations > 0);
    assert_eq!(count, 0);
}

#[test]
#[ignore = "1000-DOF release workload: run with --release --ignored --nocapture"]
fn thousand_dof_dense_solve() {
    let mut graph = Graph::new();
    let target = Rc::new(Cell::new(1.0));
    let keys: Vec<_> = (0..1000)
        .map(|_| {
            let key = graph.add(Scalar(0.0));
            graph
                .add_factor(Prior {
                    key,
                    target: target.clone(),
                })
                .unwrap();
            key
        })
        .collect();
    // Couple neighboring variables as well as anchoring each one: exercise
    // cross-block assembly, not just a diagonal normal matrix.
    for pair in keys.windows(2) {
        graph
            .add_factor(Difference {
                left: pair[0],
                right: pair[1],
            })
            .unwrap();
    }
    graph.optimize().unwrap();
    target.set(2.0);
    let start = std::time::Instant::now();
    let (result, count) = allocations(|| graph.optimize());
    let elapsed = start.elapsed();
    let report = result.unwrap();
    assert_eq!(count, 0);
    assert_eq!(report.iterations, 1);
    for key in keys {
        assert!((graph.get(key).unwrap().0 - 2.0).abs() < 1e-12);
    }
    println!("1000-DOF prepared dense GN: {elapsed:?}, {count} allocations");
}

#[test]
#[ignore = "solver comparison: run with --release --ignored --nocapture --test-threads=1"]
fn compare_cholesky_lsmr() {
    use fagra::{__private::LeastSquaresBackend, DenseNormalCholesky};
    use std::time::{Duration, Instant};

    fn measure<B: LeastSquaresBackend<Scalar = f64>>(
        n: usize,
        anchor_stride: usize,
        backend: B,
    ) -> (Duration, Vec<f64>, f64) {
        let mut graph = Graph::new();
        let keys: Vec<_> = (0..n).map(|_| graph.add(Scalar(0.0))).collect();
        let mut random = 42_u64;
        let targets: Vec<_> = (0..n)
            .step_by(anchor_stride)
            .map(|i| {
                // Deterministic varying RHS: a uniform target makes LSMR trivial.
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                let target = Rc::new(Cell::new((random >> 32) as f64 / u32::MAX as f64 - 0.5));
                graph
                    .add_factor(Prior {
                        key: keys[i],
                        target: target.clone(),
                    })
                    .unwrap();
                (i, target)
            })
            .collect();
        for pair in keys.windows(2) {
            graph
                .add_factor(Difference {
                    left: pair[0],
                    right: pair[1],
                })
                .unwrap();
        }
        let mut method = GaussNewton::new(backend);
        let options = OptimizeOptions::default();
        let mut samples = Vec::with_capacity(9);
        for sample in 0..12 {
            // Flip the objective outside the timer, forcing a real solve each time.
            for (_, target) in &targets {
                target.set(-target.get());
            }
            let start = Instant::now();
            let (result, count) = allocations(|| graph.optimize_with(&mut method, &options));
            let elapsed = start.elapsed();
            assert_eq!(result.unwrap().iterations, 1);
            if sample >= 3 {
                assert_eq!(count, 0, "warmed optimization allocated");
                samples.push(elapsed);
            }
        }
        // Independently check stationarity, outside the measured region.
        let values: Vec<_> = keys.iter().map(|&key| graph.get(key).unwrap().0).collect();
        let mut gradient = vec![0.0; n];
        for (i, target) in targets {
            gradient[i] += values[i] - target.get();
        }
        for i in 1..n {
            let residual = values[i - 1] - values[i];
            gradient[i - 1] += residual;
            gradient[i] -= residual;
        }
        assert!(
            gradient
                .iter()
                .all(|g| g.is_finite() && g.abs() <= options.gradient_tolerance)
        );
        let gradient_norm = gradient.iter().map(|g| g.abs()).fold(0.0, f64::max);
        samples.sort_unstable();
        (samples[samples.len() / 2], values, gradient_norm)
    }

    assert!(
        !cfg!(debug_assertions),
        "use --release for meaningful timings"
    );
    println!(
        "Median of 9 warmed optimize_with calls; 3 warmups; default tolerances; sequential kernels"
    );
    println!(
        "anchor_stride,n,residuals,cholesky_ms,lsmr_ms,cholesky/lsmr,max_solution_difference,lsmr_gradient_inf"
    );
    for stride in [1, 32] {
        for n in [10, 30, 100, 300, 1000, 3000, 5000] {
            let (cholesky, expected, _) = measure(n, stride, DenseNormalCholesky::default());
            let (lsmr, actual, gradient) = measure(n, stride, Lsmr::default());
            let difference = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(difference < 1e-7, "solvers disagree by {difference}");
            println!(
                "{stride},{n},{},{:.6},{:.6},{:.3},{difference:.3e},{gradient:.3e}",
                n - 1 + n.div_ceil(stride),
                cholesky.as_secs_f64() * 1000.0,
                lsmr.as_secs_f64() * 1000.0,
                cholesky.as_secs_f64() / lsmr.as_secs_f64()
            );
        }
    }
}
