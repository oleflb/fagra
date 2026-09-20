//! Reproducible warm marginalization benchmark; insertion is outside measurement.
use fagra::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorSelection, LinearizationSink, Solver,
    StateStore,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    collections::VecDeque,
    time::Instant,
};
#[allow(dead_code)]
#[path = "../examples/marginalization.rs"]
mod example;
use example::{Difference, Prior, Scalar, States};

struct Counter;
thread_local! { static COUNT: Cell<Option<usize>> = const { Cell::new(None) }; }
fn record() {
    let _ = COUNT.try_with(|c| {
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
fn measured<T>(f: impl FnOnce() -> T) -> (T, usize, u128) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            COUNT.with(|c| c.set(None));
        }
    }
    COUNT.with(|c| {
        assert!(c.get().is_none());
        c.set(Some(0));
    });
    let reset = Reset;
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed().as_nanos();
    let count = COUNT.with(|c| c.get().unwrap());
    drop(reset);
    (result, count, elapsed)
}

fagra::factors! { Factors { priors: Prior, edges: Difference, batches: Batch<Batched, DifferencePayload> } }
// A separate payload type preserves the schema's unique factor routing.
struct DifferencePayload(Difference);
struct Batched;
impl<S: StateStore<Scalar>> FactorBatch<S> for Batched {
    type Scalar = f64;
    type Factor = DifferencePayload;
    fn visit_variables(&self, f: &DifferencePayload, v: impl FnMut(BlockId)) {
        <Difference as Factor<S>>::visit_variables(&f.0, v);
    }
    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, DifferencePayload>,
    ) -> Result<f64, EvaluationError> {
        factors.map(|(_, f)| f.0.cost(states)).sum()
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        states: &S,
        factors: FactorSelection<'_, DifferencePayload>,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        for (id, f) in factors {
            sink.factor(id, |out| f.0.linearize(states, out))?;
        }
        Ok(())
    }
}

fn workload(width: usize, batched: bool, samples: usize, steps: usize) -> (f64, usize, usize) {
    let mut graph = Solver::<States<f64>, Factors>::new();
    let batch = graph.add_batch(Batched);
    let mut window = VecDeque::new();
    let mut timings = Vec::new();
    let mut min_alloc = usize::MAX;
    let mut max_alloc = 0;
    let warmup = width * 2;
    let mut sample_ns = 0;
    for time in 0..width + warmup + samples * steps {
        let next = graph.add(Scalar(time as f64 + 0.1));
        // Unary measurements prevent the retained system becoming progressively
        // less constrained; the true solution remains x_t = t.
        graph
            .add_factor(Prior {
                variable: next,
                measurement: time as f64,
            })
            .unwrap();
        for &(key, timestamp) in &window {
            let edge = Difference {
                x: key,
                y: next,
                measurement: (time - timestamp) as f64,
            };
            if batched {
                graph.add_factor_to(batch, DifferencePayload(edge)).unwrap();
            } else {
                graph.add_factor(edge).unwrap();
            }
        }
        window.push_back((next, time));
        if window.len() <= width {
            continue;
        }
        let (oldest, _) = window.pop_front().unwrap();
        let (report, allocations, ns) =
            measured(|| graph.marginalize(&[oldest.block_id()]).unwrap());
        assert_eq!(report.separator_dof, width);
        assert_eq!(report.prior_rows, width);
        let iteration = time - width;
        if iteration >= warmup {
            min_alloc = min_alloc.min(allocations);
            max_alloc = max_alloc.max(allocations);
            sample_ns += ns;
            if (iteration - warmup + 1).is_multiple_of(steps) {
                timings.push(sample_ns as f64 / steps as f64 / 1000.0);
                sample_ns = 0;
            }
        }
    }
    let report = graph.optimize().unwrap();
    assert!(report.final_cost < 1e-12, "cost {}", report.final_cost);
    for (key, timestamp) in window {
        assert!((graph.get(key).unwrap().0 - timestamp as f64).abs() < 1e-7);
    }
    timings.sort_by(f64::total_cmp);
    (timings[timings.len() / 2], min_alloc, max_alloc)
}

#[test]
fn warmed_window_marginalization_does_not_allocate() {
    for batched in [false, true] {
        let (_, min, max) = workload(8, batched, 3, 12);
        assert_eq!((min, max), (0, 0), "batched={batched}");
    }
}

#[test]
fn warmed_covariance_with_batches_and_history_does_not_allocate() {
    let mut graph = Solver::<States<f64>, Factors>::new();
    let mut keys = Vec::new();
    let batch = graph.add_batch(Batched);
    for i in 0..128 {
        let x = graph.add(Scalar(i as f64));
        graph
            .add_factor(Prior {
                variable: x,
                measurement: i as f64,
            })
            .unwrap();
        if let Some(&y) = keys.last() {
            graph
                .add_factor_to(
                    batch,
                    DifferencePayload(Difference {
                        x: y,
                        y: x,
                        measurement: 1.,
                    }),
                )
                .unwrap();
        }
        keys.push(x);
    }
    graph
        .marginalize(&[keys[0].block_id(), keys[1].block_id()])
        .unwrap();
    let blocks: Vec<_> = keys[3..24].iter().rev().map(|k| k.block_id()).collect();
    graph.joint_covariance(&blocks).unwrap();
    for width in [21, 3, 9, 21] {
        let (_, allocations, _) = measured(|| {
            graph.set(keys[3], Scalar(99.)).unwrap();
            let cov = graph.joint_covariance(&blocks[..width]).unwrap();
            assert_eq!(cov.ncols(), width);
            assert!(cov[(0, 0)] > 0.);
        });
        assert_eq!(allocations, 0);
    }
}

#[test]
fn changing_prior_shapes_and_empty_separators_reuse_capacity() {
    let mut graph = Solver::<States<f64>, Factors>::new();
    for cycle in 0..48 {
        let width = [12, 4, 8, 6][cycle % 4];
        let x = graph.add(Scalar(0.1));
        graph
            .add_factor(Prior {
                variable: x,
                measurement: 0.0,
            })
            .unwrap();
        let mut remaining = Vec::new();
        for i in 0..width {
            let y = graph.add(Scalar(i as f64 + 0.1));
            graph
                .add_factor(Difference {
                    x,
                    y,
                    measurement: i as f64,
                })
                .unwrap();
            graph
                .add_factor(Prior {
                    variable: y,
                    measurement: i as f64,
                })
                .unwrap();
            remaining.push(y.block_id());
        }
        let (_, allocations, _) = measured(|| graph.marginalize(&[x.block_id()]).unwrap());
        if cycle >= 12 {
            assert_eq!(allocations, 0, "replacement cycle {cycle}");
        }
        assert!(graph.optimize().unwrap().final_cost < 1e-12);
        let (report, allocations, _) = measured(|| graph.marginalize(&remaining).unwrap());
        assert_eq!(report.separator_dof, 0);
        if cycle >= 12 {
            assert_eq!(allocations, 0, "empty separator cycle {cycle}");
        }
    }
}

#[test]
fn alternating_covariance_dimensions_retain_both_capacities() {
    let mut graph = Solver::<States<f64>, Factors>::new();
    let mut keys = Vec::new();
    for _ in 0..32 {
        let key = graph.add(Scalar(0.));
        graph
            .add_factor(Prior {
                variable: key,
                measurement: 0.,
            })
            .unwrap();
        keys.push(key.block_id());
    }
    for sample in 0..8 {
        let width = if sample % 2 == 0 {
            for _ in 32..128 {
                let key = graph.add(Scalar(0.));
                graph
                    .add_factor(Prior {
                        variable: key,
                        measurement: 0.,
                    })
                    .unwrap();
                keys.push(key.block_id());
            }
            3
        } else {
            graph.marginalize(&keys[32..]).unwrap();
            keys.truncate(32);
            21
        };
        // Alternate (n,k)=(128,3) and (32,21). Growing either dimension must
        // preserve the other dimension's high-water mark, even after shrinking.
        let (_, count, _) = measured(|| {
            let cov = graph.joint_covariance(&keys[..width]).unwrap();
            assert_eq!(cov.shape(), (width, width));
            assert_eq!(cov[(0, 0)], 1.);
        });
        if sample >= 2 {
            assert_eq!(count, 0, "sample {sample}");
        }
    }
}

// Fail after QR, while capturing a pending reference, to exercise recycling on
// rollback rather than only early key/evaluator validation failures.
thread_local! { static BAD_REFERENCE: Cell<bool> = const { Cell::new(false) }; }
struct Probe(f64);
impl fagra::Variable for Probe {
    type Scalar = f64;
    type Dim = faer_ext::nalgebra::Const<1>;
    type Allocator = faer_ext::nalgebra::DefaultAllocator;
    fn identity() -> Self {
        Self(0.0)
    }
    fn compose(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }
    fn inverse(&self) -> Self {
        Self(if BAD_REFERENCE.with(Cell::get) {
            f64::NAN
        } else {
            -self.0
        })
    }
    fn exp(v: &fagra::Tangent<Self>) -> Self {
        Self(v[0])
    }
    fn log(&self) -> fagra::Tangent<Self> {
        fagra::Tangent::<Self>::from_element(self.0)
    }
    fn adjoint(&self) -> fagra::Jacobian<Self> {
        fagra::Jacobian::<Self>::identity()
    }
    fn right_jacobian(_: &fagra::Tangent<Self>) -> fagra::Jacobian<Self> {
        fagra::Jacobian::<Self>::identity()
    }
    fn right_jacobian_inverse(_: &fagra::Tangent<Self>) -> fagra::Jacobian<Self> {
        fagra::Jacobian::<Self>::identity()
    }
}
struct Constraint {
    keys: [fagra::StateKey<Probe>; 2],
    weights: [f64; 2],
    target: f64,
}
impl<S: StateStore<Probe>> Factor<S> for Constraint {
    type Scalar = f64;
    fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
        for key in self.keys {
            visit(key.block_id());
        }
    }
    fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
        let r = self.weights[0] * states.get(self.keys[0])?.0
            + self.weights[1] * states.get(self.keys[1])?.0
            - self.target;
        Ok(0.5 * r * r)
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        states: &S,
        out: &mut L,
    ) -> Result<(), EvaluationError> {
        use faer_ext::nalgebra::{SMatrix, SVector};
        let r = SVector::<f64, 1>::new(
            self.weights[0] * states.get(self.keys[0])?.0
                + self.weights[1] * states.get(self.keys[1])?.0
                - self.target,
        );
        let a = SMatrix::<f64, 1, 1>::new(self.weights[0]);
        let b = SMatrix::<f64, 1, 1>::new(self.weights[1]);
        out.residual(
            &r,
            &[
                fagra::JacobianBlock::new(self.keys[0], &a),
                fagra::JacobianBlock::new(self.keys[1], &b),
            ],
        )
    }
}
fagra::states! { ProbeStates { values: Probe } }
fagra::factors! { ProbeFactors { constraints: Constraint } }

#[test]
fn failed_pending_priors_are_recycled_without_allocating() {
    let mut graph = Solver::<ProbeStates, ProbeFactors>::new();
    for cycle in 0..24 {
        let x = graph.add(Probe(0.0));
        let y = graph.add(Probe(0.0));
        let keep = graph
            .add_factor(Constraint {
                keys: [x, y],
                weights: [1.0, 0.0],
                target: 1.0,
            })
            .unwrap();
        graph
            .add_factor(Constraint {
                keys: [x, y],
                weights: [-1.0, 1.0],
                target: 2.0,
            })
            .unwrap();
        graph
            .add_factor(Constraint {
                keys: [x, y],
                weights: [0.0, 1.0],
                target: 6.0,
            })
            .unwrap();
        BAD_REFERENCE.with(|flag| flag.set(true));
        let (result, count, _) = measured(|| graph.marginalize(&[x.block_id()]));
        BAD_REFERENCE.with(|flag| flag.set(false));
        assert!(result.is_err());
        assert_eq!(graph.factor_cost(keep).unwrap(), 0.5);
        assert_eq!(graph.get(x).unwrap().0, 0.0);
        if cycle >= 8 {
            assert_eq!(count, 0, "rollback");
        }
        let (_, count, _) = measured(|| graph.marginalize(&[x.block_id()]).unwrap());
        if cycle >= 8 {
            assert_eq!(count, 0, "retry");
        }
        assert!((graph.optimize().unwrap().final_cost - 1.5 * (cycle + 1) as f64).abs() < 1e-10);
        assert!((graph.get(y).unwrap().0 - 5.0).abs() < 1e-10);
        graph.marginalize(&[y.block_id()]).unwrap();
    }
}

#[test]
#[ignore = "release benchmark; run with --ignored --nocapture --test-threads=1"]
fn benchmark() {
    println!("mode,separator_dof,median_us,alloc_min,alloc_max");
    for batched in [false, true] {
        for width in [8, 32, 96] {
            let (us, min, max) = workload(width, batched, 9, 20);
            println!(
                "{},{width},{us:.3},{min},{max}",
                if batched { "batch" } else { "ordinary" }
            );
        }
    }
}
