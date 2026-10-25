use fagra::{
    CovarianceOptions, DenseSchur, LevenbergMarquardt, OptimizeOptions, Problem, SolverError,
};

#[allow(dead_code)]
#[path = "../examples/marginalization.rs"]
mod example;
use example::{Difference, Prior, Scalar, States};

fagra::factors! { Factors { priors: Prior<f64>, edges: Difference<f64> } }

// Same thread-local allocator instrumentation as marginalization_allocations.rs.
struct Counter;
thread_local! { static ALLOCATIONS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) }; }
fn record_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(n) = count.get() {
            count.set(Some(n + 1));
        }
    });
}
unsafe impl std::alloc::GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        record_allocation();
        unsafe { std::alloc::GlobalAlloc::alloc(&std::alloc::System, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        record_allocation();
        unsafe { std::alloc::GlobalAlloc::alloc_zeroed(&std::alloc::System, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        record_allocation();
        unsafe { std::alloc::GlobalAlloc::realloc(&std::alloc::System, ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::GlobalAlloc::dealloc(&std::alloc::System, ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;

fn workload<B: fagra::__private::CovarianceBackend<Scalar = f64>>(
    backend: B,
    retained: usize,
    landmarks: usize,
    samples: usize,
) -> (u128, u128, usize) {
    let mut graph = Problem::<States<f64>, Factors>::new();
    let keys: Vec<_> = (0..retained + landmarks)
        .map(|_| graph.add(Scalar(0.)))
        .collect();
    for (i, &key) in keys.iter().enumerate() {
        graph
            .add_factor(Prior {
                variable: key,
                measurement: i as f64 * 0.01,
            })
            .unwrap();
        if i > 0 && i < retained {
            graph
                .add_factor(Difference {
                    x: keys[i - 1],
                    y: key,
                    measurement: 0.01,
                })
                .unwrap();
        }
        if i >= retained {
            for offset in 0..3.min(retained) {
                graph
                    .add_factor(Difference {
                        x: keys[(i + offset) % retained],
                        y: key,
                        measurement: 0.2,
                    })
                    .unwrap();
            }
        }
    }
    let selected = [keys[retained - 1].block_id(), keys[0].block_id()];
    let mut method = LevenbergMarquardt::new(backend);
    let mut durations = Vec::with_capacity(samples);
    let mut max_allocations = 0;
    for step in 0..samples + 5 {
        // Force real accepted steps, not a converged zero-gradient timing probe.
        graph.set(keys[0], Scalar(-1.)).unwrap();
        ALLOCATIONS.with(|count| count.set(Some(0)));
        let start = std::time::Instant::now();
        let result = method.solve_batch_with_covariance(
            &mut graph,
            &OptimizeOptions::default(),
            &selected,
            &CovarianceOptions::default(),
        );
        let elapsed = start.elapsed().as_nanos();
        let allocations = ALLOCATIONS.with(|count| count.replace(None).unwrap());
        let (report, covariance) = result.unwrap();
        assert!(report.iterations > 0);
        assert!(covariance[(0, 0)] > 0.);
        if step >= 5 {
            durations.push(elapsed);
            max_allocations = max_allocations.max(allocations);
        }
    }
    durations.sort_unstable();
    (
        durations[samples / 2],
        durations[(samples * 95).div_ceil(100) - 1],
        max_allocations,
    )
}

#[test]
fn warmed_solves_with_retained_covariance_allocate_nothing() {
    let (_, _, allocations) = workload(DenseSchur::new(4, [1; 16]), 4, 16, 3);
    assert_eq!(allocations, 0);
}

#[test]
#[ignore = "release timing benchmark; run alone with --ignored --nocapture"]
fn benchmark_solve_and_retained_covariance() {
    for (name, (median, p95, allocations)) in [
        (
            "dense",
            workload(fagra::DenseNormalCholesky::default(), 48, 192, 31),
        ),
        (
            "dense-schur",
            workload(DenseSchur::new(48, [1; 192]), 48, 192, 31),
        ),
    ] {
        eprintln!(
            "{name}: retained=48 landmarks=192 median_us={:.3} p95_us={:.3} max_allocations={allocations}",
            median as f64 / 1000.,
            p95 as f64 / 1000.
        );
    }
}

#[test]
fn retained_covariance_preserves_order_cross_terms_and_marginal_history() {
    let mut graph = Problem::<States<f64>, Factors>::new();
    let keys = std::array::from_fn::<_, 4, _>(|_| graph.add(Scalar(0.)));
    for (i, &key) in keys.iter().enumerate() {
        graph
            .add_factor(Prior {
                variable: key,
                measurement: i as f64,
            })
            .unwrap();
    }
    for (a, b) in [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)] {
        graph
            .add_factor(Difference {
                x: keys[a],
                y: keys[b],
                measurement: 0.3,
            })
            .unwrap();
    }
    let selected = [keys[1].block_id(), keys[0].block_id()];
    let expected = graph.joint_covariance(&selected).unwrap().to_owned();
    let options = OptimizeOptions::default();
    let mut method = LevenbergMarquardt::new(DenseSchur::new(2, [1, 1]));
    let (_, covariance) = method
        .solve_batch_with_covariance(
            &mut graph,
            &options,
            &selected,
            &CovarianceOptions::default(),
        )
        .unwrap();
    for col in 0..2 {
        for row in 0..2 {
            assert!((covariance[(row, col)] - expected[(row, col)]).abs() < 1e-12);
        }
    }
    // Retiring landmarks first leaves an arbitrary retained-only marginal prior.
    graph
        .marginalize(&[keys[2].block_id(), keys[3].block_id()])
        .unwrap();
    let mut method = LevenbergMarquardt::new(DenseSchur::new(2, []));
    let (_, covariance) = method
        .solve_batch_with_covariance(
            &mut graph,
            &options,
            &selected,
            &CovarianceOptions::default(),
        )
        .unwrap();
    for col in 0..2 {
        for row in 0..2 {
            assert!((covariance[(row, col)] - expected[(row, col)]).abs() < 1e-12);
        }
    }
}

#[test]
fn covariance_rejects_landmark_selection_and_unobservable_information() {
    for prior in [false, true] {
        let mut graph = Problem::<States<f64>, Factors>::new();
        let retained = graph.add(Scalar(0.));
        let landmark = graph.add(Scalar(0.));
        graph
            .add_factor(Difference {
                x: retained,
                y: landmark,
                measurement: 0.,
            })
            .unwrap();
        if prior {
            graph
                .add_factor(Prior {
                    variable: retained,
                    measurement: 0.,
                })
                .unwrap();
        }
        let mut method = LevenbergMarquardt::new(DenseSchur::new(1, [1]));
        let result = method.solve_batch_with_covariance(
            &mut graph,
            &OptimizeOptions::default(),
            &[retained.block_id()],
            &CovarianceOptions::default(),
        );
        if prior {
            assert!(result.is_ok());
        } else {
            assert!(matches!(result, Err(SolverError::SingularInformation)));
        }
        let result = method.solve_batch_with_covariance(
            &mut graph,
            &OptimizeOptions::default(),
            &[landmark.block_id()],
            &CovarianceOptions::default(),
        );
        assert!(result.is_err());
    }
}
