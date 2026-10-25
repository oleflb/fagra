use super::*;
use crate::covariance::CovarianceBackend;

#[path = "../../../tests/support/allocations.rs"]
mod allocations;

const CASES: [(usize, usize); 4] = [(24, 12), (48, 48), (72, 48), (96, 96)];

struct Emission<R: Real> {
    residual: DVector<R>,
    blocks: Vec<(usize, DMatrix<R>)>,
}

fn emissions<R: Real>(retained: usize, widths: &[usize]) -> Vec<Emission<R>> {
    let c = R::from_f64_impl;
    let mut emissions = Vec::new();
    let mut offset = 0;
    for width in std::iter::once(retained).chain(widths.iter().copied()) {
        let prior = DMatrix::identity(width, width) * c(2.);
        let residual = DVector::from_fn(width, |i, _| c((offset + i) as f64 * 0.01 - 0.5));
        emissions.push(Emission {
            residual,
            blocks: vec![(offset, prior)],
        });
        if offset > 0 {
            // Dense retained coupling and a complete 3/4/6-wide landmark per emission.
            let left = DMatrix::from_fn(width + 2, retained, |i, j| {
                c(((i * 7 + j * 11 + offset) % 23) as f64 / 23. - 0.5)
            });
            let right = DMatrix::from_fn(width + 2, width, |i, j| {
                c(if i == j {
                    1.5
                } else {
                    (i + j + 1) as f64 * 0.03
                })
            });
            let residual = DVector::from_fn(width + 2, |i, _| c(i as f64 * 0.1 - 0.3));
            emissions.push(Emission {
                residual,
                blocks: vec![(0, left), (offset, right)],
            });
        }
        offset += width;
    }
    emissions
}

fn linearize<B: LeastSquaresBackend>(
    backend: &mut B,
    emissions: &[Emission<B::Scalar>],
) -> Result<(), EvaluationError> {
    backend.clear();
    for emission in emissions {
        backend.accumulate(
            emission.residual.column(0),
            emission.blocks.iter().map(|(col, j)| (*col, j.as_view())),
        )?;
    }
    Ok(())
}

fn mixed_system<R: Real>(
    retained: usize,
    widths: &[usize],
    emissions: &[Emission<R>],
) -> (DenseSchur<R>, DenseNormalCholesky<R>) {
    let c = R::from_f64_impl;
    let dimension = retained + widths.iter().sum::<usize>();
    let mut schur = DenseSchur::new(retained, widths.iter().copied());
    let mut dense = DenseNormalCholesky::default();
    schur.prepare(dimension).unwrap();
    dense.prepare(dimension).unwrap();
    linearize(&mut schur, emissions).unwrap();
    linearize(&mut dense, emissions).unwrap();
    schur.prepare_damping(c(1e-6)).unwrap();
    dense.prepare_damping(c(1e-6)).unwrap();
    (schur, dense)
}

fn selected_covariance<R: Real>(retained: usize, dimension: usize) -> SelectedCovariance<R> {
    use crate::{
        Problem,
        optimization::{Block, Layout},
        variable::test_support::Vector,
    };
    crate::states! { States { poses: Vector<6> } }
    crate::factors! { Factors {} }
    let mut graph = Problem::<States, Factors>::new();
    let pose = graph.add(Vector::<6>::identity());
    let mut layout = Layout {
        dimension,
        ..Default::default()
    };
    layout.block_index.insert(pose.block_id(), 0);
    layout.blocks.push(Block {
        id: pose.block_id(),
        offset: retained - 6,
        width: 6,
    });
    let mut selected = SelectedCovariance::default();
    selected.prepare(&layout, &[pose.block_id()]).unwrap();
    selected
}

#[test]
fn mixed_size_steps_and_six_column_covariance_match_dense() {
    fn check<R: Real + Into<f64>>(tolerance: f64) {
        let c = R::from_f64_impl;
        for (retained, landmarks) in CASES {
            let widths: Vec<_> = [3, 4, 6].into_iter().cycle().take(landmarks).collect();
            let dimension = retained + widths.iter().sum::<usize>();
            let emissions = emissions::<R>(retained, &widths);
            let (mut schur, mut dense) = mixed_system(retained, &widths, &emissions);
            for lambda in [0., 0.001, 10., 0.] {
                let actual = schur.solve_damped(c(lambda)).unwrap();
                let expected = dense.solve_damped(c(lambda)).unwrap();
                for (&a, &b) in actual.delta.iter().zip(expected.delta) {
                    let (a, b) = (Into::<f64>::into(a), Into::<f64>::into(b));
                    assert!((a - b).abs() <= tolerance * (1. + b.abs()));
                }
                let a = Into::<f64>::into(actual.predicted_reduction);
                let b = Into::<f64>::into(expected.predicted_reduction);
                assert!((a - b).abs() <= tolerance * (1. + b.abs()));
            }
            let mut actual = selected_covariance(retained, dimension);
            let mut expected = selected_covariance(retained, dimension);
            schur
                .covariance(&mut DenseNormalCholesky::default(), &mut actual)
                .unwrap();
            expected.compute(&mut dense).unwrap();
            for col in 0..6 {
                for row in 0..6 {
                    let a = Into::<f64>::into(actual.matrix()[(row, col)]);
                    let b = Into::<f64>::into(expected.matrix()[(row, col)]);
                    assert!((a - b).abs() <= tolerance * (1. + b.abs()));
                }
            }
        }
    }
    check::<f64>(1e-10);
    check::<f32>(2e-4);
}

// Re-accumulate frozen Jacobians: dense covariance consumes its normal matrix.
fn repeated_solves<B: CovarianceBackend<Scalar = f64>>(
    backend: &mut B,
    retained: usize,
    dimension: usize,
    emissions: &[Emission<f64>],
    samples: usize,
) -> (u128, u128, usize, usize) {
    let mut selected = selected_covariance(retained, dimension);
    let mut unused = DenseNormalCholesky::default();
    let mut timings = Vec::with_capacity(samples);
    let mut cold_allocations = 0;
    let mut warm_allocations = 0;
    for step in 0..samples + 5 {
        let (result, count, elapsed) = allocations::measured(|| {
            linearize(backend, emissions)?;
            backend.prepare_damping(1e-6)?;
            std::hint::black_box(backend.solve_damped(0.001)?.delta);
            backend.covariance(&mut unused, &mut selected)
        });
        result.unwrap();
        assert!(selected.matrix()[(0, 0)] > 0.);
        if step == 0 {
            cold_allocations = count;
        }
        if step >= 5 {
            timings.push(elapsed);
            warm_allocations = warm_allocations.max(count);
        }
    }
    timings.sort_unstable();
    (
        timings[samples / 2],
        timings[(samples * 95).div_ceil(100) - 1],
        cold_allocations,
        warm_allocations,
    )
}

#[cfg(target_arch = "aarch64")]
#[test]
fn warmed_mixed_size_solve_and_pose_covariance_allocate_nothing() {
    let widths: Vec<_> = [3, 4, 6].into_iter().cycle().take(48).collect();
    let emissions = emissions::<f64>(48, &widths);
    let (mut schur, _) = mixed_system(48, &widths, &emissions);
    let (_, _, cold, warm) = repeated_solves(
        &mut schur,
        48,
        48 + widths.iter().sum::<usize>(),
        &emissions,
        3,
    );
    eprintln!(
        "mixed retained=48 eliminated=208 selected=6 first_workspace_allocations={cold} max_warm_allocations={warm}"
    );
    assert_eq!(warm, 0);
}

#[test]
#[ignore = "release size benchmark; run alone with --ignored --nocapture"]
fn benchmark_mixed_sizes() {
    for (retained, landmarks) in CASES {
        let widths: Vec<_> = [3, 4, 6].into_iter().cycle().take(landmarks).collect();
        let eliminated = widths.iter().sum::<usize>();
        let emissions = emissions::<f64>(retained, &widths);
        let (mut schur, mut dense) = mixed_system(retained, &widths, &emissions);
        for (name, (p50, p95, cold, warm)) in [
            (
                "dense",
                repeated_solves(&mut dense, retained, retained + eliminated, &emissions, 31),
            ),
            (
                "dense-schur",
                repeated_solves(&mut schur, retained, retained + eliminated, &emissions, 31),
            ),
        ] {
            eprintln!(
                "mixed backend={name} retained={retained} landmarks={landmarks} eliminated={eliminated} widths=3/4/6 selected=6 p50_us={:.3} p95_us={:.3} first_workspace_allocations={cold} max_warm_allocations={warm}",
                p50 as f64 / 1000.,
                p95 as f64 / 1000.
            );
        }
    }
}
