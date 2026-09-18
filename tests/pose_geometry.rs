//! Independent matrix-exponential oracle and a dependency-free geometry benchmark.

#[allow(dead_code)]
#[path = "../examples/slam/geometry.rs"]
mod geometry;

use faer_ext::nalgebra::{RealField, SMatrix, SVector, Vector3};
use fagra::{Jacobian, Tangent, Variable};
use geometry::{InterpolatedPose, Pose, interpolate, interpolate_with_jacobians, log_jacobian};

// Original implementation: Jr = integral_0^1 exp(-s * ad(delta)) ds.
fn reference_jacobian<R: RealField + Copy>(delta: &Tangent<Pose<R>>) -> Jacobian<Pose<R>> {
    let omega = delta.fixed_rows::<3>(3).into_owned().cross_matrix();
    let velocity = delta.fixed_rows::<3>(0).into_owned().cross_matrix();
    let mut generator = SMatrix::<R, 12, 12>::zeros();
    generator.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-omega));
    generator.fixed_view_mut::<3, 3>(3, 3).copy_from(&(-omega));
    generator
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-velocity));
    generator
        .fixed_view_mut::<6, 6>(0, 6)
        .copy_from(&SMatrix::identity());
    generator.exp().fixed_view::<6, 6>(0, 6).into_owned()
}

fn near<R: RealField + Copy, const M: usize, const N: usize>(
    actual: &SMatrix<R, M, N>,
    expected: &SMatrix<R, M, N>,
    tolerance: f64,
) {
    let tolerance = R::from_f64(tolerance).unwrap();
    for (a, b) in actual.iter().zip(expected.iter()) {
        assert!(
            a.is_finite() && b.is_finite() && (*a - *b).abs() <= tolerance * (R::one() + b.abs()),
            "actual={actual:?}, expected={expected:?}"
        );
    }
}

fn check_values<R: RealField + Copy>(tolerance: f64) {
    let c = |x| R::from_f64(x).unwrap();
    let axis = Vector3::new(c(2.0), c(-3.0), c(1.0)).normalize();
    for angle in [
        0.0,
        1e-9,
        0.019999,
        0.020001,
        0.099999,
        0.100001,
        0.499999,
        0.5,
        0.500001,
        0.9,
        std::f64::consts::PI - 1e-7,
        4.0,
        6.0,
    ] {
        for scale in [0.0, 1.0, 100.0] {
            let mut delta = Tangent::<Pose<R>>::zeros();
            delta
                .fixed_rows_mut::<3>(0)
                .copy_from(&(Vector3::new(c(0.7), c(0.2), c(-0.4)) * c(scale)));
            delta.fixed_rows_mut::<3>(3).copy_from(&(axis * c(angle)));
            let reference = reference_jacobian(&delta);
            near(&Pose::<R>::right_jacobian(&delta), &reference, tolerance);
            near(
                &Pose::<R>::right_jacobian_inverse(&delta),
                &reference.try_inverse().unwrap(),
                tolerance * 10.0,
            );

            // Independent homogeneous-matrix exponential checks the value path too.
            let mut hat = SMatrix::<R, 4, 4>::zeros();
            hat.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(axis * c(angle)).cross_matrix());
            hat.fixed_view_mut::<3, 1>(0, 3)
                .copy_from(&delta.fixed_rows::<3>(0));
            let expected = hat.exp();
            let pose = Pose::<R>::exp(&delta);
            near(
                pose.rotation.to_rotation_matrix().matrix(),
                &expected.fixed_view::<3, 3>(0, 0).into_owned(),
                tolerance,
            );
            near(
                &pose.translation,
                &expected.fixed_view::<3, 1>(0, 3).into_owned(),
                tolerance,
            );
        }
    }
}

#[test]
fn specialized_geometry_matches_matrix_exponentials() {
    check_values::<f64>(2e-11);
    check_values::<f32>(3e-5);
    for value in [f64::NAN, f64::INFINITY, f64::MAX] {
        assert!(log_jacobian(&Tangent::<Pose>::from_element(value)).is_err());
    }
}

#[test]
fn jacobian_dual_derivatives_match_matrix_exponential() {
    // Differentiate the Jacobian itself, including at a pure translation.
    // This catches small-angle branches that preserve values but lose derivatives.
    macro_rules! check {
        ($dual:ty, $tolerance:expr) => {
            for angle in [
                0.0,
                1e-7,
                0.099999,
                0.100001,
                0.499999,
                0.500001,
                std::f64::consts::PI - 1e-5,
            ] {
                for column in 0..6 {
                    let values = [0.7, -0.4, 0.2, angle, 0.0, 0.0];
                    let delta = SVector::<$dual, 6>::from_fn(|i, _| {
                        <$dual>::new(values[i] as _, if i == column { 1.0 } else { 0.0 })
                    });
                    // nalgebra's matrix exponential can call ceil(), unsupported
                    // by num-dual. Differentiate the real oracle with a five-point
                    // stencil instead, at the actual (possibly f32-rounded) input.
                    let sample = |step: f64, inverse: bool| {
                        let mut value = delta.map(|x| x.re as f64);
                        value[column] += step;
                        let j = reference_jacobian(&value);
                        if inverse { j.try_inverse().unwrap() } else { j }
                    };
                    for (inverse, actual) in [
                        (false, Pose::<$dual>::right_jacobian(&delta)),
                        (true, Pose::<$dual>::right_jacobian_inverse(&delta)),
                    ] {
                        let h = 1e-3;
                        let derivative = (sample(-2.0 * h, inverse) - sample(-h, inverse) * 8.0
                            + sample(h, inverse) * 8.0
                            - sample(2.0 * h, inverse))
                            / (12.0 * h);
                        near(
                            &actual.map(|x| x.re as f64),
                            &sample(0.0, inverse),
                            $tolerance,
                        );
                        near(&actual.map(|x| x.eps as f64), &derivative, $tolerance);
                    }
                }
            }
        };
    }
    check!(num_dual::Dual64, 2e-11);
    check!(num_dual::Dual32, 3e-5);
}

// Previous interpolation Jacobian path, with the same value-level interpolation.
fn reference_interpolation<R: RealField + Copy>(
    a: &Pose<R>,
    b: &Pose<R>,
    u: R,
) -> InterpolatedPose<R> {
    let delta = Pose {
        rotation: a.rotation.inverse() * b.rotation,
        translation: Vector3::zeros(),
    }
    .log();
    let pose = interpolate(a, b, u);
    let inverse = reference_jacobian(&delta).try_inverse().unwrap();
    let rotation = pose.rotation.inverse().to_rotation_matrix();
    let r0 = a.rotation.to_rotation_matrix();
    let r1 = b.rotation.to_rotation_matrix();
    let k = (reference_jacobian(&(delta * u)) * inverse * u)
        .fixed_view::<3, 3>(3, 3)
        .into_owned();
    let mut jacobians = [Jacobian::<Pose<R>>::zeros(); 2];
    jacobians[0]
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(rotation.matrix() * r0.matrix() * (R::one() - u)));
    jacobians[1]
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(rotation.matrix() * r1.matrix() * u));
    jacobians[0]
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&(rotation.matrix() * r0.matrix() - k * r1.matrix().transpose() * r0.matrix()));
    jacobians[1].fixed_view_mut::<3, 3>(3, 3).copy_from(&k);
    InterpolatedPose { pose, jacobians }
}

#[test]
fn rotation_only_interpolation_matches_full_se3_reference() {
    fn check<R: RealField + Copy>(tolerance: f64) {
        let c = |x| R::from_f64(x).unwrap();
        let a = Pose::exp(&SVector::<R, 6>::from(
            [0.3, -0.5, 0.7, 0.2, -0.4, 0.1].map(c),
        ));
        for angle in [
            0.0,
            1e-9,
            0.099999,
            0.100001,
            0.499999,
            0.500001,
            std::f64::consts::PI - 1e-7,
        ] {
            let b = a.retract(&SVector::<R, 6>::from(
                [0.7, -0.4, 0.2, angle, 0.0, 0.0].map(c),
            ));
            for u in [0.0, 0.37, 1.0] {
                let actual = interpolate_with_jacobians(&a, &b, c(u)).unwrap();
                let expected = reference_interpolation(&a, &b, c(u));
                for i in 0..2 {
                    near(&actual.jacobians[i], &expected.jacobians[i], tolerance);
                }
            }
        }
    }
    check::<f64>(2e-11);
    check::<f32>(3e-5);
}

#[test]
#[ignore = "run with --release --test pose_geometry benchmark -- --ignored --nocapture"]
fn benchmark() {
    use std::{hint::black_box, time::Instant};
    assert!(!cfg!(debug_assertions), "benchmark requires --release");
    fn median_ns<T>(mut evaluate: impl FnMut(usize) -> T) -> f64 {
        for i in 0..100 {
            black_box(evaluate(i));
        }
        let mut samples = [0.0_f64; 9];
        for sample in &mut samples {
            let start = Instant::now();
            for i in 0..2000 {
                black_box(evaluate(i));
            }
            *sample = start.elapsed().as_nanos() as f64 / 2000.0;
        }
        samples.sort_by(f64::total_cmp);
        samples[4]
    }
    fn report(label: &str, old: f64, new: f64) {
        println!(
            "{label}: old {old:.1} ns, new {new:.1} ns, {:.2}x speedup",
            old / new
        );
    }
    let deltas = [0.0, 0.01, 0.9, std::f64::consts::PI - 1e-4]
        .map(|angle| SVector::<f64, 6>::new(0.7, -0.4, 0.2, angle, 0.0, 0.0));
    report(
        "f64 right Jacobian",
        median_ns(|i| reference_jacobian(black_box(&deltas[i % 4]))),
        median_ns(|i| Pose::right_jacobian(black_box(&deltas[i % 4]))),
    );
    report(
        "f64 inverse right Jacobian",
        median_ns(|i| {
            reference_jacobian(black_box(&deltas[i % 4]))
                .try_inverse()
                .unwrap()
        }),
        median_ns(|i| Pose::right_jacobian_inverse(black_box(&deltas[i % 4]))),
    );
    let a = Pose::exp(&SVector::<f64, 6>::new(0.3, -0.5, 0.7, 0.2, -0.4, 0.1));
    let endpoints = deltas.map(|delta| a.retract(&delta));
    report(
        "f64 interpolation with Jacobians",
        median_ns(|i| {
            reference_interpolation(black_box(&a), black_box(&endpoints[i % 4]), black_box(0.37))
        }),
        median_ns(|i| {
            interpolate_with_jacobians(black_box(&a), black_box(&endpoints[i % 4]), black_box(0.37))
                .unwrap()
        }),
    );
}
