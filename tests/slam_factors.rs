//! Reference geometry, graph behavior, and optional dual-number factor checks.

#[allow(dead_code)]
#[path = "../examples/slam.rs"]
mod slam;

use faer_ext::nalgebra::{SMatrix, SVector, UnitQuaternion, Vector2, Vector3};
use fagra::{Solver, Tangent, Variable};
use slam::geometry::{
    interpolate, interpolate_with_jacobians, log_jacobian, project, project_with_jacobians,
};
use slam::{
    CameraIntrinsics, FrameReprojections, Landmark, Pose, PosePrior, Reprojection, SlamFactors,
    SlamStates,
};

#[test]
fn interpolation_values_and_right_jacobians() {
    let a = Pose::exp(&SVector::<f64, 6>::new(0.3, -0.5, 0.7, 0.2, -0.4, 0.1));
    let b = Pose::exp(&SVector::<f64, 6>::new(-0.4, 0.3, 0.1, -0.1, 0.3, 0.5));
    let negated = Pose {
        rotation: UnitQuaternion::new_unchecked(-b.rotation.into_inner()),
        translation: b.translation,
    };
    for b in [&a, &b, &negated] {
        for u in [0.0, 0.37, 1.0] {
            let actual = interpolate_with_jacobians(&a, b, u).unwrap();
            let expected = Pose {
                rotation: a.rotation.slerp(&b.rotation, u),
                translation: a.translation * (1.0 - u) + b.translation * u,
            };
            assert!(actual.pose.local(&expected).norm() < 1e-12);
            assert!(actual.pose.local(&interpolate(&a, b, u)).norm() < 1e-12);
            for endpoint in 0..2 {
                for column in 0..6 {
                    let mut step = Tangent::<Pose>::zeros();
                    step[column] = 1e-6;
                    let sample = |step: &Tangent<Pose>| {
                        let value = if endpoint == 0 {
                            interpolate(&a.retract(step), b, u)
                        } else {
                            interpolate(&a, &b.retract(step), u)
                        };
                        actual.pose.local(&value)
                    };
                    let derivative = (sample(&step) - sample(&(-step))) / 2e-6;
                    assert!((derivative - actual.jacobians[endpoint].column(column)).norm() < 1e-8);
                }
            }
        }
    }
    // The principal branch follows the shorter arc even past pi.
    let end = Pose {
        rotation: UnitQuaternion::from_scaled_axis(Vector3::z() * (std::f64::consts::PI + 0.1)),
        translation: Vector3::zeros(),
    };
    let halfway = interpolate(&Pose::identity(), &end, 0.5);
    assert!((halfway.rotation.scaled_axis().z + (std::f64::consts::PI - 0.1) / 2.0).abs() < 1e-12);
}

#[test]
fn projection_reference_and_right_jacobians() {
    let pose = Pose {
        rotation: UnitQuaternion::from_scaled_axis(Vector3::y() * std::f64::consts::FRAC_PI_2),
        translation: Vector3::new(2.0, -1.0, 0.5),
    };
    let camera = CameraIntrinsics::new(400.0, 420.0, 320.0, 240.0);
    let landmark = Landmark(pose.rotation * Vector3::new(1.0, 2.0, 4.0) + pose.translation);
    assert!(
        (project(&pose, &camera, &landmark).unwrap() - Vector2::new(420.0, 450.0)).norm() < 1e-10
    );
    let projection = project_with_jacobians(&pose, &camera, &landmark).unwrap();
    let mut jacobian = SMatrix::<f64, 2, 9>::zeros();
    jacobian
        .fixed_view_mut::<2, 6>(0, 0)
        .copy_from(&projection.j_pose);
    jacobian
        .fixed_view_mut::<2, 3>(0, 6)
        .copy_from(&projection.j_landmark);
    for column in 0..9 {
        let sample = |h| {
            let mut delta = SVector::<f64, 9>::zeros();
            delta[column] = h;
            project(
                &pose.retract(&delta.fixed_rows::<6>(0).into_owned()),
                &camera,
                &landmark.retract(&delta.fixed_rows::<3>(6).into_owned()),
            )
            .unwrap()
        };
        let derivative = (sample(1e-6) - sample(-1e-6)) / 2e-6;
        assert!((derivative - jacobian.column(column)).norm() < 1e-6);
    }
    assert_eq!(
        project(
            &Pose::identity(),
            &camera,
            &Landmark(Vector3::new(0.0, 0.0, 5.0))
        )
        .unwrap(),
        Vector2::new(320.0, 240.0)
    );
}

#[test]
fn scene_preconditions_and_invalid_evaluations() {
    let mut solver = Solver::<SlamStates, SlamFactors>::new();
    let a = solver.add(Pose::identity());
    let b = solver.add(Pose::identity());
    let camera = CameraIntrinsics::new(400.0, 420.0, 320.0, 240.0);
    for fx in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            std::panic::catch_unwind(|| CameraIntrinsics::new(fx, 420.0, 320.0, 240.0)).is_err()
        );
    }
    for u in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(std::panic::catch_unwind(|| FrameReprojections::new([a, b], camera, u)).is_err());
    }
    assert!(std::panic::catch_unwind(|| FrameReprojections::new([a, a], camera, 0.5)).is_err());
    for z in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let point = Landmark(Vector3::new(0.0, 0.0, z));
        assert!(project(&Pose::identity(), &camera, &point).is_err());
        assert!(project_with_jacobians(&Pose::identity(), &camera, &point).is_err());
    }
    assert!(log_jacobian(&Tangent::<Pose>::from_element(f64::NAN)).is_err());
    // Finiteness of emitted costs/residuals remains the solver's responsibility.
    let landmark = solver.add(Landmark(Vector3::new(0.0, 0.0, 5.0)));
    let frame = solver.add_batch(FrameReprojections::new([a, b], camera, 0.5));
    let invalid = solver
        .add_factor_to(
            frame,
            Reprojection {
                landmark,
                pixel: Vector2::from_element(f64::NAN),
            },
        )
        .unwrap();
    assert!(solver.factor_cost(invalid).is_err());
    assert!(solver.optimize().is_err());
}

#[test]
fn synthetic_scene_converges_and_removal_preserves_siblings() {
    let mut solver = Solver::<SlamStates, SlamFactors>::new();
    let camera = CameraIntrinsics::new(500.0, 510.0, 320.0, 240.0);
    let endpoints = [
        Pose::identity(),
        Pose::exp(&SVector::<f64, 6>::new(1.0, 0.2, 0.1, -0.03, 0.07, 0.02)),
    ];
    let trajectory = std::array::from_fn(|i| {
        solver.add(endpoints[i].retract(&SVector::<f64, 6>::new(
            0.01, -0.01, 0.02, 0.002, -0.003, 0.001,
        )))
    });
    let points = [
        Vector3::new(-1.0, -0.7, 4.0),
        Vector3::new(0.8, -0.5, 5.0),
        Vector3::new(-0.6, 0.9, 6.0),
    ];
    let landmarks = points.map(|p| solver.add(Landmark(p + Vector3::new(0.03, -0.02, 0.04))));
    let mut observations = Vec::new();
    for u in [0.0, 0.5, 1.0] {
        let pose = interpolate(&endpoints[0], &endpoints[1], u);
        let frame = solver.add_batch(FrameReprojections::new(trajectory, camera, u));
        for (&landmark, point) in landmarks.iter().zip(points) {
            observations.push(
                solver
                    .add_factor_to(
                        frame,
                        Reprojection {
                            landmark,
                            pixel: project(&pose, &camera, &Landmark(point)).unwrap(),
                        },
                    )
                    .unwrap(),
            );
        }
    }
    for (&key, pose) in trajectory.iter().zip(&endpoints) {
        solver
            .add_factor(PosePrior {
                pose: key,
                measurement: pose.retract(&Tangent::<Pose>::zeros()),
            })
            .unwrap();
    }
    let report = solver.optimize().unwrap();
    assert!(report.iterations > 0 && report.final_cost < report.initial_cost);
    assert!(report.final_cost < 1e-8);
    for (&key, pose) in trajectory.iter().zip(&endpoints) {
        assert!(pose.local(solver.get(key).unwrap()).norm() < 1e-4);
    }
    for (&landmark, point) in landmarks.iter().zip(points) {
        assert!((solver.get(landmark).unwrap().0 - point).norm() < 1e-4);
    }
    let sibling_cost = solver.factor_cost(observations[1]).unwrap();
    solver.remove_factor(observations[0]).unwrap();
    assert!(solver.factor_cost(observations[0]).is_err());
    assert_eq!(solver.factor_cost(observations[1]).unwrap(), sibling_cost);
    solver.optimize().unwrap();
}

#[cfg(feature = "test-support")]
mod properties {
    use super::*;
    use faer_ext::nalgebra::RealField;
    use fagra::testing::{TestFactor, TestFactorBatch, TestStates, proptest::prelude::*};

    fn pose<R: RealField + Copy>(coords: [f64; 6]) -> Pose<R> {
        Pose::exp(&SVector::from(coords.map(|x| R::from_f64(x).unwrap())))
    }

    #[derive(Debug)]
    struct PriorCase {
        value: [f64; 6],
        measurement: [f64; 6],
    }

    impl TestFactor for PriorCase {
        type Factor<R: RealField + Copy> = PosePrior<R>;

        fn cases() -> impl Strategy<Value = Self> {
            (
                proptest::array::uniform6(-0.5..0.5),
                proptest::array::uniform6(-0.5..0.5),
                any::<bool>(),
            )
                .prop_map(|(value, measurement, equal)| Self {
                    value,
                    measurement: if equal { value } else { measurement },
                })
        }

        fn build<R: RealField + Copy>(&self, states: &mut TestStates<R>) -> PosePrior<R> {
            PosePrior {
                pose: states.insert(pose(self.value)),
                measurement: pose(self.measurement),
            }
        }
    }

    #[derive(Debug)]
    struct FrameCase {
        a: [f64; 6],
        b: [f64; 6],
        u: f64,
        points: Vec<[f64; 3]>,
        pixel_offset: f64,
    }

    impl TestFactorBatch for FrameCase {
        type Batch<R: RealField + Copy> = FrameReprojections<R>;

        fn cases() -> impl Strategy<Value = Self> {
            (
                proptest::array::uniform6(-0.2..0.2),
                proptest::array::uniform6(-0.2..0.2),
                prop_oneof![Just(0.0), Just(1.0), 0.0..1.0],
                any::<bool>(),
                proptest::collection::vec((-1.0..1.0, -1.0..1.0, 4.0..8.0), 2..4),
                prop_oneof![Just(0.0), -2.0..2.0],
            )
                .prop_map(|(a, b, u, equal, points, pixel_offset)| Self {
                    a,
                    b: if equal { a } else { b },
                    u,
                    points: points.into_iter().map(|(x, y, z)| [x, y, z]).collect(),
                    pixel_offset,
                })
        }

        fn build<R: RealField + Copy>(
            &self,
            states: &mut TestStates<R>,
        ) -> (FrameReprojections<R>, Vec<Reprojection<R>>) {
            let a = states.insert(pose(self.a));
            let b = states.insert(pose(self.b));
            // Small image units keep f32 residual and cost-derivative tolerances
            // comparable. Full pixel-scale reference geometry is checked above.
            let [fx, fy, cx, cy] = [4.0, 4.2, 3.2, 2.4].map(|x| R::from_f64(x).unwrap());
            let camera = CameraIntrinsics::new(fx, fy, cx, cy);
            // Measurements use unperturbed fixture data, not the seeded store.
            let start = pose::<f64>(self.a);
            let end = pose::<f64>(self.b);
            let rotation = start.rotation.slerp(&end.rotation, self.u);
            let translation = start.translation * (1.0 - self.u) + end.translation * self.u;
            let observations = self
                .points
                .iter()
                .map(|&point| {
                    let local = rotation.inverse() * (Vector3::from(point) - translation);
                    let pixel = Vector2::new(
                        4.0 * local.x / local.z + 3.2 + self.pixel_offset,
                        4.2 * local.y / local.z + 2.4 - self.pixel_offset,
                    );
                    Reprojection {
                        landmark: states.insert(Landmark(Vector3::from(
                            point.map(|x| R::from_f64(x).unwrap()),
                        ))),
                        pixel: pixel.map(|x| R::from_f64(x).unwrap()),
                    }
                })
                .collect();
            (
                FrameReprojections::new([a, b], camera, R::from_f64(self.u).unwrap()),
                observations,
            )
        }

        fn config() -> proptest::test_runner::Config {
            // Each case differentiates all coordinates and singleton selections.
            proptest::test_runner::Config {
                cases: 32,
                ..Default::default()
            }
        }
    }

    fagra::factor_tests!(pose_double, PriorCase, f64);
    fagra::factor_tests!(pose_single, PriorCase, f32);
    fagra::factor_batch_tests!(frame_double, FrameCase, f64);
    fagra::factor_batch_tests!(frame_single, FrameCase, f32);
}
