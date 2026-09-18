//! Batched reprojections: share pose interpolation across a frame's observations.
//! Run with `cargo run --example slam`; start with `scalar_prior` for ordinary factors.
//! Geometry lives in `slam/geometry.rs`; calibration is fixed, poses and landmarks vary.

#[path = "slam/geometry.rs"]
pub mod geometry;
pub use geometry::{CameraIntrinsics, Landmark, Pose};

use faer_ext::nalgebra::{RealField, SVector, UnitQuaternion, Vector2, Vector3};
use fagra::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorSelection, JacobianBlock,
    LinearizationSink, Solver, SolverError, StateKey, StateStore, Variable,
};
use geometry::{
    interpolate, interpolate_with_jacobians, log_jacobian, project, project_with_jacobians,
};

fagra::states! { pub SlamStates { poses: Pose, landmarks: Landmark } }
fagra::factors! {
    pub SlamFactors {
        priors: PosePrior,
        reprojections: Batch<FrameReprojections, Reprojection>,
    }
}

pub struct PosePrior<R: RealField + Copy = f64> {
    pub pose: StateKey<Pose<R>>,
    pub measurement: Pose<R>,
}

impl<R: RealField + Copy, S: StateStore<Pose<R>>> Factor<S> for PosePrior<R> {
    type Scalar = R;

    fn visit_variables(&self, mut visitor: impl FnMut(BlockId)) {
        visitor(self.pose.block_id());
    }

    fn cost(&self, states: &S) -> Result<R, EvaluationError> {
        let residual = self.measurement.local(states.get(self.pose)?);
        Ok(R::from_f64(0.5).unwrap() * residual.norm_squared())
    }

    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let residual = self.measurement.local(states.get(self.pose)?);
        let jacobian = log_jacobian(&residual)?;
        // Ordinary factors receive an already-scoped sink.
        sink.residual(&residual, &[JacobianBlock::new(self.pose, &jacobian)])
    }
}

/// A frame at normalized time u between two distinct camera-to-world pose states.
pub struct FrameReprojections<R: RealField + Copy = f64> {
    trajectory: [StateKey<Pose<R>>; 2],
    camera: CameraIntrinsics<R>,
    u: R,
}

impl<R: RealField + Copy> FrameReprojections<R> {
    pub fn new(trajectory: [StateKey<Pose<R>>; 2], camera: CameraIntrinsics<R>, u: R) -> Self {
        assert_ne!(trajectory[0], trajectory[1]);
        assert!(u.is_finite() && u >= R::zero() && u <= R::one());
        Self {
            trajectory,
            camera,
            u,
        }
    }
}

// Payloads need no Factor implementation: the batch supplies shared evaluation.
pub struct Reprojection<R: RealField + Copy = f64> {
    pub landmark: StateKey<Landmark<R>>,
    pub pixel: Vector2<R>,
}

impl<R: RealField + Copy, S> FactorBatch<S> for FrameReprojections<R>
where
    S: StateStore<Pose<R>> + StateStore<Landmark<R>>,
{
    type Scalar = R;
    type Factor = Reprojection<R>;

    fn visit_variables(&self, factor: &Self::Factor, mut visitor: impl FnMut(BlockId)) {
        for pose in self.trajectory {
            visitor(pose.block_id());
        }
        visitor(factor.landmark.block_id());
    }

    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
    ) -> Result<R, EvaluationError> {
        if factors.is_empty() {
            return Ok(R::zero());
        }
        let pose = interpolate(
            states.get(self.trajectory[0])?,
            states.get(self.trajectory[1])?,
            self.u,
        );
        let mut cost = R::zero();
        for (_, factor) in factors {
            let residual =
                project(&pose, &self.camera, states.get(factor.landmark)?)? - factor.pixel;
            cost += R::from_f64(0.5).unwrap() * residual.norm_squared();
        }
        Ok(cost)
    }

    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        if factors.is_empty() {
            return Ok(());
        }
        // Shared trajectory work happens once per frame, not once per landmark.
        let interp = interpolate_with_jacobians(
            states.get(self.trajectory[0])?,
            states.get(self.trajectory[1])?,
            self.u,
        )?;
        for (id, factor) in factors {
            let projection =
                project_with_jacobians(&interp.pose, &self.camera, states.get(factor.landmark)?)?;
            let residual = projection.pixel - factor.pixel; // Unit measurement covariance.
            let j0 = projection.j_pose * interp.jacobians[0];
            let j1 = projection.j_pose * interp.jacobians[1];
            let blocks = [
                JacobianBlock::new(self.trajectory[0], &j0),
                JacobianBlock::new(self.trajectory[1], &j1),
                JacobianBlock::new(factor.landmark, &projection.j_landmark),
            ];
            sink.factor(id, |out| out.residual(&residual, &blocks))?;
        }
        Ok(())
    }
}

fn main() -> Result<(), SolverError> {
    let mut solver = Solver::<SlamStates, SlamFactors>::new();
    let camera = CameraIntrinsics::new(500.0, 510.0, 320.0, 240.0);
    let endpoints = [
        Pose {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::new(-0.5, 0.0, 0.0),
        },
        Pose {
            rotation: UnitQuaternion::from_scaled_axis(Vector3::new(-0.02, 0.08, -0.03)),
            translation: Vector3::new(0.8, 0.1, 0.15),
        },
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
        Vector3::new(1.2, 0.8, 4.5),
    ];
    let landmarks = points.map(|p| solver.add(Landmark(p + Vector3::new(0.03, -0.02, 0.04))));

    let mut observations = Vec::new();
    for u in [0.0, 0.5, 1.0] {
        let pose = interpolate(&endpoints[0], &endpoints[1], u);
        let frame = solver.add_batch(FrameReprojections::new(trajectory, camera, u));
        for (landmark, point) in landmarks.iter().zip(points) {
            observations.push(solver.add_factor_to(
                frame,
                Reprojection {
                    landmark: *landmark,
                    pixel: project(&pose, &camera, &Landmark(point))?,
                },
            )?);
        }
    }
    // Endpoint priors fix the world frame and scale; multiple views constrain landmarks.
    for (pose, measurement) in trajectory.into_iter().zip(endpoints) {
        solver.add_factor(PosePrior { pose, measurement })?;
    }

    let report = solver.optimize()?;
    println!(
        "cost: {:.6} -> {:.6e}, steps: {}",
        report.initial_cost, report.final_cost, report.iterations
    );
    println!(
        "second camera translation: {:?}",
        solver.get(trajectory[1])?.translation
    );
    println!("observation cost: {}", solver.factor_cost(observations[0])?);
    solver.remove_factor(observations[0])?;
    Ok(())
}
