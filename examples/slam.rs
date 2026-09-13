//! Advanced API walkthrough: standalone priors and independently removable reprojections.
//! Start with `examples/scalar_prior.rs` for an ordinary one-variable factor.
//!
//! Compile with `cargo check --example slam`. Optimization, marginalization, and
//! application geometry below are placeholders, so this example is not runnable yet.

use faer_ext::nalgebra::{SMatrix, SVector, UnitQuaternion, Vector2, Vector3};
use fagra::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorSelection, JacobianBlock,
    LinearizationSink, Solver, SolverError, StateKey, StateStore, Variable,
};

pub struct Pose {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
}

impl Variable for Pose {
    type Tangent = SVector<f64, 6>;
    const DOF: usize = 6;

    fn tangent_from_slice(delta: &[f64]) -> Self::Tangent {
        SVector::from_column_slice(delta)
    }

    fn retract(&self, _delta: &Self::Tangent) -> Self {
        todo!("Application geometry: choose a consistent pose perturbation convention")
    }
}

pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Variable for CameraIntrinsics {
    type Tangent = SVector<f64, 4>;
    const DOF: usize = 4;

    fn tangent_from_slice(delta: &[f64]) -> Self::Tangent {
        SVector::from_column_slice(delta)
    }

    fn retract(&self, delta: &Self::Tangent) -> Self {
        Self {
            fx: self.fx + delta[0],
            fy: self.fy + delta[1],
            cx: self.cx + delta[2],
            cy: self.cy + delta[3],
        }
    }
}

pub struct Landmark(pub Vector3<f64>);

impl Variable for Landmark {
    type Tangent = Vector3<f64>;
    const DOF: usize = 3;

    fn tangent_from_slice(delta: &[f64]) -> Self::Tangent {
        Vector3::from_column_slice(delta)
    }

    fn retract(&self, delta: &Self::Tangent) -> Self {
        Self(self.0 + delta)
    }
}

pub struct PosePrior {
    pub pose: StateKey<Pose>,
    pub measurement: Pose,
}

impl<S: StateStore<Pose>> Factor<S> for PosePrior {
    fn visit_variables(&self, mut visitor: impl FnMut(BlockId)) {
        visitor(self.pose.block_id());
    }

    fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
        let _pose = states.get(self.pose)?;
        todo!("Application geometry: prior cost without Jacobians")
    }

    fn linearize<L: LinearizationSink>(
        &self,
        states: &S,
        _sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let _pose = states.get(self.pose)?;
        // Emit directly: the solver has already established this factor's scope.
        todo!("Application geometry: prior residual and Jacobian")
    }
}

pub struct FrameReprojections {
    pub trajectory: [StateKey<Pose>; 4],
    pub camera: StateKey<CameraIntrinsics>,
}

// No Factor implementation: the batch supplies shared inputs and evaluation.
pub struct Reprojection {
    pub landmark: StateKey<Landmark>,
    pub pixel: Vector2<f64>,
}

impl<S> FactorBatch<S> for FrameReprojections
where
    S: StateStore<Pose> + StateStore<CameraIntrinsics> + StateStore<Landmark>,
{
    type Factor = Reprojection;

    fn visit_variables(&self, factor: &Reprojection, mut visitor: impl FnMut(BlockId)) {
        for pose in self.trajectory {
            visitor(pose.block_id());
        }
        visitor(self.camera.block_id());
        visitor(factor.landmark.block_id());
    }

    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, Reprojection>,
    ) -> Result<f64, EvaluationError> {
        if factors.is_empty() {
            return Ok(0.0);
        }

        let pose = interpolate(states, self.trajectory)?;
        let camera = states.get(self.camera)?;
        let mut cost = 0.0;
        for (_, factor) in factors {
            let landmark = states.get(factor.landmark)?;
            let residual = project(&pose, camera, landmark)? - factor.pixel;
            cost += 0.5 * residual.norm_squared();
        }
        Ok(cost)
    }

    fn linearize<L: LinearizationSink>(
        &self,
        states: &S,
        factors: FactorSelection<'_, Reprojection>,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        if factors.is_empty() {
            return Ok(());
        }

        // Shared trajectory work happens once, not once per landmark.
        let interp = interpolate_with_jacobians(states, self.trajectory)?;
        let camera = states.get(self.camera)?;

        for (id, factor) in factors {
            let landmark = states.get(factor.landmark)?;
            let projection = project_with_jacobians(&interp.pose, camera, landmark)?;
            // Unit measurement covariance in this example.
            let residual = projection.pixel - factor.pixel;
            let j0 = projection.j_pose * interp.jacobians[0];
            let j1 = projection.j_pose * interp.jacobians[1];
            let j2 = projection.j_pose * interp.jacobians[2];
            let j3 = projection.j_pose * interp.jacobians[3];

            let blocks = [
                JacobianBlock::new(self.trajectory[0], &j0),
                JacobianBlock::new(self.trajectory[1], &j1),
                JacobianBlock::new(self.trajectory[2], &j2),
                JacobianBlock::new(self.trajectory[3], &j3),
                JacobianBlock::new(self.camera, &projection.j_camera),
                JacobianBlock::new(factor.landmark, &projection.j_landmark),
            ];

            sink.factor(id, |out| out.residual(&residual, &blocks))?;
        }
        Ok(())
    }
}

struct InterpolatedPose {
    pose: Pose,
    jacobians: [SMatrix<f64, 6, 6>; 4],
}

struct Projection {
    pixel: Vector2<f64>,
    j_pose: SMatrix<f64, 2, 6>,
    j_camera: SMatrix<f64, 2, 4>,
    j_landmark: SMatrix<f64, 2, 3>,
}

fn interpolate<S: StateStore<Pose>>(
    _states: &S,
    _trajectory: [StateKey<Pose>; 4],
) -> Result<Pose, EvaluationError> {
    todo!("Application geometry: value-only trajectory interpolation")
}

fn interpolate_with_jacobians<S: StateStore<Pose>>(
    _states: &S,
    _trajectory: [StateKey<Pose>; 4],
) -> Result<InterpolatedPose, EvaluationError> {
    todo!("Application geometry: trajectory interpolation and Jacobians")
}

fn project(
    _pose: &Pose,
    _camera: &CameraIntrinsics,
    _landmark: &Landmark,
) -> Result<Vector2<f64>, EvaluationError> {
    todo!("Application geometry: value-only projection")
}

fn project_with_jacobians(
    _pose: &Pose,
    _camera: &CameraIntrinsics,
    _landmark: &Landmark,
) -> Result<Projection, EvaluationError> {
    todo!("Application geometry: projection and Jacobians")
}

fagra::states! {
    /// State families used by this application.
    pub SlamStates {
        poses: Pose,
        cameras: CameraIntrinsics,
        landmarks: Landmark,
    }
}

fagra::factors! {
    /// Ordinary priors and batches of reprojection factors.
    pub SlamFactors {
        priors: PosePrior,
        reprojections: Batch<FrameReprojections, Reprojection>,
    }
}

type SlamSolver = Solver<SlamStates, SlamFactors>;

fn main() -> Result<(), SolverError> {
    let mut solver = SlamSolver::new();
    let camera = solver.add(CameraIntrinsics {
        fx: 500.0,
        fy: 500.0,
        cx: 320.0,
        cy: 240.0,
    });
    let trajectory = std::array::from_fn(|_| {
        solver.add(Pose {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        })
    });
    let landmark = solver.add(Landmark(Vector3::new(0.0, 0.0, 5.0)));

    let _prior = solver.add_factor(PosePrior {
        pose: trajectory[0],
        measurement: Pose {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        },
    })?;

    let frame = solver.add_batch(FrameReprojections { trajectory, camera });
    let observation = solver.add_factor_to(
        frame,
        Reprojection {
            landmark,
            pixel: Vector2::new(320.0, 240.0),
        },
    )?;

    solver.optimize()?;
    let _estimate: &Pose = solver.get(trajectory[2])?;
    let _cost = solver.factor_cost(observation)?;
    solver.remove_factor(observation)?;
    solver.marginalize(trajectory[0])?;
    Ok(())
}
