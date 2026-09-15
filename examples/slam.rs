//! Advanced API walkthrough: standalone priors and independently removable reprojections.
//! Start with `examples/scalar_prior.rs` for an ordinary one-variable factor.
//!
//! Compile with `cargo check --example slam`. State Lie-group geometry is implemented;
//! factor geometry and marginalization remain placeholders, so this is not runnable yet.

use faer_ext::nalgebra::{
    Const, DefaultAllocator, SMatrix, SVector, UnitQuaternion, Vector2, Vector3,
};
use fagra::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorSelection, Jacobian, JacobianBlock,
    LinearizationSink, Solver, SolverError, StateKey, StateStore, Tangent, Variable,
};

/// SE(3) transform acting as `rotation * point + translation`.
///
/// Tangents are `[vx, vy, vz, wx, wy, wz]`: translation in the right/body frame
/// followed by rotation in radians. The exponential couples the two parts.
/// `log` uses the principal rotation branch (angle at most pi), with a branch
/// discontinuity at pi. The inverse right Jacobian is singular at nonzero
/// multiples of 2*pi rotation; callers must stay away from those singularities.
pub struct Pose {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
}

impl Variable for Pose {
    type Scalar = f64;
    type Dim = Const<6>;
    type Allocator = DefaultAllocator;

    fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        Self {
            rotation: self.rotation * other.rotation,
            translation: self.rotation * other.translation + self.translation,
        }
    }

    fn inverse(&self) -> Self {
        let rotation = self.rotation.inverse();
        Self {
            rotation,
            translation: -(rotation * self.translation),
        }
    }

    fn exp(delta: &Tangent<Self>) -> Self {
        let omega = delta.fixed_rows::<3>(3).into_owned();
        let mut generator = SMatrix::<f64, 4, 4>::zeros();
        generator
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&omega.cross_matrix());
        generator
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&delta.fixed_rows::<3>(0));
        Self {
            rotation: UnitQuaternion::from_scaled_axis(omega),
            translation: generator.exp().fixed_view::<3, 1>(0, 3).into_owned(),
        }
    }

    fn log(&self) -> Tangent<Self> {
        let mut delta = Tangent::<Self>::zeros();
        delta
            .fixed_rows_mut::<3>(3)
            .copy_from(&self.rotation.scaled_axis());
        // t = Jl(omega) * v. Its inverse is regular on the principal branch.
        let jl = Self::right_jacobian(&(-delta))
            .fixed_view::<3, 3>(0, 0)
            .into_owned();
        let v = jl
            .lu()
            .solve(&self.translation)
            .expect("principal SO(3) Jacobian is invertible");
        delta.fixed_rows_mut::<3>(0).copy_from(&v);
        delta
    }

    fn adjoint(&self) -> Jacobian<Self> {
        let rotation = self.rotation.to_rotation_matrix();
        let mut adjoint = Jacobian::<Self>::zeros();
        adjoint
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(rotation.matrix());
        adjoint
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(rotation.matrix());
        adjoint
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(self.translation.cross_matrix() * rotation.matrix()));
        adjoint
    }

    fn right_jacobian(delta: &Tangent<Self>) -> Jacobian<Self> {
        let omega = delta.fixed_rows::<3>(3).into_owned().cross_matrix();
        let velocity = delta.fixed_rows::<3>(0).into_owned().cross_matrix();
        // Jr(delta) = integral_0^1 exp(-s * ad(delta)) ds. It is the top-right
        // block of exp([ -ad(delta), I; 0, 0 ]), including at delta = 0.
        // ponytail: a 12x12 exponential favors clarity; use closed-form SE(3)
        // Jacobians if profiling shows geometry dominates frame evaluation.
        let mut generator = SMatrix::<f64, 12, 12>::zeros();
        generator.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-omega));
        generator.fixed_view_mut::<3, 3>(3, 3).copy_from(&(-omega));
        generator
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-velocity));
        generator
            .fixed_view_mut::<6, 6>(0, 6)
            .copy_from(&Jacobian::<Self>::identity());
        generator.exp().fixed_view::<6, 6>(0, 6).into_owned()
    }

    fn right_jacobian_inverse(delta: &Tangent<Self>) -> Jacobian<Self> {
        Self::right_jacobian(delta)
            .try_inverse()
            .expect("singular SE(3) right Jacobian")
    }
}

/// Additive coordinates `[fx, fy, cx, cy]`, all measured in pixels.
///
/// The group is all of R^4; its zero identity is not a usable camera calibration.
/// Projection factors must validate the calibration required by their model.
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Variable for CameraIntrinsics {
    type Scalar = f64;
    type Dim = Const<4>;
    type Allocator = DefaultAllocator;

    fn identity() -> Self {
        Self::exp(&Tangent::<Self>::zeros())
    }

    fn compose(&self, other: &Self) -> Self {
        Self::exp(&(self.log() + other.log()))
    }

    fn inverse(&self) -> Self {
        Self::exp(&(-self.log()))
    }

    fn exp(delta: &Tangent<Self>) -> Self {
        Self {
            fx: delta[0],
            fy: delta[1],
            cx: delta[2],
            cy: delta[3],
        }
    }

    fn log(&self) -> Tangent<Self> {
        SVector::<f64, 4>::new(self.fx, self.fy, self.cx, self.cy)
    }

    fn adjoint(&self) -> Jacobian<Self> {
        SMatrix::identity()
    }

    fn right_jacobian(_: &Tangent<Self>) -> Jacobian<Self> {
        SMatrix::identity()
    }

    fn right_jacobian_inverse(_: &Tangent<Self>) -> Jacobian<Self> {
        SMatrix::identity()
    }
}

/// Additive world-space coordinates `[x, y, z]`, in the pose's translation units.
pub struct Landmark(pub Vector3<f64>);

impl Variable for Landmark {
    type Scalar = f64;
    type Dim = Const<3>;
    type Allocator = DefaultAllocator;

    fn identity() -> Self {
        Self(Vector3::zeros())
    }

    fn compose(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }

    fn inverse(&self) -> Self {
        Self(-self.0)
    }

    fn exp(delta: &Tangent<Self>) -> Self {
        Self(*delta)
    }

    fn log(&self) -> Tangent<Self> {
        self.0
    }

    fn adjoint(&self) -> Jacobian<Self> {
        SMatrix::identity()
    }

    fn right_jacobian(_: &Tangent<Self>) -> Jacobian<Self> {
        SMatrix::identity()
    }

    fn right_jacobian_inverse(_: &Tangent<Self>) -> Jacobian<Self> {
        SMatrix::identity()
    }
}

pub struct PosePrior {
    pub pose: StateKey<Pose>,
    pub measurement: Pose,
}

impl<S: StateStore<Pose>> Factor<S> for PosePrior {
    type Scalar = f64;
    fn visit_variables(&self, mut visitor: impl FnMut(BlockId)) {
        visitor(self.pose.block_id());
    }

    fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
        let _pose = states.get(self.pose)?;
        todo!("Application geometry: prior cost without Jacobians")
    }

    fn linearize<L: LinearizationSink<Scalar = f64>>(
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
    type Scalar = f64;
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

    fn linearize<L: LinearizationSink<Scalar = f64>>(
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
