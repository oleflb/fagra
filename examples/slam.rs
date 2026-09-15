//! Advanced API walkthrough: standalone priors and independently removable reprojections.
//! Start with `examples/scalar_prior.rs` for an ordinary one-variable factor.
//!
//! Compile with `cargo check --example slam`. State Lie-group geometry is implemented;
//! factor geometry and marginalization remain placeholders, so this is not runnable yet.

use faer_ext::nalgebra::{
    Const, DefaultAllocator, Quaternion, RealField, SMatrix, SVector, UnitQuaternion, Vector2,
    Vector3,
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
#[derive(Debug)]
pub struct Pose<R: RealField + Copy = f64> {
    pub rotation: UnitQuaternion<R>,
    pub translation: Vector3<R>,
}

impl<R: RealField + Copy> Variable for Pose<R> {
    type Scalar = R;
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
        let c = |x| R::from_f64(x).unwrap();
        let omega = delta.fixed_rows::<3>(3).into_owned();
        let velocity = delta.fixed_rows::<3>(0).into_owned();
        let t2 = omega.norm_squared();
        // Series in squared angle preserve dual derivatives at exactly zero.
        // Returning constant identity in that branch would lose the rotation derivative.
        let (w, a, b, d) = if t2 < c(1e-2) {
            (
                R::one() + t2 * (c(-1.0 / 8.0) + t2 * (c(1.0 / 384.0) - t2 * c(1.0 / 46080.0))),
                c(0.5) + t2 * (c(-1.0 / 48.0) + t2 * (c(1.0 / 3840.0) - t2 * c(1.0 / 645120.0))),
                c(0.5) + t2 * (c(-1.0 / 24.0) + t2 * (c(1.0 / 720.0) - t2 * c(1.0 / 40320.0))),
                c(1.0 / 6.0)
                    + t2 * (c(-1.0 / 120.0) + t2 * (c(1.0 / 5040.0) - t2 * c(1.0 / 362880.0))),
            )
        } else {
            let angle = t2.sqrt();
            let half = angle * c(0.5);
            (
                half.cos(),
                half.sin() / angle,
                (R::one() - angle.cos()) / t2,
                (angle - angle.sin()) / (angle * t2),
            )
        };
        let cross = omega.cross(&velocity);
        Self {
            rotation: UnitQuaternion::new_normalize(Quaternion::from_parts(w, omega * a)),
            translation: velocity + cross * b + omega.cross(&cross) * d,
        }
    }

    fn log(&self) -> Tangent<Self> {
        let c = |x| R::from_f64(x).unwrap();
        let q = self.rotation.quaternion();
        let q = if q.w < R::zero() { -*q } else { *q };
        let vector = q.imag();
        let s2 = vector.norm_squared();
        let scale = if s2 < c(1e-4) {
            c(2.0) + s2 * (c(1.0 / 3.0) + s2 * (c(3.0 / 20.0) + s2 * c(5.0 / 56.0)))
        } else {
            let s = s2.sqrt();
            c(2.0) * s.atan2(q.w) / s
        };
        let omega = vector * scale;
        let t2 = omega.norm_squared();
        // Value-level SE(3) logarithm, independent of the Jacobian hooks tested by AD.
        let coefficient = if t2 < c(1e-2) {
            c(1.0 / 12.0)
                + t2 * (c(1.0 / 720.0) + t2 * (c(1.0 / 30240.0) + t2 * c(1.0 / 1209600.0)))
        } else {
            let half = t2.sqrt() * c(0.5);
            (R::one() - half / half.tan()) / t2
        };
        let cross = omega.cross(&self.translation);
        let velocity = self.translation - cross * c(0.5) + omega.cross(&cross) * coefficient;
        let mut delta = Tangent::<Self>::zeros();
        delta.fixed_rows_mut::<3>(3).copy_from(&omega);
        delta.fixed_rows_mut::<3>(0).copy_from(&velocity);
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
        let mut generator = SMatrix::<R, 12, 12>::zeros();
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
#[derive(Debug)]
pub struct CameraIntrinsics<R: RealField + Copy = f64> {
    pub fx: R,
    pub fy: R,
    pub cx: R,
    pub cy: R,
}

impl<R: RealField + Copy> Variable for CameraIntrinsics<R> {
    type Scalar = R;
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
        SVector::<R, 4>::new(self.fx, self.fy, self.cx, self.cy)
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
#[derive(Debug)]
pub struct Landmark<R: RealField + Copy = f64>(pub Vector3<R>);

impl<R: RealField + Copy> Variable for Landmark<R> {
    type Scalar = R;
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
