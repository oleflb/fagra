//! Geometry supporting the SLAM walkthrough. All derivatives use right increments.

use faer_ext::nalgebra::{
    Const, DefaultAllocator, Quaternion, RealField, SMatrix, UnitQuaternion, Vector2, Vector3,
};
use fagra::{EvaluationError, Jacobian, Tangent, Variable};

/// SE(3) transform acting as `rotation * point + translation`.
/// Tangents are body-frame `[vx, vy, vz, wx, wy, wz]`.
/// `log` takes the principal rotation branch, discontinuous at pi; the inverse
/// right Jacobian is singular at nonzero multiples of 2*pi rotation.
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
            let a = half.sin() / angle;
            (
                half.cos(),
                a,
                c(2.0) * a * a,
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
        // Value-level logarithm, independent of the Jacobian hooks tested by AD.
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
        let (j, q) = right_jacobian_blocks(delta);
        assemble_jacobian(j, q)
    }

    fn right_jacobian_inverse(delta: &Tangent<Self>) -> Jacobian<Self> {
        log_jacobian(delta).expect("invalid or singular SE(3) right Jacobian")
    }
}

// B = (1-cos(theta))/theta², C = (theta-sin(theta))/theta³.
// The wider series interval also protects the radial derivatives used below,
// particularly for f32. At theta=0.5 the omitted B term is less than 3e-15.
// Keep these independent of the value-level exp/log for derivative checks.
fn so3_coefficients<R: RealField + Copy>(t2: R) -> (R, R) {
    let c = |x| R::from_f64(x).unwrap();
    if t2 < c(0.25) {
        (
            c(0.5)
                + t2 * (c(-1.0 / 24.0)
                    + t2 * (c(1.0 / 720.0)
                        + t2 * (c(-1.0 / 40320.0)
                            + t2 * (c(1.0 / 3628800.0) - t2 * c(1.0 / 479001600.0))))),
            c(1.0 / 6.0)
                + t2 * (c(-1.0 / 120.0)
                    + t2 * (c(1.0 / 5040.0)
                        + t2 * (c(-1.0 / 362880.0)
                            + t2 * (c(1.0 / 39916800.0) - t2 * c(1.0 / 6227020800.0))))),
        )
    } else {
        let angle = t2.sqrt();
        let a = (angle * c(0.5)).sin() / angle;
        (c(2.0) * a * a, (R::one() - angle.sin() / angle) / t2)
    }
}

fn so3_right_jacobian<R: RealField + Copy>(omega: &Vector3<R>) -> SMatrix<R, 3, 3> {
    let (b, c) = so3_coefficients(omega.norm_squared());
    let w = omega.cross_matrix();
    SMatrix::identity() - w * b + w * w * c
}

fn right_jacobian_blocks<R: RealField + Copy>(
    delta: &Tangent<Pose<R>>,
) -> (SMatrix<R, 3, 3>, SMatrix<R, 3, 3>) {
    let c = |x| R::from_f64(x).unwrap();
    let omega = delta.fixed_rows::<3>(3).into_owned();
    let velocity = delta.fixed_rows::<3>(0).into_owned();
    let t2 = omega.norm_squared();
    let (b, d) = so3_coefficients(t2);
    // Radial derivatives B'(theta)/theta and C'(theta)/theta. Polynomials
    // avoid both cancellation and sqrt(0) when evaluating with dual numbers.
    let (db, dd) = if t2 < c(0.25) {
        (
            c(-1.0 / 12.0)
                + t2 * (c(1.0 / 180.0)
                    + t2 * (c(-1.0 / 6720.0)
                        + t2 * (c(1.0 / 453600.0) - t2 * c(1.0 / 47900160.0)))),
            c(-1.0 / 60.0)
                + t2 * (c(1.0 / 1260.0)
                    + t2 * (c(-1.0 / 60480.0)
                        + t2 * (c(1.0 / 4989600.0) - t2 * c(1.0 / 622702080.0)))),
        )
    } else {
        ((R::one() - t2 * d - c(2.0) * b) / t2, (b - c(3.0) * d) / t2)
    };
    let w = omega.cross_matrix();
    let w2 = w * w;
    let j = SMatrix::<R, 3, 3>::identity() - w * b + w2 * d;
    // Qr = R^T * d(Jl(omega) * velocity)/d(omega), in [v, omega] order.
    // See GTSAM's so3::Kernel::applyLeft and DexpFunctor::tangentExpmap.
    let cross = omega.cross(&velocity);
    let derivative = -velocity.cross_matrix() * b
        + (omega * velocity.transpose() + SMatrix::identity() * omega.dot(&velocity)
            - velocity * omega.transpose() * c(2.0))
            * d
        + (cross * db + omega.cross(&cross) * dd) * omega.transpose();
    let rotation_inverse = SMatrix::<R, 3, 3>::identity() - w * (R::one() - t2 * d) + w2 * b;
    (j, rotation_inverse * derivative)
}

fn assemble_jacobian<R: RealField + Copy>(
    j: SMatrix<R, 3, 3>,
    q: SMatrix<R, 3, 3>,
) -> Jacobian<Pose<R>> {
    let mut result = Jacobian::<Pose<R>>::zeros();
    result.fixed_view_mut::<3, 3>(0, 0).copy_from(&j);
    result.fixed_view_mut::<3, 3>(3, 3).copy_from(&j);
    result.fixed_view_mut::<3, 3>(0, 3).copy_from(&q);
    result
}

/// Additive world-space coordinates, in the pose's translation units.
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

/// Fixed pinhole calibration in pixels, validated once during scene construction.
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics<R: RealField + Copy = f64> {
    fx: R,
    fy: R,
    cx: R,
    cy: R,
}

impl<R: RealField + Copy> CameraIntrinsics<R> {
    pub fn new(fx: R, fy: R, cx: R, cy: R) -> Self {
        assert!([fx, fy, cx, cy].iter().all(|x| x.is_finite()) && fx > R::zero() && fy > R::zero());
        Self { fx, fy, cx, cy }
    }
}

pub struct InterpolatedPose<R: RealField + Copy> {
    pub pose: Pose<R>,
    pub jacobians: [SMatrix<R, 6, 6>; 2],
}

pub struct Projection<R: RealField + Copy> {
    pub pixel: Vector2<R>,
    pub j_pose: SMatrix<R, 2, 6>,
    pub j_landmark: SMatrix<R, 2, 3>,
}

/// Guard numerical work before the sink can check the emitted coefficients.
pub fn log_jacobian<R: RealField + Copy>(
    delta: &Tangent<Pose<R>>,
) -> Result<Jacobian<Pose<R>>, EvaluationError> {
    if !delta.iter().all(|x| x.is_finite()) {
        return Err(EvaluationError::InvalidEvaluation);
    }
    let (j, q) = right_jacobian_blocks(delta);
    // Only invert the 3x3 diagonal block; preserve singularity detection.
    let inverse = j.try_inverse().ok_or(EvaluationError::InvalidEvaluation)?;
    let result = assemble_jacobian(inverse, -inverse * q * inverse);
    if !result.iter().all(|x| x.is_finite()) {
        return Err(EvaluationError::InvalidEvaluation);
    }
    Ok(result)
}

// Rotation-only log/exp implements dual-safe SLERP. Translation is linear in
// world coordinates. Callers supply finite endpoint poses and u in [0, 1].
fn interpolation_value<R: RealField + Copy>(
    a: &Pose<R>,
    b: &Pose<R>,
    u: R,
) -> (Pose<R>, Tangent<Pose<R>>) {
    let delta = Pose {
        rotation: a.rotation.inverse() * b.rotation,
        translation: Vector3::zeros(),
    }
    .log();
    let pose = Pose {
        rotation: a.rotation * Pose::exp(&(delta * u)).rotation,
        translation: a.translation * (R::one() - u) + b.translation * u,
    };
    (pose, delta)
}

pub fn interpolate<R: RealField + Copy>(a: &Pose<R>, b: &Pose<R>, u: R) -> Pose<R> {
    interpolation_value(a, b, u).0
}

pub fn interpolate_with_jacobians<R: RealField + Copy>(
    a: &Pose<R>,
    b: &Pose<R>,
    u: R,
) -> Result<InterpolatedPose<R>, EvaluationError> {
    let (pose, delta) = interpolation_value(a, b, u);
    let omega = delta.fixed_rows::<3>(3).into_owned();
    if !omega.iter().all(|x| x.is_finite()) {
        return Err(EvaluationError::InvalidEvaluation);
    }
    let inverse = so3_right_jacobian(&omega)
        .try_inverse()
        .ok_or(EvaluationError::InvalidEvaluation)?;
    let rotation = pose.rotation.inverse().to_rotation_matrix();
    let r0 = a.rotation.to_rotation_matrix();
    let r1 = b.rotation.to_rotation_matrix();
    // Rotation-only K = Jr(u*w) * u * Jr(w)^-1. The endpoint derivatives
    // are R(u)^T R0 - K R1^T R0 and K; translation stays independent.
    let k = so3_right_jacobian(&(omega * u)) * inverse * u;
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
    Ok(InterpolatedPose { pose, jacobians })
}

fn projection_value<R: RealField + Copy>(
    pose: &Pose<R>,
    camera: &CameraIntrinsics<R>,
    landmark: &Landmark<R>,
) -> Result<(Vector2<R>, Vector3<R>), EvaluationError> {
    let point = pose.rotation.inverse() * (landmark.0 - pose.translation);
    // Depth can become invalid during optimization even with valid calibration.
    if !point.z.is_finite() || point.z <= R::zero() {
        return Err(EvaluationError::InvalidEvaluation);
    }
    let pixel = Vector2::new(
        camera.fx * (point.x / point.z) + camera.cx,
        camera.fy * (point.y / point.z) + camera.cy,
    );
    Ok((pixel, point))
}

pub fn project<R: RealField + Copy>(
    pose: &Pose<R>,
    camera: &CameraIntrinsics<R>,
    landmark: &Landmark<R>,
) -> Result<Vector2<R>, EvaluationError> {
    Ok(projection_value(pose, camera, landmark)?.0)
}

pub fn project_with_jacobians<R: RealField + Copy>(
    pose: &Pose<R>,
    camera: &CameraIntrinsics<R>,
    landmark: &Landmark<R>,
) -> Result<Projection<R>, EvaluationError> {
    let (pixel, point) = projection_value(pose, camera, landmark)?;
    let dx = camera.fx / point.z;
    let dy = camera.fy / point.z;
    let d = SMatrix::<R, 2, 3>::new(
        dx,
        R::zero(),
        -dx * (point.x / point.z),
        R::zero(),
        dy,
        -dy * (point.y / point.z),
    );
    let mut j_pose = SMatrix::<R, 2, 6>::zeros();
    j_pose.fixed_view_mut::<2, 3>(0, 0).copy_from(&(-d));
    j_pose
        .fixed_view_mut::<2, 3>(0, 3)
        .copy_from(&(d * point.cross_matrix()));
    Ok(Projection {
        pixel,
        j_pose,
        j_landmark: d * pose.rotation.inverse().to_rotation_matrix().matrix(),
    })
}
