#![cfg(feature = "test-support")]

use faer_ext::nalgebra::{
    Const, DefaultAllocator, Quaternion, RealField, SVector, UnitQuaternion, Vector3,
};
use fagra::testing::num_dual::{Dual32, Dual64};
use fagra::testing::{
    self, TestScalar, TestVariable, Tolerance, VariableProperty,
    proptest::{prelude::*, test_runner::Config},
};
use fagra::{Jacobian, Tangent, Variable};

#[allow(dead_code)]
#[path = "../examples/scalar_prior.rs"]
mod scalar;
#[allow(dead_code)]
#[path = "../examples/slam.rs"]
mod slam;

impl<R: TestScalar> TestVariable for slam::Pose<R> {
    type Dual = slam::Pose<R::Dual>;

    fn states() -> impl Strategy<Value = Self> {
        // Construct stored transforms independently of Variable::exp/log.
        (
            proptest::array::uniform3(-2.0..2.0),
            proptest::array::uniform3(-1.2..1.2),
        )
            .prop_map(|(translation, angles)| Self {
                translation: Vector3::from(translation.map(R::from_test_value)),
                rotation: UnitQuaternion::from_euler_angles(
                    R::from_test_value(angles[0]),
                    R::from_test_value(angles[1]),
                    R::from_test_value(angles[2]),
                ),
            })
    }

    fn increments() -> impl Strategy<Value = Tangent<Self>> {
        proptest::array::uniform6(-0.8..0.8)
            .prop_map(|coordinates| SVector::from(coordinates.map(R::from_test_value)))
    }

    fn to_dual(&self) -> Self::Dual {
        let q = self.rotation.quaternion();
        slam::Pose {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                q.w.dual(0.0),
                q.i.dual(0.0),
                q.j.dual(0.0),
                q.k.dual(0.0),
            )),
            translation: self.translation.map(|x| x.dual(0.0)),
        }
    }

    fn equivalent(&self, other: &Self, tolerance: Tolerance) -> bool {
        let a = self.rotation.coords.map(R::test_value);
        let b = other.rotation.coords.map(R::test_value);
        let sign = if a.dot(&b) < 0.0 { -1.0 } else { 1.0 };
        a.iter()
            .zip(b.iter())
            .all(|(&x, &y)| tolerance.close(x, sign * y))
            && self
                .translation
                .iter()
                .zip(other.translation.iter())
                .all(|(&x, &y)| tolerance.close(x.test_value(), y.test_value()))
    }

    fn log_is_smooth(&self) -> bool {
        self.rotation.w.test_value().abs() > 1e-4
    }

    fn config() -> Config {
        Config {
            cases: 64,
            ..Default::default()
        }
    }
}

use fagra as renamed;
renamed::variable_tests!(pose_double, slam::Pose<f64>);
fagra::variable_tests!(pose_single, slam::Pose<f32>);

#[test]
fn dual_geometry_preserves_derivatives_at_identity() {
    let x = scalar::Scalar::<Dual32>::exp(&SVector::from_element(Dual32::new(0.0, 1.0)));
    assert_eq!(x.log()[0].eps, 1.0);
    for column in 0..6 {
        let mut delta = SVector::<Dual64, 6>::from_element(Dual64::new(0.0, 0.0));
        delta[column].eps = 1.0;
        let actual = slam::Pose::<Dual64>::exp(&delta).log();
        for row in 0..6 {
            assert_eq!(actual[row].re, 0.0);
            assert!((actual[row].eps - if row == column { 1.0 } else { 0.0 }).abs() < 1e-12);
        }
    }
}

// Each mutation keeps the real and dual implementation shared. In particular,
// mutation 2 preserves J * J_inverse = I; only an independent derivative detects it.
#[derive(Debug)]
struct Broken<R: RealField + Copy, const KIND: u8>(slam::Pose<R>);

impl<R: RealField + Copy, const KIND: u8> Variable for Broken<R, KIND> {
    type Scalar = R;
    type Dim = Const<6>;
    type Allocator = DefaultAllocator;
    fn identity() -> Self {
        Self(slam::Pose::identity())
    }
    fn compose(&self, other: &Self) -> Self {
        Self(self.0.compose(&other.0))
    }
    fn inverse(&self) -> Self {
        if KIND == 4 {
            Self(slam::Pose {
                rotation: self.0.rotation,
                translation: self.0.translation,
            })
        } else {
            Self(self.0.inverse())
        }
    }
    fn exp(delta: &Tangent<Self>) -> Self {
        if KIND == 3 && delta.iter().all(|&v| v == R::zero()) {
            Self::identity() // Drops a dual seed whose real part is zero.
        } else {
            Self(slam::Pose::exp(delta))
        }
    }
    fn log(&self) -> Tangent<Self> {
        self.0.log()
    }
    fn adjoint(&self) -> Jacobian<Self> {
        if KIND == 1 {
            -self.0.adjoint()
        } else {
            self.0.adjoint()
        }
    }
    fn right_jacobian(delta: &Tangent<Self>) -> Jacobian<Self> {
        let j = slam::Pose::<R>::right_jacobian(delta);
        if KIND == 2 && delta.iter().any(|&v| v != R::zero()) {
            -j
        } else {
            j
        }
    }
    fn right_jacobian_inverse(delta: &Tangent<Self>) -> Jacobian<Self> {
        let j = slam::Pose::<R>::right_jacobian_inverse(delta);
        if KIND == 2 && delta.iter().any(|&v| v != R::zero()) {
            -j
        } else {
            j
        }
    }
}

impl<const KIND: u8> TestVariable for Broken<f64, KIND> {
    type Dual = Broken<Dual64, KIND>;
    fn states() -> impl Strategy<Value = Self> {
        slam::Pose::<f64>::states().prop_map(Self)
    }
    fn increments() -> impl Strategy<Value = Tangent<Self>> {
        slam::Pose::<f64>::increments()
    }
    fn to_dual(&self) -> Self::Dual {
        Broken(self.0.to_dual())
    }
    fn equivalent(&self, other: &Self, t: Tolerance) -> bool {
        self.0.equivalent(&other.0, t)
    }
    fn log_is_smooth(&self) -> bool {
        if KIND == 5 {
            self.0.equivalent(
                &slam::Pose::identity(),
                Tolerance {
                    absolute: 0.0,
                    relative: 0.0,
                },
            )
        } else {
            self.0.log_is_smooth()
        }
    }
    fn config() -> Config {
        Config {
            cases: 8,
            max_global_rejects: 8,
            max_shrink_iters: 32,
            failure_persistence: None, // Expected failures must not write regressions.
            ..Default::default()
        }
    }
}

#[test]
fn incorrect_implementations_are_rejected() {
    fn rejected<T: TestVariable>(property: VariableProperty, expected: &str) {
        let failure = std::panic::catch_unwind(|| testing::check_variable::<T>(property))
            .expect_err("broken variable passed");
        let message = failure
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| failure.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(message.contains(expected), "unexpected failure: {message}");
    }
    rejected::<Broken<f64, 1>>(VariableProperty::Jacobians, "adjoint derivative");
    rejected::<Broken<f64, 2>>(VariableProperty::Jacobians, "Exp derivative");
    rejected::<Broken<f64, 3>>(VariableProperty::Jacobians, "Exp derivative");
    rejected::<Broken<f64, 4>>(VariableProperty::GroupLaws, "left inverse");
    // A domain predicate cannot hide bad group laws or pass by rejecting every random case.
    testing::check_variable::<Broken<f64, 5>>(VariableProperty::GroupLaws);
    rejected::<Broken<f64, 5>>(VariableProperty::ExpLog, "Too many global rejects");
}

#[test]
fn tolerance_rejects_nonfinite_coefficients() {
    let tolerance = Tolerance::for_scalar::<f64>();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!tolerance.close(value, value));
        assert!(!tolerance.close(value, 0.0));
    }
}
