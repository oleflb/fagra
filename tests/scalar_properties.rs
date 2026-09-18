//! Property tests for the scalar example, registered once rather than in every
//! integration target that imports its model types.
#![cfg(feature = "test-support")]

#[allow(dead_code)]
#[path = "../examples/scalar_prior.rs"]
mod scalar;
use scalar::Prior;
pub use scalar::Scalar;

use faer_ext::nalgebra::{RealField, SVector};
use fagra::{
    Tangent,
    testing::{TestFactor, TestScalar, TestStates, TestVariable, Tolerance, proptest::prelude::*},
};

impl<R: TestScalar> TestVariable for Scalar<R> {
    type Dual = Scalar<R::Dual>;

    fn states() -> impl Strategy<Value = Self> {
        (-10.0..10.0).prop_map(|x| Self(R::from_test_value(x)))
    }

    fn increments() -> impl Strategy<Value = Tangent<Self>> {
        (-1.0..1.0).prop_map(|x| SVector::from_element(R::from_test_value(x)))
    }

    fn to_dual(&self) -> Self::Dual {
        Scalar(self.0.dual(0.0))
    }

    fn equivalent(&self, other: &Self, tolerance: Tolerance) -> bool {
        tolerance.close(self.0.test_value(), other.0.test_value())
    }
}

#[derive(Debug)]
struct PriorCase {
    value: f64,
    measurement: f64,
}

impl TestFactor for PriorCase {
    type Factor<R: RealField + Copy> = Prior<R>;

    fn cases() -> impl Strategy<Value = Self> {
        prop_oneof![
            (-10.0..10.0, -10.0..10.0),
            (-10.0..10.0).prop_map(|x| (x, x)),
        ]
        .prop_map(|(value, measurement)| Self { value, measurement })
    }

    fn build<R: RealField + Copy>(&self, states: &mut TestStates<R>) -> Prior<R> {
        Prior {
            variable: states.insert(Scalar(R::from_f64(self.value).unwrap())),
            measurement: R::from_f64(self.measurement).unwrap(),
        }
    }
}

fagra::variable_tests!(variable_double, Scalar<f64>);
fagra::variable_tests!(variable_single, Scalar<f32>);
fagra::factor_tests!(prior_double, PriorCase, f64);
fagra::factor_tests!(prior_single, PriorCase, f32);
