//! Property tests for user-defined [`Variable`] implementations.
//!
//! Enable `test-support` on a **dev-dependency**, implement [`TestVariable`] inside
//! your crate's `#[cfg(test)]` module, then invoke [`crate::variable_tests!`]. The
//! macro generates ordinary, separately named tests. Dependencies are not built
//! with their consumer's `cfg(test)`, so the feature exposes these helpers while
//! the generated tests are guarded by the consumer's `cfg(test)`.
//!
//! Checks establish consistency, not that a group represents your intended
//! physical model. Keep independent reference cases for that. State strategies
//! should cover the representation directly, rather than only calling `exp`.
//! Increments must lie in the selected local log chart. Exact identity and tiny
//! increments are also checked explicitly; AD must work at these points.
//!
//! Dual evaluations must execute the same scalar-generic geometry as real
//! evaluations. Do not implement a separate test-only geometry model or use
//! analytical Jacobian hooks in value-level operations to manufacture the AD
//! reference. In particular, a constant zero-angle branch can discard derivatives.

use std::{fmt::Debug, panic::Location};

use faer_ext::nalgebra::{DMatrix, DVector, DimName, RealField};
use proptest::{
    prelude::*,
    test_runner::{TestCaseError, TestCaseResult, TestRunner},
};

use crate::{Jacobian, Tangent, Variable};

/// The dual-number implementation used by the test runners (nalgebra 0.34 compatible).
pub use num_dual;
/// Strategies, shrinking, failure persistence, and runner configuration.
pub use proptest;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Maps the supported test precisions to matching dual numbers.
///
/// Implemented by fagra for `f32`/`Dual32` and `f64`/`Dual64`; users do not
/// implement this trait. Keeping this bound on the associated scalar avoids
/// introducing backend arithmetic or allocator bounds into test implementations.
pub trait TestScalar: RealField + Copy + sealed::Sealed {
    /// Dual number with the same underlying floating-point precision.
    type Dual: RealField + Copy;
    /// Machine epsilon, expressed as f64 for tolerance configuration.
    const EPSILON: f64;
    /// Convert a test coordinate to this precision.
    fn from_test_value(value: f64) -> Self;
    /// Convert a coefficient to f64 for diagnostics and comparisons.
    fn test_value(self) -> f64;
    /// Attach a derivative seed (the runner uses zero or one).
    fn dual(self, derivative: f64) -> Self::Dual;
    /// Extract value and derivative independently; comparisons must inspect both.
    fn parts(value: Self::Dual) -> (f64, f64);
}

macro_rules! test_scalar {
    ($real:ty, $dual:ty) => {
        impl TestScalar for $real {
            type Dual = $dual;
            const EPSILON: f64 = <$real>::EPSILON as f64;
            fn from_test_value(value: f64) -> Self {
                value as Self
            }
            fn test_value(self) -> f64 {
                self as f64
            }
            fn dual(self, derivative: f64) -> Self::Dual {
                <$dual>::new(self, derivative as Self)
            }
            fn parts(value: Self::Dual) -> (f64, f64) {
                (value.re as f64, value.eps as f64)
            }
        }
    };
}
test_scalar!(f32, num_dual::Dual32);
test_scalar!(f64, num_dual::Dual64);

/// Absolute and relative tolerances for finite coefficients.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    /// Absolute error allowed near zero.
    pub absolute: f64,
    /// Relative error allowed at larger magnitudes.
    pub relative: f64,
}

impl Tolerance {
    /// Defaults for a precision: each tolerance is `max(1e-9, 128 * epsilon)`.
    pub fn for_scalar<R: TestScalar>() -> Self {
        let tolerance = (128.0 * R::EPSILON).max(1e-9);
        Self {
            absolute: tolerance,
            relative: tolerance,
        }
    }

    /// Check `|actual - expected| <= absolute + relative * max(|actual|, |expected|)`.
    /// Nonfinite inputs always fail, including matching infinities.
    pub fn close(self, actual: f64, expected: f64) -> bool {
        actual.is_finite()
            && expected.is_finite()
            && (actual - expected).abs()
                <= self.absolute + self.relative * actual.abs().max(expected.abs())
    }
}

/// Describes how to test a [`Variable`]; implement this in your test module.
///
/// The scalar must be `f32` or `f64`. The associated dual variable must use the
/// matching dual scalar and tangent dimension. Both should come from the same
/// scalar-generic `Variable` implementation. No allocator selection is required
/// by the runners; the variable's existing allocator supplies its coordinates.
pub trait TestVariable: Variable<Scalar: TestScalar> + Debug {
    /// This variable evaluated with dual numbers of matching precision.
    type Dual: Variable<Scalar = <Self::Scalar as TestScalar>::Dual, Dim = Self::Dim>;

    /// Generate finite valid states, including nonidentity and noncommuting cases
    /// where applicable. Proptest shrinks these values and prints failing cases.
    fn states() -> impl Strategy<Value = Self>;

    /// Generate finite tangent increments inside the chosen log chart, with
    /// nonsingular exponential Jacobians. Strategies must preserve this domain
    /// while shrinking. This restriction does not restrict the group-law tests.
    fn increments() -> impl Strategy<Value = Tangent<Self>>;

    /// Copy the represented state into dual coefficients with zero derivatives.
    /// Lift stored coefficients directly, rather than rebuilding through `log`.
    fn to_dual(&self) -> Self::Dual;

    /// Compare represented states independently of the operations under test.
    ///
    /// Honor the tolerance, reject nonfinite state coefficients, and handle
    /// equivalent representations (e.g. quaternion `q` and `-q`). Do not implement
    /// this by calling the `log` or `local` that this suite is meant to verify.
    fn equivalent(&self, other: &Self, tolerance: Tolerance) -> bool;

    /// Whether `log` is smooth at this state (e.g. exclude a rotation's pi seam).
    ///
    /// Only checks using a log chart may reject these inputs. Group laws,
    /// coordinate conversion, and adjoint properties never use this predicate.
    /// Excessive random rejections fail through proptest's rejection limit.
    /// Identity and the explicit tiny-increment checks must never be rejected.
    fn log_is_smooth(&self) -> bool {
        true
    }

    /// Override case count, seed, shrinking, persistence, or rejection limits.
    /// Defaults honor proptest's environment variables and failure persistence.
    fn config() -> proptest::test_runner::Config {
        Default::default()
    }

    /// Coefficient and state-comparison tolerances, defaulting to this precision's
    /// [`Tolerance::for_scalar`] values. Both must be finite and nonnegative.
    fn tolerance() -> Tolerance {
        Tolerance::for_scalar::<Self::Scalar>()
    }
}

/// Independently runnable groups of variable properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Property {
    /// Tangent slice conversion and invalid-length rejection.
    Coordinates,
    /// Identity, inverse, and associativity, without log-chart filtering.
    GroupLaws,
    /// Exponential/log round trips and real/dual value agreement.
    ExpLog,
    /// Retraction/local round trips and the declared right-side convention.
    RetractLocal,
    /// Adjoint identity, composition, and conjugation.
    Adjoint,
    /// Analytical geometry Jacobians versus forward-mode dual derivatives.
    Jacobians,
}

#[derive(Debug)]
struct Case<T: TestVariable> {
    x: T,
    y: T,
    z: T,
    delta: Tangent<T>,
}

fn tangent<T: Variable<Scalar: TestScalar>>(values: &[f64]) -> Tangent<T> {
    T::tangent_from_slice(
        &values
            .iter()
            .map(|&v| T::Scalar::from_test_value(v))
            .collect::<Vec<_>>(),
    )
}

fn constant<T: TestVariable>(value: &Tangent<T>) -> Tangent<T::Dual> {
    T::Dual::tangent_from_slice(&value.iter().map(|&v| v.dual(0.0)).collect::<Vec<_>>())
}

fn near(tolerance: Tolerance, label: &str, actual: f64, expected: f64) -> TestCaseResult {
    prop_assert!(
        tolerance.close(actual, expected),
        "{label}: actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
    );
    Ok(())
}

fn vector<T: TestVariable>(
    t: Tolerance,
    label: &str,
    actual: &Tangent<T>,
    expected: &Tangent<T>,
) -> TestCaseResult {
    for i in 0..T::Dim::DIM {
        near(
            t,
            &format!("{label}[{i}]"),
            actual[i].test_value(),
            expected[i].test_value(),
        )?;
    }
    Ok(())
}

fn state<T: TestVariable>(t: Tolerance, label: &str, actual: &T, expected: &T) -> TestCaseResult {
    prop_assert!(
        actual.equivalent(expected, t),
        "{label}: actual={actual:?}, expected={expected:?}"
    );
    Ok(())
}

// Only diagnostic arithmetic uses f64; geometry and AD keep the variable's precision.
fn coefficients<T: TestVariable>(jacobian: &Jacobian<T>) -> DMatrix<f64> {
    DMatrix::from_iterator(
        T::Dim::DIM,
        T::Dim::DIM,
        jacobian.iter().map(|v| v.test_value()),
    )
}

fn matrix(
    t: Tolerance,
    label: &str,
    actual: &DMatrix<f64>,
    expected: &DMatrix<f64>,
) -> TestCaseResult {
    prop_assert_eq!(
        actual.shape(),
        expected.shape(),
        "{}: shape mismatch",
        label
    );
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            near(
                t,
                &format!("{label}[{i},{j}]"),
                actual[(i, j)],
                expected[(i, j)],
            )?;
        }
    }
    Ok(())
}

fn identity(i: usize, j: usize) -> f64 {
    if i == j { 1.0 } else { 0.0 }
}

fn smooth<T: TestVariable>(value: &T) -> TestCaseResult {
    if value.log_is_smooth() {
        Ok(())
    } else {
        Err(TestCaseError::reject("outside smooth log domain"))
    }
}

fn properties<T: TestVariable>(property: Property, case: &Case<T>, t: Tolerance) -> TestCaseResult {
    let Case { x, y, z, delta } = case;
    let zero = tangent::<T>(&vec![0.0; T::Dim::DIM]);
    let id = T::identity();
    match property {
        Property::Coordinates => {
            vector::<T>(
                t,
                "coordinate round trip",
                &T::tangent_from_slice(delta.as_slice()),
                delta,
            )?;
        }
        Property::GroupLaws => {
            state(t, "left identity", &id.compose(x), x)?;
            state(t, "right identity", &x.compose(&id), x)?;
            state(t, "left inverse", &x.inverse().compose(x), &id)?;
            state(t, "right inverse", &x.compose(&x.inverse()), &id)?;
            state(t, "double inverse", &x.inverse().inverse(), x)?;
            state(
                t,
                "associativity",
                &x.compose(y).compose(z),
                &x.compose(&y.compose(z)),
            )?;
            state(
                t,
                "product inverse",
                &x.compose(y).inverse(),
                &y.inverse().compose(&x.inverse()),
            )?;
        }
        Property::ExpLog => {
            smooth(x)?;
            smooth(y)?;
            smooth(z)?;
            let exp = T::exp(delta);
            smooth(&exp)?;
            state(t, "Exp(0)", &T::exp(&zero), &id)?;
            vector::<T>(t, "Log(identity)", &id.log(), &zero)?;
            vector::<T>(t, "Log(Exp(delta))", &exp.log(), delta)?;
            for value in [x, y, z] {
                let log = value.log();
                state(t, "Exp(Log(x))", &T::exp(&log), value)?;
                let dual_log = value.to_dual().log();
                for i in 0..T::Dim::DIM {
                    let (primal, derivative) = T::Scalar::parts(dual_log[i]);
                    near(t, "lift/log value", primal, log[i].test_value())?;
                    near(t, "lift/log unseeded derivative", derivative, 0.0)?;
                }
            }
        }
        Property::RetractLocal => {
            smooth(&x.inverse().compose(y))?;
            smooth(&T::exp(delta))?;
            state(t, "zero retraction", &x.retract(&zero), x)?;
            state(
                t,
                "right retraction",
                &x.retract(delta),
                &x.compose(&T::exp(delta)),
            )?;
            vector::<T>(t, "local(x,x)", &x.local(x), &zero)?;
            vector::<T>(
                t,
                "local convention",
                &x.local(y),
                &x.inverse().compose(y).log(),
            )?;
            vector::<T>(
                t,
                "local(retract(delta))",
                &x.local(&x.retract(delta)),
                delta,
            )?;
            state(t, "retract(local(y))", &x.retract(&x.local(y)), y)?;
        }
        Property::Adjoint => {
            let ax = coefficients::<T>(&x.adjoint());
            let ay = coefficients::<T>(&y.adjoint());
            matrix(
                t,
                "Ad(identity)",
                &coefficients::<T>(&id.adjoint()),
                &DMatrix::identity(T::Dim::DIM, T::Dim::DIM),
            )?;
            matrix(
                t,
                "Ad(x*y)",
                &coefficients::<T>(&x.compose(y).adjoint()),
                &(&ax * ay),
            )?;
            let coordinates =
                DVector::from_iterator(delta.len(), delta.iter().map(|v| v.test_value()));
            let transformed = tangent::<T>((&ax * coordinates).as_slice());
            state(
                t,
                "adjoint conjugation",
                &x.compose(&T::exp(delta)).compose(&x.inverse()),
                &T::exp(&transformed),
            )?;
        }
        Property::Jacobians => jacobians(case, t)?,
    }
    Ok(())
}

fn derivative<T: TestVariable>(
    t: Tolerance,
    label: &str,
    input: &Tangent<T>,
    value: &Tangent<T>,
    expected: &DMatrix<f64>,
    evaluate: impl Fn(&Tangent<T::Dual>) -> Tangent<T::Dual>,
) -> TestCaseResult {
    prop_assert_eq!(
        expected.shape(),
        (T::Dim::DIM, T::Dim::DIM),
        "{}: shape mismatch",
        label
    );
    for column in 0..T::Dim::DIM {
        let seeded = T::Dual::tangent_from_slice(
            &input
                .iter()
                .enumerate()
                .map(|(i, &v)| v.dual(identity(i, column)))
                .collect::<Vec<_>>(),
        );
        let output = evaluate(&seeded);
        for row in 0..T::Dim::DIM {
            let (primal, slope) = T::Scalar::parts(output[row]);
            near(
                t,
                &format!("{label} value[{row}]"),
                primal,
                value[row].test_value(),
            )?;
            near(
                t,
                &format!("{label} derivative[{row},{column}]"),
                slope,
                expected[(row, column)],
            )?;
        }
    }
    Ok(())
}

// Measure dual outputs around a separately evaluated and lifted native output.
fn output_chart<T: TestVariable>(output: T) -> impl Fn(T::Dual) -> Tangent<T::Dual> {
    let inverse = output.to_dual().inverse();
    move |value| inverse.compose(&value).log()
}

fn jacobians<T: TestVariable>(case: &Case<T>, t: Tolerance) -> TestCaseResult {
    let Case { x, y, delta, .. } = case;
    smooth(x)?;
    smooth(&x.inverse().compose(y))?;
    let exp = T::exp(delta);
    smooth(&exp)?;
    let zero = tangent::<T>(&vec![0.0; T::Dim::DIM]);
    let jr = coefficients::<T>(&T::right_jacobian(delta));
    let inverse = coefficients::<T>(&T::right_jacobian_inverse(delta));
    let identity = DMatrix::identity(T::Dim::DIM, T::Dim::DIM);
    matrix(
        t,
        "Jr(0)",
        &coefficients::<T>(&T::right_jacobian(&zero)),
        &identity,
    )?;
    matrix(
        t,
        "Jr_inverse(0)",
        &coefficients::<T>(&T::right_jacobian_inverse(&zero)),
        &identity,
    )?;
    matrix(t, "Jr * Jr_inverse", &(&jr * &inverse), &identity)?;
    let dx = x.to_dual();
    let dy = y.to_dual();
    let base_jacobian = coefficients::<T>(&exp.inverse().adjoint());
    let dexp = exp.to_dual();
    let exp_chart = output_chart(exp);
    let ddelta = constant::<T>(delta);
    derivative::<T>(t, "Exp", delta, &zero, &jr, |v| exp_chart(T::Dual::exp(v)))?;
    derivative::<T>(t, "Log(Exp(delta))", &zero, delta, &inverse, |e| {
        dexp.retract(e).log()
    })?;
    let log = x.log();
    let jlog = coefficients::<T>(&T::right_jacobian_inverse(&log));
    derivative::<T>(t, "Log(x)", &zero, &log, &jlog, |e| dx.retract(e).log())?;
    let ax = coefficients::<T>(&x.adjoint());
    derivative::<T>(t, "adjoint", &zero, &zero, &ax, |e| {
        dx.compose(&T::Dual::exp(e)).compose(&dx.inverse()).log()
    })?;
    let product_chart = output_chart(x.compose(y));
    let jx = coefficients::<T>(&y.inverse().adjoint());
    derivative::<T>(t, "compose/left input", &zero, &zero, &jx, |e| {
        product_chart(dx.retract(e).compose(&dy))
    })?;
    derivative::<T>(t, "compose/right input", &zero, &zero, &identity, |e| {
        product_chart(dx.compose(&dy.retract(e)))
    })?;
    let inverse_chart = output_chart(x.inverse());
    derivative::<T>(t, "inverse", &zero, &zero, &(-ax), |e| {
        inverse_chart(dx.retract(e).inverse())
    })?;
    let retracted_chart = output_chart(x.retract(delta));
    derivative::<T>(t, "retract/base", &zero, &zero, &base_jacobian, |e| {
        retracted_chart(dx.retract(e).retract(&ddelta))
    })?;
    derivative::<T>(t, "retract/increment", delta, &zero, &jr, |v| {
        retracted_chart(dx.retract(v))
    })?;
    let local = x.local(y);
    let mut negative_local = local.clone();
    negative_local.neg_mut();
    let left = coefficients::<T>(&T::right_jacobian_inverse(&negative_local));
    let right = coefficients::<T>(&T::right_jacobian_inverse(&local));
    derivative::<T>(t, "local/base", &zero, &local, &(-left), |e| {
        dx.retract(e).local(&dy)
    })?;
    derivative::<T>(t, "local/target", &zero, &local, &right, |e| {
        dx.local(&dy.retract(e))
    })?;
    Ok(())
}

/// Run one property group, panicking with a shrunk counterexample on failure.
///
/// Called by [`crate::variable_tests!`], or directly from a custom `#[test]`.
/// Identity is always checked. Jacobian tests also check signed tiny increments
/// along every coordinate. Those fixed cases cannot be rejected by a domain
/// predicate. Random cases use proptest's shrinking and bounded rejection.
#[track_caller]
pub fn check<T: TestVariable>(property: Property) {
    let mut config = T::config();
    let t = T::tolerance();
    assert!(
        config.cases > 0,
        "property tests require a nonzero case count"
    );
    assert!(
        [t.absolute, t.relative]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.0),
        "invalid property-test tolerance"
    );
    if config.source_file.is_none() {
        config.source_file = Some(Location::caller().file());
    }
    let mut fixed = Case::<T> {
        x: T::identity(),
        y: T::identity(),
        z: T::identity(),
        delta: tangent::<T>(&vec![0.0; T::Dim::DIM]),
    };
    properties(property, &fixed, t).unwrap_or_else(|e| panic!("{property:?} at identity: {e:?}"));
    if property == Property::Coordinates {
        for (label, length) in [
            ("long", Some(T::Dim::DIM + 1)),
            ("short", T::Dim::DIM.checked_sub(1)),
        ] {
            let Some(length) = length else { continue };
            let coordinates = vec![T::Scalar::from_test_value(0.0); length];
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| T::tangent_from_slice(
                    &coordinates
                )))
                .is_err(),
                "{label} tangent slice was accepted"
            );
        }
    }
    if property == Property::Jacobians {
        for axis in 0..T::Dim::DIM {
            for sign in [-1.0, 1.0] {
                let mut coordinates = vec![0.0; T::Dim::DIM];
                coordinates[axis] = sign * 1e-7;
                fixed.delta = tangent::<T>(&coordinates);
                properties(property, &fixed, t).unwrap_or_else(|e| {
                    panic!("{property:?} at tiny increment {coordinates:?}: {e:?}")
                });
            }
        }
    }
    let cases = (T::states(), T::states(), T::states(), T::increments())
        .prop_map(|(x, y, z, delta)| Case::<T> { x, y, z, delta });
    TestRunner::new(config)
        .run(&cases, |case| properties(property, &case, t))
        .unwrap_or_else(|e| panic!("{}::{property:?}: {e}", std::any::type_name::<T>()));
}

/// Register property tests for a type implementing [`TestVariable`](crate::testing::TestVariable).
///
/// Enable the `test-support` feature on your dev-dependency. The generated module
/// is guarded by `#[cfg(test)]` in the consuming crate. The type is resolved in
/// the invocation's scope; multiple invocations need distinct module names.
///
/// ```ignore
/// // After implementing TestVariable inside your #[cfg(test)] module:
/// fagra::variable_tests!(pose_properties, Pose<f64>);
/// fagra::variable_tests!(pose_f32_properties, Pose<f32>);
/// ```
#[macro_export]
macro_rules! variable_tests {
    ($name:ident, $variable:ty $(,)?) => {
        #[cfg(test)]
        mod $name {
            #[allow(unused_imports)]
            use super::*;
            #[test]
            fn coordinates() {
                $crate::testing::check::<$variable>($crate::testing::Property::Coordinates);
            }
            #[test]
            fn group_laws() {
                $crate::testing::check::<$variable>($crate::testing::Property::GroupLaws);
            }
            #[test]
            fn exp_log() {
                $crate::testing::check::<$variable>($crate::testing::Property::ExpLog);
            }
            #[test]
            fn retract_local() {
                $crate::testing::check::<$variable>($crate::testing::Property::RetractLocal);
            }
            #[test]
            fn adjoint() {
                $crate::testing::check::<$variable>($crate::testing::Property::Adjoint);
            }
            #[test]
            fn jacobians() {
                $crate::testing::check::<$variable>($crate::testing::Property::Jacobians);
            }
        }
    };
}
