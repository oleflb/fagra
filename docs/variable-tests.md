# Testing a variable implementation

Fagra supplies reusable property checks through `testing::TestVariable` and the
small `variable_tests!` registration macro. The suite checks the Lie-group
contract and its analytical Jacobians using the actual scalar-generic geometry.

## Enable test support

```toml
[dependencies]
fagra = "0.1"

[dev-dependencies]
fagra = { version = "0.1", features = ["test-support"] }
```

Cargo does not compile dependencies with their consumer's `cfg(test)`. The
feature therefore exposes the macro and helpers; the macro generates a
`#[cfg(test)]` module **in your crate**. Put your `TestVariable` implementation in
a test module too. The optional proptest and num-dual dependencies are absent
from ordinary builds without `test-support`.

The test module re-exports `proptest` and `num_dual`, so applications can use the
same compatible versions without declaring those dependencies separately.

## Use one geometry implementation for real and dual numbers

`Variable::Scalar` requires `nalgebra::RealField + Copy`. A generic implementation
can therefore be evaluated with either real numbers or dual numbers:

```rust,ignore
impl<R: faer_ext::nalgebra::RealField + Copy> Variable for Pose<R> {
    type Scalar = R;
    // ... the same implementation for f64, f32, Dual64, and Dual32 ...
}
```

The solver still requires `fagra::Real` for its schema/backend arithmetic.
Dual numbers can evaluate geometry and factors in tests. A variable
hard-coded to `f64` must first make its geometry scalar-generic; a macro cannot
differentiate an opaque real-valued function.

## Implement `TestVariable`

Here is the test-side implementation for the scalar variable from the
[complete runnable example](../examples/scalar_prior.rs):

```rust,ignore
#[cfg(test)]
mod tests {
    use super::*;
    use fagra::{Tangent, testing::{TestScalar, TestVariable, Tolerance}};
    use fagra::testing::proptest::prelude::*;
    use faer_ext::nalgebra::SVector;

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

    fagra::variable_tests!(double, Scalar<f64>);
    fagra::variable_tests!(single, Scalar<f32>);
}
```

`TestScalar` is already implemented for `f32` and `f64`, mapping them to `Dual32`
and `Dual64` respectively. A nongeneric implementation may simply select
`type Dual = Pose<fagra::testing::num_dual::Dual64>` for a `Pose<f64>`.

### Choosing strategies and comparisons

- **`states()`**: generate valid finite representations. Include nonidentity
  states and, for noncommutative groups, states that do not commute. Construct
  some states independently of `Variable::exp` so an error in `exp` cannot define
  the entire test domain. States need `Debug` so counterexamples can be printed.
- **`increments()`**: generate coordinates inside your chosen log chart, away
  from exponential-Jacobian singularities. Ensure shrinking preserves that
  domain. Rotation coordinates and translation coordinates may need different
  ranges. Arbitrary full-range floating-point sampling is usually unsuitable.
- **`to_dual()`**: copy stored coefficients, attaching zero derivative to each.
  Do not rebuild the state through the `log` being tested. The runner seeds input
  coordinates itself and compares both value and derivative components.
- **`equivalent()`**: compare represented states directly, with the supplied
  tolerance. Reject nonfinite coefficients. For quaternion rotations, handle
  both `q` and `-q`. Do not use the `local` or `log` being tested as the comparator.

Override `log_is_smooth()` if the log has a branch seam. For a principal
quaternion log, for example, exclude a small region around rotations of pi.
The runner consults this predicate only for properties that use a log chart;
group laws, coordinate conversion, and adjoint laws never discard inputs this
way. Proptest's rejection limit prevents an all-rejected run from passing.

Identity and signed tiny increments along each axis are mandatory cases for
the derivative checks. Rejecting one of those cases is a failure: the local
chart must be smooth around identity.

Each property generates only its required inputs: coordinates use increments;
group laws use three states; exp/log uses one state and an increment; the other
properties use two states and an increment. Fixed identity/zero assertions run
once per property rather than being repeated for every random case.

## What gets tested

Each macro invocation generates six ordinary tests:

| Test | Checks |
| --- | --- |
| `coordinates` | Slice round trip; rejection of too-short and too-long inputs |
| `group_laws` | Left/right identity and inverse; double inverse; associativity; inverse of a product |
| `exp_log` | Both local round trips; identity; real/dual log values and unseeded derivatives |
| `retract_local` | Right-side convention; zero update; local-coordinate round trips |
| `adjoint` | Identity, composition law, conjugation identity |
| `jacobians` | Values at zero; inverse consistency; all geometry derivatives against dual-number AD |

Derivative tests cover `exp`, `log`, the adjoint, both inputs to composition,
inverse, both inputs to retraction, and both inputs to `local`. They use the
right-side coordinate convention documented on `Variable`. For example, the
derivative of

```text
epsilon -> Log(Exp(delta)^-1 * Exp(delta + epsilon))
```

at zero is compared entry by entry to `right_jacobian(delta)`. Scalar dual
numbers seed one column at a time. There is no finite-difference step size.
Comparisons use absolute and relative tolerances and reject nonfinite results.

AD needs correct value-level geometry. A branch that returns constant identity
for a zero rotation can lose the derivative even if ordinary values look
correct. Use small-angle expressions in squared angle that preserve the first
derivative. Value-level geometry should not call the analytical Jacobian hook
under test to construct its derivative reference. The
[SE(3) example](../examples/slam.rs) follows both rules.

## Configuration and reproducing failures

The default case count, seed, shrinking, and failure persistence come from
proptest's `Config::default()` (including its environment variables). Both
tolerances default to `max(1e-9, 128 * scalar_epsilon)`. Override `config()` for
runner settings and `tolerance()` for coefficient and state comparisons:

```rust,ignore
fn config() -> fagra::testing::proptest::test_runner::Config {
    fagra::testing::proptest::test_runner::Config {
        cases: 512,
        ..Default::default()
    }
}

fn tolerance() -> fagra::testing::Tolerance {
    fagra::testing::Tolerance { absolute: 1e-10, relative: 1e-9 }
}
```

Failures identify the property, operation, and matrix entry and report the
shrunk case. Proptest normally persists regression seeds beside the test source.
Keep those regressions to reproduce failures. For a custom test, call
`testing::check_variable::<YourVariable>(testing::VariableProperty::Jacobians)` directly.

These checks establish consistency over the tested domain, not mathematical
proof or agreement with a physical model. Keep independent reference cases,
such as a known transform acting on a known point, alongside the generated suite.

## Repository checks

```sh
cargo test --features test-support
cargo test --example scalar_prior --features test-support
cargo test --no-default-features
cargo doc --no-deps --features test-support
```

The integration tests exercise noncommutative poses in both precisions and
deliberately faulty implementations: wrong adjoints, paired Jacobian sign
errors, broken inverses, and zero-angle branches that drop dual derivatives.
