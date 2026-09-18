# Testing factors and batches

Enable the optional test helpers on your dev-dependency:

```toml
[dev-dependencies]
fagra = { version = "0.1", features = ["test-support"] }
```

Implement `testing::TestFactor` or `TestFactorBatch` on a small input fixture,
then register it with a macro. Both macros generate ordinary `cost` and
`jacobians` tests in a `#[cfg(test)]` module. The testing module re-exports
compatible versions of `proptest` and `num_dual`.

## Make evaluation scalar-generic

Variables, factors, batches, and their residual calculations must support
`faer_ext::nalgebra::RealField + Copy`. The runner evaluates the actual `cost()`
and `linearize()` implementations with `f32`/`Dual32` or `f64`/`Dual64`.
Use the same value-level computation for both scalars, without manufacturing
residual derivatives from the analytical Jacobians under test.

`fagra::Real` additionally requires faer backend operations and is too restrictive
for dual evaluation. Keep that bound on solver/backend code. For constants in
geometry-generic code, use `R::from_f64(value).unwrap()` rather than faer's
`R::from_f64_impl(value)`.

Factors must accept compatible `StateStore` implementations, for example:

```rust,ignore
impl<R: RealField + Copy, S: StateStore<Scalar<R>>> Factor<S> for Prior<R> {
    type Scalar = R;
    // Production cost and linearize implementations.
}
```

## Ordinary factor fixture

The fixture owns input values; the production factor can store only keys.
`cases()` generates and shrinks values. `build()` inserts them into the supplied
`TestStates<R>` and returns the production factor with valid keys.

```rust,ignore
#[cfg(test)]
mod tests {
    use super::*;
    use faer_ext::nalgebra::RealField;
    use fagra::testing::{TestFactor, TestStates, proptest::prelude::*};

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
            ].prop_map(|(value, measurement)| Self { value, measurement })
        }

        fn build<R: RealField + Copy>(
            &self, states: &mut TestStates<R>,
        ) -> Prior<R> {
            Prior {
                variable: states.insert(Scalar(R::from_f64(self.value).unwrap())),
                measurement: R::from_f64(self.measurement).unwrap(),
            }
        }
    }

    fagra::factor_tests!(double, PriorCase, f64);
    fagra::factor_tests!(single, PriorCase, f32);
}
```

The complete model is in [examples/scalar_prior.rs](../examples/scalar_prior.rs),
with tests registered once in [tests/scalar_properties.rs](../tests/scalar_properties.rs):

```sh
cargo test --test scalar_properties --features test-support
```

### Builder requirements

- Generate finite inputs in a smooth residual domain. Shrinking must preserve
  validity; evaluation errors fail the case rather than silently rejecting it.
  Include zero residuals, nonzero residuals, and relevant boundary cases.
- Build the same states, dimensions, measurements, and topology for every scalar
  type. State identities correspond by insertion order across builds.
- Construct measurements from fixture data, not from values read back from the
  supplied store: the store may already have perturbed an inserted state.
- Insert shared states once and reuse their keys. Aliased factor inputs must
  combine their Jacobian contributions before emission, as the solver requires.
- Within each factor, preserve residual-coordinate order across scalar types and
  batch selections. Block boundaries may differ: one two-row block and two
  one-row blocks compare equally. Factor scopes may be emitted in any order.

The store supports heterogeneous variable types through `StateStore<T>`; no
solver schema, `TestVariable` implementation, or custom dual conversion is needed.
The runner applies dual seeds through the variable's actual `retract()` method.

## Batch fixture

`TestFactorBatch` has the same `cases()` method, plus this associated type and
builder shape:

```rust,ignore
type Batch<R: RealField + Copy> = SharedModel<R>;

fn build<R: RealField + Copy>(
    &self, states: &mut TestStates<R>,
) -> (SharedModel<R>, Vec<Observation<R>>) {
    // Insert shared and per-payload states; construct the model and payloads.
}
```

Register with `fagra::factor_batch_tests!(batch, BatchCase, f64)`. The runner
supplies valid `FactorSelection` identities and exercises the full selection,
the empty selection, and every singleton. Include multi-payload cases in the
strategy so shared preparation is exercised. The payload type is inferred from
the production model's `FactorBatch::Factor`; no test-side declaration is needed.

See [tests/factor_properties.rs](../tests/factor_properties.rs) for a complete
batch fixture with nonlinear shared preparation, multiple residual blocks, and
aliased inputs:

```sh
cargo test --test factor_properties --features test-support
```

## Checks and configuration

`cost` requires a finite, nonnegative objective matching `0.5 * sum(r_i²)` over
all emitted whitened residuals. Individual residual coefficients may be negative.
Batch selections must agree with the full batch's per-factor emissions.

`jacobians` additionally compares dual primal residuals with real residuals and
dual derivatives with every analytical Jacobian column. It seeds one tangent
coordinate at a time at zero increment, differentiating `r(x.retract(delta))`.
All inserted states are seeded; missing Jacobian blocks mean zero derivatives.
An unseeded dual evaluation also runs, including for factors without states.
Dual cost values and derivatives must agree with the emitted residual objective.

The capture sink asserts valid scopes, dependencies, dimensions, and finite real
coefficients; proptest catches and shrinks assertion failures. Relative and
absolute tolerance comparisons reject nonfinite values. The dual Jacobians
themselves are not used as the derivative reference.

Both fixture traits provide defaults matching the variable suite:

```rust,ignore
fn config() -> fagra::testing::proptest::test_runner::Config {
    fagra::testing::proptest::test_runner::Config {
        cases: 512,
        ..Default::default()
    }
}

fn tolerance<R: fagra::testing::TestScalar>() -> fagra::testing::Tolerance {
    fagra::testing::Tolerance::for_scalar::<R>()
}
```

Defaults honor proptest environment variables and failure persistence; failures
report shrunk fixtures. Run a property directly with
`testing::check_factor::<PriorCase, f64>(testing::FactorProperty::Jacobians)` or
`testing::check_factor_batch`. These are consistency checks over the generated
domain; keep independent reference cases for the intended measurement model.
