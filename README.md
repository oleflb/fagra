# Fagra

A typed Rust factor-graph API with homogeneous storage, static dispatch, and
monomorphized `f32`/`f64` arithmetic.

Define variables and constraints, declare their types, insert values, optimize,
and read estimates through typed keys.

**Status:** graph construction, stable generational handles, state access, ordinary
and batched factor insertion, cost evaluation, removal, and Gauss–Newton
optimization work. `marginalize()` remains a `todo!()` stub.

## Quick start: one scalar and one prior

This model minimizes `0.5 * (x - measurement)²`. Its solution is `x = measurement`.
The walkthrough below uses `f64`. The complete, generic version in
[examples/scalar_prior.rs](examples/scalar_prior.rs) runs in both precisions:

```sh
cargo run --example scalar_prior
```

### 1. Define the variable

`Variable` describes a Lie group and its geometry derivatives. A scalar is the
additive group: composition is addition, inverse is negation, and its geometry
Jacobians are identity. The default retraction applies the increment on the right.

```rust
use fagra::{
    BlockId, EvaluationError, Factor, Jacobian, JacobianBlock, LinearizationSink,
    Solver, SolverError, StateKey, StateStore, Tangent, Variable,
};
use faer_ext::nalgebra::{Const, DefaultAllocator, SMatrix, SVector};

struct Scalar(f64);

impl Variable for Scalar {
    type Scalar = f64;
    type Dim = Const<1>;
    type Allocator = DefaultAllocator;

    fn identity() -> Self { Self(0.0) }
    fn compose(&self, other: &Self) -> Self { Self(self.0 + other.0) }
    fn inverse(&self) -> Self { Self(-self.0) }
    fn exp(delta: &Tangent<Self>) -> Self { Self(delta[0]) }
    fn log(&self) -> Tangent<Self> { SVector::<f64, 1>::new(self.0) }

    fn adjoint(&self) -> Jacobian<Self> { SMatrix::identity() }
    fn right_jacobian(_: &Tangent<Self>) -> Jacobian<Self> {
        SMatrix::identity()
    }
    fn right_jacobian_inverse(_: &Tangent<Self>) -> Jacobian<Self> {
        SMatrix::identity()
    }
}
```

`Dim` determines the solver coordinate count and the shapes of `Tangent<T>` and
`Jacobian<T>`; a one-dimensional tangent is a nalgebra vector, even if the stored
state is a scalar. `tangent_from_slice`, `retract`, and `local` have defaults.
`Allocator = DefaultAllocator` selects fixed-size array storage for these geometry
values. Generic state access and geometry calls need only `T: Variable`; allocator
bounds are local to generic numerical code that constructs nalgebra-owned results.
The [`Variable` documentation](src/variable.rs)
defines the adjoint, exponential Jacobians, log branches, and derivative conventions.

### Test your variable implementation

Enable `test-support` on your **dev-dependency**, implement
`fagra::testing::TestVariable` in a `#[cfg(test)]` module, then register the suite:

```rust,ignore
fagra::variable_tests!(pose_properties, Pose<f64>);
```

The generated tests use proptest for sampling/shrinking and matching-precision dual
numbers to verify the analytical Jacobians. See the [variable testing guide](docs/variable-tests.md)
and the [scalar example's test fixtures](tests/scalar_properties.rs):

```sh
cargo test --test scalar_properties --features test-support
```

### 2. Define the constraint

An ordinary `Factor` declares dependencies, evaluates its cost without Jacobians,
and emits a linearization. It reads estimates through `StateStore<T>`.

```rust
struct Prior {
    variable: StateKey<Scalar>,
    measurement: f64,
}

impl<S: StateStore<Scalar>> Factor<S> for Prior {
    type Scalar = f64;

    fn visit_variables(&self, mut visitor: impl FnMut(BlockId)) {
        visitor(self.variable.block_id());
    }

    fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
        let residual = states.get(self.variable)?.0 - self.measurement;
        let cost = 0.5 * residual * residual;
        if !cost.is_finite() {
            return Err(EvaluationError::InvalidEvaluation);
        }
        Ok(cost)
    }

    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let residual = SVector::<f64, 1>::new(states.get(self.variable)?.0 - self.measurement);
        let jacobian = SMatrix::<f64, 1, 1>::new(1.0);
        sink.residual(&residual, &[JacobianBlock::new(self.variable, &jacobian)])
    }
}
```

The residual is `x - measurement`; its derivative under additive retraction is
`1`. The sink rejects invalid dimensions and nonfinite coefficients.
`JacobianBlock::new` requires the state's `Dim` as its column dimension during
type checking. It accepts owned matrices and fixed-size views into existing
storage, preserving row and column strides without allocation or coefficient
copies. Ordinary factors emit directly: the solver has already opened their
factor scope.

Use nalgebra through `faer_ext::nalgebra` to match the interoperability crate's
version. The solver backend can borrow these coefficients as faer matrices with
`block.jacobian().into_faer()`; see [backend interoperability](docs/internals.md#backend-interoperability).

### Test your factors and batches

With `test-support`, implement `testing::TestFactor` or `TestFactorBatch` on an
input fixture. Its `cases()` generates state values and measurements; `build()`
inserts states into the supplied test store and constructs the evaluator from
their keys. Register the fixture and precision:

```rust,ignore
fagra::factor_tests!(prior, PriorCase, f64);
fagra::factor_batch_tests!(reprojections, ReprojectionCase, f64);
```

The generated `cost` and `jacobians` tests check nonnegative least-squares cost
and analytical Jacobians against dual-number residual derivatives. Batch tests
also compare full, empty, and singleton selections. See the
[factor testing guide](docs/factor-tests.md) for complete setup and examples.

### 3. Declare the graph's types

The macros register types and generate their storage. Each state or factor
payload type identifies one family; a family can hold many instances.

```rust
fagra::states! { States { scalars: Scalar } }
fagra::factors! { Factors { priors: Prior } }
```

### 4. Insert, optimize, and read

```rust
fn main() -> Result<(), SolverError> {
    let mut solver = Solver::<States, Factors>::new();
    let x = solver.add(Scalar(0.0));
    let prior = solver.add_factor(Prior { variable: x, measurement: 3.0 })?;

    let report = solver.optimize()?;
    let estimate = solver.get(x)?;
    let cost = solver.factor_cost(prior)?;
    println!("estimate = {}, cost = {cost}, steps = {}", estimate.0, report.iterations);
    Ok(())
}
```

`x` is a `StateKey<Scalar>` and `prior` is a `FactorKey<Prior>`. Keys select the
correct type without casts. `optimize` means optimization to convergence;
`factor_cost` returns a numerical objective contribution, while failures use
Rust's `Result`. A factor can be removed with `solver.remove_factor(prior)?`.
Before optimization, this example's initial estimate is `0.0` and its prior cost
is `4.5`; after one GN step, the estimate is `3.0` and cost is zero.
Handles survive storage growth and compaction. Removed handles are rejected even
after their slots are reused, and keys from another solver are rejected.

## Scalar precision

Each graph uses one scalar type throughout: tangent coordinates, residuals,
Jacobians, cost accumulation, backend storage, steps, tolerances, and reports.
`f32` and `f64` graphs can coexist in one binary; Rust monomorphizes both.

For generic application types `Scalar<R>` and `Prior<R>`, declare:

```rust
fagra::states! { States<R> { scalars: Scalar<R> } }
fagra::factors! { Factors<R> { priors: Prior<R> } }

let mut single = Solver::<States<f32>, Factors<f32>>::new();
let mut double = Solver::<States<f64>, Factors<f64>>::new();
```

The macros introduce `R: fagra::Real`, which combines faer's and nalgebra's
real-number traits with `Copy`. This bound is required for backend arithmetic.
Variables, factors, and batch models declare `type Scalar = R`; evaluators accept
`L: LinearizationSink<Scalar = R>`. A variable's `tangent_from_slice` receives
`&[R]`, and costs return `Result<R, EvaluationError>`. See the
[generic scalar-prior example](examples/scalar_prior.rs) for a complete implementation.

For dual-number tests, implement variables, factors, and batches over
`R: faer_ext::nalgebra::RealField + Copy`. Evaluation traits support these geometry
scalars; solver schemas and numerical backends still require `R: fagra::Real`.

Both schemas and the selected backend must agree on precision. A Jacobian's
coefficients must match its variable's scalar type. These are compile-time checks.
Batch syntax also accepts generic types: `Batch<Model<R>, Observation<R>>`.
Non-generic schema declarations use `f64`; existing implementations need
`type Scalar = f64` and the corresponding sink bound.

The retained default optimizer uses the schema's precision. To select LSMR:

```rust
let mut method = fagra::GaussNewton::new(fagra::Lsmr::<f32>::default());
let report: fagra::OptimizeReport<f32> =
    single.optimize_with(&mut method, &fagra::OptimizeOptions::<f32>::default())?;
```

Defaults preserve the existing `f64` tolerances and account for `f32` rounding.
With `ε` denoting the scalar's machine epsilon:

| Control | Default |
| --- | --- |
| Gradient tolerance | `max(1e-8, 8ε)` |
| Step tolerance | `max(1e-10, ε)` |
| Cost tolerance | `max(1e-12, ε)` |
| LSMR relative tolerance | `max(1e-10, 128ε)` |

These remain configurable. Gradient and step tolerances are absolute, so choose
values appropriate to the model's units and scaling.

## Optimization controls and workspace reuse

`optimize()` retains its default workspace. For explicit stopping controls or a
reusable optimizer instance:

```rust
use fagra::{GaussNewton, OptimizeOptions};

let mut method = GaussNewton::default(); // Dense normal equations + Cholesky.
let options = OptimizeOptions {
    max_iterations: 100,
    gradient_tolerance: 1e-8,
    step_tolerance: 1e-10,
    cost_tolerance: 1e-12,
};
let report = solver.optimize_with(&mut method, &options)?;
```

The report includes accepted step count, initial/final costs, and the stopping
reason. Gradient and step tolerances use absolute infinity norms. Cost stopping
requires a nonnegative decrease at most `cost_tolerance * max(1, previous_cost)`;
an exactly zero cost also terminates. Invalid options are rejected before evaluation.

This is **full-step GN**, with no damping or line search: finite uphill steps are
accepted. Evaluation failure rolls back the current trial, while already accepted
steps remain. Iteration exhaustion returns `NoConvergence`. Cholesky failure returns
`LinearSolveFailed`; every stored variable contributes coordinates, so unobserved
variables and gauge freedoms can make a required solve singular.

`GaussNewton::new(DenseNormalCholesky::default())` explicitly selects the dense
backend. For matrix-free LSMR:

```rust
use fagra::{GaussNewton, Lsmr, OptimizeOptions};

let mut method = GaussNewton::new(Lsmr::default());
let report = solver.optimize_with(&mut method, &OptimizeOptions::default())?;
```

`Lsmr` uses faer's `BiLinOp` with cached local Jacobian blocks: factors linearize
once per nonlinear iteration, and inner iterations reuse those coefficients.
No global Jacobian or normal matrix is assembled. Storage is O(nnz + m + n),
including copied block coefficients and reusable workspace. Products are sequential
and unpreconditioned. Before constructing the optimizer, set `Lsmr::max_iterations`
(default 1000) and `Lsmr::relative_tolerance` (default `max(1e-10, 128ε)`) to tune the inner solve.
Linear iteration exhaustion returns `LinearSolveFailed` before applying a step.
Each optimization call rebuilds the cache, including after graph or model edits.

## Shared evaluation and further reading

- [Batching guide](docs/batching.md): shared models, individual payloads, and the
  ordinary-versus-batched factor-scope rules.
- [SLAM example](examples/slam.rs): manifold variables and shared trajectory
  computation across six-variable reprojections. State Lie-group geometry is implemented;
  factor geometry remains placeholder code.
- [Internal design](docs/internals.md): typed pools, schema visitors, and numerical solver passes.

## Checks and storage timings

```sh
cargo check --all-targets
cargo test
cargo doc --no-deps
```

A small release-mode storage workload measures checked state lookup, packed
factor iteration, and reserved-capacity insertion/removal:

```sh
cargo test --release --test storage storage_workload_timing -- --ignored --nocapture
```

The 1,000-DOF workload uses unary priors and neighboring-variable constraints. It
checks that a warmed call, including dense assembly, factorization, and a real
accepted step, performs zero heap allocations:

```sh
cargo test --release --test optimizer thousand_dof_dense_solve -- --ignored --nocapture
```

Compare dense Cholesky and LSMR across 10–5,000 variables and two anchor spacings:

```sh
cargo test --release --test optimizer compare_cholesky_lsmr -- --ignored --nocapture --test-threads=1
```

See [solver performance](docs/solver-performance.md) for measured results and methodology.
