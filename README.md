# Fagra

A typed Rust factor-graph API with homogeneous storage and static dispatch.

Define variables and constraints, declare their types, insert values, optimize,
and read estimates through typed keys.

**Status:** graph construction, stable generational handles, state access, ordinary
and batched factor insertion, cost evaluation, removal, and Gauss–Newton
optimization work. `marginalize()` remains a `todo!()` stub.

## Quick start: one scalar and one prior

This model minimizes `0.5 * (x - measurement)²`. Its solution is `x = measurement`.
The complete, runnable source is [examples/scalar_prior.rs](examples/scalar_prior.rs):

```sh
cargo run --example scalar_prior
```

### 1. Define the variable

`Variable` describes the optimization coordinates and how to apply an increment.
A scalar has one coordinate and uses addition for retraction.

```rust
use fagra::{
    BlockId, EvaluationError, Factor, JacobianBlock, LinearizationSink,
    Solver, SolverError, StateKey, StateStore, Variable,
};
use faer_ext::nalgebra::{SMatrix, SVector};

struct Scalar(f64);

impl Variable for Scalar {
    type Tangent = f64;
    const DOF: usize = 1;

    fn tangent_from_slice(delta: &[f64]) -> f64 {
        delta[0] // The solver guarantees exactly DOF coordinates.
    }

    fn retract(&self, delta: &f64) -> Self {
        Self(self.0 + *delta)
    }
}
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

    fn linearize<L: LinearizationSink>(
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
`JacobianBlock::new` checks the column count against the state's `DOF` during
code generation. It accepts owned matrices and fixed-size views into existing
storage, preserving row and column strides without allocation or coefficient
copies. Ordinary factors emit directly: the solver has already opened their
factor scope.

Use nalgebra through `faer_ext::nalgebra` to match the interoperability crate's
version. The solver backend can borrow these coefficients as faer matrices with
`block.jacobian().into_faer()`; see [backend interoperability](docs/internals.md#backend-interoperability).

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
(default 1000) and `Lsmr::relative_tolerance` (default 1e-10) to tune the inner solve.
Linear iteration exhaustion returns `LinearSolveFailed` before applying a step.
Each optimization call rebuilds the cache, including after graph or model edits.

## Shared evaluation and further reading

- [Batching guide](docs/batching.md): shared models, individual payloads, and the
  ordinary-versus-batched factor-scope rules.
- [SLAM example](examples/slam.rs): manifold variables and shared trajectory
  computation across six-variable reprojections. Its application geometry is placeholder code.
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
