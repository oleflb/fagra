# Fagra

A typed Rust factor-graph API with homogeneous storage, static dispatch, and
monomorphized `f32`/`f64` arithmetic.

Define variables and constraints, declare their types, insert values, optimize,
and read estimates through typed keys.

**Status:** graph construction, stable generational handles, state access, ordinary
and batched factor insertion, cost evaluation, removal, and Gauss–Newton and
Levenberg–Marquardt optimization work. Bulk square-root marginalization replaces selected states and
their incident factors with internal manifold priors.
Checked estimate replacement, selected joint covariance, and user-defined robust
IRLS local models are supported.

## Problem ownership and batch solvers

`Problem<S, F>` owns states, factors, batches, and historical marginal priors.
Nonlinear solvers own their linear backend and reusable optimization workspace:

```rust,ignore
let mut problem = fagra::Problem::<States, Factors>::new();
let x = problem.add(initial);
problem.add_factor(measurement)?;

let mut solver = fagra::LevenbergMarquardt::new(fagra::Lsmr::default());
let report = solver.solve_batch(&mut problem, &Default::default())?;
let estimate = problem.get(x)?;
```

`GaussNewton` supports the same `solve_batch` entry point. Both methods also
provide `solve_batch_with_covariance(problem, options, blocks, covariance_options)`
to reuse the final model. Covariance and marginalization query workspaces remain
on the problem, retaining their buffers across calls. The existing `Solver` graph
API remains available as a compatibility wrapper with a default GN optimizer.
See [examples/scalar_prior.rs](examples/scalar_prior.rs) for a runnable example.

### Opt-in change tracking

`TrackedProblem::new(problem)` owns a problem and records successful state/factor
edits. Ordinary `Problem` instances have no tracking fields or mutation hooks.
Read access is shared with `Problem`; mutable graph access goes through the wrapper.

```rust,ignore
let mut tracked = fagra::TrackedProblem::new(problem);
tracked.set(x, replacement)?;
solver.solve_batch(&mut tracked, &Default::default())?;
let problem = tracked.into_inner();
```

Batch solving invalidates the tracked numerical model before evaluation, including
on errors or unwinding. Successful nonempty marginalization also requests a rebuild.
`changes()` exposes the pending journal read-only; `invalidate_all()` marks external
changes to shared evaluator data. There is no journal-consuming incremental solver
or Bayes tree yet, so edits remain pending until superseded by a rebuild marker.

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

Use `solver.set(x, replacement)?` to replace an estimate while preserving its key,
attached factors, and historical marginal-prior anchors. This validates the key;
the application remains responsible for constructing valid variable values.

## Selected joint covariance

```rust,ignore
let blocks = [start.block_id(), end.block_id(), alignment.block_id()];
let covariance = solver.joint_covariance(&blocks)?;
```

The matrix concatenates the requested right-tangent coordinates in that order,
including cross-correlations and the information retained by marginalized history.
It includes ordinary factors, batches, and internal priors, excludes optimizer
damping, and rejects invalid or numerically singular information. Every live
coordinate must be observable, including unselected variables. Duplicate blocks
are rejected; an empty selection still checks the full information matrix.

For the tracking hot path, reuse the final optimizer model within the same call:

```rust,ignore
let (report, covariance) = solver.optimize_with_covariance(
    &mut method,
    &fagra::OptimizeOptions::default(),
    &blocks,
    &fagra::CovarianceOptions::default(),
)?;
```

This supports GN with dense Cholesky, LSMR, or Schur, and LM with LSMR or Schur.
Successful LM convergence already has current Jacobians; extraction does not
evaluate factors again. GN refreshes after cost-based termination if necessary.
Covariance failure preserves accepted estimates. The standalone query always
evaluates a fresh model without optimizing or moving estimates.

Both return a borrowed `faer::MatRef` backed by reusable solver storage. Warmed
calls within retained capacities allocate no library heap memory; user evaluators
and geometry must also avoid allocations. Call `.to_owned()` only when an owned
copy is needed beyond the next mutable use of the solver.

Extraction factors the undamped information `H = L Lᵀ`, solves `L Y = E` for the
selected coordinate columns, and returns `Yᵀ Y`; it never forms the full inverse.
Dense GN reuses its current factorization when available. LSMR/Schur assemble
dense information from their cached, unscaled Jacobians once at extraction time.
This implementation uses **O(n²) storage and O(n³) factorization**, even for small
selections; selected extraction costs O(n² k + n k²). It is intended for bounded
windows, not as a sparse covariance backend for arbitrarily large graphs.

`joint_covariance_with(&blocks, &options)` accepts explicit rank controls. The
default rejects normalized squared Cholesky pivots at or below `64 * epsilon`.
This ordering-dependent numerical rank screen is not a condition-number estimate.
No damping, diagonal repair, or pseudoinverse is substituted for singular information.

## User-defined robust factors

Implement the true robust objective in `cost()` and its frozen-weight local model
in `linearize()`. For blockwise Huber, with whitened residual norm `s` and threshold
`d > 0`, use cost `s²/2` for `s <= d`, otherwise `d * (s - d/2)`. Emit
`sqrt(w) * r` and `sqrt(w) * J`, where `w = 1` inside the threshold and `w = d/s`
outside. Apply one weight per observation (for example a 2D pixel or 6D odometry
residual), to all its Jacobian blocks. Do not differentiate the weight in that model.

The true cost gradient must equal the emitted `Jᵀr`. LM uses active factors' true
costs for trial acceptance. Marginalization absorbs the weighted local model and
never robustifies the resulting prior again. It does not infer or retain the
additive difference between true robust cost and weighted row cost. Reported costs
thereafter use the retained surrogate objective, up to that omitted additive robust
constant; states, local gradients, information, and covariance are unaffected by
the constant. The existing irreducible least-squares constant produced by QR is
still retained. Covariance uses inverse IRLS information, not a robust sandwich
estimator.

No graph-level loss registration or built-in loss helper is necessary. See the
user-defined implementation and checks in [tests/robust.rs](tests/robust.rs), and
the [robust-factor testing guide](docs/factor-tests.md#robust-local-models).

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

### Levenberg–Marquardt

For nonlinear problems where full GN steps overshoot or leave the model's domain:

```rust,ignore
use fagra::{LevenbergMarquardt, Lsmr, OptimizeOptions};

let mut method = LevenbergMarquardt::new(Lsmr::default());
method.options.max_trials = 16; // Solve attempts per cached linearization.
let result = solver.optimize_with(&mut method, &OptimizeOptions::default());
let progress = method.statistics(); // Also available when result is an error.
let linear_work = method.backend().statistics();
let report = result?;
```

LM solves `||J delta + r||² + lambda ||D delta||²`, with `D` derived from
Jacobian column norms. Damping rows are implicit: no normal equations or dense
diagonal matrix are formed. Rejected attempts reuse the cached Jacobians and all
buffers; they only repeat the linear solve, retraction, and cost evaluation.
Warm calls within retained capacity are allocation-free when user code is too.

Damped LSMR uses diagonal right-preconditioning by default: solve in coordinates
`z = D delta` with operator `[J D^-1; sqrt(lambda) I]`, then recover `delta`.
Set `Lsmr::diagonal_preconditioning = false` before constructing LM to use the
original coordinates. This changes the inner stopping norm, not the objective.
Preconditioner selection is fixed during damping preparation. Changing the effective
mode via the public flags requires preparing damping again; a solve with a stale mode
returns `InvalidOptions`. Normal optimizer calls prepare this automatically.
`backend().diagnostics()` retains termination, iteration count, finite-step status,
column-scale bounds, and estimated/explicit normal residuals after errors too.
`Lsmr::on_solve` optionally observes each attempt. Explicit residuals are computed
in both original and solver coordinates for damped solves.

For correlated variable coordinates, set `Lsmr::block_preconditioning = true`.
This adds small variable-block Cholesky preconditioners on top of diagonal
scaling, with diagonal fallback when a block cannot be factored. It preserves
the damped objective; the inner residual norm changes with the preconditioner.
Set `collect_timings = true` and inspect `timings()` to measure assembly,
factorization, operator products, preconditioning, and residual verification.
Timing is opt-in. `LevenbergMarquardt::on_accept` observes accepted progress.

`LmOptions` controls damping bounds, the column-norm floor, acceptance ratio, and
retry count. Damping starts fresh each call. Only cost-decreasing steps with
adequate actual/predicted agreement are accepted. Invalid trial geometry can be
retried; invalid accepted-state geometry, keys, or emissions return an error.
Retry exhaustion or stagnation returns `NoProgress`, retaining accepted estimates.

On nonempty graphs LM reports success only after checking gradient tolerance.
Small accepted steps with small cost changes trigger a fresh gradient check rather
than declaring an over-damped step converged. Set `step_tolerance = 0` to disable
that stagnation check; `cost_tolerance = 0` disables its additional cost condition.
Marginal priors participate in costs and Jacobians, but optimizer damping is never
marginalized. The default `solver.optimize()` remains GN with Cholesky; LM supports
LSMR and the bipartite Schur backend below. See [GN versus LM measurements](docs/lm-performance.md).

### Iterative Schur for bipartite graphs

For a schema ordered as 9D cameras followed by 3D points:

```rust,ignore
let mut backend = fagra::Schur::new(camera_count * 9, 9, 3);
backend.relative_tolerance = 1e-3;
backend.collect_timings = true;
let mut method = fagra::LevenbergMarquardt::new(backend);
let result = solver.optimize_with(&mut method, &fagra::OptimizeOptions::default());
let diagnostics = method.backend().diagnostics();
```

The constructor selects a retained coordinate prefix, its variable-block width,
and the eliminated block width. Schema/pool order defines the coordinate order;
reconstruct the backend when the partition changes. Each emitted residual may
touch at most one full block in each partition. Unary factors are supported;
other couplings return `EvaluationError::UnsupportedStructure` (including general
marginal priors that violate this structure).

Schur caches normalized local Gram and cross blocks, eliminates points with small
Cholesky solves, and applies `B - E C^-1 E^T` implicitly. Faer's CG solves the
reduced system using retained B-block preconditioning. No global normal or reduced
matrix is assembled, but the algebra uses normal equations and can lose accuracy
on poorly conditioned problems. Prediction uses the original cached Jacobians.
The reduced stopping norm differs from LSMR's preconditioned normal-residual norm.
See [Schur versus block-LSMR results](docs/schur-performance.md): this first Schur
backend is not generally faster on the measured BAL problems.

## Marginalization

The application chooses which states to remove, including heterogeneous types:

```rust,ignore
let report = solver.marginalize(&[old_pose.block_id(), old_bias.block_id()])?;
```

The solver linearizes incident factors at the current estimates and eliminates the
selected coordinates with rank-aware QR on Jacobian rows. It does not optimize
first or impose a time-window policy. Surviving-only factors stay nonlinear;
absorbed information becomes a fixed-reference prior with the correct Lie-group
coordinate derivatives. Removed state and factor handles become stale.

Run the [checked scalar example](examples/marginalization.rs):

```sh
cargo run --example marginalization
```

It compares the full and reduced problems against the analytical answer: `y = 5`,
cost `1.5`. Both marginalization and its LSMR optimization avoid normal equations.
The default `optimize()` method still uses normal-equation Cholesky.

`marginalize_with` accepts `MarginalizationOptions` for the relative numerical rank
tolerance. Empty batches can be retired explicitly with `remove_batch`.
This first implementation scans graph metadata and uses a dense affected front;
it reuses planning, QR, and prior storage for allocation-free warm calls within
retained capacities, but is not an incremental sparse marginalizer. See
[the internal design](docs/internals.md#square-root-marginalization) for semantics
and scaling limits, and [marginalization performance](docs/marginalization-performance.md)
for before/after measurements and allocation checks.

## Shared evaluation and further reading

- [Batching guide](docs/batching.md): shared models, individual payloads, and the
  ordinary-versus-batched factor-scope rules.
- [SLAM example](examples/slam.rs): manifold variables and shared trajectory
  computation across three-variable reprojections, using fixed calibration and
  linear translation/shortest-path rotation interpolation between two poses. Run with
  `cargo run --example slam` to optimize a synthetic scene and remove an observation.
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

### BAL bundle adjustment

Run the nine-parameter BAL camera model on a decompressed official dataset:

```sh
cargo run --release --example bal -- /path/to/problem-49-7776-pre.txt 20 1000 3
```

Arguments select the accepted-step limit, LSMR iteration limit, and repetitions.
The runner uses LM + LSMR, batches observations by camera, independently checks
the objective, and emits timings, errors, solver work, and Linux peak RSS as CSV.
See [BAL measurements](docs/bal-performance.md) for downloads and reproduction.
The original unpreconditioned baseline accepted no steps. Diagonal preconditioning
now enables 20 accepted steps on all three datasets, reducing RMSE to 0.86–1.18
pixels, but still reaches the configured iteration limit. Append `identity` to
reproduce the unpreconditioned mode, or set `BAL_TRACE=1` for per-attempt diagnostics.

The runner also accepts a preconditioner and inner tolerance, and reports
component timings and time to 5%/2% of the initial objective:

```sh
cargo run --release --example bal -- /path/to/problem-49-7776-pre.txt 20 1000 1 block 1e-3
```

The [24-run tolerance sweep](docs/bal-performance.md#block-jacobi-and-inner-tolerance-sweep)
found block mode at `1e-3` substantially faster than diagonal scaling on these
datasets, while all runs still reached the 20-step limit. Library defaults remain
diagonal preconditioning and the precision-aware default inner tolerance.
