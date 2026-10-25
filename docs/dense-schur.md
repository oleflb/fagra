# Dense Schur for bounded-window BA

`DenseSchur` is a CPU backend separate from the existing BAL-oriented iterative
`Schur`. It accepts arbitrary heterogeneous retained blocks and couplings, including
four spline controls, biases, camera calibration, field parameters, and retained-only
marginal priors. Eliminated landmarks may have different tangent dimensions.
No GPU implementation, incremental solver, or new dependency is introduced.

## Consumer API

```rust,ignore
// Schema/pool order must put all retained variables before all landmarks.
// These widths are examples, not a recommended landmark parameterization.
let backend = fagra::DenseSchur::<f64>::new(retained_dof, [3, 3, 4]);
let mut method = fagra::LevenbergMarquardt::new(backend);
let (report, covariance) = method.solve_batch_with_covariance(
    &mut graph,
    &fagra::OptimizeOptions::default(),
    &retained_keys, // &[BlockId], requested output order, including cross terms
    &fagra::CovarianceOptions::default(),
)?;
```

- `new(retained_dimension, eliminated_widths)` defines the coordinate partition.
  Widths must be positive and sum to graph DOF minus retained DOF.
- Retained widths need not be supplied. Any nonoverlapping retained Jacobian blocks
  are accepted, including multiple blocks in the same residual emission.
- Each residual emission may touch **at most one complete eliminated block**.
  Retained-only, landmark-only, and empty emissions are supported. Two landmarks in
  an emission, a partial landmark block, or a block spanning the partition boundary
  returns `EvaluationError::UnsupportedStructure`.
- Ordinary factors, batches, robust frozen-weight local models, and internal priors
  use the existing evaluation API. A factor may emit several admissible residuals.
- Reconstruct the backend when the retained dimension or ordered landmark widths
  change. The partition uses current schema/pool coordinate order, not track IDs.
  Declare retained families before landmark families; account for pool compaction.
- GN `solve_batch` and LM `solve_batch`/`solve_batch_with_covariance` work unchanged.
  The integration tests in `tests/dense_schur.rs` exercise the public graph path.

## Elimination and covariance

Normalize columns with D and assemble `[B E; Eᵀ C]`, where C has independent dense
landmark blocks. Each LM trial adds `lambda I` to **both** scaled partitions. Local
Cholesky solves produce `C_lambda^-1 [Eᵀ, -g_l]`; faer's dense products assemble
`S = B_lambda - E C_lambda^-1 Eᵀ` and the reduced RHS. Dense Cholesky solves retained
increments, then local back-substitution recovers landmarks. Original Jacobians
are retained for undamped step prediction and finite-step verification.

Scoped covariance rebuilds this reduction with **zero damping**, factors S, and
solves only for requested retained columns. The inverse scaling D is applied to
the selected RHS. It neither assembles/factors a full landmark-sized Hessian nor
allocates a full-coordinate selected RHS. The full-normal fallback workspace is
unused, covered by a regression assertion.

All landmarks must have full-rank information, including unobserved landmarks and
when no covariance blocks are selected. Both local C and reduced S pass the
configured relative pivot screen. S pivots are compared with the original B
diagonal, not a cancellation-small reduced diagonal, so eliminating a free
landmark cannot hide a retained gauge. This is an ordering-dependent numerical
screen, not a condition estimate or an FEJ/observability guarantee. No damping,
pseudoinverse, or diagonal repair is substituted for missing information.
Rank failure returns `SingularInformation`; a landmark covariance selection returns
`Evaluation(UnsupportedStructure)` via the existing solver error conversion.
As before, covariance failure does not roll back accepted optimizer updates.

`Problem::joint_covariance[_with]` remains the existing **full dense** standalone
query. Reduced extraction is currently available through successful scoped optimizer
calls only; no cached covariance query after a failed/partial solve is added here.

## Retirement contract

The existing square-root marginalizer is unchanged. It creates general separator
priors, not landmark-independent priors. Retiring a trajectory state while keeping
several connected landmarks may create a prior that this backend must reject.

Before/with old trajectory states, retire the connected landmark episodes so each
replacement prior touches at most one surviving eliminated block, preferably none.
Retained-only priors may couple arbitrary controls, biases, calibration and field
parameters; their correlations are not discarded. Check provisional/dynamic tracks
before prior formation. Deleting original factors later does not erase their
information already absorbed into a prior. Track lifetime/admission and gauge
consistency remain application policy.

## Cost and measurements

For retained DOF r, eliminated DOF l and landmark widths w, numerical storage is
`O(r² + r*l + sum(w²))` plus cached Jacobians and selected RHS `O(r*k)`.
Cross and solved-cross arrays are dense: this is intentionally a bounded-window
implementation, not a sparse map solver. Reduction work includes `O(r²*l)` and
the retained factorization `O(r³)`. Normal equations can lose weak-depth directions;
reject/defer poorly observable landmarks rather than invent precise priors.

Buffers/scratch are reused after warm-up within retained capacities. This does not
promise allocation-free structural growth or a faster full localization pipeline.
CPU is the initial placement: existing sequential faer kernels avoid GPU transfers
and synchronization for a small, frequently changing graph. Revisit placement
only with target-hardware full-pipeline evidence.

Reproduce separately from other benchmarks:

```sh
cargo test --test dense_schur
taskset -c 2 cargo test --release --test dense_schur benchmark_solve_and_retained_covariance -- --ignored --nocapture --test-threads=1
```

The benchmark uses 48 retained scalar states, 192 scalar landmarks, retained chain
edges, priors and three retained connections per landmark. It measures complete LM
solves **including selected retained covariance**, with five warm-up calls and 31
samples. Every call perturbs an estimate first and asserts accepted iterations.
Insertion/perturbation is outside timing; allocation instrumentation is inside.
This synthetic linear graph is not a visual-inertial BA workload or a robot budget.

Measured 2026-10-25 on AMD Ryzen AI 9 HX PRO 370, Linux x86-64, Rust 1.99.1,
faer 0.24.4, release kernels sequential, process pinned to logical CPU 2:

| Backend | Median (µs) | p95 (µs) | Maximum warmed allocations/call |
| --- | ---: | ---: | ---: |
| DenseNormalCholesky | 710.750 | 843.791 | 0 |
| DenseSchur | 359.948 | 372.803 | 0 |

One 31-sample run, including allocator instrumentation; no confidence interval or
hardware real-time claim. The benchmark processes were not run concurrently with
other tests/benchmarks by this session; unrelated system contention was not controlled.
