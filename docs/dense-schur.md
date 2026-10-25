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
`S = B_lambda - E C_lambda^-1 Eᵀ` and the reduced RHS. On AArch64, the lower-triangle
product uses faer's SIMD column dot products: faer 0.24.4 otherwise dispatches larger
products to gemm-common 0.19.0, which allocates packing buffers and status vectors
on every call, even with sequential execution and warmed fagra scratch. Other
architectures keep the original triangular GEMM path. Dense Cholesky solves retained
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
promise allocation-free structural growth, allocation-free dependency kernels at
arbitrary fixed dimensions, or a faster full localization pipeline.
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

### Orin NX: AArch64 allocation fix and size limits

Measured 2026-10-25 on standalone Booster 99, Jetson Orin NX 8 GB / Cortex-A78AE,
L4T 36.4.3, Rust 1.98.0 SDK release build with `-C target-cpu=cortex-a78`.
Existing MAXN_SUPER mode and CPU min=max=1.984 GHz were unchanged; benchmark pinned
to CPU 2, sequential faer kernels, no running HULK pipeline, camera, or GPU work.
Services, power modes and clocks were not changed.

Allocation backtraces at the original 48+192 scalar shape identified all 56 warmed
DenseSchur allocations: seven NEON GEMM calls per reduction, each allocating a
packing buffer (`gemm-common/src/gemm.rs:437`) and packing-status vector (`:490`).
Three LM reductions plus one undamped covariance reduction give `4 * 7 * 2 = 56`.
The reduced Cholesky and two-column selected covariance extraction allocated
nothing at that shape. More fagra scratch cannot replace these internal buffers.
The AArch64 column-dot change removes this specific allocation source, not every
possible allocation in the solver/dependencies.

Original commit `20ca45b` executable and patched executable, ten sequential paired
process runs with alternating process order; five warm-ups and 31 samples/backend
per process. Values are medians of the ten per-run percentiles, not pooled p95:

| Scalar LM solve + covariance | p50 (µs) | p95 (µs) | Maximum warmed allocations/call |
| --- | ---: | ---: | ---: |
| DenseNormalCholesky, patched-build control | 2995.407 | 3004.991 | 302 |
| DenseSchur, original AArch64 GEMM | 1184.437 | 1193.189 | 56 |
| DenseSchur, AArch64 SIMD dots | 1241.479 | 1248.903 | 0 |

About 4.8% slower p50 buys zero warmed allocations for this shape. An exploratory
scalar-loop alternative was about 52.5% slower and was not adopted. The x86 path
is unchanged. Allocation counters count calling-thread Rust alloc/alloc_zeroed/
realloc calls, not bytes, C-library allocations, or process-wide activity.

#### Mixed-width, densely coupled size check

`src/dense_schur/tests/sizes.rs` uses full-rank priors, dense retained couplings and
independent landmark blocks cycling through widths **3, 4, 6**. Every landmark
emission touches the entire retained partition: a deliberate dense-coupling stress
case, not a representative visual-inertial observation pattern. Six retained
covariance columns represent a pose-sized output. Numerical tests compare every
step coordinate, predicted reduction and all 36 covariance entries against dense
for damping `0, 0.001, 10, 0`, in f64 and f32 (error bounds `1e-10 * (1+|reference|)`
and `2e-4 * (1+|reference|)`). They pass on both x86 and Orin.

The ignored size benchmark times re-accumulation of frozen Jacobians, preparation,
**one damped step**, and undamped selected covariance. Re-accumulation is required
because dense covariance consumes its normal matrix. Jacobian construction,
initial backend preparation and selection setup are outside timing/counting;
graph evaluation, accepted LM iterations/retries and the full pipeline are absent.
Ten runs, five warm-ups plus 31 measured calls per case/backend/run:

| Retained DOF | Landmark blocks / DOF | Backend | p50 (ms) | p95 (ms) | First measured allocations | Maximum warmed allocations |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 24 | 12 / 52 | DenseNormalCholesky | 0.142 | 0.144 | 1004 | 6 |
| 24 | 12 / 52 | DenseSchur | 0.514 | 0.521 | 3 | 0 |
| 48 | 48 / 208 | DenseNormalCholesky | 1.871 | 1.880 | 211 | 210 |
| 48 | 48 / 208 | DenseSchur | 5.749 | 5.760 | 3 | 0 |
| 72 | 48 / 208 | DenseNormalCholesky | 2.542 | 2.551 | 437 | 436 |
| 72 | 48 / 208 | DenseSchur | 12.173 | 12.183 | 7 | 4 |
| 96 | 96 / 416 | DenseNormalCholesky | 11.207 | 11.245 | 1219 | 1218 |
| 96 | 96 / 416 | DenseSchur | 39.372 | 39.400 | 23 | 20 |

First measured counts are **not full cold-start counts**: dimensions and Jacobian
storage are already prepared; process/dependency initialization also depends on
case order (dense 24 runs first). Warmed counts exclude the first five calls.
The target allocation regressions assert zero at scalar 48+192 and mixed 48+208
with six covariance columns. Larger retained factors/solves still reach allocating
faer kernels. Additional allocation backtraces attributed the four calls at 72
retained DOF to reduced Cholesky (two) and selected triangular solve (two); at 96,
the split was 18 and two. None came from the replaced reduction product. No
arbitrary-size zero-allocation guarantee is made.

**A 33 ms / 30 Hz end-to-end budget is not established.** The fully coupled 96+416
stress case exceeds 33 ms even for one step plus covariance, and dense is faster
than Schur in these dense-block assembly cases. Actual residual sparsity,
Jacobian evaluation, iteration count, marginalization, data association and
pipeline contention must be measured before choosing a window size. Do not
multiply the scalar benchmark into a full-BA throughput claim or treat these
short-run percentiles as a worst-case bound.

During the final approximately 30-second series, 286 tegrastats samples reported
only 1.984 GHz CPU clocks, CPU 52.625–53.562 °C, junction 54.000–54.812 °C and
whole-board input power 7.806–8.210 W. No sampled clock drop; this is neither a
thermal-soak guarantee nor per-backend energy measurement.

Reproduce the size benchmark separately (cross-build and run its test executable
on the target when using the SDK):

```sh
taskset -c 2 cargo test --release --all-features --lib benchmark_mixed_sizes -- --ignored --nocapture --test-threads=1
```

Raw snapshot, build logs, executable hashes, ten-run datasets, correctness results
and telemetry: `/tmp/opencode/orin-aarch64-live-20ca45b-LElBzLJg/` on the build host;
target run: `/tmp/opencode/orin-aarch64-live-v2-DfMovM2w/`. Earlier baseline/trace
artifacts remain separate. These temporary paths are local evidence, not shipped
fixtures or durable external benchmark archives.
