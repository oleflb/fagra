# Implicit Schur versus block-preconditioned LSMR on BAL

## Summary

This first Schur backend works and preserves the BAL objective, but **does not
generally beat block-LSMR on these three datasets**. At relative tolerance `1e-3`
it is 1.23–1.89× slower after 20 accepted steps, with slightly lower final costs
and gradients. At `1e-2` it wins the 73-camera case by about 10%, but loses the
other two. Neither backend reaches the nonlinear gradient tolerance in this budget.

## Implementation and constraints

`Schur::new(retained_dimension, retained_block_width, eliminated_block_width)`
selects a retained coordinate prefix and independent eliminated blocks. BAL uses
`Schur::new(9 * camera_count, 9, 3)` with cameras declared before points.
Each residual may involve at most one complete camera block and one complete point
block, or a unary block. Dimensions, alignment, and unsupported couplings are
checked. General camera-camera edges or marginal priors coupling multiple blocks
are rejected as `EvaluationError::UnsupportedStructure`, not approximated away.

With column-norm coordinates z = D delta, the backend caches local Gram blocks
and sparse cross blocks E. Every damping attempt factors local B and C blocks,
forms `rhs = -g_c + E C^-1 g_p`, and applies the reduced operator
`S = B - E C^-1 E^T` implicitly. B and C include `lambda I` in these normalized
coordinates. Faer's CG uses the inverse B blocks as a preconditioner. Back-substitution
recovers points, followed by delta = D^-1 z. This reduces the Krylov system from
23,769/33,753/60,876 coordinates to 441/657/1,242 camera coordinates.

Point work remains in each Schur product: two sparse cross-block products and
local C triangular solves. There is no assembled global Hessian, explicit inverse,
or dense reduced matrix. The normal-equation algebra can amplify conditioning
problems; damping is not a guarantee of numerical positive definiteness. Failed
local Cholesky or CG attempts are returned as errors. In particular, exact point
elimination cannot use the diagonal fallback used by LSMR's preconditioner.

Schur and LSMR share cached-model, column-norm and block-factor implementations;
each backend owns its own workspace. Original-Jacobian products independently verify the recovered
full normal residual and calculate the undamped predicted reduction. It also
recomputes the reduced residual after CG. Warm-workspace calls allocate no library
heap memory in the integration test, including reuse across fresh graphs.

Two faer 0.24.4 CG issues are handled explicitly:

- Its `Zero` initial-guess branch leaves the residual uninitialized. The backend
  supplies an explicitly zero step with `MaybeNonZero`, computing b-A*0 correctly.
- Its iteration-limit error returns a shadowed, initial residual. Schur reports
  no recurrence estimate on exhaustion and uses its fresh explicit residual.

## Measurement protocol

Measured 2026-09-18, AMD Ryzen AI 9 HX PRO 370, Linux x86-64, Rust 1.98.1,
faer 0.24.4, release build, sequential kernels pinned to logical CPU 2.
Base revision `2019426` plus the Schur implementation accompanying this report.
Input URLs, decompressed SHA-256 checksums and objective conventions are in
[the BAL report](bal-performance.md). No Ceres comparison was performed.

36 measured optimization calls: 3 datasets × 2 backends × 2 inner tolerances ×
3 repetitions. Each dataset/backend/tolerance group is a separate process;
each repetition creates a fresh graph and backend from the original input.
There is no explicit warm-up. Loading is once per process. Graph construction,
initial/final independent objective evaluations and output formatting are outside
the optimization timer. All benchmark processes run sequentially; no other
benchmark or test processes were launched concurrently.

Both methods use the same LM options: 20 accepted steps maximum, gradient tolerance
`1e-8`, step/cost tolerances zero, and default LM damping/retry controls. Inner
solves permit at most 1,000 iterations, with configured relative tolerance `1e-3`
or `1e-2`. The free similarity gauge, original initialization and non-robust
least-squares objective are preserved. All final independent objective checks pass.

**The stopping norms differ.** LSMR tests a block-preconditioned full normal residual;
CG tests the reduced-system residual relative to its reduced RHS. Equal numbers
in the tolerance argument do not mean equal original-coordinate linear accuracy.
The two nonlinear trajectories therefore also differ. Treat the table as a
comparison of configurations, not identical-accuracy linear solves.

Component timing is enabled on both backends, including clock/counter overhead.
Per-attempt text tracing is disabled. Peak RSS is Linux process-wide `VmHWM` and
includes input, verification, workspace and allocator retention across repetitions.
Three samples support medians/ranges, not reliable tail-latency estimates.
A preliminary 49-camera Schur `1e-3` run (15.092311 s) is excluded.

## Results

Every configuration has 20 accepted steps and 21 Jacobian passes and returns
`NoConvergence` at the outer limit. None is a converged solution measurement.

| Cameras | Tolerance | Block-LSMR median (s) | Schur median (s) | Schur / LSMR | LSMR / Schur final cost |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 49 | 1e-3 | 7.905 | 14.911 | 1.89× | 13346.65676 / 13346.51299 |
| 73 | 1e-3 | 18.865 | 23.284 | 1.23× | 17099.47198 / 17099.47197 |
| 138 | 1e-3 | 36.300 | 55.198 | 1.52× | 59378.63119 / 59377.10756 |
| 49 | 1e-2 | 4.508 | 6.897 | 1.53× | 13347.16755 / 13346.51867 |
| 73 | 1e-2 | 16.859 | 15.247 | 0.90× | 17099.78372 / 17099.47198 |
| 138 | 1e-2 | 15.451 | 23.827 | 1.54× | 59528.12602 / 59377.32780 |

For reference, initial objectives are 850912.4606808, 963738.9372180, and
2185545.840628. Schur `1e-3` final RMSE is 0.91557076, 0.86109741 and 1.18048751
pixels, versus LSMR's 0.91557569, 0.86109741 and 1.18050266.

| Cameras | Tolerance | Inner iterations, LSMR / Schur | Rejections, LSMR / Schur | Final gradient infinity norm, LSMR / Schur |
| ---: | ---: | ---: | ---: | ---: |
| 49 | 1e-3 | 1783 / 3703 | 0 / 0 | 282.2449 / 267.8557 |
| 73 | 1e-3 | 3045 / 3963 | 1 / 0 | 3.8200 / 1.7714 |
| 138 | 1e-3 | 3135 / 5126 | 5 / 2 | 272.4652 / 115.3650 |
| 49 | 1e-2 | 943 / 1563 | 2 / 0 | 288.1445 / 255.2551 |
| 73 | 1e-2 | 2661 / 2423 | 7 / 1 | 119.1235 / 5.1675 |
| 138 | 1e-2 | 1243 / 2037 | 3 / 4 | 135.0927 / 132.1918 |

All attempted solves in these runs completed successfully (cost evaluations =
solve attempts + 1); the rejections were nonlinear acceptance rejections, not
exhausted linear solves. Solve attempts equal 20 plus rejected attempts.

### Time to common objective targets

Medians below are computed independently for each accepted-step crossing time.
Targets are fractions of the same initial objective, not relative improvement
from different initializations. 5% and 2% mean 95% and 98% cost reduction. These
are early-progress targets; no interpolation between accepted states is used.

| Cameras | Tolerance | To 5%, LSMR / Schur (s) | To 2%, LSMR / Schur (s) |
| ---: | ---: | ---: | ---: |
| 49 | 1e-3 | 0.126827 / 0.278307 | 0.440102 / 0.647429 |
| 73 | 1e-3 | 0.185738 / 0.354408 | 0.595835 / 0.924283 |
| 138 | 1e-3 | 0.419630 / 0.723768 | not reached / not reached |
| 49 | 1e-2 | 0.065868 / 0.180362 | 0.204396 / 0.384809 |
| 73 | 1e-2 | 0.099389 / 0.180414 | 0.272992 / 0.519737 |
| 138 | 1e-2 | 0.197374 / 0.401205 | not reached / not reached |

Block-LSMR reaches these coarse targets faster in every configuration, including
the 73-camera `1e-2` case where Schur wins the overall 20-step runtime.

### Component timings at 1e-3

These correspond to each configuration's median-total-time run; units are seconds.

| Cameras / backend | Prepare | Factor/RHS | Linear solve | Products | Preconditioner | Verify |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 49 / LSMR | 0.359442 | 0.020776 | 7.251601 | 6.239206 | 0.533639 | 0.108443 |
| 49 / Schur | 0.487859 | 0.055998 | 13.938518 | 13.795912 | 0.109366 | 0.257357 |
| 73 / LSMR | 0.529369 | 0.032973 | 17.883412 | 15.333243 | 1.302787 | 0.166895 |
| 73 / Schur | 0.715940 | 0.080083 | 21.864468 | 21.610205 | 0.195438 | 0.366806 |
| 138 / LSMR | 0.982687 | 0.067913 | 34.387644 | 29.744285 | 2.432178 | 0.362759 |
| 138 / Schur | 1.332202 | 0.160736 | 52.499401 | 51.775203 | 0.597815 | 0.738616 |

Products and preconditioner time are nested inside linear-solve time. For LSMR,
products = forward + transpose augmented Jacobian products. For Schur, products
include B application, E/E^T products and eliminated C solves. The separate Schur
preconditioner column measures only retained B solves. Verify includes point
back-substitution and independent reduced/full residual checks. Remaining total
time includes nonlinear evaluation, retraction, layout and convergence checks.

The reduced Krylov vectors are much smaller, but the sparse products still visit
every observation and point solve. These dominate Schur runtime. The initial B-block
preconditioner is cheap but does not give enough iteration reduction to compensate
in most configurations. The next measured experiments would be an exact Schur-diagonal
preconditioner or assembling the reduced system for these modest camera counts.
These results do not establish how a sparse-direct Schur solver or much larger
BAL collections would compare.

### Raw runtimes and maximum process RSS

Final costs and iteration counts were identical across each group's repetitions.

```csv
cameras,mode,tolerance,run1_s,run2_s,run3_s,max_peak_rss_kib
49,block,1e-3,7.890958,8.181159,7.905324,30260
49,schur,1e-3,14.842158,14.911309,15.008347,34112
73,block,1e-3,18.562865,18.864802,19.065906,47924
73,schur,1e-3,23.269970,23.283932,23.434243,50948
138,block,1e-3,36.413024,36.299531,36.273836,75344
138,schur,1e-3,54.611512,55.523216,55.198268,93092
49,block,1e-2,4.438354,4.753668,4.507780,30072
49,schur,1e-2,6.833315,6.897069,7.020977,34112
73,block,1e-2,16.859307,16.592120,16.884601,47916
73,schur,1e-2,14.786336,15.437586,15.247100,51144
138,block,1e-2,15.237625,15.450898,15.912374,75272
138,schur,1e-2,23.417898,24.505480,23.826924,93148
```

Schur retains Jacobians for prediction/verification plus sparse cross blocks and
local normal blocks. Its higher memory here is expected; eliminating unknowns
from the Krylov vector does not imply lower total memory for this implementation.

## Reproduce

Use the downloads/checksums in [BAL reproduction](bal-performance.md#reproduce).

```sh
cargo build --release --example bal
for tolerance in 1e-3 1e-2; do
  for problem in 49-7776 73-11032 138-19878; do
    for mode in block schur; do
      taskset -c 2 target/release/examples/bal \
        "/tmp/fagra-bal/problem-${problem}-pre.txt" 20 1000 3 "$mode" "$tolerance"
    done
  done
done
```

The measurements above used the earlier CSV schema (`forward_s` for Schur products,
with `transpose_s` zero). The cleaned-up runner now emits backend-neutral
`products_s`: forward + transpose products for LSMR, complete reduced products
for Schur. `verification_s` replaces `diagnostics_s`. `precondition_s` measures retained
B solves in Schur mode, and `factor_s` includes reduced RHS construction. Diagnostic stderr
identifies the backend; set `BAL_TRACE=1` for per-attempt diagnostics. Tracing I/O
would be inside the timed call and was disabled for the comparison.

Checks: `cargo test --lib schur`, `cargo test --test levenberg_marquardt`, and their
release equivalents. They compare both precisions against dense augmented SVD,
exercise retries and reversed/repeated camera-point emissions, unary factors,
unused blocks, empty partitions, invalid partition/structure, failed local
factorization, CG exhaustion, and fresh-graph warm workspace reuse. All-feature
tests and the documentation build also pass.

## Shared-model cleanup regression

After separating the shared model/block workspace from solver-specific state,
four single-run checks used the same hardware, pinning, `1e-3` tolerance and
20-step budget. The final printed costs, gradients, iteration/solve counts,
acceptance/rejection counts and termination statuses matched the earlier runs.

| Cameras / backend | Earlier median (s) | Cleanup check (s) | Inner iterations | Final cost |
| --- | ---: | ---: | ---: | ---: |
| 49 / block-LSMR | 7.905324 | 7.702052 | 1783 | 13346.65675979 |
| 49 / Schur | 14.911309 | 15.038667 | 3703 | 13346.51299283 |
| 138 / block-LSMR | 36.299531 | 36.537138 | 3135 | 59378.63118544 |
| 138 / Schur | 55.198268 | 55.926452 | 5126 | 59377.10755877 |

These are regression checks, not a new speedup study: a single run is being
compared to earlier three-run medians. Observed runtime differences are within
about 3%. Schur no longer allocates LSMR-only damping/augmented-RHS buffers; its
first-run peak RSS was 32,808/83,072 KiB versus the earlier 33,744/85,364 KiB on
the 49/138-camera inputs. Allocator/process effects also influence RSS.

The cleanup retains separate solver termination criteria and fallback policies.
Tests additionally verify that changing an LSMR preconditioner mode requires a
fresh damping preparation and that actual solve attempts notify observers once.
