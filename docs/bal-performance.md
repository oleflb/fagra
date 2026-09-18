# BAL: LM + LSMR preconditioning experiments

The initial identity-preconditioned baseline below accepted no steps. The
[diagonal-preconditioned follow-up](#diagonal-preconditioning-follow-up) accepts
20 steps on all three datasets, but does not yet meet gradient convergence.
The latest [block-Jacobi/tolerance sweep](#block-jacobi-and-inner-tolerance-sweep)
improves runtime further while keeping the same objective and 20-step budget.

Measured 2026-09-18 on AMD Ryzen AI 9 HX PRO 370, Linux x86-64,
Rust 1.98.1, faer 0.24.4, release build, sequential kernels, pinned to
logical CPU 2. Library revision: `639b1501e8b57b67d19de1a80aba502b69b72056`
plus the BAL example and tests accompanying this document. No library solver
changes were made for these measurements.

**None of the three real datasets made optimization progress in the initial baseline.** All nine runs
returned `NoProgress` after exhausting linear solves. These are measurements of
time to failure, not successful bundle-adjustment timings.

## Model and methodology

The [official BAL Ladybug datasets](https://grail.cs.washington.edu/projects/bal/ladybug.html)
are loaded without normalization, perturbation, robust loss, or added priors.
All nine camera parameters and all three point coordinates are optimized in f64.
Camera parameters use additive angle-axis/translation/focal/distortion updates.
Observations are batched by camera to share rotation calculations.

The [BAL convention](https://grail.cs.washington.edu/projects/bal/) is:

```text
P = R(angle_axis) * X + translation
p = -P.xy / P.z
r2 = dot(p, p)
prediction = focal * (1 + k1*r2 + k2*r2*r2) * p
cost = 0.5 * sum ||prediction - observation||²
```

The similarity gauge is left free. Positive LM damping regularizes trial solves;
it does not identify a unique reconstruction. No comparison of raw estimated
coordinates to ground truth is claimed.

Controls, identical across all datasets:

- LM: at most 20 accepted steps; default damping bounds `1e-12..1e12`, initial
  damping `1e-3`, column norm floor `1e-6`, acceptance threshold `1e-4`, at most
  16 trials per linearization.
- LSMR: at most 1,000 iterations per solve, relative tolerance `1e-6`, identity
  preconditioning. This tolerance is deliberately looser than the f64 backend
  default of `1e-10` but did not suffice to complete a solve.
- Gradient tolerance `1e-8`; step and cost tolerances zero.

Each dataset was measured three times in one process. Every repetition constructs
a fresh graph and optimizer with the original estimates, so these are cold
optimizer-workspace calls, not warm allocation benchmarks. Loading occurs once
and is reported separately; construction and independent objective checks are
outside the solve timer. No timed datasets run concurrently.

The objective is independently recomputed before/after optimization using a
separate projection expression and nalgebra's `Rotation3`, rather than the
factor implementation's quaternion-based rotation. Its final value is checked
against LM's reported cost. RMSE means `sqrt(2*cost / observations)`, i.e. RMS
2D reprojection distance, not per-coordinate RMSE.

`last_gradient` is the norm at LM's last linearization; it can precede accepted
estimates on some error paths. In every recorded run it is the gradient of the
unchanged initial/final state. Peak RSS comes from Linux `/proc/self/status`
`VmHWM`, includes parsing, graph storage, verification, optimizer and allocator
retention, and is process-wide across repetitions. Other platforms print `NA`.

## Results

| Ladybug problem | DOF | Observations | Median solve time (s) | Range (s) | Max reported RSS (MiB) | Successful runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 49 cameras / 7,776 points | 23,769 | 31,843 | 38.126 | 37.663–38.588 | 30.1 | 0/3 |
| 73 cameras / 11,032 points | 33,753 | 46,122 | 55.487 | 55.162–56.179 | 44.8 | 0/3 |
| 138 cameras / 19,878 points | 60,876 | 85,217 | 103.744 | 103.214–104.872 | 65.8 | 0/3 |

| Cameras | Initial = final objective | Initial = final RMSE (pixels) | Final gradient infinity norm |
| ---: | ---: | ---: | ---: |
| 49 | 850,912.4606808 | 7.31055672 | 8.56792572e6 |
| 73 | 963,738.9372180 | 6.46458477 | 1.01607290e7 |
| 138 | 2,185,545.840628 | 7.16195911 | 2.06286031e7 |

All runs had exactly:

- 0 accepted steps, 11 rejected attempts, 1 linearization, 1 cost evaluation;
- 11 linear solves and 11,000 completed inner iterations;
- last rejection `LinearSolve`, final attempted damping `1e12`;
- termination `NoProgress` when the maximum damping bound was reached.

The linear iteration counts show each attempt exhausted its budget before any
nonlinear trial evaluation. Damping alone did not resolve the difficulty.
Unpreconditioned solving of the mixed-scale BAL parameters is the next performance
and numerical investigation; these measurements do not establish which
preconditioner or solver would be sufficient. Raising limits alone is not a
demonstrated solution. A preliminary 100-iteration run on the 49-camera dataset
also accepted no steps (3.754 s, 1,100 total inner iterations).

Three repetitions support a basic range/median summary, not a reliable p95 or
confidence interval. This is a baseline on three Ladybug instances, not a claim
about every BAL collection. No Ceres comparison was performed.

### Raw timing and memory samples

| Cameras | Run | Load (s, once) | Build (s) | Solve (s) | VmHWM (KiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 49 | 1 | 0.007072 | 0.002361 | 37.663214 | 25768 |
| 49 | 2 | — | 0.001964 | 38.588422 | 29948 |
| 49 | 3 | — | 0.002000 | 38.125676 | 30792 |
| 73 | 1 | 0.010667 | 0.003869 | 55.487195 | 35452 |
| 73 | 2 | — | 0.002954 | 55.161950 | 45860 |
| 73 | 3 | — | 0.002533 | 56.179089 | 45708 |
| 138 | 1 | 0.019498 | 0.006829 | 103.743883 | 65048 |
| 138 | 2 | — | 0.005005 | 104.871601 | 67100 |
| 138 | 3 | — | 0.004730 | 103.213888 | 67412 |

## Reproduce

Download into a directory outside the repository; `curl`, `bzip2`, and Linux
`taskset` are external tools, not Rust dependencies. Dataset files are not vendored.

```sh
mkdir -p /tmp/fagra-bal
for problem in 49-7776 73-11032 138-19878; do
  curl -fL "https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-${problem}-pre.txt.bz2" \
    -o "/tmp/fagra-bal/problem-${problem}-pre.txt.bz2"
  bzip2 -dk "/tmp/fagra-bal/problem-${problem}-pre.txt.bz2"
done
cargo build --release --example bal
for problem in 49-7776 73-11032 138-19878; do
  taskset -c 2 target/release/examples/bal \
    "/tmp/fagra-bal/problem-${problem}-pre.txt" 20 1000 3 identity
done
```

Omit `taskset` where unavailable or choose an allowed CPU. The positional controls
are accepted-step limit, inner iteration limit, repetitions, `scaled` (default),
`block`, or `identity` preconditioning, and the relative inner tolerance (default
`1e-6`). Metadata goes
to stderr and per-run CSV to stdout. Optimization errors are recorded in CSV;
the runner exits successfully after reporting them. Invalid input or a failed
independent objective check produces an unsuccessful process exit.

SHA-256 of **decompressed** input files:

```text
96ca2845519d89d0727953d983427ab38a42c54991cd4d73e46a4221da3c61b4  problem-49-7776-pre.txt
58525e22314ccd72f5baec575f04cdd0be329dd90a23d82489108aef441af5b4  problem-73-11032-pre.txt
0921d835d31752af15cec103866535290cb40a9a7aa50102bda47c57ebd387f2  problem-138-19878-pre.txt
```

Checks require no downloaded data:

```sh
cargo test --test bal
cargo test --release --test bal
```

They cover malformed/truncated input, out-of-range indices, nonfinite numbers,
projection sign/distortion, zero and small rotations, additive analytical
Jacobians against central differences, and a synthetic batched graph that
actually converges with LM.

## Diagonal preconditioning follow-up

Measured on the same machine/toolchain and CPU on 2026-09-18, with the working-tree
diagonal-preconditioning and diagnostic changes accompanying this document.
No changes to dataset initialization, damping controls, objective, or iteration
budgets. The relative linear tolerance remains `1e-6`, now measured in scaled
coordinates. Each entry below is **one run**, not a median. The 49-camera run had
per-attempt stderr tracing enabled; the others printed final diagnostics only.
All include explicit residual recomputation inside solve timing.

The backend now solves in `z = D delta` coordinates with operator
`[J D^-1; sqrt(lambda) I]` and recovers the original step afterward. It computes
predicted reduction using the original model. Both f32/f64 tests compare damped
solutions against dense SVD, including mixed scales, zero columns, repeated
damping values, identity/scaled modes, and preserved undamped solves. Exhaustion
still returns an error. Warm LM allocation checks continue to pass.

### What the diagnostics established

For the first 49-camera linearization:

- Floored column norms span `1.2048834 .. 175163.9545` (about 145,378×).
- Identity mode at `lambda=0.001` exhausts 1,000 iterations, with estimated
  relative residual `8.1495995e-5`; explicit recomputation agrees.
- Scaled mode at the same damping converges in **284 iterations**, with estimated
  relative residual `9.98956736e-7`, explicitly recomputed scaled relative residual
  `9.98956736e-7`, and original-coordinate relative residual `8.77517281e-7`.
- The identity rerun still accepts no steps: 38.443400 s, 11 failed solves,
  11,000 inner iterations. At its last attempt (`lambda=1e12`), the recurrence
  and explicit relative residual both remain about `9.3334933e-4`.

The failed identity steps were finite. The recurrence residuals agreed with
explicit stationarity calculations: this is slow convergence, not an observed
NaN breakdown or false residual estimate. Right scaling resolves the initial
linear-solve barrier without changing the objective.

### End-to-end results

| Cameras | Solve (s) | Initial objective | Final objective | RMSE before → after (pixels) | Accepted / rejected | Inner iterations | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 49 | 133.370303 | 850912.460681 | 13441.722283 | 7.310557 → 0.918831 | 20 / 21 | 35442 | NoConvergence |
| 73 | 206.915158 | 963738.937218 | 17116.463572 | 6.464585 → 0.861525 | 20 / 23 | 37546 | NoConvergence |
| 138 | 350.523472 | 2185545.840628 | 59658.018689 | 7.161959 → 1.183277 | 20 / 20 | 33985 | NoConvergence |

Independent objective checks passed. All runs reached the configured 20 accepted
steps; `NoConvergence` is the outer iteration limit, not the former `NoProgress`.
Final gradient infinity norms were respectively `47.6125534`, `299.589042`, and
`278.203204`, still above the strict `1e-8` tolerance. These are progress results,
not converged solutions or a time-to-solution comparison. The old baseline's
shorter runtimes represent early failure.

| Cameras | Load (s) | Build (s) | VmHWM (KiB) | Linearizations / solves / costs | Last scaled relative residual | Last original relative residual |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 49 | 0.008224 | 0.002414 | 25772 | 21 / 41 / 21 | 9.91146222e-7 | 1.99286996e-6 |
| 73 | 0.012781 | 0.003835 | 35744 | 21 / 43 / 21 | 9.92416183e-7 | 1.48752090e-6 |
| 138 | 0.020813 | 0.006749 | 65584 | 21 / 40 / 22 | 9.90914256e-7 | 2.37333034e-6 |

Last linear residuals refer to the solve preceding the final accepted step;
final nonlinear gradients come from the fresh convergence-check linearization.
An inner scaled convergence result does not imply the same relative threshold
in original coordinates, or nonlinear convergence.

At lower damping, many solves still reach the inner iteration limit, so retries
increase damping to obtain a solvable step. Diagonal scaling is a useful first
fix, but block preconditioning or a carefully validated inexact-LM stopping
policy remain candidates to reduce this repeated work.

```sh
# One run each, same budgets as the original experiment:
for problem in 49-7776 73-11032 138-19878; do
  taskset -c 2 target/release/examples/bal \
    "/tmp/fagra-bal/problem-${problem}-pre.txt" 20 1000 1 scaled
done
# Inspect every attempt on the first problem; use identity for the control:
BAL_TRACE=1 taskset -c 2 target/release/examples/bal \
  /tmp/fagra-bal/problem-49-7776-pre.txt 20 1000 1 scaled
```

`Lsmr::diagnostics()` retains the last attempt's termination, finite-step status,
iterations, damping, floored scale bounds, faer recurrence residuals, and explicit
original/solver-coordinate residuals. `Lsmr::on_solve` optionally observes every
attempt without retaining a history or allocating library memory. Tracing I/O
is part of the measured call when enabled.

## Block-Jacobi and inner-tolerance sweep

Same datasets, hardware, release toolchain, CPU pinning and initialization as above.
24 runs: three datasets × two preconditioners × four tolerances. Each configuration
ran once in a fresh process, sequentially, with 20 accepted steps and at most 1,000
inner iterations per solve. This is an exploratory sweep, not a confidence interval
or a converged time-to-solution comparison. A preliminary 49-camera/block/1e-3
run (7.735562 s) is excluded; the table contains the subsequent complete sweep.
The implementation is the working-tree block-Jacobi/timing changes accompanying
this document, based on the library revision recorded at the top.

Block mode uses normalized per-variable Gram blocks and per-retry Cholesky
factorizations, with triangular applications through faer's native right
preconditioner interface. The full solve remains Jacobian-based. No adaptive
forcing policy or automatic acceptance of exhausted solves was introduced.

All runs returned `NoConvergence` after **20 accepted steps and 21 linearizations**;
independent final objective checks passed. The tolerance controls the normal
residual in the chosen preconditioner's coordinates, so identical numeric
tolerances do not imply identical original-coordinate accuracy. Nonlinear
gradient tolerance remains `1e-8`.

### Main comparison

| Cameras | Diagonal, 1e-3 (s) | Block, 1e-3 (s) | Runtime ratio | Final block RMSE (pixels) | Inner iterations, diagonal → block |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 49 | 81.438 | 7.864 | 10.4× | 0.915576 | 22130 → 1783 |
| 73 | 140.728 | 18.628 | 7.6× | 0.861097 | 25731 → 3045 |
| 138 | 159.256 | 36.431 | 4.4× | 1.180503 | 15470 → 3135 |

At `1e-6`, block mode took 44.492, 74.576, and 253.360 s versus diagonal's
132.240, 205.322, and 342.862 s (3.0×, 2.8×, and 1.4× faster). Block mode helps at
the same configured tolerance; relaxing the tolerance is a separate additional
gain. Final cost is comparable or lower in those pairs, but final gradients
vary and remain above convergence tolerance.

The fastest 20-step runs used block mode at `1e-2` (4.401, 17.091, 15.387 s).
However, `1e-3` reaches a better final cost on all three, particularly the
138-camera problem, and is a useful starting point for further experiments.
This does not justify changing the library's default tolerance for every model.

### Complete sweep

`5%` and `2%` are elapsed times to the first **accepted** state at or below that
fraction of the initial objective. The runner observes accepted costs via
`on_accept`; it does not interpolate between steps. Final costs are independently
recomputed. `NA` means the target was not reached within the budget, not that it
is unattainable. All 138-camera runs miss 2%. These coarse targets measure early
progress; they do not establish equal near-solution accuracy.

```csv
cameras,mode,tolerance,solve_s,final_cost,final_gradient,inner_iterations,rejected,target_5pct_s,target_2pct_s,peak_rss_kib
49,scaled,1e-2,63.138851,13377.56729455,141.392494,16980,10,0.067968,0.437370,25840
49,scaled,1e-3,81.438106,13354.28428372,790.301776,22130,9,0.208563,1.049449,25956
49,scaled,1e-4,125.061722,13413.82646997,152.814054,33733,20,0.507267,1.630314,26028
49,scaled,1e-6,132.240269,13441.72228344,47.6125534,35442,21,1.109104,3.506721,26008
49,block,1e-2,4.400985,13347.16754901,288.144466,943,2,0.066449,0.208008,27284
49,block,1e-3,7.863717,13346.65675979,282.244885,1783,0,0.124963,0.507040,27092
49,block,1e-4,24.065204,13346.57168376,262.367538,5930,0,0.282041,0.877070,27360
49,block,1e-6,44.492070,13346.55526324,281.004490,10848,0,0.809690,2.119950,27112
73,scaled,1e-2,93.251107,17099.82436002,9.25131005,16988,11,0.098800,0.467772,35832
73,scaled,1e-3,140.727604,17099.56329003,18.2536979,25731,14,0.271006,1.265886,35908
73,scaled,1e-4,167.996069,17100.03492027,15.5453514,30985,17,0.667007,2.346633,35784
73,scaled,1e-6,205.321874,17116.46357182,299.589042,37546,23,1.611222,4.721821,35784
73,block,1e-2,17.091252,17099.78371673,119.123516,2661,7,0.106456,0.281127,38084
73,block,1e-3,18.628195,17099.47197710,3.81995170,3045,1,0.180287,0.581300,37888
73,block,1e-4,30.444781,17099.47196988,1.69060888,4992,0,0.419916,1.282412,37780
73,block,1e-6,74.575717,17099.47197002,1.75191914,12472,0,1.137514,3.337981,37960
138,scaled,1e-2,79.566608,59582.17166673,424.421874,7760,7,0.888924,NA,65604
138,scaled,1e-3,159.255841,59381.19635964,26.8676970,15470,8,0.591649,NA,65424
138,scaled,1e-4,281.128175,59593.11768965,95.2895948,27464,15,1.451263,NA,65624
138,scaled,1e-6,342.861551,59658.01868896,278.203204,33985,20,3.200292,NA,65424
138,block,1e-2,15.386631,59528.12602157,135.092718,1243,3,0.191203,NA,68868
138,block,1e-3,36.430796,59378.63118544,272.465186,3135,5,0.423085,NA,68924
138,block,1e-4,73.752278,59531.00296426,345.720735,6586,4,0.882003,NA,68872
138,block,1e-6,253.359551,59382.03449768,35.3404288,22941,10,2.224191,NA,68752
```

At the common 2% target, block `1e-3` takes 0.507/0.581 s versus diagonal
`1e-3`'s 1.049/1.266 s on the first two datasets. This improvement is smaller
than the full 20-step runtime ratios: much of the saved work is in later solves.
The 73-camera diagonal `1e-2` run even reaches 5% slightly sooner than block
`1e-2`; block assembly has a real setup cost.

### Component timings

All sweep runs enable `collect_timings`. Measurements include clock/counter
overhead; per-attempt text tracing is disabled. Operator timers use synchronized
counters for faer's `Sync` trait contract. This instrumentation is opt-in for
library users. Times below are the `1e-3` runs; all units are seconds.

| Cameras / mode | Total | Prepare | Factor | Linear solve | Forward | Transpose | Precondition | Verify |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 49 / diagonal | 81.438106 | 0.205696 | 0.000011 | 80.921148 | 42.878630 | 32.376236 | 0 | 0.151335 |
| 49 / block | 7.863717 | 0.357698 | 0.021662 | 7.208066 | 3.509606 | 2.683154 | 0.533098 | 0.112947 |
| 73 / diagonal | 140.727604 | 0.309506 | 0.000015 | 139.908724 | 73.658934 | 55.839319 | 0 | 0.264192 |
| 73 / block | 18.628195 | 0.515487 | 0.031652 | 17.684609 | 8.624616 | 6.499547 | 1.281145 | 0.168265 |
| 138 / diagonal | 159.255841 | 0.550173 | 0.000036 | 157.850299 | 83.466825 | 63.081792 | 0 | 0.401526 |
| 138 / block | 36.430796 | 0.979346 | 0.069810 | 34.562733 | 16.943152 | 12.917681 | 2.457616 | 0.369668 |

Forward/transpose/preconditioner columns are **subsets** of linear-solve time;
do not add them to it. Their remainder includes faer recurrence/vector work,
scratch setup and timing overhead. The remainder of total time includes layout,
Jacobians, nonlinear cost/retraction, gradient checks and reporting hooks.
Preparation is column-norm plus optional normalized Gram assembly; Verify
includes recovery, prediction and explicit normal-residual checks.

At `1e-3`, linear solves consume about 92–95% of block-mode runtime. Triangular
preconditioning costs about 7% of the linear-solve time; most time remains in
Jacobian products. Setup costs and retained block matrices are modest on these
9D camera / 3D point workloads. Large custom variable dimensions can have much
higher O(sum d_i²) memory and O(sum d_i³) factorization costs.

### Reproduce and use

```sh
cargo build --release --example bal
for problem in 49-7776 73-11032 138-19878; do
  for mode in scaled block; do
    for tolerance in 1e-2 1e-3 1e-4 1e-6; do
      taskset -c 2 target/release/examples/bal \
        "/tmp/fagra-bal/problem-${problem}-pre.txt" 20 1000 1 "$mode" "$tolerance"
    done
  done
done
```

```rust,ignore
let mut backend = fagra::Lsmr::default();
backend.block_preconditioning = true; // Also enables diagonal normalization.
backend.relative_tolerance = 1e-3;     // Application-specific, not a new default.
backend.collect_timings = true;
let mut lm = fagra::LevenbergMarquardt::new(backend);
// graph.optimize_with(&mut lm, &options)?;
let timings = lm.backend().timings();
let diagnostics = lm.backend().diagnostics();
```

Checks cover dense-SVD agreement in both precisions, repeated damping, coupled
blocks, untouched coordinates, transpose consistency with multiple RHS columns,
singular/nonfinite factorization fallback, overlapping-range rejection, and zero
warm-call allocations in both diagonal and block modes. Defaults remain diagonal
preconditioning and the existing precision-aware inner tolerance. A Schur backend
or adaptive inexact-LM policy remains future work; neither is needed to obtain
the improvements measured here.
