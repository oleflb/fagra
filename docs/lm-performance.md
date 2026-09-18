# Gauss–Newton versus Levenberg–Marquardt

These are historical measurements with identity preconditioning. Damped LSMR
now defaults to diagonal right-preconditioning and recomputes residual diagnostics;
the current benchmark command therefore measures the updated implementation.
See [BAL follow-up](bal-performance.md#diagonal-preconditioning-follow-up) for
measurements of the new solver on real data.

Measured on 2026-09-18 with an AMD Ryzen AI 9 HX PRO 370, Rust 1.98.1,
faer 0.24.4, release build, sequential kernels, pinned to logical CPU 2.

Both methods use the same cached-block LSMR backend and its default relative
tolerance (`1e-10`) and inner iteration limit (1000). Both use identity
preconditioning. LM adds column-norm-scaled damping with default `LmOptions`;
it does not change coordinate scaling or the physical objective.

## End-to-end optimization results

Each entry is 31 measured optimization calls after three warm-ups. Units are
microseconds. p95 is the empirical nearest-rank percentile of those 31 calls,
not a worst-case bound. Success requires an `Ok` result, independently measured
maximum state error below `1e-6`, finite cost, and gradient infinity norm at most
`1.01e-8`.

| Problem | DOF | GN median / p95 (µs) | LM median / p95 (µs) | GN successes | LM successes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Square, near solution | 32 | 134.183 / 153.409 | 143.952 / 219.003 | 31/31 | 31/31 |
| Square, near solution | 256 | 569.504 / 878.126 | 628.925 / 730.277 | 31/31 | 31/31 |
| Atan, poor initialization | 32 | 659.183 / 751.066 | 334.962 / 393.451 | 0/31 | 31/31 |
| Atan, poor initialization | 256 | 16330.519 / 18946.191 | 1443.572 / 1674.978 | 0/31 | 31/31 |
| Log, invalid full step | 32 | 73.929 / 78.268 | 689.360 / 1044.529 | 0/31 | 31/31 |
| Log, invalid full step | 256 | 307.910 / 353.436 | 2693.268 / 2860.944 | 0/31 | 31/31 |

On the easy square problems, LM took the same three accepted steps and was about
7% and 10% slower. Both methods met the same accuracy requirement; GN's residual
error happened to be smaller at termination. All measured calls allocated zero
heap memory, including rejected LM trials and the unsuccessful GN calls.

On atan, GN moved far from the solution into residual saturation and returned
`Ok` through its existing stopping rules; independent checks rejected those
answers (maximum state errors about `1.7e10` and `1.8e10`). This is why reporting
API success or execution time alone is misleading. LM converged in every run.
On log, GN's first trial left the positive domain and returned an error, retaining
the initial state. Its shorter runtime is time to failure, not a performance win.

## Work performed

Counts below correspond to the median-time run. A Jacobian pass includes all
factors; cost counts include the initial evaluation and attempted invalid trials.

| Problem | DOF | Method | Jacobian passes | Cost evaluations | Linear solves | Inner iterations | Rejections |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Square | 32 / 256 | GN | 4 | 4 | 3 | 42 | 0 |
| Square | 32 / 256 | LM | 4 | 4 | 3 | 42 | 0 |
| Atan | 32 | GN | 7 | 8 | 7 | 238 | 0 |
| Atan | 256 | GN | 7 | 8 | 7 | 1572 | 0 |
| Atan | 32 | LM | 8 | 12 | 11 | 95 | 4 |
| Atan | 256 | LM | 8 | 12 | 11 | 94 | 4 |
| Log | 32 / 256 | GN | 1 | 2 | 1 | 24 | 0 |
| Log | 32 | LM | 8 | 13 | 12 | 226 | 5 |
| Log | 256 | LM | 8 | 13 | 12 | 227 | 5 |

LM's rejection counts do not add Jacobian passes. Damping also changes LSMR
convergence: more solves need not imply more total inner iterations.

Across all successful LM runs, maximum state error was below `1.1e-9`, maximum
gradient norm below `9.4e-10`, and cost below `4.6e-17`.

## Reproducible workload

`tests/levenberg_marquardt.rs` builds scalar chains with targets
`t_i = 0.8 + 0.025 * (i % 17)`, plus neighboring residuals
`0.1 * (x_i - x_(i-1) - t_i + t_(i-1))`. The unary residuals and starts are:

| Problem | Unary residual | Initial value |
| --- | --- | --- |
| Square | `x_i² - t_i²` | `1.05 * t_i` |
| Atan | `atan(x_i - t_i)` | `t_i + 2 + 0.1 * sin(i)` |
| Log | `ln(x_i / t_i)` | `10 * t_i` |

The intended solution is `x_i = t_i`, with zero objective. Each sample uses a fresh
graph with identical initial estimates. Graph construction and a linear reset pass
that prepares graph-owned trial capacity happen outside timing. The measured
optimizer persists across graphs, retaining its workspace. This tests warm
capacity and safe graph switching; it is not a complete sensor/fixed-lag replay.

Timing covers the entire `optimize_with` call: layout preparation, linearizations,
solves, trial evaluation, rejection, acceptance, and final convergence checks.
Independent correctness checks run afterward. Thread-local allocator instrumentation
counts allocations and reallocations on the measured thread for both methods.
Nonlinear controls are identical: at most 100 accepted steps, gradient tolerance
`1e-8`, and step/cost tolerances zero. The latter disable LM stagnation detection
and GN's nonzero small-step/cost thresholds for this accuracy-focused comparison.

These are small synthetic workloads. The easy square case is a near-solution
proxy, not a measured real-world warm-start sequence. There is no statistical
confidence interval; frequency scaling and scheduling affect the timing tails.
Production sensor geometry, preconditioning, and separator size can change the
relative costs substantially.

## Reproduce and verify

```sh
taskset -c 2 cargo test --release --test levenberg_marquardt benchmark -- --ignored --nocapture --test-threads=1
cargo test --test levenberg_marquardt
cargo test --release --test levenberg_marquardt
cargo test --lib damped_retries
```

Normal tests verify both precisions, retry caching, invalid-domain recovery,
overdamping, bounded inner-solve failures, preservation of accepted progress,
marginal prior handling, and zero warm-call allocations. A separate backend check
compares augmented solves and undamped predictions with an independent dense SVD.
