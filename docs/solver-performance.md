# Dense Cholesky versus cached-block LSMR

Measured on 2026-09-13 with an AMD Ryzen 9 9900X, Rust 1.98.1, faer 0.24.4,
and `cargo --release`. The process was pinned to logical CPU 2; both backends
use sequential kernels.

## Workload and measurement

`compare_cholesky_lsmr` in `tests/optimizer.rs` builds scalar-variable chains
with unit-weight neighboring differences and unary priors. Two anchor spacings
exercise different conditioning:

- Every variable has a prior: `2n - 1` scalar residuals, well-conditioned.
- Every 32nd variable has a prior, starting at zero:
  `n - 1 + ceil(n / 32)` residuals, more weakly constrained.

Prior targets are deterministic pseudorandom values. Each call negates the
targets before timing, forcing a real solve rather than checking an already
converged graph. The same target sequence is used for both backends. Uniform
targets are avoided because they make the fully anchored chain trivial for LSMR.

Times are medians of nine complete `optimize_with` calls after three warmups.
They include ordering preparation, evaluation, linearization, assembly/caching,
linear solving, retraction, and convergence checks. Graph construction and
target changes are excluded. All solver tolerances are defaults: LSMR relative
tolerance `1e-10`, maximum 1000 inner iterations; GN gradient tolerance `1e-8`.

Every measured call accepted exactly one GN step and allocated zero heap memory.
Final gradients were independently checked against `1e-8`, and the two solvers'
estimates differed by at most `2.20e-9` across these workloads.

## Results

All times are milliseconds. Speedup is Cholesky time divided by LSMR time;
values below one favor Cholesky. Small-problem timings are particularly sensitive
to CPU frequency and system noise; crossover ranges are approximate.

### Prior on every variable

| Variables / DOF | Dense Cholesky | LSMR | LSMR speedup |
| ---: | ---: | ---: | ---: |
| 10 | 0.00710 | 0.01175 | 0.60× |
| 30 | 0.02285 | 0.03481 | 0.66× |
| 100 | 0.09311 | 0.09922 | 0.94× |
| 300 | 0.4184 | 0.2030 | 2.06× |
| 1,000 | 4.748 | 0.6601 | 7.19× |
| 3,000 | 119.08 | 1.7115 | 69.57× |
| 5,000 | 520.84 | 3.3045 | 157.62× |

### Prior on every 32nd variable

| Variables / DOF | Dense Cholesky | LSMR | LSMR speedup |
| ---: | ---: | ---: | ---: |
| 10 | 0.00820 | 0.01696 | 0.48× |
| 30 | 0.02412 | 0.03616 | 0.67× |
| 100 | 0.06813 | 0.2185 | 0.31× |
| 300 | 0.3192 | 0.8702 | 0.37× |
| 1,000 | 4.539 | 2.3853 | 1.90× |
| 3,000 | 115.18 | 7.4966 | 15.37× |
| 5,000 | 509.82 | 12.0579 | 42.28× |

LSMR wins between 100–300 variables on the fully anchored chain, and between
300–1,000 with sparse anchors. Dense Cholesky has O(n²) matrix storage and
O(n³) factorization; LSMR traverses O(n) cached coefficients per iteration on
these chains. Its iteration count depends on conditioning. At 5,000 variables,
the dense normal matrix alone occupies approximately 200 MB (decimal).

These results compare the current **dense** Cholesky implementation. The chain's
normal matrix is tridiagonal; a sparse or banded direct solver would have different
scaling. Results also need not carry over to graphs with large dense blocks,
different connectivity, or expensive nonlinear evaluations.

## Reproduce

```sh
taskset -c 2 cargo test --release --test optimizer compare_cholesky_lsmr -- --ignored --nocapture --test-threads=1
```

Omit `taskset -c 2` on systems without Linux CPU affinity support. The benchmark
prints CSV rows, including solution differences and LSMR gradient norms.

The first run exposed an upstream scratch-size underestimate at 300 variables.
`src/lsmr.rs` now reserves the two omitted `n × 1` vectors (`wbar` and `vold`),
with a regression test. All results above include that correction.
