# Small-row dense normal assembly

Measured on 2026-09-20 with an AMD Ryzen AI 9 HX PRO 370, Rust 1.98.1,
faer 0.24.4, and release builds. Runs were sequential, without CPU affinity;
no other tests or benchmarks were launched concurrently with the timings.

`DenseNormalCholesky` now uses direct dot products for residuals with one
through four rows, with a static row count for compiler unrolling. Larger
residuals retain the existing faer products. Assembly still consumes each
observation separately, including its already-applied robust weight, and writes
only the lower triangle. There are no new allocations or dependencies.

`LeastSquaresBackend::accumulate<Rows: Dim>` takes a borrowed
`VectorView<'_, R, Rows>`. The checked sink preserves the factor's row dimension:
`Const<N>` selects the kernel during monomorphization, while `Dyn` retains
runtime small-row dispatch for marginal priors and cached covariance rows.
Both paths share the same dimension-generic direct kernel. LSMR and Schur
consume the view's contiguous slice internally.

## Assembly microbenchmark

`normal::tests::small_row_timings` assembles 100 separate observations with
six Jacobian blocks of widths `[6, 6, 6, 6, 3, 4]`, matching
`localization-fagra`'s four pose controls, field alignment, and intrinsics.
Inputs are deterministic, preweighted, and constructed outside timing.
Each pass clears a preallocated 31-coordinate backend, accumulates all
observations, and exposes the matrix/RHS to `black_box`; it does not solve.
The row-count sweep uses synthetic Jacobians, not camera derivatives.

The initial two-kernel benchmark used 20 warmups per kernel and seven samples
each timing 100 passes, alternating kernel order between samples. The table
gives median microseconds per pass before the dimension-preserving refactor.

| Rows | f32 faer | f32 direct | Speedup | f64 faer | f64 direct | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 242.78 | 88.52 | 2.74× | 215.66 | 86.95 | 2.48× |
| 2 | 279.95 | 80.76 | 3.47× | 369.29 | 73.95 | 4.99× |
| 3 | 291.14 | 91.50 | 3.18× | 373.75 | 92.20 | 4.05× |
| 4 | 313.17 | 123.22 | 2.54× | 373.41 | 95.48 | 3.91× |
| 8 | 330.13 | 211.36 | 1.56× | 410.59 | 162.19 | 2.53× |
| 16 | 426.47 | 539.17 | 0.79× | 510.29 | 365.19 | 1.40× |
| 32 | 624.39 | 942.82 | 0.66× | 682.88 | 903.28 | 0.76× |

An earlier build/run measured two-row speedups of 3.49× / 4.44× (f32 / f64),
but eight-row f32 direct accumulation took 633.57 µs versus 326.04 µs for
faer. The cutoff stays conservatively at four: one through four improved in
both runs and precisions. These measurements do not establish a universal
crossover for other hardware or block widths.

After the refactor, the benchmark also measures the actual `accumulate` entry
point with `Const<N>` and `Dyn`. It rotates all four paths between samples and
passes the dynamic row count through `black_box`. Warmup and sample counts
are unchanged. Selected medians in microseconds per pass:

| Scalar | Rows | faer | Direct kernel | Static dispatch | Dynamic dispatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| f32 | 2 | 281.62 | 76.35 | 75.31 | 77.58 |
| f64 | 2 | 363.51 | 71.08 | 75.05 | 75.20 |
| f32 | 4 | 299.83 | 123.70 | 125.14 | 100.57 |
| f64 | 4 | 370.24 | 105.56 | 105.38 | 104.24 |
| f32 | 8 | 332.91 | 387.49 | 328.54 | 323.65 |
| f64 | 8 | 428.26 | 173.93 | 409.41 | 403.65 |

Static and dynamic dispatch retain the two-row speedup and use faer above
four rows. Timings vary with compilation and machine state; these results
do not establish an additional speedup from preserving the dimension type.

```sh
cargo test --release --lib normal::tests::small_row_timings -- --ignored --nocapture --test-threads=1
```

## User-crate workload

`localization-fagra/tests/spline_performance.rs::factor_workload::all_factor_timings`
measures cost, Jacobians, and dense assembly for 138 factors, including 100
two-row reprojections. Its gradient tolerance stops the pass before solving.
It takes seven samples of 20,000 passes after 1,000 warmups.

| 138-factor f64 pass | Time | Speedup over baseline |
| --- | ---: | ---: |
| Original faer-only assembly | 664.38 µs | 1.00× |
| Small-row assembly, slice interface | 376.57 µs | 1.76× |
| Small-row assembly, typed view interface | 374.88 µs | 1.77× |

This is a 43.6% time reduction for the full evaluation/assembly pass, not a
measurement of total optimization time. Before/after are separate runs using
the same working tree apart from the normal-assembly change.

Run from `/home/ole/hulk-stuff/hulk` (the crate's dependency points at this checkout):

```sh
cargo test -p localization-fagra --release --test spline_performance all_factor_timings -- --ignored --nocapture --test-threads=1
cargo test -p localization-fagra --release --test spline_performance
```

## Checks

- `cargo test --features test-support`: passes, including robust observations,
  solvers, covariance, and marginalization.
- `cargo test --release --lib normal::tests`: passes. The dense-reference check
  covers f32/f64, zero through five rows and larger residuals, both kernels and
  static/dynamic dispatch,
  strided views, shuffled blocks, empty blocks, repeated weighted observations,
  and an untouched NaN-filled upper triangle.
- User-crate warmed allocation checks: pass for f32 and f64.
- Strict Clippy reports existing `assertions_on_constants` and
  `field_reassign_with_default` warnings outside `normal.rs`; checking library
  and tests with only those two lints allowed passes.
