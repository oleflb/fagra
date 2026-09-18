# Marginalization: before and after buffer reuse

Measured on 2026-09-18, AMD Ryzen AI 9 HX PRO 370, Rust 1.98.1,
faer 0.24.4, release build, pinned to logical CPU 2. Kernels are sequential.

The baseline was the working-tree square-root marginalizer before this allocation
refactor, measured before changing its implementation. The final implementation
was then measured with the same workload, warm-up, sample count, and affinity.
These measurements include allocator instrumentation in both versions.

## Results

Times are microseconds per complete `marginalize` call. The reported statistic is
the median of nine sample means, each covering 20 successive calls.

| Factor mode | Separator DOF | Before (µs) | After (µs) | Speedup | Allocations/call before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary | 8 | 31.199 | 23.848 | 1.31× | 56 → 0 |
| Ordinary | 32 | 242.634 | 215.331 | 1.13× | 90 → 0 |
| Ordinary | 96 | 2068.836 | 1692.735 | 1.22× | 114 → 0 |
| Batched | 8 | 32.846 | 25.083 | 1.31× | 58 → 0 |
| Batched | 32 | 251.749 | 198.734 | 1.27× | 94 → 0 |
| Batched | 96 | 2211.758 | 1721.425 | 1.28× | 120 → 0 |

All 180 measured calls in each workload had the listed allocation count: minimum
and maximum were equal. Latency fell 11–24% on these workloads. Measurements are
single-machine comparisons, not a statistical confidence interval or a claim
about all graph topologies. CPU frequency and scheduling still introduce noise.

## Workload

`tests/marginalization_allocations.rs` advances a fixed window of `f64` scalar
states. Each step inserts one state, adds its unary prior and relative factors to
every existing window state, and marginalizes the oldest state. The separator
contains 8, 32, or 96 coordinates. An existing dense marginal prior is absorbed
and replaced on every steady-state call.

- The true trajectory is `x_t = t`; initial estimates are `t + 0.1`.
- Unary measurements keep the system constrained as the window advances.
- The ordinary and batched variants describe the same objective. The batched
  variant uses one persistent batch, exercising partial selection and compaction.
- Warm-up consists of twice the window size in marginalization calls, after
  initially filling the window.
- Measurement covers key selection/validation, graph metadata scanning, factor
  evaluation, both QR passes, reference capture, and graph publication/removal.
- Graph insertion, final optimization, and correctness assertions are outside
  the measured interval. This is marginalization latency, not total frame latency.
- Thread-local allocator instrumentation counts `alloc`, `alloc_zeroed`, and
  `realloc` calls; it does not count frees. Factor and geometry code is included.
- After each workload, optimization must recover every retained state within
  `1e-7` and a total objective below `1e-12`.

## What changed

1. Retain selection sets, both graph layouts, validation buffers, and batch swap
   logs instead of rebuilding their allocations on every call.
2. Recycle prior matrices/vectors and dependency lists on both success and failure.
   Keep active and pending storage separate to preserve transactional behavior.
3. Capture typed references directly into retained state-pool storage.
4. Preserve both row and column capacity high-water marks when matrix shapes change.

The elimination algorithm and rank tolerance are unchanged. No normal equations
were introduced. Metadata scanning and the dense affected front still determine
the asymptotic cost; this change removes allocation churn and prior-buffer clones.
The tradeoff is retaining peak workspace and recyclable prior capacity until the
solver is dropped.

## Allocation contract and regression checks

Allocation-free means **after warm-up, within retained capacities**, with
allocation-free user factors and geometry. Bigger graphs/fronts or more concurrent
priors may require growth. This does not promise allocation-free graph insertion,
arbitrary new workloads, or first-call execution.

Three normal tests verify zero allocations after warm-up:

- Repeated ordinary and partially selected batched windows.
- Alternating prior sizes and elimination to an empty separator.
- Failed reference capture after QR, graph rollback, and successful retry.

Existing checks also cover rank deficiency, SVD objective equivalence, both
precisions/backends, manifold derivatives, heterogeneous keys, and batch rollback.

## Reproduce

```sh
taskset -c 2 cargo test --release --test marginalization_allocations benchmark -- --ignored --nocapture --test-threads=1
cargo test --test marginalization_allocations
cargo test --release --test marginalization_allocations
```

Omit `taskset -c 2` on platforms without Linux CPU affinity support. The benchmark
prints CSV with timing and minimum/maximum allocation counts for each workload.
