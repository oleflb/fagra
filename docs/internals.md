# Internal storage and visitors

Application and factor authors can start with the [public quick start](../README.md).
This document describes the interfaces connecting generated schemas to the solver.

## Typed access and traversal

| Trait | Responsibility |
| --- | --- |
| `PoolAccess<P>` | Borrow one concrete pool with `pool` or `pool_mut` |
| `FactorStore<S, P>` | Route a payload-typed factor key for cost evaluation or removal |
| `StateSchema` | Traverse the declared state pools |
| `StateVisitor` | Implement one bulk state operation |
| `FactorSchema<S>` | Traverse factor families compatible with state schema `S` |
| `FactorVisitor<S>` | Implement one bulk ordinary/batched factor operation |

The macros generate library-owned `StatePool<T>`, `FactorPool<T>`, and
`BatchPool<B, P>` fields, empty initialization, typed access, and traversal.
`factors!` also generates factor-key routing. A blanket implementation provides
the public `StateStore<T>` for any schema with `PoolAccess<StatePool<T>>`; the
macro does not generate a separate `get` implementation for every state type.

These interfaces live under `fagra::__private` solely to support downstream
macro expansion. They are documented in source but hidden from normal rustdoc.
They are technically public, not sealed or protected by `doc(hidden)`, and are
not intended as application extension points. The storage entry for an individual
batch is private; `Batch<Model, Payload>` in declarations is only macro syntax.

Individual insertion and lookup use typed pool access, not traversal. Factor
cost/removal by key use `FactorStore<S, P>` because a payload-typed key does not
identify its batch model at the type level.

## Dense generational storage

States, ordinary factors, and batch models share one `DensePool<T>` implementation:

```text
values:      Vec<T>         packed live values
keys:        Vec<LocalKey>  parallel (slot, generation) pairs
slots:       Vec<Slot>      generation and current dense position
free_slots:  Vec<u32>       reusable logical slots
```

A key combines a unique 64-bit pool identity with a 32-bit slot and generation.
Typed handles, `BlockId`, and `FactorId` carry the same identity representation.
They implement equality and hashing independently of payload traits. Pool IDs
are assigned with a checked relaxed atomic counter at construction; lookups and
insertion need no atomics or runtime type registry. These are process-local
identities, not persistent serialized keys or addresses.

Checked lookup validates the pool, slot bounds, generation, and occupancy before
resolving the dense position. It requires a slot-table indirection. Bulk iteration
instead zips packed values with their local keys without resolving handles.
Removal uses `swap_remove` on the parallel arrays and repairs the moved entry's
slot mapping. It then advances the removed slot's generation; exhausted slots
are retired permanently. Pool IDs and index arithmetic never silently wrap.

Reservation covers all parallel buffers, including enough free-list capacity
for removal. Capacity is retained after deletion; pool insertion/removal within
reserved capacity and live iteration require no storage allocations. User payload
construction, evaluation, and destruction may have their own allocations.

Batch payloads remain contiguous within each batch, alongside their local keys.
A family-wide generational directory, also backed by `DensePool`, maps each
factor identity to a stable batch slot and dense payload position. Removing a
payload repairs its moved sibling's directory entry. Empty models stay reusable;
there is currently no public batch-removal operation.

Internal pool methods provide reservation, insertion, checked access/removal,
and live iteration. Batch pools additionally expose models and borrowed selections.
`FactorSelection` borrows contiguous slices of identities and payloads: a whole
batch for optimization, or one payload for individual cost evaluation. Its exact
remaining length and `as_slice()` access are O(1), with no allocation.

Solver insertion validates dependencies before publishing a factor, including
shared batch inputs. This currently uses state visitors: O(state families) per
dependency at edit time. A pool-routing index can replace this if insertion
throughput warrants it; the checked state-read path stays O(1). There is no
numerical cache or persistent incidence index yet. Public state deletion is
reserved for graph-aware operations so it cannot leave dangling dependencies.

## Traversal contract

- Every declared pool is visited once, even when empty, in declaration order.
- A batch visit receives one family of batch instances, not one payload or instance.
- Calls are statically dispatched and borrow pools mutably. Read-only passes may
  reborrow them immutably; evaluator traits still receive shared references.
- The first visitor error stops traversal. Earlier mutations are not rolled back.
- Empty schemas return success without invoking the visitor.
- Visitors choose their error type. Acceptance and rejection passes should use
  `std::convert::Infallible` so cleanup cannot fail.

For example, this pass counts declared state families without calling operational
stubs. It assumes `States` from the scalar-prior quick start:

```rust
use fagra::{Variable, __private::{StatePool, StateSchema, StateVisitor}};
use std::convert::Infallible;

struct CountFamilies(usize);

impl StateVisitor for CountFamilies {
    type Error = Infallible;

    fn pool<T: Variable>(&mut self, _: &mut StatePool<T>) -> Result<(), Infallible> {
        self.0 += 1;
        Ok(())
    }
}

let mut states = States::default();
let mut count = CountFamilies(0);
states.visit(&mut count).unwrap(); // Infallible error type.
assert_eq!(count.0, 1);
```

## Backend interoperability

The numerical backend uses faer. Application-side geometry and small Jacobians
use nalgebra through `faer_ext::nalgebra`, keeping their types compatible with
`faer_ext::IntoFaer`:

```rust
use faer_ext::IntoFaer;

fn consume(block: &fagra::JacobianBlock<'_>) {
    let matrix: faer::MatRef<'_, f64> = block.jacobian().into_faer();
    // Use matrix to accumulate into solver-owned numerical workspace.
}
```

Conversion copies only view metadata, preserving the data pointer, dimensions,
element strides, and immutable borrow lifetime. It allocates no coefficient
buffer and supports noncontiguous views. `into_faer` checks conversion of nalgebra's
unsigned strides to faer's signed strides and panics if they do not fit `isize`.

The sink consumes these views during emission. Persistent caches, assembled
systems, and mutable factorization scratch space need solver-owned storage;
zero-copy view conversion does not eliminate that workspace.

Cargo also declares nalgebra 0.34 solely to enable `std`, which faer-ext does not
forward. Cargo unifies this with faer-ext's dependency; all Rust imports still
go through the re-export. The checked sink rejects strides that cannot fit faer's
signed representation before invoking the conversion.

Faer enables only `std` and `linalg`; the sequential backends do not require its
default parallel, sparse-solver, random-generation, or NumPy I/O features.

## Optimizer and backend separation

`src/optimization.rs` contains the shared interfaces, layout, checked sink, and
state/cost visitors. Nonlinear methods live in child modules such as
`src/optimization/gauss_newton.rs`, which owns GN's workspace and iteration policy.
The dense Cholesky backend lives separately in `src/normal.rs`.

`Solver<S, F>` owns the graph and a retained default `GaussNewton` instance.
`optimize_with` accepts a caller-owned method and stopping controls. The hidden
`Optimizer<S, F>` interface separates nonlinear orchestration from graph ownership.
`LeastSquaresBackend` consumes validated residual/Jacobian emissions and resolved
column offsets, supplies a gradient norm, and returns a borrowed solution slice.
Its representation is unconstrained: QR can retain rows, while CG can use a matrix
or block operator. LM will need regularization and model-prediction extensions;
those methods and alternate algorithms are not implemented yet.

The first backend, `DenseNormalCholesky`, streams small dense block products into
one preallocated normal matrix's lower triangle and accumulates `b = -Jᵀr`.
Faer factorizes that matrix in place and overwrites `b` with delta. There is no
global Jacobian, mirrored upper triangle, fagra-owned block-product temporary,
matrix inverse, or separate copied step vector. Faer scratch and matrix/vector
capacities are retained; kernels may pack data internally.
The implementation uses sequential kernels and disables dynamic pivot regularization.
The dense allocation costs approximately `8 * n²` bytes; only the lower triangle
is cleared, accumulated, and read. Normal equations square J's condition number.

## Solver passes

Each optimization call rebuilds ordering and dependency metadata using retained
vectors and maps. This handles graph edits and reusing one method across unrelated
graphs without stale caches. State schema declaration order and each pool's dense
order determine scalar offsets; all stored states contribute their `DOF` coordinates.
No-variable problems return their evaluated cost without a linear solve.

GN then uses ordinary library visitor implementations:

1. **Linearize:** visit factor families, establish ordinary-factor scopes, and
   invoke `Factor::linearize` or `FactorBatch::linearize` with batch selections.
2. **Solve:** check gradient convergence, then use faer Cholesky to compute delta;
   stop before applying an already-small step. No schema visitor is needed for the solve.
3. **Stage:** visit state pools, take each pool's prepared delta range, convert
   tangents, and retract into trial storage without destroying accepted values.
4. **Evaluate:** visit factor families to sum trial costs through `Factor::cost`
   and `FactorBatch::cost`. `StateStore::get` exposes trial values during this pass.
5. **Accept or reject:** commit each finite full step, including uphill steps;
   discard trials if staging or evaluation fails. There is no damping or line search.

The shared sink enforces factor scope ownership/completeness, dependency membership,
matrix dimensions, finite coefficients, and representable strides. Duplicate
variable blocks in a single emission are rejected. Ordered scopes and blocks
take a comparison-based fast path; reordered emissions use the prepared maps.
Per-emission markers reject duplicate variables without allocating a set. Metadata
buffers are sized during preparation, never grown by valid residual emissions.

Each state pool retains a second value buffer, reserved to at least the accepted
buffer's capacity. Retraction writes new trial values without `T: Clone`.
Acceptance swaps the value buffers while leaving identity metadata untouched,
then drops old estimates. A guard discards all partial trials on evaluation errors
or retraction/evaluator unwinding. The cost visitor always sees the trial estimates
during evaluation and the accepted estimates after cleanup.

Staging or trial evaluation failure must trigger rejection across **all** pools
before returning an error, including when staging stopped partway through.
Previously accepted iterations remain accepted. The optimizer owns convergence,
damping, and acceptance rules; visitors only implement the passes.

The hot path reuses workspace after preparation. Tests count allocations for a
warmed optimization that performs actual steps, including a release workload with
1,000 DOFs. User evaluators, retractions, and destructors must also avoid allocation
to satisfy the end-to-end contract. Structural growth may allocate. Controls are
documented on `OptimizeOptions`; failures preserve previously accepted iterations.

Marginalization's manifold-coordinate contract and heterogeneous bulk API remain
open; `marginalize()` is still a stub. There is no incremental relinearization cache.
