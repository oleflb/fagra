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
`FactorSelection` supports contiguous ranges and prepared index lists. Indices
are validated and their order is preserved; scheduling callers select each factor
once. Its exact remaining length is O(1). Testing contiguity of an indexed suffix
is O(selected entries), and never scans unrelated payloads.

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
go through the re-export. The solver's numerical operations remain API stubs.

## Intended solver passes

`Solver::optimize` will use ordinary library visitor implementations:

1. **Linearize:** visit factor families, establish ordinary-factor scopes, and
   invoke `Factor::linearize` or `FactorBatch::linearize` with batch selections.
2. **Solve:** use the numerical backend to compute delta; no schema visitor is needed.
3. **Stage:** visit state pools, map each `BlockId` to its delta range, convert
   tangents, and retract into trial storage without destroying accepted values.
4. **Evaluate:** visit factor families to sum trial costs through `Factor::cost`
   and `FactorBatch::cost`. `StateStore::get` exposes trial values during this pass.
5. **Accept or reject:** visit all state pools to commit or discard trial values.

Staging or trial evaluation failure must trigger rejection across **all** pools
before returning an error, including when staging stopped partway through.
Previously accepted iterations remain accepted. The optimizer owns convergence,
damping, and acceptance rules; visitors only implement the passes.

The intended hot path reuses workspace after structure preparation. Graph edits
may allocate; user evaluators must also avoid allocations. Marginalization's
manifold-coordinate contract and heterogeneous bulk API are still open.

Storage and identity management are implemented. Numerical assembly, trial-state
storage, optimization, and marginalization are the remaining solver work.
