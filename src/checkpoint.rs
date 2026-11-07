//! Logical graph snapshots and rollback lineage.

use std::sync::{Arc, Weak};

use crate::{
    marginalization::PriorCheckpoint,
    states::{CheckpointIdentity, StatePools, StateSchema},
};

/// An owned full-graph checkpoint, consumed by [`crate::Problem::restore`].
///
/// Captures states, factors, batch models/payloads, marginal priors and anchors,
/// but not numerical workspaces. Available when every stored state, factor,
/// batch model and payload implements `Clone`; schema syntax stays unchanged.
/// Cloning must independently preserve all mutable data affecting evaluation;
/// shallow `Arc`/`Rc` clones of mutable data do not satisfy this contract.
/// Immutable sharing is allowed. External aliases are not themselves rolled back,
/// and independent clones need not preserve sharing relationships between objects.
///
/// Restoring consumes this checkpoint and invalidates all younger graph checkpoints.
/// Older checkpoints remain valid. Dropping a checkpoint simply discards it.
///
/// ```
/// use fagra::Problem;
/// fagra::states! { States {} }
/// fagra::factors! { Factors {} }
/// let mut graph = Problem::<States, Factors>::new();
/// let checkpoint = graph.checkpoint();
/// graph.restore(checkpoint)?;
/// # Ok::<(), fagra::GraphCheckpointError>(())
/// ```
pub struct GraphCheckpoint<S: StateSchema + SnapshotSchema, F: SnapshotSchema> {
    pub(crate) owner: CheckpointIdentity,
    pub(crate) token: Arc<()>,
    pub(crate) states: <S as SnapshotSchema>::Snapshot,
    pub(crate) factors: F::Snapshot,
    pub(crate) priors: PriorCheckpoint<S::Scalar>,
}

/// Why a full-graph rollback was rejected, before changing graph contents.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum GraphCheckpointError {
    /// The checkpoint belongs to another graph, including an empty graph.
    #[error("checkpoint belongs to a different graph")]
    ForeignGraph,
    /// An older checkpoint was already restored, abandoning this checkpoint's branch.
    #[error("checkpoint was invalidated by an older rollback")]
    Invalidated,
    /// A pool cannot reserve a generation above its identity high-water mark.
    #[error("checkpoint rollback generation space exhausted")]
    GenerationExhausted,
}

/// Full-graph snapshot capability available for cloneable state and factor schemas.
///
/// All stored values must implement `Clone` with independent mutable-value semantics.
/// Schema declarations and ordinary graph operations require neither this capability nor `Clone`.
pub trait SnapshotSchema {
    #[doc(hidden)]
    type Snapshot;
    #[doc(hidden)]
    fn snapshot_graph(&self) -> Self::Snapshot;
    #[doc(hidden)]
    fn preflight_restore(&self) -> Result<(), GraphCheckpointError>;
    #[doc(hidden)]
    fn restore_graph(&mut self, snapshot: Self::Snapshot);
}

/// Conditional full-graph snapshot traversal over nested pool views; macro plumbing.
#[doc(hidden)]
pub trait GraphSnapshotPools: StatePools {
    type Snapshot;
    fn snapshot(pools: Self::Ref<'_>) -> Self::Snapshot;
    fn preflight(pools: Self::Ref<'_>) -> Result<(), GraphCheckpointError>;
    fn restore(pools: Self::Mut<'_>, snapshot: Self::Snapshot);
}

impl GraphSnapshotPools for () {
    type Snapshot = ();
    fn snapshot(_: ()) {}
    fn preflight(_: ()) -> Result<(), GraphCheckpointError> {
        Ok(())
    }
    fn restore(_: (), _: ()) {}
}

impl<Head: SnapshotSchema, Tail: GraphSnapshotPools> GraphSnapshotPools for (Head, Tail) {
    type Snapshot = (Head::Snapshot, Tail::Snapshot);
    fn snapshot(pools: Self::Ref<'_>) -> Self::Snapshot {
        (pools.0.snapshot_graph(), Tail::snapshot(pools.1))
    }
    fn preflight(pools: Self::Ref<'_>) -> Result<(), GraphCheckpointError> {
        pools.0.preflight_restore()?;
        Tail::preflight(pools.1)
    }
    fn restore(pools: Self::Mut<'_>, snapshot: Self::Snapshot) {
        pools.0.restore_graph(snapshot.0);
        Tail::restore(pools.1, snapshot.1);
    }
}

pub(crate) struct Checkpoints {
    pub(crate) owner: CheckpointIdentity,
    live: Vec<Weak<()>>,
}

impl Checkpoints {
    pub(crate) fn new(owner: CheckpointIdentity) -> Self {
        Self {
            owner,
            live: Vec::new(),
        }
    }

    fn prune(&mut self) {
        self.live.retain(|token| token.strong_count() != 0);
    }

    pub(crate) fn capture(&mut self) -> Arc<()> {
        self.prune();
        let token = Arc::new(());
        self.live.push(Arc::downgrade(&token));
        token
    }

    pub(crate) fn position(&self, token: &Arc<()>) -> Result<usize, GraphCheckpointError> {
        let token = Arc::downgrade(token);
        self.live
            .iter()
            .position(|entry| entry.ptr_eq(&token))
            .ok_or(GraphCheckpointError::Invalidated)
    }

    pub(crate) fn restored(&mut self, position: usize) {
        self.live.truncate(position);
        self.prune();
    }
}

// A token's allocation is its non-reusable identity while any checkpoint holds it;
// weak registry entries prevent address reuse even after a checkpoint is dropped.

/// Internal generation-floor preflight shared by pool implementations.
pub(crate) fn floor<T>(pool: &crate::dense::DensePool<T>) -> Result<u32, GraphCheckpointError> {
    pool.restore_floor()
        .ok_or(GraphCheckpointError::GenerationExhausted)
}

#[doc(hidden)]
#[macro_export]
macro_rules! __snapshot_schema {
    ([$name:ident $($scalar:ident)?] { $($field:ident: $pool:ty,)* }) => {
        const _: () = {
            trait __FagraGraphAccess {
                type Pools: $crate::__private::StatePools;
                fn pools(&self) -> <Self::Pools as $crate::__private::StatePools>::Ref<'_>;
                fn pools_mut(&mut self) -> <Self::Pools as $crate::__private::StatePools>::Mut<'_>;
            }
            impl$(<$scalar: $crate::Real>)? __FagraGraphAccess for $name$(<$scalar>)? {
                type Pools = $crate::__snapshot_schema!(@pool_types $($pool,)*);
                fn pools(&self) -> <Self::Pools as $crate::__private::StatePools>::Ref<'_> {
                    $crate::states!(@pool_refs self; $($field,)*)
                }
                fn pools_mut(&mut self) -> <Self::Pools as $crate::__private::StatePools>::Mut<'_> {
                    $crate::states!(@pool_muts self; $($field,)*)
                }
            }
            #[doc(hidden)]
            pub struct __FagraGraphSnapshot<T>(T);

            $crate::__snapshot_schema!(@conditional [$name] [$($scalar)?]);
        };
    };
    (@conditional [$name:ident] [$scalar:ident]) => {
        $crate::__snapshot_schema!(@conditional_impl [$name] [__Scalar]);
    };
    (@conditional [$name:ident] []) => {
        $crate::__snapshot_schema!(@conditional_impl [$name] []);
    };
    (@conditional_impl [$name:ident] [$($scalar:ident)?]) => {
        impl<$($scalar: $crate::Real,)? __Pools> $crate::SnapshotSchema for $name$(<$scalar>)?
        where
            Self: __FagraGraphAccess<Pools = __Pools>,
            __Pools: $crate::__private::GraphSnapshotPools,
        {
            type Snapshot = __FagraGraphSnapshot<__Pools::Snapshot>;
            fn snapshot_graph(&self) -> Self::Snapshot {
                __FagraGraphSnapshot(__Pools::snapshot(__FagraGraphAccess::pools(self)))
            }
            fn preflight_restore(&self) -> ::core::result::Result<(), $crate::GraphCheckpointError> {
                __Pools::preflight(__FagraGraphAccess::pools(self))
            }
            fn restore_graph(&mut self, snapshot: Self::Snapshot) {
                __Pools::restore(__FagraGraphAccess::pools_mut(self), snapshot.0);
            }
        }
    };
    (@pool_types) => { () };
    (@pool_types $pool:ty, $($rest:ty,)*) => {
        ($pool, $crate::__snapshot_schema!(@pool_types $($rest,)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropped_tokens_are_pruned_and_live_descendants_cannot_rejoin() {
        let mut registry = Checkpoints::new(CheckpointIdentity::default());
        let oldest = registry.capture();
        for _ in 0..1_000 {
            drop(registry.capture());
            assert_eq!(registry.live.len(), 2);
        }
        let younger = registry.capture();
        assert_eq!(registry.live.len(), 2);
        registry.restored(registry.position(&oldest).unwrap());
        assert_eq!(
            registry.position(&younger),
            Err(GraphCheckpointError::Invalidated)
        );
        let new_branch = registry.capture();
        assert_eq!(
            registry.position(&younger),
            Err(GraphCheckpointError::Invalidated)
        );
        assert_eq!(registry.position(&new_branch), Ok(0));
    }
}
