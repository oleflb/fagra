use crate::storage::{PoolAccess, StatePool};
use crate::{KeyError, Real, StateKey, Variable};

/// An owned, schema-typed snapshot of state values and their identities.
///
/// Created by [`Problem::checkpoint_states`](crate::Problem::checkpoint_states).
/// Restore consumes the checkpoint. It contains no factors, priors, anchors, or
/// numerical workspaces. Values are copied with `Clone`, so variable authors must
/// ensure cloning preserves the value independently of subsequent mutations.
///
/// ```
/// use fagra::Problem;
/// fagra::states! { States {} }
/// fagra::factors! { Factors {} }
/// let mut graph = Problem::<States, Factors>::new();
/// let checkpoint = graph.checkpoint_states();
/// graph.restore_states(checkpoint)?;
/// # Ok::<(), fagra::StateCheckpointError>(())
/// ```
///
/// Checkpoints from different schema types cannot be mixed:
/// ```compile_fail
/// use fagra::Problem;
/// fagra::states! { First {} }
/// fagra::states! { Second {} }
/// fagra::factors! { Factors {} }
/// let first = Problem::<First, Factors>::new();
/// let mut second = Problem::<Second, Factors>::new();
/// second.restore_states(first.checkpoint_states()).unwrap();
/// ```
pub struct StateCheckpoint<S: SnapshotStates> {
    pub(crate) owner: CheckpointIdentity,
    pub(crate) values: S::Snapshot,
}

/// Why a state checkpoint cannot be restored. Rejection leaves all states intact.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum StateCheckpointError {
    /// The checkpoint was captured from another graph, even if both are empty.
    #[error("state checkpoint belongs to a different graph")]
    ForeignGraph,
    /// Live state identities no longer match the checkpoint.
    #[error("state membership changed since the checkpoint")]
    MembershipChanged,
}

/// State snapshot capability available for cloneable [`states!`](crate::states!) schemas.
///
/// Checkpoint calls require every declared variable type to implement `Clone`.
/// Schema declarations and [`Variable`] do not require it. Generic code can use
/// this bound to enable checkpoints.
pub trait SnapshotStates: StateSchema {
    #[doc(hidden)]
    type Snapshot;
    #[doc(hidden)]
    fn snapshot(&self) -> Self::Snapshot;
    #[doc(hidden)]
    fn validate_snapshot(&self, snapshot: &Self::Snapshot) -> Result<(), StateCheckpointError>;
    #[doc(hidden)]
    fn restore_snapshot(&mut self, snapshot: Self::Snapshot);
}

/// Identity allocated on first capture, including for empty schemas.
#[doc(hidden)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct CheckpointIdentity(crate::key::PoolId);

impl Default for CheckpointIdentity {
    fn default() -> Self {
        Self(crate::key::PoolId::new())
    }
}

// Nested tuples describe the pools without changing the schema's named-field storage.
/// Borrowed views of an arbitrary-length pool list; internal macro plumbing.
#[doc(hidden)]
pub trait StatePools {
    type Ref<'a>
    where
        Self: 'a;
    type Mut<'a>
    where
        Self: 'a;
}

impl StatePools for () {
    type Ref<'a> = ();
    type Mut<'a> = ();
}

impl<Head, Tail: StatePools> StatePools for (Head, Tail) {
    type Ref<'a>
        = (&'a Head, Tail::Ref<'a>)
    where
        Self: 'a;
    type Mut<'a>
        = (&'a mut Head, Tail::Mut<'a>)
    where
        Self: 'a;
}

/// Conditional snapshot traversal of a pool list; internal macro plumbing.
#[doc(hidden)]
pub trait SnapshotPools: StatePools {
    type Snapshot;
    fn snapshot(pools: Self::Ref<'_>) -> Self::Snapshot;
    fn validate(
        pools: Self::Ref<'_>,
        snapshot: &Self::Snapshot,
    ) -> Result<(), StateCheckpointError>;
    fn restore(pools: Self::Mut<'_>, snapshot: Self::Snapshot);
}

impl SnapshotPools for () {
    type Snapshot = ();
    fn snapshot(_: ()) {}
    fn validate(_: (), _: &()) -> Result<(), StateCheckpointError> {
        Ok(())
    }
    fn restore(_: (), _: ()) {}
}

impl<Head: crate::storage::StatePoolSnapshot, Tail: SnapshotPools> SnapshotPools for (Head, Tail) {
    type Snapshot = (Head::Snapshot, Tail::Snapshot);

    fn snapshot(pools: Self::Ref<'_>) -> Self::Snapshot {
        (pools.0.checkpoint(), Tail::snapshot(pools.1))
    }

    fn validate(
        pools: Self::Ref<'_>,
        snapshot: &Self::Snapshot,
    ) -> Result<(), StateCheckpointError> {
        pools.0.validate_checkpoint(&snapshot.0)?;
        Tail::validate(pools.1, &snapshot.1)
    }

    fn restore(pools: Self::Mut<'_>, snapshot: Self::Snapshot) {
        pools.0.restore_checkpoint(snapshot.0);
        Tail::restore(pools.1, snapshot.1);
    }
}

/// Statically selected, checked access to variables of type `T`.
///
/// Provided automatically for schemas declared with [`states!`](crate::states!).
/// Factor authors only need this read-only trait; typed pool access and bulk
/// traversal belong to the solver's internal interfaces.
pub trait StateStore<T: Variable> {
    /// Borrow the current estimate, rather than its linearization anchor.
    ///
    /// Returns an error for a removed, unknown, or foreign key.
    fn get(&self, key: StateKey<T>) -> Result<&T, KeyError>;
}

impl<S, T> StateStore<T> for S
where
    T: Variable,
    S: PoolAccess<StatePool<T>>,
{
    fn get(&self, key: StateKey<T>) -> Result<&T, KeyError> {
        <S as PoolAccess<StatePool<T>>>::pool(self).get(key)
    }
}

/// Static traversal of the state pools declared by [`states!`](crate::states!).
///
/// Internal macro plumbing. `Default` constructs empty pools without requiring
/// variables to implement `Default`.
pub trait StateSchema: Default {
    /// Scalar shared by every variable in this schema.
    type Scalar: Real;

    /// Visit every declared pool once, including empty pools, in declaration order.
    ///
    /// Stops at the first visitor error without rolling back earlier mutations.
    /// A solver must run an infallible rejection pass after failed trial staging.
    /// An empty schema returns `Ok(())` without calling the visitor.
    fn visit<V: StateVisitor<Self::Scalar>>(&mut self, visitor: &mut V) -> Result<(), V::Error>;
}

/// One statically dispatched solver pass over heterogeneous state pools.
///
/// Implement operations such as staging, acceptance, and rejection in ordinary
/// library code. Use [`std::convert::Infallible`] for passes that cannot fail.
/// Visitors receive pools, not individual values; typed iteration belongs to the pool.
pub trait StateVisitor<R: Real = f64> {
    /// Failure returned by this pass.
    type Error;

    /// Process one state family. Read-only passes may simply reborrow the pool.
    fn pool<T: Variable<Scalar = R>>(&mut self, pool: &mut StatePool<T>)
    -> Result<(), Self::Error>;
}

/// Declare a homogeneous state pool for each variable type.
///
/// Each type may appear once. Fields are private; access uses [`StateStore`].
/// The declaration generates `Default` without requiring variables to implement it.
/// Internally, fields are library-owned pools. The macro generates typed pool
/// access and one static traversal; the library supplies `StateStore::get`.
/// Traversal visits even empty pools in declaration order and stops on error.
/// A declaration `States<R> { values: Value<R> }` introduces one scalar parameter
/// bounded by [`Real`]. Every variable must have `Scalar = R`. Declarations
/// without a parameter use `f64`. Empty generic schemas are supported.
///
/// State and full-graph checkpoint capabilities are available when all listed types implement
/// `Clone`. Schemas containing non-cloneable values still support ordinary graph operations.
///
/// ```no_run
/// use fagra::{states, Variable, Tangent, Jacobian};
/// # use faer_ext::nalgebra::{Const, DefaultAllocator};
/// # struct Pose;
/// # impl Variable for Pose {
/// #     type Scalar = f64;
/// #     type Dim = Const<6>;
/// #     type Allocator = DefaultAllocator;
/// #     fn identity() -> Self { todo!() }
/// #     fn compose(&self, _: &Self) -> Self { todo!() }
/// #     fn inverse(&self) -> Self { todo!() }
/// #     fn exp(_: &Tangent<Self>) -> Self { todo!() }
/// #     fn log(&self) -> Tangent<Self> { todo!() }
/// #     fn adjoint(&self) -> Jacobian<Self> { todo!() }
/// #     fn right_jacobian(_: &Tangent<Self>) -> Jacobian<Self> { todo!() }
/// #     fn right_jacobian_inverse(_: &Tangent<Self>) -> Jacobian<Self> { todo!() }
/// # }
/// states! {
///     /// Variables used by this application.
///     SlamStates {
///         poses: Pose,
///     }
/// }
/// ```
#[macro_export]
macro_rules! states {
    ($(#[$attr:meta])* $vis:vis $name:ident<$scalar:ident> { $($fields:tt)* }) => {
        $crate::states!(@impl [$(#[$attr])* $vis $name] [$scalar] [$scalar] { $($fields)* });
    };
    ($(#[$attr:meta])* $vis:vis $name:ident { $($fields:tt)* }) => {
        $crate::states!(@impl [$(#[$attr])* $vis $name] [] [f64] { $($fields)* });
    };
    (@snapshot [$name:ident] [$($scalar:ident)?] {
        $($field:ident: $ty:ty),* $(,)?
    }) => {
        const _: () = {
            // The associated equality defers Clone checking, even for concrete non-Clone schemas.
            trait __FagraStateAccess {
                type Pools: $crate::__private::StatePools;
                fn pools(&self) -> <Self::Pools as $crate::__private::StatePools>::Ref<'_>;
                fn pools_mut(&mut self) -> <Self::Pools as $crate::__private::StatePools>::Mut<'_>;
            }

            impl$(<$scalar: $crate::Real>)? __FagraStateAccess for $name$(<$scalar>)? {
                type Pools = $crate::states!(@pool_types $($ty,)*);

                fn pools(&self) -> <Self::Pools as $crate::__private::StatePools>::Ref<'_> {
                    $crate::states!(@pool_refs self; $($field,)*)
                }

                fn pools_mut(&mut self) -> <Self::Pools as $crate::__private::StatePools>::Mut<'_> {
                    $crate::states!(@pool_muts self; $($field,)*)
                }
            }

            // A public opaque carrier keeps private variable types out of the associated type.
            #[doc(hidden)]
            pub struct __FagraStateSnapshot<T>(T);

            $crate::states!(@conditional [$name] [$($scalar)?]);
        };
    };
    (@conditional [$name:ident] [$scalar:ident]) => {
        $crate::states!(@conditional_impl [$name] [__Scalar]);
    };
    (@conditional [$name:ident] []) => {
        $crate::states!(@conditional_impl [$name] []);
    };
    (@conditional_impl [$name:ident] [$($scalar:ident)?]) => {
        impl<$($scalar: $crate::Real,)? __Pools> $crate::SnapshotStates for $name$(<$scalar>)?
        where
            Self: __FagraStateAccess<Pools = __Pools>,
            __Pools: $crate::__private::SnapshotPools,
        {
            type Snapshot = __FagraStateSnapshot<__Pools::Snapshot>;

            fn snapshot(&self) -> Self::Snapshot {
                __FagraStateSnapshot(__Pools::snapshot(__FagraStateAccess::pools(self)))
            }

            fn validate_snapshot(
                &self,
                snapshot: &Self::Snapshot,
            ) -> ::core::result::Result<(), $crate::StateCheckpointError> {
                __Pools::validate(__FagraStateAccess::pools(self), &snapshot.0)
            }

            fn restore_snapshot(&mut self, snapshot: Self::Snapshot) {
                __Pools::restore(__FagraStateAccess::pools_mut(self), snapshot.0);
            }
        }
    };
    (@pool_types) => { () };
    (@pool_types $ty:ty, $($rest:ty,)*) => {
        ($crate::__private::StatePool<$ty>, $crate::states!(@pool_types $($rest,)*))
    };
    (@pool_refs $self:ident;) => { () };
    (@pool_refs $self:ident; $field:ident, $($rest:ident,)*) => {
        (&$self.$field, $crate::states!(@pool_refs $self; $($rest,)*))
    };
    (@pool_muts $self:ident;) => { () };
    (@pool_muts $self:ident; $field:ident, $($rest:ident,)*) => {
        (&mut $self.$field, $crate::states!(@pool_muts $self; $($rest,)*))
    };
    (@impl [$(#[$attr:meta])* $vis:vis $name:ident] [$($scalar:ident)?] [$real:ty] {
        $($field:ident: $ty:ty),* $(,)?
    }) => {
        $(#[$attr])*
        #[allow(dead_code)]
        $vis struct $name$(<$scalar: $crate::Real>)? {
            $($field: $crate::__private::StatePool<$ty>,)*
            __fagra_scalar: ::core::marker::PhantomData<$real>,
        }

        impl$(<$scalar: $crate::Real>)? ::core::default::Default for $name$(<$scalar>)? {
            fn default() -> Self {
                Self {
                    $($field: ::core::default::Default::default(),)*
                    __fagra_scalar: ::core::marker::PhantomData,
                }
            }
        }

        impl$(<$scalar: $crate::Real>)? $crate::__private::StateSchema for $name$(<$scalar>)? {
            type Scalar = $real;

            fn visit<__Visitor: $crate::__private::StateVisitor<$real>>(
                &mut self,
                _visitor: &mut __Visitor,
            ) -> ::core::result::Result<(), __Visitor::Error> {
                $(_visitor.pool(&mut self.$field)?;)*
                ::core::result::Result::Ok(())
            }
        }

        $crate::states!(@access [$name $($scalar)?] $($field: $ty,)*);
        $crate::states!(@snapshot [$name] [$($scalar)?] { $($field: $ty,)* });
        $crate::__snapshot_schema!([$name $($scalar)?] {
            $($field: $crate::__private::StatePool<$ty>,)*
        });
    };
    (@access [$name:ident $($scalar:ident)?] $field:ident: $ty:ty, $($rest:tt)*) => {
        impl$(<$scalar: $crate::Real>)? $crate::__private::PoolAccess<$crate::__private::StatePool<$ty>> for $name$(<$scalar>)? {
            fn pool(&self) -> &$crate::__private::StatePool<$ty> {
                &self.$field
            }

            fn pool_mut(&mut self) -> &mut $crate::__private::StatePool<$ty> {
                &mut self.$field
            }
        }
        $crate::states!(@access [$name $($scalar)?] $($rest)*);
    };
    (@access [$name:ident $($scalar:ident)?]) => {};
}
