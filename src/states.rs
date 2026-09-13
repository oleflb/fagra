use crate::storage::{PoolAccess, StatePool};
use crate::{KeyError, StateKey, Variable};

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
    /// Visit every declared pool once, including empty pools, in declaration order.
    ///
    /// Stops at the first visitor error without rolling back earlier mutations.
    /// A solver must run an infallible rejection pass after failed trial staging.
    /// An empty schema returns `Ok(())` without calling the visitor.
    fn visit<V: StateVisitor>(&mut self, visitor: &mut V) -> Result<(), V::Error>;
}

/// One statically dispatched solver pass over heterogeneous state pools.
///
/// Implement operations such as staging, acceptance, and rejection in ordinary
/// library code. Use [`std::convert::Infallible`] for passes that cannot fail.
/// Visitors receive pools, not individual values; typed iteration belongs to the pool.
pub trait StateVisitor {
    /// Failure returned by this pass.
    type Error;

    /// Process one state family. Read-only passes may simply reborrow the pool.
    fn pool<T: Variable>(&mut self, pool: &mut StatePool<T>) -> Result<(), Self::Error>;
}

/// Declare a homogeneous state pool for each variable type.
///
/// Each type may appear once. Fields are private; access uses [`StateStore`].
/// The declaration generates `Default` without requiring variables to implement it.
/// Internally, fields are library-owned pools. The macro generates typed pool
/// access and one static traversal; the library supplies `StateStore::get`.
/// Traversal visits even empty pools in declaration order and stops on error.
///
/// ```no_run
/// use fagra::{states, Variable};
/// # struct Pose;
/// # impl Variable for Pose {
/// #     type Tangent = [f64; 6];
/// #     const DOF: usize = 6;
/// #     fn tangent_from_slice(_: &[f64]) -> Self::Tangent { todo!() }
/// #     fn retract(&self, _: &Self::Tangent) -> Self { todo!() }
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
    ($(#[$attr:meta])* $vis:vis $name:ident {
        $($field:ident: $ty:ty),* $(,)?
    }) => {
        $(#[$attr])*
        #[derive(Default)]
        #[allow(dead_code)]
        $vis struct $name {
            $($field: $crate::__private::StatePool<$ty>,)*
        }

        impl $crate::__private::StateSchema for $name {
            fn visit<V: $crate::__private::StateVisitor>(
                &mut self,
                _visitor: &mut V,
            ) -> ::core::result::Result<(), V::Error> {
                $(_visitor.pool(&mut self.$field)?;)*
                ::core::result::Result::Ok(())
            }
        }

        $(impl $crate::__private::PoolAccess<$crate::__private::StatePool<$ty>> for $name {
            fn pool(&self) -> &$crate::__private::StatePool<$ty> {
                &self.$field
            }

            fn pool_mut(&mut self) -> &mut $crate::__private::StatePool<$ty> {
                &mut self.$field
            }
        })*
    };
}
