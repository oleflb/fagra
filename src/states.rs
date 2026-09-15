use crate::storage::{PoolAccess, StatePool};
use crate::{KeyError, Real, StateKey, Variable};

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
