use std::{
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PoolId(NonZeroU64);

impl PoolId {
    pub(crate) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self::allocate(&NEXT)
    }

    fn allocate(counter: &AtomicU64) -> Self {
        let id = counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("pool identity space exhausted");
        Self(NonZeroU64::new(id).expect("pool identities must be nonzero"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LocalKey {
    pub(crate) slot: u32,
    pub(crate) generation: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RawKey {
    pub(crate) pool: PoolId,
    pub(crate) local: LocalKey,
}

/// A typed state handle issued by [`Problem::add`](crate::Problem::add).
///
/// Identity is solver-local and survives pool growth and dense-storage moves.
/// Keys are process-local identities, not persistent addresses or matrix offsets.
///
/// ```compile_fail
/// use fagra::StateKey;
/// struct Pose;
/// struct Landmark;
/// fn expects_pose(_: StateKey<Pose>) {}
/// fn wrong_kind(landmark: StateKey<Landmark>) {
///     expects_pose(landmark);
/// }
/// ```
pub struct StateKey<T> {
    pub(crate) raw: RawKey,
    _private: PhantomData<fn() -> T>,
}

impl<T> StateKey<T> {
    /// Return the stable numerical identity, independent of elimination ordering.
    pub fn block_id(self) -> BlockId {
        BlockId(self.raw)
    }
}

/// A typed handle to one ordinary or batched factor.
///
/// Removing a sibling factor does not invalidate this handle.
pub struct FactorKey<T> {
    pub(crate) raw: RawKey,
    _private: PhantomData<fn() -> T>,
}

impl<T> FactorKey<T> {
    /// Stable identity used by change tracking and numerical factor caches.
    pub fn factor_id(self) -> FactorId {
        FactorId(self.raw)
    }
}

/// A typed handle to one shared evaluator and its factor storage.
///
/// The type parameter is the batch model, not its factor payload.
pub struct BatchKey<T> {
    pub(crate) raw: RawKey,
    _private: PhantomData<fn() -> T>,
}

// Deriving these traits on generic handles would impose unnecessary bounds on T.
macro_rules! impl_key {
    ($($name:ident),*) => {$(
        impl<T> $name<T> {
            pub(crate) fn from_raw(raw: RawKey) -> Self {
                Self { raw, _private: PhantomData }
            }
        }

        impl<T> Copy for $name<T> {}

        impl<T> Clone for $name<T> {
            fn clone(&self) -> Self { *self }
        }

        impl<T> PartialEq for $name<T> {
            fn eq(&self, other: &Self) -> bool { self.raw == other.raw }
        }

        impl<T> Eq for $name<T> {}

        impl<T> Hash for $name<T> {
            fn hash<H: Hasher>(&self, state: &mut H) { self.raw.hash(state); }
        }

        impl<T> fmt::Debug for $name<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($name)).field(&self.raw).finish()
            }
        }
    )*};
}

impl_key!(StateKey, FactorKey, BatchKey);

/// Opaque variable identity used by the numerical backend.
///
/// Obtained from [`StateKey::block_id`]; never a dense index or scalar matrix offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub(crate) RawKey);

/// Opaque identity tagging one factor's linearization emissions.
///
/// Supplied by [`FactorSelection`](crate::FactorSelection), not constructed by evaluators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FactorId(pub(crate) RawKey);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_never_wrap_and_handles_stay_compact() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(PoolId::allocate(&counter).0.get(), u64::MAX - 1);
        for _ in 0..2 {
            assert!(std::panic::catch_unwind(|| PoolId::allocate(&counter)).is_err());
            assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        }
        assert_eq!(std::mem::size_of::<RawKey>(), 16);
        assert_eq!(std::mem::size_of::<StateKey<()>>(), 16);
        assert_eq!(std::mem::size_of::<FactorKey<()>>(), 16);
        assert_eq!(std::mem::size_of::<BatchKey<()>>(), 16);
    }
}
