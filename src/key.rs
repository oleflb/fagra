use std::marker::PhantomData;

/// A typed state handle issued by [`Solver::add`](crate::Solver::add).
///
/// Identity is solver-local and survives dense-storage moves.
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
#[derive(Debug)]
pub struct StateKey<T> {
    _private: PhantomData<fn() -> T>,
}

impl<T> Copy for StateKey<T> {}

impl<T> Clone for StateKey<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> StateKey<T> {
    /// Return the stable numerical identity, independent of elimination ordering.
    pub fn block_id(self) -> BlockId {
        todo!("API only: variable block identity")
    }
}

/// A typed handle to one ordinary or batched factor.
///
/// Removing a sibling factor does not invalidate this handle.
#[derive(Debug)]
pub struct FactorKey<T> {
    _private: PhantomData<fn() -> T>,
}

impl<T> Copy for FactorKey<T> {}

impl<T> Clone for FactorKey<T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A typed handle to one shared evaluator and its factor storage.
///
/// The type parameter is the batch model, not its factor payload.
#[derive(Debug)]
pub struct BatchKey<T> {
    _private: PhantomData<fn() -> T>,
}

impl<T> Copy for BatchKey<T> {}

impl<T> Clone for BatchKey<T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Opaque variable identity used by the numerical backend.
///
/// Obtained from [`StateKey::block_id`]; never a dense index or scalar matrix offset.
#[derive(Debug, Clone, Copy)]
pub struct BlockId {
    _private: (),
}

/// Opaque identity tagging one factor's linearization emissions.
///
/// Supplied by [`FactorSelection`](crate::FactorSelection), not constructed by evaluators.
#[derive(Debug, Clone, Copy)]
pub struct FactorId {
    _private: (),
}
