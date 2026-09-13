use crate::{Factor, FactorBatch, FactorKey, KeyError, SolverError, StateKey, Variable};

/// Direct access to one concrete pool in a macro-generated schema.
///
/// The pool type selects the field at compile time. This is internal macro
/// plumbing, not the read-only interface used by factor authors.
pub trait PoolAccess<P> {
    /// Borrow the selected pool without traversing other families.
    fn pool(&self) -> &P;

    /// Mutably borrow the selected pool for solver-controlled operations.
    fn pool_mut(&mut self) -> &mut P;
}

/// Library-owned homogeneous state storage.
///
/// `Default` creates an empty pool without requiring `T: Default`. Identity
/// management, accepted/trial storage, and checked lookup are not implemented yet.
pub struct StatePool<T: Variable> {
    _values: Vec<T>,
}

impl<T: Variable> Default for StatePool<T> {
    fn default() -> Self {
        Self {
            _values: Vec::new(),
        }
    }
}

impl<T: Variable> StatePool<T> {
    /// Borrow the current estimate after validating the key.
    ///
    /// Removed, unknown, and foreign keys must be rejected. This API-only stub
    /// currently panics for every call.
    pub fn get(&self, _key: StateKey<T>) -> Result<&T, KeyError> {
        todo!("API only: checked state access")
    }
}

/// Library-owned homogeneous storage for ordinary factors.
///
/// `Default` creates an empty pool without requiring `T: Default`. Operational
/// methods remain API-only stubs; stable identities and compaction are not implemented.
pub struct FactorPool<T> {
    _factors: Vec<T>,
}

impl<T> Default for FactorPool<T> {
    fn default() -> Self {
        Self {
            _factors: Vec::new(),
        }
    }
}

impl<T> FactorPool<T> {
    /// Validate a handle and evaluate the ordinary factor's nonlinear cost.
    ///
    /// This API-only stub currently panics for every call.
    pub fn factor_cost<S>(&self, _states: &S, _key: FactorKey<T>) -> Result<f64, SolverError>
    where
        T: Factor<S>,
    {
        todo!("API only: ordinary factor cost")
    }

    /// Remove one factor, rejecting invalid keys and preserving sibling handles.
    ///
    /// This edits pool storage only; the solver owns graph and cache updates.
    /// This API-only stub currently panics for every call.
    pub fn remove_factor(&mut self, _key: FactorKey<T>) -> Result<(), KeyError> {
        todo!("API only: ordinary factor removal")
    }
}

/// One batch instance's shared model and contiguous payload storage.
struct Batch<B, P> {
    _model: B,
    _factors: Vec<P>,
}

/// Library-owned family of batches sharing model type `B` and payload type `P`.
///
/// Each batch instance owns its payload buffer. `Default` creates an empty
/// family without requiring either type to implement `Default`. Evaluation
/// requires `B: FactorBatch<S, Factor = P>`, but storage is independent of `S`.
pub struct BatchPool<B, P> {
    _batches: Vec<Batch<B, P>>,
}

impl<B, P> Default for BatchPool<B, P> {
    fn default() -> Self {
        Self {
            _batches: Vec::new(),
        }
    }
}

impl<B, P> BatchPool<B, P> {
    /// Locate the payload's batch and evaluate just that factor through its model.
    ///
    /// Invalid keys must be rejected without scanning unrelated payloads.
    /// This API-only stub currently panics for every call.
    pub fn factor_cost<S>(&self, _states: &S, _key: FactorKey<P>) -> Result<f64, SolverError>
    where
        B: FactorBatch<S, Factor = P>,
    {
        todo!("API only: batched factor cost")
    }

    /// Remove one payload, rejecting invalid keys and preserving sibling handles.
    ///
    /// This edits pool storage only; the solver owns graph and cache updates.
    /// This API-only stub currently panics for every call.
    pub fn remove_factor(&mut self, _key: FactorKey<P>) -> Result<(), KeyError> {
        todo!("API only: batched factor removal")
    }
}
