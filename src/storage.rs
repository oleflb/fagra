use faer_ext::nalgebra::DimName;

use crate::{
    BatchKey, BlockId, EvaluationError, Factor, FactorBatch, FactorId, FactorKey, FactorSelection,
    KeyError, Real, SolverError, StateKey, Variable,
    dense::DensePool,
    key::{LocalKey, RawKey},
    marginalization::Anchor,
};

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
/// Payloads and identity metadata occupy separate dense arrays. `Default` creates
/// an empty pool with a fresh identity, without requiring `T: Default`.
pub struct StatePool<T: Variable> {
    entries: DensePool<T>,
    pub(crate) anchors: Vec<Anchor<T>>,
    trial: Vec<T>,
    trial_active: bool,
}

impl<T: Variable> Default for StatePool<T> {
    fn default() -> Self {
        Self {
            entries: DensePool::default(),
            anchors: Vec::new(),
            trial: Vec::new(),
            trial_active: false,
        }
    }
}

impl<T: Variable> StatePool<T> {
    pub(crate) fn values_and_anchors(
        &mut self,
    ) -> (impl Iterator<Item = (BlockId, &T)>, &mut Vec<Anchor<T>>) {
        // Marginalization captures accepted values, never trial values.
        assert!(!self.trial_active);
        (
            self.entries
                .iter()
                .map(|(key, value)| (BlockId(key), value)),
            &mut self.anchors,
        )
    }
    pub(crate) fn remove_selected(&mut self, ids: &std::collections::HashSet<BlockId>) {
        for index in (0..self.entries.keys.len()).rev() {
            let raw = RawKey {
                pool: self.entries.id,
                local: self.entries.keys[index],
            };
            if ids.contains(&BlockId(raw)) {
                self.entries.remove(raw).expect("live state");
            }
        }
    }

    /// Borrow the current estimate after validating the key.
    ///
    /// Removed, unknown, and foreign keys are rejected in constant time.
    pub fn get(&self, key: StateKey<T>) -> Result<&T, KeyError> {
        let index = self.entries.index(key.raw)?;
        Ok(&self.current_values()[index])
    }

    /// Reserve room for additional states and their identity metadata.
    ///
    /// Panics on capacity/index exhaustion. Existing handles survive reallocation.
    pub fn reserve(&mut self, additional: usize) {
        self.entries.reserve(additional);
    }

    /// Insert a state and issue a solver-local handle.
    pub fn insert(&mut self, value: T) -> StateKey<T> {
        StateKey::from_raw(self.entries.insert(value))
    }

    /// Remove a state from storage, preserving surviving handles and capacity.
    ///
    /// Internal storage operation: graph-aware callers must handle incident
    /// factors first. The public solver does not expose raw state deletion.
    pub fn remove(&mut self, key: StateKey<T>) -> Result<T, KeyError> {
        self.entries.remove(key.raw)
    }

    /// Traverse live states in dense order without resolving handles again.
    /// Order may change on removal. Estimate updates must preserve these identities.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (BlockId, &T)> {
        self.entries
            .keys
            .iter()
            .copied()
            .zip(self.current_values())
            .map(|(local, value)| {
                (
                    BlockId(RawKey {
                        pool: self.entries.id,
                        local,
                    }),
                    value,
                )
            })
    }

    pub(crate) fn validate(&self, id: BlockId) -> Option<Result<(), KeyError>> {
        (id.0.pool == self.entries.id).then(|| self.entries.index(id.0).map(|_| ()))
    }

    fn current_values(&self) -> &[T] {
        if self.trial_active {
            &self.trial
        } else {
            &self.entries.values
        }
    }

    pub(crate) fn prepare_trial(&mut self) {
        self.reject_trial();
        // Preserve user-reserved capacity when acceptance swaps the two buffers.
        self.trial.reserve(self.entries.values.capacity());
    }

    pub(crate) fn stage(&mut self, delta: &[T::Scalar]) -> Result<(), EvaluationError> {
        let expected = self
            .entries
            .values
            .len()
            .checked_mul(T::Dim::DIM)
            .ok_or(EvaluationError::DimensionMismatch)?;
        if delta.len() != expected {
            return Err(EvaluationError::DimensionMismatch);
        }
        self.reject_trial();
        for (index, value) in self.entries.values.iter().enumerate() {
            let start = index * T::Dim::DIM;
            let tangent = T::tangent_from_slice(&delta[start..start + T::Dim::DIM]);
            self.trial.push(value.retract(&tangent));
        }
        self.trial_active = true;
        Ok(())
    }

    pub(crate) fn accept_trial(&mut self) {
        if self.trial_active {
            std::mem::swap(&mut self.entries.values, &mut self.trial);
            self.trial_active = false;
        }
    }

    pub(crate) fn reject_trial(&mut self) {
        self.trial_active = false;
        self.trial.clear();
    }
}

/// Library-owned homogeneous storage for ordinary factors.
///
/// `Default` creates an empty pool without requiring `T: Default`. Dense payloads
/// and parallel identities support sequential evaluation without slot lookups.
pub struct FactorPool<T> {
    entries: DensePool<T>,
}

impl<T> Default for FactorPool<T> {
    fn default() -> Self {
        Self {
            entries: DensePool::default(),
        }
    }
}

impl<T> FactorPool<T> {
    pub(crate) fn remove_selected(&mut self, ids: &std::collections::HashSet<FactorId>) {
        for index in (0..self.entries.keys.len()).rev() {
            let raw = RawKey {
                pool: self.entries.id,
                local: self.entries.keys[index],
            };
            if ids.contains(&FactorId(raw)) {
                self.entries.remove(raw).expect("live factor");
            }
        }
    }

    /// Reserve payload and identity capacity for additional factors.
    pub fn reserve(&mut self, additional: usize) {
        self.entries.reserve(additional);
    }

    /// Insert a factor whose dependencies have already been validated by the solver.
    pub fn insert(&mut self, factor: T) -> FactorKey<T> {
        FactorKey::from_raw(self.entries.insert(factor))
    }

    /// Traverse live factors and their identities in dense order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (FactorId, &T)> {
        self.entries
            .iter()
            .map(|(key, value)| (FactorId(key), value))
    }

    /// Validate a handle and evaluate the ordinary factor's nonlinear cost.
    ///
    /// Nonfinite or negative objective contributions are rejected.
    pub fn factor_cost<S>(&self, states: &S, key: FactorKey<T>) -> Result<T::Scalar, SolverError>
    where
        T: Factor<S>,
        T::Scalar: Real,
    {
        checked_cost(self.entries.get(key.raw)?.cost(states))
    }

    /// Remove one factor, rejecting invalid keys and preserving sibling handles.
    ///
    /// This edits pool storage only; the solver owns graph and cache updates.
    pub fn remove_factor(&mut self, key: FactorKey<T>) -> Result<(), KeyError> {
        self.entries.remove(key.raw).map(drop)
    }
}

/// One batch instance's shared model and contiguous payload storage.
struct Batch<B, P> {
    model: B,
    values: Vec<P>,
    keys: Vec<LocalKey>,
}

#[derive(Clone, Copy)]
struct FactorLocation {
    batch_slot: u32,
    dense_index: u32,
}

/// Library-owned family of batches sharing model type `B` and payload type `P`.
///
/// Each batch instance owns its payload buffer. `Default` creates an empty
/// family without requiring either type to implement `Default`. Evaluation
/// requires `B: FactorBatch<S, Factor = P>`, but storage is independent of `S`.
pub struct BatchPool<B, P> {
    batches: DensePool<Batch<B, P>>,
    // Family-wide generational directory; all payload buffers share its identity.
    locations: DensePool<FactorLocation>,
}

impl<B, P> Default for BatchPool<B, P> {
    fn default() -> Self {
        Self {
            batches: DensePool::default(),
            locations: DensePool::default(),
        }
    }
}

impl<B, P> BatchPool<B, P> {
    /// Pack selected payloads temporarily, restoring the exact order even on panic.
    /// The directory remains unchanged: evaluators only receive states and slices.
    pub(crate) fn with_selected<E>(
        &mut self,
        selected: &std::collections::HashSet<FactorId>,
        swaps: &mut Vec<(usize, usize)>,
        mut evaluate: impl FnMut(&B, FactorSelection<'_, P>) -> Result<(), E>,
    ) -> Result<(), E> {
        struct Packed<'a, B, P> {
            batch: &'a mut Batch<B, P>,
            swaps: &'a mut Vec<(usize, usize)>,
        }
        impl<B, P> Drop for Packed<'_, B, P> {
            fn drop(&mut self) {
                for &(a, b) in self.swaps.iter().rev() {
                    self.batch.values.swap(a, b);
                    self.batch.keys.swap(a, b);
                }
            }
        }
        for batch in &mut self.batches.values {
            swaps.clear();
            let packed = Packed { batch, swaps };
            let mut count = 0;
            for index in 0..packed.batch.keys.len() {
                let id = FactorId(RawKey {
                    pool: self.locations.id,
                    local: packed.batch.keys[index],
                });
                if selected.contains(&id) {
                    if index != count {
                        packed.swaps.push((index, count));
                        packed.batch.values.swap(index, count);
                        packed.batch.keys.swap(index, count);
                    }
                    count += 1;
                }
            }
            if count > 0 {
                evaluate(
                    &packed.batch.model,
                    FactorSelection::contiguous(
                        self.locations.id,
                        &packed.batch.keys[..count],
                        &packed.batch.values[..count],
                    ),
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn remove_selected(&mut self, ids: &std::collections::HashSet<FactorId>) {
        for batch in 0..self.batches.values.len() {
            for index in (0..self.batches.values[batch].keys.len()).rev() {
                let raw = RawKey {
                    pool: self.locations.id,
                    local: self.batches.values[batch].keys[index],
                };
                if ids.contains(&FactorId(raw)) {
                    self.remove_factor(FactorKey::from_raw(raw))
                        .expect("live payload");
                }
            }
        }
    }

    /// Remove an empty batch and invalidate its handle. Reject nonempty batches.
    pub fn remove_empty(&mut self, key: BatchKey<B>) -> Result<(), SolverError> {
        if !self.batches.get(key.raw)?.values.is_empty() {
            return Err(SolverError::BatchNotEmpty);
        }
        self.batches.remove(key.raw)?;
        Ok(())
    }

    /// Reserve capacity for additional batch models.
    pub fn reserve(&mut self, additional: usize) {
        self.batches.reserve(additional);
    }

    /// Create an empty batch. Shared dependencies are checked on payload insertion.
    /// Empty batches stay alive and reusable until this pool is dropped.
    pub fn insert_batch(&mut self, model: B) -> BatchKey<B> {
        BatchKey::from_raw(self.batches.insert(Batch {
            model,
            values: Vec::new(),
            keys: Vec::new(),
        }))
    }

    /// Borrow a batch's shared model after validating its key.
    pub fn model(&self, batch: BatchKey<B>) -> Result<&B, KeyError> {
        Ok(&self.batches.get(batch.raw)?.model)
    }

    /// Reserve additional payloads in a batch, including directory and identity storage.
    pub fn reserve_factors(
        &mut self,
        batch: BatchKey<B>,
        additional: usize,
    ) -> Result<(), KeyError> {
        let index = self.batches.index(batch.raw)?;
        self.locations.reserve(additional);
        let batch = &mut self.batches.values[index];
        batch.values.reserve(additional);
        batch.keys.reserve(additional);
        Ok(())
    }

    /// Insert a payload after the solver has validated shared and local dependencies.
    pub fn insert_into(&mut self, key: BatchKey<B>, factor: P) -> Result<FactorKey<P>, KeyError> {
        self.reserve_factors(key, 1)?;
        let batch = self.batches.get_mut(key.raw)?;
        let raw = self.locations.insert(FactorLocation {
            batch_slot: key.raw.local.slot,
            dense_index: batch.values.len() as u32,
        });
        batch.values.push(factor);
        batch.keys.push(raw.local);
        Ok(FactorKey::from_raw(raw))
    }

    /// Traverse batches in dense order, borrowing each model and all of its payloads.
    /// Empty batches yield empty selections. No payloads are copied.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (BatchKey<B>, &B, FactorSelection<'_, P>)> {
        self.batches.iter().map(|(key, batch)| {
            (
                BatchKey::from_raw(key),
                &batch.model,
                FactorSelection::contiguous(self.locations.id, &batch.keys, &batch.values),
            )
        })
    }

    /// Locate the payload's batch and evaluate just that factor through its model.
    ///
    /// Invalid keys must be rejected without scanning unrelated payloads.
    /// Nonfinite or negative objective contributions are rejected.
    pub fn factor_cost<S>(&self, states: &S, key: FactorKey<P>) -> Result<B::Scalar, SolverError>
    where
        B: FactorBatch<S, Factor = P>,
        B::Scalar: Real,
    {
        let location = self.locations.get(key.raw)?;
        let batch = self.batches.at_slot(location.batch_slot);
        let index = location.dense_index as usize;
        let selection = FactorSelection::contiguous(
            self.locations.id,
            &batch.keys[index..index + 1],
            &batch.values[index..index + 1],
        );
        checked_cost(batch.model.cost(states, selection))
    }

    /// Remove one payload, rejecting invalid keys and preserving sibling handles.
    ///
    /// This edits pool storage only; the solver owns graph and cache updates.
    pub fn remove_factor(&mut self, key: FactorKey<P>) -> Result<(), KeyError> {
        let location = *self.locations.get(key.raw)?;
        let batch = self.batches.at_slot_mut(location.batch_slot);
        let index = location.dense_index as usize;
        let removed = batch.values.swap_remove(index);
        batch.keys.swap_remove(index);
        if let Some(&local) = batch.keys.get(index) {
            self.locations
                .get_mut(RawKey {
                    pool: self.locations.id,
                    local,
                })
                .expect("live payload directory entry")
                .dense_index = index as u32;
        }
        self.locations.remove(key.raw)?;
        drop(removed);
        Ok(())
    }
}

pub(crate) fn checked_cost<R: Real>(result: Result<R, EvaluationError>) -> Result<R, SolverError> {
    let cost = result?;
    if !cost.is_finite() || cost < R::zero() {
        return Err(EvaluationError::InvalidEvaluation.into());
    }
    Ok(cost)
}
