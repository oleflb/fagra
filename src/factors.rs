use std::iter::FusedIterator;

use crate::key::{LocalKey, PoolId, RawKey};
use crate::storage::{BatchPool, FactorPool};
use crate::{
    BlockId, EvaluationError, FactorId, FactorKey, KeyError, LinearizationSink, SolverError,
};

/// An independent graph contribution evaluated against compatible state storage.
///
/// One factor may emit multiple residual blocks; all share its identity and
/// lifetime. For shared computation across independently removable factors,
/// implement [`FactorBatch`] instead.
pub trait Factor<S> {
    /// Visit every variable dependency, including inputs shared by its residuals.
    ///
    /// This describes incidence, not a dense clique in the numerical system.
    fn visit_variables(&self, visitor: impl FnMut(BlockId));

    /// Evaluate `0.5 * sum(squared whitened residuals)` without Jacobians.
    ///
    /// The objective must match [`linearize`](Self::linearize). Report invalid
    /// model evaluations rather than silently dropping measurements.
    fn cost(&self, states: &S) -> Result<f64, EvaluationError>;

    /// Emit residuals and Jacobians into an already established factor scope.
    ///
    /// Do not call [`LinearizationSink::factor`] here: the solver owns this scope.
    /// See [factor scopes](crate::LinearizationSink#factor-scopes) for a comparison
    /// with batch emission.
    fn linearize<L: LinearizationSink>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError>;
}

/// Shared evaluation of independently identified factor payloads.
///
/// Implement this trait once, generically over compatible `S`. Payloads need no
/// [`Factor`] implementation. Prepare shared values inside each call, then
/// evaluate only the supplied selection. Empty selections do no work.
pub trait FactorBatch<S> {
    /// One factor's local data; shared inputs live in `Self`.
    type Factor;

    /// Visit one factor's complete dependencies, including shared batch inputs.
    fn visit_variables(&self, factor: &Self::Factor, visitor: impl FnMut(BlockId));

    /// Sum the selected factors' costs, sharing value-only preparation.
    ///
    /// Use the same whitened least-squares objective as [`Factor::cost`].
    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
    ) -> Result<f64, EvaluationError>;

    /// Prepare shared intermediates once and linearize the selected factors.
    ///
    /// Open one [`LinearizationSink::factor`] scope per selected identity. Each
    /// scope owns all residual blocks emitted for that factor.
    /// See [factor scopes](crate::LinearizationSink#factor-scopes) for a comparison
    /// with ordinary factor emission.
    fn linearize<L: LinearizationSink>(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
        sink: &mut L,
    ) -> Result<(), EvaluationError>;
}

/// Borrowed selection of live factors from one batch.
///
/// Iteration yields `(FactorId, &F)`. The solver groups requests by batch and
/// reuses scheduling storage; selecting a few factors must not scan the full pool.
/// Identities remain valid even when removal changes dense storage positions.
pub struct FactorSelection<'a, F> {
    pool: PoolId,
    keys: &'a [LocalKey],
    values: &'a [F],
    indices: Option<&'a [usize]>,
    position: usize,
}

impl<'a, F> FactorSelection<'a, F> {
    pub(crate) fn contiguous(pool: PoolId, keys: &'a [LocalKey], values: &'a [F]) -> Self {
        assert_eq!(keys.len(), values.len());
        Self {
            pool,
            keys,
            values,
            indices: None,
            position: 0,
        }
    }

    pub(crate) fn indexed(
        pool: PoolId,
        keys: &'a [LocalKey],
        values: &'a [F],
        indices: &'a [usize],
    ) -> Result<Self, KeyError> {
        if indices.iter().any(|&index| index >= values.len()) {
            return Err(KeyError::Unknown);
        }
        Ok(Self {
            indices: Some(indices),
            ..Self::contiguous(pool, keys, values)
        })
    }

    /// Number of factors remaining in this selection.
    pub fn len(&self) -> usize {
        self.indices.map_or(self.values.len(), <[usize]>::len) - self.position
    }

    /// Whether the selection has no remaining factors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow remaining payloads when contiguous, for bulk evaluation kernels.
    ///
    /// Returns `None` for a noncontiguous selection. The slice and subsequent
    /// iteration have exactly the same order. Checking an indexed selection
    /// examines only its remaining indices, never the full payload pool.
    pub fn as_slice(&self) -> Option<&'a [F]> {
        match self.indices {
            None => Some(&self.values[self.position..]),
            Some(indices) => {
                let remaining = &indices[self.position..];
                match remaining.first() {
                    None => Some(&self.values[..0]),
                    Some(&first)
                        if remaining
                            .windows(2)
                            .all(|pair| pair[0].checked_add(1) == Some(pair[1])) =>
                    {
                        Some(&self.values[first..first + remaining.len()])
                    }
                    Some(_) => None,
                }
            }
        }
    }
}

impl<'a, F> Iterator for FactorSelection<'a, F> {
    type Item = (FactorId, &'a F);

    fn next(&mut self) -> Option<Self::Item> {
        if self.len() == 0 {
            return None;
        }
        let index = self
            .indices
            .map_or(self.position, |indices| indices[self.position]);
        self.position += 1;
        Some((
            FactorId(RawKey {
                pool: self.pool,
                local: self.keys[index],
            }),
            &self.values[index],
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<F> ExactSizeIterator for FactorSelection<'_, F> {}
impl<F> FusedIterator for FactorSelection<'_, F> {}

/// Route payload-typed handles to their registered ordinary or batch family.
///
/// Internal macro plumbing generated once per payload type. Unlike typed pool
/// access, this interface does not require the caller to know a batch's model.
pub trait FactorStore<S, P> {
    /// Validate the key and evaluate one factor's nonlinear cost against `states`.
    fn factor_cost(&self, states: &S, key: FactorKey<P>) -> Result<f64, SolverError>;

    /// Remove one payload from storage, preserving sibling handles.
    ///
    /// Invalid keys are rejected. Graph connectivity and cache updates remain
    /// the solver's responsibility, outside this storage operation.
    fn remove_factor(&mut self, key: FactorKey<P>) -> Result<(), KeyError>;
}

/// Static traversal of the factor families declared by [`factors!`](crate::factors!).
///
/// Internal macro plumbing. Storage remains reusable across compatible state
/// schemas; evaluator bounds are checked for each `S`.
pub trait FactorSchema<S>: Default {
    /// Visit each declared family once, including empty families, in declaration order.
    ///
    /// Stops at the first visitor error without rolling back earlier mutations.
    /// An empty schema returns `Ok(())`. Mutable access also supports internal
    /// editing passes; evaluation passes can reborrow pools immutably.
    fn visit<V: FactorVisitor<S>>(&mut self, visitor: &mut V) -> Result<(), V::Error>;
}

/// One statically dispatched solver pass over ordinary and batched factor families.
///
/// Linearization and cost evaluation are separate visitor implementations, not
/// generated macro code. A batch call receives the whole family; the pass groups
/// selected payloads by batch before invoking [`FactorBatch`].
pub trait FactorVisitor<S> {
    /// Failure returned by this pass; use [`std::convert::Infallible`] if none.
    type Error;

    /// Process one homogeneous ordinary-factor pool.
    fn standalone<T: Factor<S>>(&mut self, pool: &mut FactorPool<T>) -> Result<(), Self::Error>;

    /// Process one family of batches sharing a model and payload type.
    fn batch<B: FactorBatch<S>>(
        &mut self,
        pool: &mut BatchPool<B, B::Factor>,
    ) -> Result<(), Self::Error>;
}

/// Declare ordinary factor pools and explicit `Batch<Model, Payload>` pools.
///
/// Use the literal `Batch<B, F>` spelling for batched fields; the macro recognizes
/// it as syntax, not a public Rust type to import or construct. Each payload and
/// batch model may identify only one registered family; any number of batch
/// instances may inhabit that family.
/// Fields stay private.
/// `Default` creates empty pools without requiring models or payloads to implement it.
/// Internally, the macro generates typed pool access, payload-key routing, and
/// one static traversal over library-owned ordinary and batch-family pools.
/// Traversal visits even empty families in declaration order and stops on error.
///
/// ```no_run
/// use fagra::factors;
/// # struct PosePrior;
/// # struct FrameReprojections;
/// # struct Reprojection;
/// factors! {
///     SlamFactors {
///         priors: PosePrior,
///         reprojections: Batch<FrameReprojections, Reprojection>,
///     }
/// }
/// ```
///
/// Registering one payload in two families is rejected:
///
/// ```compile_fail
/// use fagra::factors;
/// struct Frame;
/// struct RollingFrame;
/// struct Reprojection;
/// factors! {
///     AmbiguousFactors {
///         frames: Batch<Frame, Reprojection>,
///         rolling: Batch<RollingFrame, Reprojection>,
///     }
/// }
/// ```
#[macro_export]
macro_rules! factors {
    ($(#[$attr:meta])* $vis:vis $name:ident { $($fields:tt)* }) => {
        $crate::factors!(@parse [$(#[$attr])* $vis $name] [] [] []; $($fields)* ,);
    };
    (@parse [$(#[$attr:meta])* $vis:vis $name:ident]
        [$($field:ident: $pool:ty,)*] [$($bounds:tt)*]
        [$($kind:ident $visit_field:ident,)*];
    ) => {
        $(#[$attr])*
        #[derive(Default)]
        #[allow(dead_code)]
        $vis struct $name {
            $($field: $pool,)*
        }

        impl<S> $crate::__private::FactorSchema<S> for $name
        where $($bounds)*
        {
            fn visit<V: $crate::__private::FactorVisitor<S>>(
                &mut self,
                _visitor: &mut V,
            ) -> ::core::result::Result<(), V::Error> {
                $(_visitor.$kind(&mut self.$visit_field)?;)*
                ::core::result::Result::Ok(())
            }
        }
    };
    (@parse [$($header:tt)*] [$($fields:tt)*] [$($bounds:tt)*]
        [$($visits:tt)*]; , $($rest:tt)*
    ) => {
        $crate::factors!(@parse [$($header)*] [$($fields)*] [$($bounds)*]
            [$($visits)*]; $($rest)*);
    };
    (@parse [$(#[$attr:meta])* $vis:vis $name:ident]
        [$($fields:tt)*] [$($bounds:tt)*] [$($visits:tt)*];
        $field:ident: Batch<$model:ty, $factor:ty>, $($rest:tt)*
    ) => {
        $crate::factors!(@access $name, $field,
            $crate::__private::BatchPool<$model, $factor>);

        impl<S> $crate::__private::FactorStore<S, $factor> for $name
        where $model: $crate::FactorBatch<S, Factor = $factor>,
        {
            fn factor_cost(
                &self, states: &S, key: $crate::FactorKey<$factor>,
            ) -> ::core::result::Result<f64, $crate::SolverError> {
                self.$field.factor_cost(states, key)
            }

            fn remove_factor(
                &mut self, key: $crate::FactorKey<$factor>,
            ) -> ::core::result::Result<(), $crate::KeyError> {
                self.$field.remove_factor(key)
            }
        }

        $crate::factors!(@parse [$(#[$attr])* $vis $name]
            [$($fields)* $field: $crate::__private::BatchPool<$model, $factor>,]
            [$($bounds)* $model: $crate::FactorBatch<S, Factor = $factor>,]
            [$($visits)* batch $field,]; $($rest)*);
    };
    (@parse [$(#[$attr:meta])* $vis:vis $name:ident]
        [$($fields:tt)*] [$($bounds:tt)*] [$($visits:tt)*];
        $field:ident: $factor:ty, $($rest:tt)*
    ) => {
        $crate::factors!(@access $name, $field, $crate::__private::FactorPool<$factor>);

        impl<S> $crate::__private::FactorStore<S, $factor> for $name
        where $factor: $crate::Factor<S>,
        {
            fn factor_cost(
                &self, states: &S, key: $crate::FactorKey<$factor>,
            ) -> ::core::result::Result<f64, $crate::SolverError> {
                self.$field.factor_cost(states, key)
            }

            fn remove_factor(
                &mut self, key: $crate::FactorKey<$factor>,
            ) -> ::core::result::Result<(), $crate::KeyError> {
                self.$field.remove_factor(key)
            }
        }

        $crate::factors!(@parse [$(#[$attr])* $vis $name]
            [$($fields)* $field: $crate::__private::FactorPool<$factor>,]
            [$($bounds)* $factor: $crate::Factor<S>,]
            [$($visits)* standalone $field,]; $($rest)*);
    };
    (@access $name:ident, $field:ident, $pool:ty) => {
        impl $crate::__private::PoolAccess<$pool> for $name {
            fn pool(&self) -> &$pool {
                &self.$field
            }

            fn pool_mut(&mut self) -> &mut $pool {
                &mut self.$field
            }
        }
    };
}
