use std::iter::FusedIterator;

use faer_ext::nalgebra::RealField;

use crate::key::{LocalKey, PoolId, RawKey};
use crate::storage::{BatchPool, FactorPool};
use crate::{
    BlockId, EvaluationError, FactorId, FactorKey, KeyError, LinearizationSink, Real, SolverError,
};

/// An independent graph contribution evaluated against compatible state storage.
///
/// One factor may emit multiple residual blocks; all share its identity and
/// lifetime. For shared computation across independently removable factors,
/// implement [`FactorBatch`] instead.
pub trait Factor<S> {
    /// Scalar used for this factor's cost, residuals, and Jacobians.
    /// Geometry scalars, including dual numbers, are supported; solvers require [`Real`].
    type Scalar: RealField + Copy;

    /// Visit every variable dependency, including inputs shared by its residuals.
    ///
    /// This describes incidence, not a dense clique in the numerical system.
    fn visit_variables(&self, visitor: impl FnMut(BlockId));

    /// Evaluate the finite, nonnegative nonlinear objective without Jacobians.
    ///
    /// Ordinary least squares uses `0.5 * sum(squared whitened residuals)`.
    /// Robust factors may instead evaluate a robust objective and emit its frozen-
    /// weight IRLS model in [`linearize`](Self::linearize). The emitted model must
    /// have gradient `Jᵀr` equal to this objective's derivative in right coordinates;
    /// its value may differ by an additive constant. Report invalid evaluations
    /// rather than silently dropping measurements.
    fn cost(&self, states: &S) -> Result<Self::Scalar, EvaluationError>;

    /// Emit residuals and Jacobians into an already established factor scope.
    ///
    /// Do not call [`LinearizationSink::factor`] here: the solver owns this scope.
    /// See [factor scopes](crate::LinearizationSink#factor-scopes) for a comparison
    /// with batch emission.
    ///
    /// For blockwise Huber with whitened residual `r`, norm `s`, and threshold `d > 0`,
    /// `cost` is `s²/2` for `s <= d`, otherwise `d * (s - d/2)`. Emit `sqrt(w) * r`
    /// and `sqrt(w) * J` for every associated Jacobian, where `w = 1` for `s <= d`
    /// and `w = d/s` otherwise. Compute one weight per observation block, not per
    /// coordinate or entire batch. Do not differentiate the weight in this local
    /// model. Marginalization absorbs these rows with frozen weights; it does not
    /// robustify the resulting prior again or infer the additive difference between
    /// true robust cost and weighted row cost. Subsequent reported costs use the
    /// retained surrogate objective, omitting that additive robust constant.
    /// QR's irreducible least-squares constant is retained. The omitted constant
    /// does not affect the local model's gradient, information, or covariance.
    ///
    /// Test raw residual derivatives separately. For robust models use
    /// `testing::FactorProperty::LocalModel` to check the true cost gradient against
    /// `Jᵀr`, and independently verify the intended weighting/curvature. Ordinary
    /// residual-derivative tests do not apply to state-dependent IRLS weights.
    fn linearize<L: LinearizationSink<Scalar = Self::Scalar>>(
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
    /// Scalar used by the shared evaluator and its selected factors.
    /// Geometry scalars, including dual numbers, are supported; solvers require [`Real`].
    type Scalar: RealField + Copy;

    /// One factor's local data; shared inputs live in `Self`.
    type Factor;

    /// Visit one factor's complete dependencies, including shared batch inputs.
    fn visit_variables(&self, factor: &Self::Factor, visitor: impl FnMut(BlockId));

    /// Sum the selected factors' costs, sharing value-only preparation.
    ///
    /// Use the same nonlinear objective contract as [`Factor::cost`], including
    /// robust objectives with frozen-weight local models.
    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
    ) -> Result<Self::Scalar, EvaluationError>;

    /// Prepare shared intermediates once and linearize the selected factors.
    ///
    /// Open one [`LinearizationSink::factor`] scope per selected identity. Each
    /// scope owns all residual blocks emitted for that factor.
    /// See [factor scopes](crate::LinearizationSink#factor-scopes) for a comparison
    /// with ordinary factor emission.
    fn linearize<L: LinearizationSink<Scalar = Self::Scalar>>(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
        sink: &mut L,
    ) -> Result<(), EvaluationError>;
}

/// Borrowed contiguous selection of live factors from one batch.
///
/// Iteration yields `(FactorId, &F)`. The solver borrows a whole batch for
/// optimization or a single payload for individual cost evaluation.
/// Identities remain valid even when removal changes dense storage positions.
pub struct FactorSelection<'a, F> {
    pool: PoolId,
    keys: &'a [LocalKey],
    values: &'a [F],
    position: usize,
}

impl<F> Clone for FactorSelection<'_, F> {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool,
            keys: self.keys,
            values: self.values,
            position: self.position,
        }
    }
}

impl<'a, F> FactorSelection<'a, F> {
    pub(crate) fn contiguous(pool: PoolId, keys: &'a [LocalKey], values: &'a [F]) -> Self {
        assert_eq!(keys.len(), values.len());
        Self {
            pool,
            keys,
            values,
            position: 0,
        }
    }

    /// Number of factors remaining in this selection.
    pub fn len(&self) -> usize {
        self.values.len() - self.position
    }

    /// Whether the selection has no remaining factors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow remaining payloads for bulk evaluation kernels.
    ///
    /// The slice and subsequent iteration have exactly the same order.
    pub fn as_slice(&self) -> &'a [F] {
        &self.values[self.position..]
    }
}

impl<'a, F> Iterator for FactorSelection<'a, F> {
    type Item = (FactorId, &'a F);

    fn next(&mut self) -> Option<Self::Item> {
        if Self::is_empty(self) {
            return None;
        }
        let index = self.position;
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
    /// Scalar returned by the registered evaluator.
    type Scalar: Real;

    /// Validate the key and evaluate one factor's nonlinear cost against `states`.
    fn factor_cost(&self, states: &S, key: FactorKey<P>) -> Result<Self::Scalar, SolverError>;

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
    /// Scalar shared by every evaluator in this schema.
    type Scalar: Real;

    /// Visit each declared family once, including empty families, in declaration order.
    ///
    /// Stops at the first visitor error without rolling back earlier mutations.
    /// An empty schema returns `Ok(())`. Mutable access also supports internal
    /// editing passes; evaluation passes can reborrow pools immutably.
    fn visit<V: FactorVisitor<S, Self::Scalar>>(&mut self, visitor: &mut V)
    -> Result<(), V::Error>;
}

/// One statically dispatched solver pass over ordinary and batched factor families.
///
/// Linearization and cost evaluation are separate visitor implementations, not
/// generated macro code. A batch call receives the whole family; the pass groups
/// selected payloads by batch before invoking [`FactorBatch`].
pub trait FactorVisitor<S, R: Real = f64> {
    /// Failure returned by this pass; use [`std::convert::Infallible`] if none.
    type Error;

    /// Process one homogeneous ordinary-factor pool.
    fn standalone<T: Factor<S, Scalar = R>>(
        &mut self,
        pool: &mut FactorPool<T>,
    ) -> Result<(), Self::Error>;

    /// Process one family of batches sharing a model and payload type.
    fn batch<B: FactorBatch<S, Scalar = R>>(
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
/// A declaration `Factors<R> { priors: Prior<R> }` introduces one scalar parameter
/// bounded by [`Real`]. All ordinary and batched evaluators must have `Scalar = R`.
/// Declarations without a parameter use `f64`; payloads need no scalar trait.
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
    ($(#[$attr:meta])* $vis:vis $name:ident<$scalar:ident> { $($fields:tt)* }) => {
        $crate::factors!(@parse [$(#[$attr])* $vis $name] [$scalar] [$scalar] [] [] []; $($fields)* ,);
    };
    ($(#[$attr:meta])* $vis:vis $name:ident { $($fields:tt)* }) => {
        $crate::factors!(@parse [$(#[$attr])* $vis $name] [] [f64] [] [] []; $($fields)* ,);
    };
    (@parse [$(#[$attr:meta])* $vis:vis $name:ident]
        [$($scalar:ident)?] [$real:ty]
        [$($field:ident: $pool:ty,)*] [$($bounds:tt)*]
        [$($kind:ident $visit_field:ident,)*];
    ) => {
        $(#[$attr])*
        #[allow(dead_code)]
        $vis struct $name$(<$scalar: $crate::Real>)? {
            $($field: $pool,)*
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

        impl<__S $(, $scalar: $crate::Real)?> $crate::__private::FactorSchema<__S> for $name$(<$scalar>)?
        where $($bounds)*
        {
            type Scalar = $real;

            fn visit<__Visitor: $crate::__private::FactorVisitor<__S, $real>>(
                &mut self,
                _visitor: &mut __Visitor,
            ) -> ::core::result::Result<(), __Visitor::Error> {
                $(_visitor.$kind(&mut self.$visit_field)?;)*
                ::core::result::Result::Ok(())
            }
        }
    };
    (@parse [$($header:tt)*] [$($scalar:ident)?] [$real:ty] [$($fields:tt)*] [$($bounds:tt)*]
        [$($visits:tt)*]; , $($rest:tt)*
    ) => {
        $crate::factors!(@parse [$($header)*] [$($scalar)?] [$real] [$($fields)*] [$($bounds)*]
            [$($visits)*]; $($rest)*);
    };
    (@parse [$(#[$attr:meta])* $vis:vis $name:ident]
        [$($scalar:ident)?] [$real:ty]
        [$($fields:tt)*] [$($bounds:tt)*] [$($visits:tt)*];
        $field:ident: Batch<$model:ty, $factor:ty>, $($rest:tt)*
    ) => {
        $crate::factors!(@access [$name $($scalar)?], $field,
            $crate::__private::BatchPool<$model, $factor>);

        impl<__S $(, $scalar: $crate::Real)?> $crate::__private::FactorStore<__S, $factor> for $name$(<$scalar>)?
        where $model: $crate::FactorBatch<__S, Scalar = $real, Factor = $factor>,
        {
            type Scalar = $real;

            fn factor_cost(
                &self, states: &__S, key: $crate::FactorKey<$factor>,
            ) -> ::core::result::Result<$real, $crate::SolverError> {
                self.$field.factor_cost(states, key)
            }

            fn remove_factor(
                &mut self, key: $crate::FactorKey<$factor>,
            ) -> ::core::result::Result<(), $crate::KeyError> {
                self.$field.remove_factor(key)
            }
        }

        $crate::factors!(@parse [$(#[$attr])* $vis $name] [$($scalar)?] [$real]
            [$($fields)* $field: $crate::__private::BatchPool<$model, $factor>,]
            [$($bounds)* $model: $crate::FactorBatch<__S, Scalar = $real, Factor = $factor>,]
            [$($visits)* batch $field,]; $($rest)*);
    };
    (@parse [$(#[$attr:meta])* $vis:vis $name:ident]
        [$($scalar:ident)?] [$real:ty]
        [$($fields:tt)*] [$($bounds:tt)*] [$($visits:tt)*];
        $field:ident: $factor:ty, $($rest:tt)*
    ) => {
        $crate::factors!(@access [$name $($scalar)?], $field, $crate::__private::FactorPool<$factor>);

        impl<__S $(, $scalar: $crate::Real)?> $crate::__private::FactorStore<__S, $factor> for $name$(<$scalar>)?
        where $factor: $crate::Factor<__S, Scalar = $real>,
        {
            type Scalar = $real;

            fn factor_cost(
                &self, states: &__S, key: $crate::FactorKey<$factor>,
            ) -> ::core::result::Result<$real, $crate::SolverError> {
                self.$field.factor_cost(states, key)
            }

            fn remove_factor(
                &mut self, key: $crate::FactorKey<$factor>,
            ) -> ::core::result::Result<(), $crate::KeyError> {
                self.$field.remove_factor(key)
            }
        }

        $crate::factors!(@parse [$(#[$attr])* $vis $name] [$($scalar)?] [$real]
            [$($fields)* $field: $crate::__private::FactorPool<$factor>,]
            [$($bounds)* $factor: $crate::Factor<__S, Scalar = $real>,]
            [$($visits)* standalone $field,]; $($rest)*);
    };
    (@access [$name:ident $($scalar:ident)?], $field:ident, $pool:ty) => {
        impl$(<$scalar: $crate::Real>)? $crate::__private::PoolAccess<$pool> for $name$(<$scalar>)? {
            fn pool(&self) -> &$pool {
                &self.$field
            }

            fn pool_mut(&mut self) -> &mut $pool {
                &mut self.$field
            }
        }
    };
}
