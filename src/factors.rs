use std::marker::PhantomData;

use crate::{BlockId, EvaluationError, FactorId, LinearizationSink};

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
    fn error(&self, states: &S) -> Result<f64, EvaluationError>;

    /// Emit residuals and Jacobians into an already established factor scope.
    ///
    /// Do not call [`LinearizationSink::factor`] here: the solver owns this scope.
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
    /// Use the same whitened least-squares objective as [`Factor::error`].
    fn error(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
    ) -> Result<f64, EvaluationError>;

    /// Prepare shared intermediates once and linearize the selected factors.
    ///
    /// Open one [`LinearizationSink::factor`] scope per selected identity. Each
    /// scope owns all residual blocks emitted for that factor.
    fn linearize<L: LinearizationSink>(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
        sink: &mut L,
    ) -> Result<(), EvaluationError>;
}

/// Library-owned storage for model `B` and its homogeneous factor payloads `F`.
///
/// Register as `Batch<B, F>` in [`factors!`](crate::factors!). Evaluation requires
/// `B: FactorBatch<S, Factor = F>`; storage itself does not depend on `S`.
/// Each batch has its own contiguous payload buffer. The solver manages identity
/// and compaction; users construct only the model and individual payloads.
pub struct Batch<B, F> {
    _model: B,
    _factors: Vec<F>,
}

/// Borrowed selection of live factors from one batch.
///
/// Iteration yields `(FactorId, &F)`. The solver groups requests by batch and
/// reuses scheduling storage; selecting a few factors must not scan the full pool.
/// Identities remain valid even when removal changes dense storage positions.
pub struct FactorSelection<'a, F> {
    _private: PhantomData<&'a F>,
}

impl<'a, F> FactorSelection<'a, F> {
    /// Number of factors remaining in this selection.
    pub fn len(&self) -> usize {
        todo!("API only: selection length")
    }

    /// Whether the selection has no remaining factors.
    pub fn is_empty(&self) -> bool {
        todo!("API only: empty selection")
    }

    /// Borrow remaining payloads when contiguous, for bulk evaluation kernels.
    ///
    /// Returns `None` for a noncontiguous selection. Use iteration to obtain the
    /// corresponding identities when emitting results.
    pub fn as_slice(&self) -> Option<&'a [F]> {
        todo!("API only: contiguous selection view")
    }
}

impl<'a, F> Iterator for FactorSelection<'a, F> {
    type Item = (FactorId, &'a F);

    fn next(&mut self) -> Option<Self::Item> {
        todo!("API only: selected factor iteration")
    }
}

#[doc(hidden)]
pub trait FactorStore<F> {}

#[doc(hidden)]
pub trait StandaloneFactorStore<F>: FactorStore<F> {}

#[doc(hidden)]
pub trait BatchStore<B, F>: FactorStore<F> {}

#[doc(hidden)]
pub trait FactorSchema<S>: Default {}

/// Declare ordinary factor pools and explicit `Batch<Model, Payload>` pools.
///
/// Use the literal `Batch<B, F>` spelling for batched fields, with [`Batch`] in
/// scope. Each payload and batch model may identify only one registered family;
/// any number of batch instances may inhabit that family. Fields stay private.
/// `Default` creates empty pools without requiring models or payloads to implement it.
///
/// ```no_run
/// use fagra::{factors, Batch};
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
/// use fagra::{factors, Batch};
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
        $crate::factors!(@struct [$(#[$attr])* $vis $name] { $($fields)* });
        $crate::factors!(@register $name []; $($fields)* ,);
    };
    (@struct [$(#[$attr:meta])* $vis:vis $name:ident] {
        $($field:ident: $ty:ty),* $(,)?
    }) => {
        $(#[$attr])*
        #[derive(Default)]
        #[allow(dead_code)]
        $vis struct $name {
            $($field: ::std::vec::Vec<$ty>,)*
        }
    };
    (@register $name:ident [$($bounds:tt)*];) => {
        impl<S> $crate::__private::FactorSchema<S> for $name
        where $($bounds)*
        {}
    };
    (@register $name:ident [$($bounds:tt)*]; , $($rest:tt)*) => {
        $crate::factors!(@register $name [$($bounds)*]; $($rest)*);
    };
    (@register $name:ident [$($bounds:tt)*];
        $field:ident: Batch<$model:ty, $factor:ty>, $($rest:tt)*
    ) => {
        impl $crate::__private::FactorStore<$factor> for $name {}
        impl $crate::__private::BatchStore<$model, $factor> for $name {}
        $crate::factors!(@register $name [
            $($bounds)* $model: $crate::FactorBatch<S, Factor = $factor>,
        ]; $($rest)*);
    };
    (@register $name:ident [$($bounds:tt)*];
        $field:ident: $factor:ty, $($rest:tt)*
    ) => {
        impl $crate::__private::FactorStore<$factor> for $name {}
        impl $crate::__private::StandaloneFactorStore<$factor> for $name {}
        $crate::factors!(@register $name [
            $($bounds)* $factor: $crate::Factor<S>,
        ]; $($rest)*);
    };
}
