use nalgebra::{DMatrixView, SMatrix, SVector};

use crate::{BlockId, EvaluationError, FactorId, StateKey, Variable};

/// A borrowed Jacobian matrix and the variable it differentiates.
///
/// Descriptors can form a stack array; the matrices remain in their original storage.
pub struct JacobianBlock<'a> {
    variable: BlockId,
    jacobian: DMatrixView<'a, f64>,
}

impl<'a> JacobianBlock<'a> {
    /// Borrow a fixed-size Jacobian without copying its coefficients.
    ///
    /// The state type determines the required column count. A mismatch with
    /// [`Variable::DOF`] fails during monomorphization, not necessarily `cargo check`.
    /// The constructor obtains the numerical identity from [`StateKey::block_id`].
    ///
    /// Factor and batch handles cannot be used as state handles:
    ///
    /// ```compile_fail
    /// use fagra::{FactorKey, JacobianBlock, Variable};
    /// use nalgebra::SMatrix;
    /// fn wrong_kind<T: Variable>(key: FactorKey<T>, jacobian: &SMatrix<f64, 2, 3>) {
    ///     JacobianBlock::new(key, jacobian);
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use fagra::{BatchKey, JacobianBlock, Variable};
    /// use nalgebra::SMatrix;
    /// fn wrong_kind<T: Variable>(key: BatchKey<T>, jacobian: &SMatrix<f64, 2, 3>) {
    ///     JacobianBlock::new(key, jacobian);
    /// }
    /// ```
    pub fn new<T: Variable, const R: usize, const C: usize>(
        state: StateKey<T>,
        jacobian: &'a SMatrix<f64, R, C>,
    ) -> Self {
        const {
            assert!(C == T::DOF, "Jacobian columns must match state DOF");
        }

        Self {
            variable: state.block_id(),
            jacobian: jacobian.as_view(),
        }
    }

    /// Stable identity of the differentiated state.
    pub fn variable(&self) -> BlockId {
        self.variable
    }

    /// Borrow the coefficients with their original lifetime and checked column count.
    pub fn jacobian(&self) -> DMatrixView<'a, f64> {
        self.jacobian
    }
}

/// Statically dispatched output for factor-tagged residuals and Jacobians.
///
/// Implementations may assemble normal equations or retain square-root factors.
/// Normal equations use `g = Jᵀr`, `H = JᵀJ`, and solve `H delta = -g`.
pub trait LinearizationSink: Sized {
    /// Emit one factor's complete linearization within an identity scope.
    ///
    /// Scopes cannot nest. Ordinary factors receive an already scoped sink;
    /// batches open one scope per selected factor. Propagate callback failures.
    fn factor(
        &mut self,
        id: FactorId,
        emit: impl FnOnce(&mut Self) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError>;

    /// Emit a whitened residual and any number of borrowed Jacobian blocks.
    ///
    /// An active factor scope is required. Reject invalid identities, dimensions,
    /// or nonfinite values. Repeated variable IDs must be combined correctly or
    /// rejected, never treated as independent variables. Borrowed data may not
    /// be retained after this call without copying into backend-owned storage.
    fn residual<const R: usize>(
        &mut self,
        residual: &SVector<f64, R>,
        jacobians: &[JacobianBlock<'_>],
    ) -> Result<(), EvaluationError>;
}
