use faer_ext::nalgebra::{
    DMatrixView, DimName, Dyn, Matrix, RealField, Storage, U1, storage::IsContiguous,
};

use crate::{BlockId, EvaluationError, FactorId, StateKey, Variable};

/// A borrowed Jacobian matrix and the variable it differentiates.
///
/// Descriptors can form a stack array; the matrices remain in their original storage.
/// Construction borrows the coefficients without allocating or copying. Arbitrary
/// row and column strides are preserved, including those of noncontiguous views.
pub struct JacobianBlock<'a, R: RealField + Copy = f64> {
    variable: BlockId,
    jacobian: DMatrixView<'a, R, Dyn, Dyn>,
}

impl<'a, R: RealField + Copy> JacobianBlock<'a, R> {
    /// Borrow a fixed-dimension Jacobian from any compatible nalgebra storage.
    ///
    /// Accepts owned matrices and immutable or mutably backed views borrowed
    /// immutably. The matrix or view passed here must outlive the descriptor.
    /// Dimensions stay known at compile time; the column dimension must be the
    /// variable's [`Variable::Dim`]. A mismatch is a type error, even in `cargo check`.
    /// The sink checks that the row count matches the emitted residual.
    /// The constructor obtains the numerical identity from [`StateKey::block_id`].
    ///
    /// A preallocated dynamic matrix can supply a fixed-size, strided view:
    ///
    /// ```no_run
    /// # use fagra::{JacobianBlock, StateKey, Variable};
    /// # use faer_ext::nalgebra::{Const, DMatrix};
    /// # fn from_workspace<T: Variable<Scalar = f64, Dim = Const<6>>>(key: StateKey<T>, workspace: &DMatrix<f64>) {
    /// // Requires a workspace large enough for the selected view.
    /// let view = workspace.fixed_view::<2, 6>(0, 0);
    /// let block = JacobianBlock::new(key, &view);
    /// # let _ = block;
    /// # }
    /// ```
    ///
    /// Factor and batch handles cannot be used as state handles:
    ///
    /// ```compile_fail
    /// use fagra::{FactorKey, JacobianBlock, Variable};
    /// use faer_ext::nalgebra::{Const, SMatrix};
    /// fn wrong_kind<T: Variable<Scalar = f64, Dim = Const<3>>>(key: FactorKey<T>, jacobian: &SMatrix<f64, 2, 3>) {
    ///     JacobianBlock::new(key, jacobian);
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use fagra::{BatchKey, JacobianBlock, Variable};
    /// use faer_ext::nalgebra::{Const, SMatrix};
    /// fn wrong_kind<T: Variable<Scalar = f64, Dim = Const<3>>>(key: BatchKey<T>, jacobian: &SMatrix<f64, 2, 3>) {
    ///     JacobianBlock::new(key, jacobian);
    /// }
    /// ```
    pub fn new<T, Rows: DimName, S>(
        state: StateKey<T>,
        jacobian: &'a Matrix<R, Rows, T::Dim, S>,
    ) -> Self
    where
        T: Variable<Scalar = R>,
        S: Storage<R, Rows, T::Dim>,
    {
        Self {
            variable: state.block_id(),
            // Zero skipped rows/columns preserves strides while erasing their types.
            jacobian: jacobian.view_with_steps((0, 0), (Rows::DIM, T::Dim::DIM), (0, 0)),
        }
    }

    /// Stable identity of the differentiated state.
    pub fn variable(&self) -> BlockId {
        self.variable
    }

    /// Borrow the coefficients with their original lifetime and strides.
    ///
    /// The dynamic dimensions and strides are metadata, not owned storage.
    /// Returning this view does not allocate or copy coefficients.
    /// Convert it to a faer view with [`IntoFaer`](faer_ext::IntoFaer), preserving
    /// its borrow lifetime, shape, and strides:
    ///
    /// ```no_run
    /// use faer_ext::IntoFaer;
    /// # fn consume(block: &fagra::JacobianBlock<'_>) {
    /// let matrix: faer::MatRef<'_, f64> = block.jacobian().into_faer();
    /// # let _ = matrix;
    /// # }
    /// ```
    ///
    /// `into_faer` panics if a stride cannot be represented as an `isize`.
    pub fn jacobian(&self) -> DMatrixView<'a, R, Dyn, Dyn> {
        self.jacobian
    }
}

/// Statically dispatched output for factor-tagged residuals and Jacobians.
///
/// Implementations may assemble normal equations or retain square-root factors.
/// Normal equations use `g = Jᵀr`, `H = JᵀJ`, and solve `H delta = -g`.
///
/// # Factor scopes
/// Ordinary [`Factor`](crate::Factor) implementations receive an already-scoped
/// sink and emit directly:
///
/// ```no_run
/// # use fagra::{EvaluationError, JacobianBlock, LinearizationSink};
/// # use faer_ext::nalgebra::SVector;
/// # fn ordinary<L: LinearizationSink<Scalar = f64>>(sink: &mut L, residual: &SVector<f64, 1>,
/// #     jacobians: &[JacobianBlock<'_>]) -> Result<(), EvaluationError> {
/// sink.residual(residual, jacobians)?;
/// # Ok(())
/// # }
/// ```
///
/// A [`FactorBatch`](crate::FactorBatch) implementation instead opens one scope
/// for each identity yielded by [`FactorSelection`](crate::FactorSelection):
///
/// ```no_run
/// # use fagra::{EvaluationError, FactorId, JacobianBlock, LinearizationSink};
/// # use faer_ext::nalgebra::SVector;
/// # fn batched<L: LinearizationSink<Scalar = f64>>(sink: &mut L, id: FactorId,
/// #     residual: &SVector<f64, 1>, jacobians: &[JacobianBlock<'_>])
/// #     -> Result<(), EvaluationError> {
/// sink.factor(id, |out| out.residual(residual, jacobians))?;
/// # Ok(())
/// # }
/// ```
///
/// All residual blocks belonging to one factor go inside its single scope.
/// Opening another scope in an ordinary factor would nest scopes; emitting
/// directly from a batch would omit the factor identity. Both are invalid.
pub trait LinearizationSink: Sized {
    /// Scalar shared by all residuals and Jacobians emitted to this sink.
    type Scalar: RealField + Copy;

    /// Emit one factor's complete linearization within an identity scope.
    ///
    /// Scopes cannot nest. Ordinary factors receive an already scoped sink;
    /// batches open one scope per selected factor. Propagate callback failures.
    fn factor(
        &mut self,
        id: FactorId,
        emit: impl FnOnce(&mut Self) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError>;

    /// Emit a local least-squares residual and borrowed Jacobian blocks.
    ///
    /// Ordinary factors emit whitened residuals; robust factors may additionally
    /// apply frozen IRLS weights as documented in [`Factor::linearize`](crate::Factor::linearize).
    ///
    /// An active factor scope is required. Reject invalid identities, dimensions,
    /// or nonfinite values. Repeated variable IDs must be combined correctly or
    /// rejected, never treated as independent variables. Borrowed data may not
    /// be retained after this call without copying into backend-owned storage.
    /// The row dimension may be a concrete `Const<N>` or a variable's associated
    /// dimension, allowing generic factors to emit [`Tangent<T>`](crate::Tangent) residuals.
    /// Residual storage is borrowed and must be contiguous; no allocator is needed.
    fn residual<Rows: DimName, S>(
        &mut self,
        residual: &Matrix<Self::Scalar, Rows, U1, S>,
        jacobians: &[JacobianBlock<'_, Self::Scalar>],
    ) -> Result<(), EvaluationError>
    where
        S: Storage<Self::Scalar, Rows, U1> + IsContiguous;
}
