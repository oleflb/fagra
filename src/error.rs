use thiserror::Error;

/// Failure to resolve an opaque handle.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum KeyError {
    /// The handle belongs to a different solver.
    #[error("handle belongs to a different solver")]
    ForeignSolver,
    /// The entry was removed or its slot was reused.
    #[error("handle refers to a removed or reused entry")]
    Stale,
    /// No entry matches the handle.
    #[error("unknown handle")]
    Unknown,
}

/// Failure during nonlinear evaluation or linearization emission.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EvaluationError {
    /// A referenced variable could not be resolved.
    #[error(transparent)]
    Key(#[from] KeyError),
    /// The measurement model is undefined or produced nonfinite output.
    #[error("measurement evaluation is undefined or nonfinite")]
    InvalidEvaluation,
    /// A residual or Jacobian has an incompatible shape.
    #[error("residual or Jacobian dimensions do not match")]
    DimensionMismatch,
    /// Factor scopes or emitted variable identities are invalid.
    #[error("invalid factor scope or emitted variable identity")]
    InvalidEmission,
    /// The selected backend cannot represent the emitted variable coupling.
    #[error("residual structure is unsupported by the selected backend")]
    UnsupportedStructure,
}

/// Failure to edit, evaluate, or optimize a graph.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SolverError {
    /// A covariance selection contains the same variable more than once.
    #[error("duplicate variable in covariance selection")]
    DuplicateCovarianceBlock,
    /// Undamped information is not positive definite or fails the pivot tolerance.
    #[error("undamped information is singular or numerically rank deficient")]
    SingularInformation,
    /// Information assembly, factorization, or covariance produced nonfinite values.
    #[error("invalid or nonfinite covariance information")]
    InvalidInformation,
    /// A numerical rank tolerance was negative, nonfinite, or at least one.
    #[error("relative rank tolerance must be finite and in [0, 1)")]
    InvalidRankTolerance,
    /// Only empty batches may be retired explicitly.
    #[error("batch still contains factors")]
    BatchNotEmpty,
    /// Iteration, convergence, or damping controls are invalid.
    #[error("invalid optimization limits, tolerances, or damping controls")]
    InvalidOptions,
    /// A supplied handle could not be resolved.
    #[error(transparent)]
    Key(#[from] KeyError),
    /// A factor or sink rejected an evaluation.
    #[error(transparent)]
    Evaluation(#[from] EvaluationError),
    /// The numerical backend could not solve the linear system.
    #[error("linear system solve failed")]
    LinearSolveFailed,
    /// Optimization did not meet its convergence criterion.
    #[error("optimization did not converge")]
    NoConvergence,
    /// LM exhausted its retries or stagnated without meeting gradient tolerance.
    #[error("optimization could not make progress; accepted estimates are retained")]
    NoProgress,
}
