use thiserror::Error;

/// Failure to resolve an opaque handle.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum KeyError {
    /// The handle belongs to a different solver.
    #[error("handle belongs to a different solver")]
    ForeignSolver,
    /// The entry was removed or its slot was reused.
    #[error("handle refers to a removed or replaced entry")]
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
}

/// Failure to edit, evaluate, or optimize a graph.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SolverError {
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
}
