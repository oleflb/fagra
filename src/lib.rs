//! Typed factor graphs with homogeneous storage and statically dispatched evaluation.
//!
//! Define variables and constraints, declare their types with [`states!`] and
//! [`factors!`], then insert values, [`optimize`](Solver::optimize), and read
//! estimates through typed keys.
//!
//! # Status
//! Graph construction, generational handles, checked state access, factor and
//! batch insertion, cost evaluation, removal, and Gauss–Newton optimization
//! are implemented. [`Solver::marginalize`] remains a `todo!()` stub.
//!
//! # Quick start: one scalar and one prior
//! [`Variable`] defines how a state changes. [`Factor`] declares its dependencies,
//! evaluates a [`cost`](Factor::cost), and emits residuals and Jacobians. Factors
//! read estimates through [`StateStore`]; the solver manages storage and handles.
//!
//! This complete, tested example optimizes the scalar to its prior measurement.
#![doc = concat!("```\n", include_str!("../examples/scalar_prior.rs"), "\n```")]
//!
//! # Scalar precision
//! Both `f32` and `f64` are supported with monomorphized arithmetic. Declare
//! `States<R>` and `Factors<R>` with the schema macros, then instantiate
//! `Solver<States<f32>, Factors<f32>>` or `Solver<States<f64>, Factors<f64>>`.
//! The macros supply the [`Real`] bound. Non-generic schemas use `f64`.
//! Variables, factors, batches, and sinks declare an associated `Scalar`; schemas
//! and backends must agree on it. Jacobians, costs, and step buffers use that
//! precision throughout. [`OptimizeOptions`] and [`Lsmr`] defaults account for
//! machine epsilon, and remain configurable for the model's scale.
//!
//! # Optimization
//! [`Solver::optimize`] uses [`GaussNewton`] with [`DenseNormalCholesky`] and default
//! stopping controls. [`Solver::optimize_with`] accepts a reusable method and
//! [`OptimizeOptions`], returning an [`OptimizeReport`]. Full steps may increase
//! cost; there is no damping or line search. A required Cholesky solve needs a
//! positive-definite normal matrix, so gauge freedoms need explicit constraints.
//! Select [`Lsmr`] with `GaussNewton::new(Lsmr::default())` for an iterative solve
//! using cached local Jacobians without assembling a global matrix.
//!
//! # Shared evaluation with batches
//! Use [`FactorBatch`] when many factors share computation. Register its model
//! and payload with `Batch<Model, Payload>` syntax in [`factors!`]; there is no
//! public `Batch` type to import or construct. Each payload retains its own
//! [`FactorKey`]. See [factor scopes](LinearizationSink#factor-scopes) for the
//! difference between ordinary and batched emission.
//!
//! # Evaluation contract
//! A factor owns an independent contribution to the objective. A batch shares
//! computation across selected factors without merging their graph identities.
//! Variables implement a Lie group with a fixed nalgebra [`Variable::Dim`].
//! [`Tangent<T>`](Tangent) and [`Jacobian<T>`](Jacobian)
//! derive their dimensions from that type. Jacobians differentiate the variable's
//! right-side [`Variable::retract`] convention; see [`Variable`] for the formulas.
//!
//! The intended allocation contract is reusable workspace for iterations over
//! prepared, unchanged structure. Structural changes may allocate; user evaluators
//! must also avoid allocation to satisfy the end-to-end contract.

#![deny(missing_docs)]

mod dense;
mod error;
mod factors;
mod key;
mod linearization;
mod lsmr;
mod normal;
mod optimization;
mod real;
mod solver;
mod states;
mod storage;
mod variable;

pub use error::{EvaluationError, KeyError, SolverError};
pub use factors::{Factor, FactorBatch, FactorSelection};
pub use key::{BatchKey, BlockId, FactorId, FactorKey, StateKey};
pub use linearization::{JacobianBlock, LinearizationSink};
pub use lsmr::Lsmr;
pub use normal::DenseNormalCholesky;
pub use optimization::{GaussNewton, OptimizeOptions, OptimizeReport, TerminationReason};
pub use real::Real;
pub use solver::Solver;
pub use states::StateStore;
pub use variable::{Jacobian, Tangent, Variable};

/// Internal support for generated schemas and optimizer/backend dispatch.
///
/// These items are public for downstream macro expansion and generic bounds. They are not
/// intended for application use; `doc(hidden)` hides documentation, not access.
/// Changes to this plumbing can affect downstream macro expansions and manual impls.
#[doc(hidden)]
pub mod __private {
    pub use crate::factors::{FactorSchema, FactorStore, FactorVisitor};
    pub use crate::optimization::{LeastSquaresBackend, Optimizer};
    pub use crate::states::{StateSchema, StateVisitor};
    pub use crate::storage::{BatchPool, FactorPool, PoolAccess, StatePool};
}
