//! Typed factor graphs with homogeneous storage and statically dispatched evaluation.
//!
//! Define variables and constraints, declare their types with [`states!`] and
//! [`factors!`], then insert values, [`optimize`](Solver::optimize), and read
//! estimates through typed keys.
//!
//! # Status
//! Graph construction, generational handles, checked state access, factor and
//! batch insertion, cost evaluation, removal, and schema traversal are implemented.
//! [`Solver::optimize`] and [`Solver::marginalize`] remain `todo!()` stubs and panic
//! when called. Numerical assembly and trial-state management are not implemented.
//!
//! # Quick start: one scalar and one prior
//! [`Variable`] defines how a state changes. [`Factor`] declares its dependencies,
//! evaluates a [`cost`](Factor::cost), and emits residuals and Jacobians. Factors
//! read estimates through [`StateStore`]; the solver manages storage and handles.
//!
//! This model works for insertion and cost evaluation; the optimization call is
//! still a stub. Its complete workflow is compile-checked below.
#![doc = concat!("```no_run\n", include_str!("../examples/scalar_prior.rs"), "\n```")]
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
//! Jacobians differentiate the variable's [`Variable::retract`] convention.
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
mod solver;
mod states;
mod storage;
mod variable;

pub use error::{EvaluationError, KeyError, SolverError};
pub use factors::{Factor, FactorBatch, FactorSelection};
pub use key::{BatchKey, BlockId, FactorId, FactorKey, StateKey};
pub use linearization::{JacobianBlock, LinearizationSink};
pub use solver::Solver;
pub use states::StateStore;
pub use variable::Variable;

/// Internal support for exported schema macros.
///
/// These items must be public for expansion in downstream crates. They are not
/// intended for application use; `doc(hidden)` hides documentation, not access.
/// Changes to this plumbing can affect downstream macro expansions and manual impls.
#[doc(hidden)]
pub mod __private {
    pub use crate::factors::{FactorSchema, FactorStore, FactorVisitor};
    pub use crate::states::{StateSchema, StateVisitor};
    pub use crate::storage::{BatchPool, FactorPool, PoolAccess, StatePool};
}
