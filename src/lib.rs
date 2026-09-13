//! Typed factor graphs with homogeneous storage and statically dispatched evaluation.
//!
//! Declare variable families with [`states!`] and factor families with [`factors!`].
//! Ordinary constraints implement [`Factor`]; shared evaluators implement [`FactorBatch`]
//! and are registered as [`Batch<B, F>`](Batch).
//!
//! # Status
//! This crate currently defines the API only. Concrete operational methods are
//! `todo!()` stubs and panic when called. Schema declarations and trait bounds can
//! be checked now; optimization, storage management, and marginalization are not implemented.
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

mod error;
mod factors;
mod key;
mod linearization;
mod solver;
mod states;
mod variable;

pub use error::{EvaluationError, KeyError, SolverError};
pub use factors::{Batch, Factor, FactorBatch, FactorSelection};
pub use key::{BatchKey, BlockId, FactorId, FactorKey, StateKey};
pub use linearization::{JacobianBlock, LinearizationSink};
pub use solver::Solver;
pub use states::StateStore;
pub use variable::Variable;

// Public only so exported macros can refer to these bounds in downstream crates.
#[doc(hidden)]
pub mod __private {
    pub use crate::factors::{BatchStore, FactorSchema, FactorStore, StandaloneFactorStore};
    pub use crate::states::StateSchema;
}
