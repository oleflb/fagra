//! Opt-in mutation tracking. Ordinary problems contain no tracking metadata.
use crate::{
    BatchKey, BlockId, Factor, FactorBatch, FactorId, FactorKey, KeyError, OptimizeOptions,
    OptimizeReport, Problem, SolverError, StateKey, Variable,
    factors::{FactorSchema, FactorStore},
    optimization::Optimizer,
    states::StateSchema,
    storage::{BatchPool, FactorPool, PoolAccess, StatePool},
};

/// Successful external changes since tracking began. Rebuild supersedes earlier edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemChange {
    /// A variable was inserted.
    StateAdded(BlockId),
    /// An accepted estimate was replaced.
    StateSet(BlockId),
    /// An ordinary or batched factor was inserted.
    FactorAdded(FactorId),
    /// A factor was removed.
    FactorRemoved(FactorId),
    /// Cached models must be reconstructed from the current problem.
    Rebuild,
}

/// An owning, opt-in change journal around a [`Problem`].
///
/// Mutations go through this wrapper; there is deliberately no mutable dereference.
/// Changes to externally shared evaluator data require [`Self::invalidate_all`].
/// No incremental numerical solver is implemented yet.
///
/// Mutable access cannot bypass the journal:
/// ```compile_fail
/// use fagra::{Problem, TrackedProblem};
/// fagra::states! { States {} }
/// fagra::factors! { Factors {} }
/// let mut tracked = TrackedProblem::new(Problem::<States, Factors>::new());
/// let untracked: &mut Problem<States, Factors> = &mut *tracked;
/// ```
pub struct TrackedProblem<S: StateSchema, F> {
    problem: Problem<S, F>,
    changes: Vec<ProblemChange>,
}

impl<S: StateSchema, F: FactorSchema<S, Scalar = S::Scalar>> TrackedProblem<S, F> {
    /// Begin tracking an existing problem. Its first numerical model needs a rebuild.
    pub fn new(problem: Problem<S, F>) -> Self {
        Self {
            problem,
            changes: vec![ProblemChange::Rebuild],
        }
    }

    /// Stop tracking and return the graph with its accepted estimates and priors.
    pub fn into_inner(self) -> Problem<S, F> {
        self.problem
    }

    /// Inspect pending changes without consuming them.
    pub fn changes(&self) -> &[ProblemChange] {
        &self.changes
    }

    /// Invalidate cached information, including externally changed evaluator data.
    pub fn invalidate_all(&mut self) {
        self.changes.clear();
        self.changes.push(ProblemChange::Rebuild);
    }

    /// Insert a variable and record its stable identity.
    pub fn add<T: Variable<Scalar = S::Scalar>>(&mut self, value: T) -> StateKey<T>
    where
        S: PoolAccess<StatePool<T>>,
    {
        let key = self.problem.add(value);
        self.changes.push(ProblemChange::StateAdded(key.block_id()));
        key
    }

    /// Replace an estimate, recording only successful edits.
    pub fn set<T: Variable<Scalar = S::Scalar>>(
        &mut self,
        key: StateKey<T>,
        value: T,
    ) -> Result<(), KeyError>
    where
        S: PoolAccess<StatePool<T>>,
    {
        self.problem.set(key, value)?;
        self.changes.push(ProblemChange::StateSet(key.block_id()));
        Ok(())
    }

    /// Insert an ordinary factor after dependency validation.
    pub fn add_factor<T: Factor<S, Scalar = S::Scalar>>(
        &mut self,
        factor: T,
    ) -> Result<FactorKey<T>, SolverError>
    where
        F: PoolAccess<FactorPool<T>>,
    {
        let key = self.problem.add_factor(factor)?;
        self.changes
            .push(ProblemChange::FactorAdded(key.factor_id()));
        Ok(key)
    }

    /// Create an empty batch; empty models contribute no numerical information.
    pub fn add_batch<B: FactorBatch<S, Scalar = S::Scalar>>(&mut self, model: B) -> BatchKey<B>
    where
        F: PoolAccess<BatchPool<B, B::Factor>>,
    {
        self.problem.add_batch(model)
    }

    /// Retire an empty batch.
    pub fn remove_batch<B: FactorBatch<S, Scalar = S::Scalar>>(
        &mut self,
        batch: BatchKey<B>,
    ) -> Result<(), SolverError>
    where
        F: PoolAccess<BatchPool<B, B::Factor>>,
    {
        self.problem.remove_batch(batch)
    }

    /// Insert a batch payload and record its factor identity.
    pub fn add_factor_to<B: FactorBatch<S, Scalar = S::Scalar>>(
        &mut self,
        batch: BatchKey<B>,
        factor: B::Factor,
    ) -> Result<FactorKey<B::Factor>, SolverError>
    where
        F: PoolAccess<BatchPool<B, B::Factor>>,
    {
        let key = self.problem.add_factor_to(batch, factor)?;
        self.changes
            .push(ProblemChange::FactorAdded(key.factor_id()));
        Ok(key)
    }

    /// Remove a factor without exposing mutable graph access.
    pub fn remove_factor<T>(&mut self, key: FactorKey<T>) -> Result<(), SolverError>
    where
        F: FactorStore<S, T, Scalar = S::Scalar>,
    {
        self.problem.remove_factor(key)?;
        self.changes
            .push(ProblemChange::FactorRemoved(key.factor_id()));
        Ok(())
    }

    /// Marginalize selected variables; successful elimination requires rebuilding.
    pub fn marginalize(
        &mut self,
        blocks: &[BlockId],
    ) -> Result<crate::MarginalizationReport, SolverError> {
        self.marginalize_with(blocks, &Default::default())
    }

    /// Marginalize with explicit rank controls.
    pub fn marginalize_with(
        &mut self,
        blocks: &[BlockId],
        options: &crate::MarginalizationOptions<S::Scalar>,
    ) -> Result<crate::MarginalizationReport, SolverError> {
        let report = self.problem.marginalize_with(blocks, options)?;
        if !blocks.is_empty() {
            self.invalidate_all();
        }
        Ok(report)
    }

    /// Evaluate fresh covariance without changing accepted estimates.
    pub fn joint_covariance(
        &mut self,
        blocks: &[BlockId],
    ) -> Result<faer::MatRef<'_, S::Scalar>, SolverError> {
        self.problem.joint_covariance(blocks)
    }

    /// Evaluate fresh covariance with explicit rank controls.
    pub fn joint_covariance_with(
        &mut self,
        blocks: &[BlockId],
        options: &crate::CovarianceOptions<S::Scalar>,
    ) -> Result<faer::MatRef<'_, S::Scalar>, SolverError> {
        self.problem.joint_covariance_with(blocks, options)
    }
}

impl<S: StateSchema, F> std::ops::Deref for TrackedProblem<S, F> {
    type Target = Problem<S, F>;
    fn deref(&self) -> &Self::Target {
        &self.problem
    }
}

mod sealed {
    pub trait Sealed {}
    impl<S: super::StateSchema, F> Sealed for super::Problem<S, F> {}
    impl<S: super::StateSchema, F> Sealed for super::TrackedProblem<S, F> {}
}

/// Sealed batch-solve access for ordinary and tracked problems.
/// Access to a tracked problem invalidates numerical caches before optimization.
pub trait BatchProblem<S: StateSchema, F>: sealed::Sealed {
    #[doc(hidden)]
    fn run_batch<O: Optimizer<S, F>>(
        &mut self,
        solver: &mut O,
        options: &OptimizeOptions<S::Scalar>,
    ) -> Result<OptimizeReport<S::Scalar>, SolverError>;

    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    fn run_batch_with_covariance<O: crate::covariance::CovarianceOptimizer<S, F>>(
        &mut self,
        solver: &mut O,
        options: &OptimizeOptions<S::Scalar>,
        blocks: &[BlockId],
        covariance_options: &crate::CovarianceOptions<S::Scalar>,
    ) -> Result<(OptimizeReport<S::Scalar>, faer::MatRef<'_, S::Scalar>), SolverError>;
}

impl<S: StateSchema, F: FactorSchema<S, Scalar = S::Scalar>> BatchProblem<S, F> for Problem<S, F> {
    fn run_batch_with_covariance<O: crate::covariance::CovarianceOptimizer<S, F>>(
        &mut self,
        solver: &mut O,
        options: &OptimizeOptions<S::Scalar>,
        blocks: &[BlockId],
        covariance_options: &crate::CovarianceOptions<S::Scalar>,
    ) -> Result<(OptimizeReport<S::Scalar>, faer::MatRef<'_, S::Scalar>), SolverError> {
        self.optimize_with_covariance(solver, options, blocks, covariance_options)
    }
    fn run_batch<O: Optimizer<S, F>>(
        &mut self,
        solver: &mut O,
        options: &OptimizeOptions<S::Scalar>,
    ) -> Result<OptimizeReport<S::Scalar>, SolverError> {
        self.optimize_with(solver, options)
    }
}

impl<S: StateSchema, F: FactorSchema<S, Scalar = S::Scalar>> BatchProblem<S, F>
    for TrackedProblem<S, F>
{
    fn run_batch_with_covariance<O: crate::covariance::CovarianceOptimizer<S, F>>(
        &mut self,
        solver: &mut O,
        options: &OptimizeOptions<S::Scalar>,
        blocks: &[BlockId],
        covariance_options: &crate::CovarianceOptions<S::Scalar>,
    ) -> Result<(OptimizeReport<S::Scalar>, faer::MatRef<'_, S::Scalar>), SolverError> {
        self.invalidate_all();
        self.problem
            .optimize_with_covariance(solver, options, blocks, covariance_options)
    }
    fn run_batch<O: Optimizer<S, F>>(
        &mut self,
        solver: &mut O,
        options: &OptimizeOptions<S::Scalar>,
    ) -> Result<OptimizeReport<S::Scalar>, SolverError> {
        self.invalidate_all();
        self.problem.optimize_with(solver, options)
    }
}
