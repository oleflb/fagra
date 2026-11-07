use crate::{
    BatchKey, BlockId, DenseNormalCholesky, Factor, FactorBatch, FactorKey, GaussNewton, KeyError,
    OptimizeOptions, OptimizeReport, Real, SolverError, StateKey, StateStore, Variable,
    factors::{BatchFactorStore, FactorSchema, FactorStore},
    marginalization::{MarginalizationOptions, MarginalizationReport, Marginalizer, Priors},
    optimization::Optimizer,
    states::{StateSchema, StateVisitor},
    storage::{BatchPool, FactorPool, PoolAccess, StatePool},
};

/// A graph over a declared state schema `S` and factor schema `F`.
///
/// Both schemas share one scalar type, inferred from `S::Scalar`.
/// This owns states, factors, historical priors, and reusable query workspaces,
/// but no nonlinear optimizer or incremental change tracking.
///
/// Created from [`states!`](crate::states!) and [`factors!`](crate::factors!)
/// declarations. Storage, checked graph insertion, factor costs, and removal work.
/// Nonlinear solvers operate on this graph through their `solve_batch` methods.
/// Bulk square-root marginalization replaces incident factors with internal priors.
pub struct Problem<S: StateSchema, F> {
    checkpoint_owner: std::sync::OnceLock<crate::states::CheckpointIdentity>,
    states: S,
    factors: F,
    priors: Priors<S::Scalar>,
    marginalizer: Marginalizer<S::Scalar>,
    covariance: crate::covariance::CovarianceWorkspace<S::Scalar>,
    covariance_blocks: Vec<crate::CovarianceBlock>,
}

impl<S, F> Problem<S, F>
where
    S: StateSchema,
    F: FactorSchema<S, Scalar = S::Scalar>,
{
    /// Create an empty graph with the declared storage families.
    pub fn new() -> Self {
        Self {
            checkpoint_owner: std::sync::OnceLock::new(),
            states: S::default(),
            factors: F::default(),
            priors: Priors::default(),
            marginalizer: Marginalizer::default(),
            covariance: crate::covariance::CovarianceWorkspace::default(),
            covariance_blocks: Vec::new(),
        }
    }

    /// Insert an application-initialized variable into its registered pool.
    pub fn add<T: Variable<Scalar = S::Scalar>>(&mut self, value: T) -> StateKey<T>
    where
        S: PoolAccess<StatePool<T>>,
    {
        self.states.pool_mut().insert(value)
    }

    /// Borrow the current estimate; reject unknown, removed, or foreign keys.
    pub fn get<T: Variable<Scalar = S::Scalar>>(&self, key: StateKey<T>) -> Result<&T, KeyError>
    where
        S: StateStore<T>,
    {
        self.states.get(key)
    }

    /// Replace an accepted estimate after checking its key, preserving its identity.
    ///
    /// Attached factors and historical marginal-prior anchors remain intact. Invalid
    /// keys leave storage unchanged. As with insertion, the caller supplies a value
    /// in the variable's valid domain; this checks identity, not arbitrary geometry.
    pub fn set<T: Variable<Scalar = S::Scalar>>(
        &mut self,
        key: StateKey<T>,
        value: T,
    ) -> Result<(), KeyError>
    where
        S: PoolAccess<StatePool<T>>,
    {
        self.states.pool_mut().set(key, value)
    }

    /// Capture exact accepted state values and identities using their `Clone` implementations.
    ///
    /// Requires all variables in the [`states!`](crate::states!) schema to implement
    /// `Clone`. Factors, historical priors, anchors, and numerical workspaces are not captured.
    pub fn checkpoint_states(&self) -> crate::StateCheckpoint<S>
    where
        S: crate::SnapshotStates,
    {
        crate::StateCheckpoint {
            owner: *self.checkpoint_owner.get_or_init(Default::default),
            values: self.states.snapshot(),
        }
    }

    /// Restore state values, consuming the checkpoint even on rejection.
    ///
    /// All families are validated before any value changes. Foreign graphs and
    /// changed live state identities are rejected, including removed/reused slots.
    /// Factor edits are allowed; factors, priors, anchors, and workspaces stay intact.
    pub fn restore_states(
        &mut self,
        checkpoint: crate::StateCheckpoint<S>,
    ) -> Result<(), crate::StateCheckpointError>
    where
        S: crate::SnapshotStates,
    {
        if self.checkpoint_owner.get() != Some(&checkpoint.owner) {
            return Err(crate::StateCheckpointError::ForeignGraph);
        }
        self.states.validate_snapshot(&checkpoint.values)?;
        self.states.restore_snapshot(checkpoint.values);
        Ok(())
    }

    /// Insert an ordinary factor after validating its variable dependencies.
    /// On validation failure, storage is unchanged and the supplied factor is dropped.
    pub fn add_factor<T>(&mut self, factor: T) -> Result<FactorKey<T>, SolverError>
    where
        T: Factor<S, Scalar = S::Scalar>,
        F: PoolAccess<FactorPool<T>>,
    {
        let mut dependencies = Dependencies {
            states: &mut self.states,
            result: Ok(()),
        };
        factor.visit_variables(|id| dependencies.check(id));
        dependencies.result?;
        Ok(self.factors.pool_mut().insert(factor))
    }

    /// Create an empty batch from its shared inputs and evaluator.
    ///
    /// Variable dependencies are validated when individual factors are added.
    /// Empty batches remain reusable until the solver is dropped.
    pub fn add_batch<B>(&mut self, model: B) -> BatchKey<B>
    where
        B: FactorBatch<S, Scalar = S::Scalar>,
        F: PoolAccess<BatchPool<B, B::Factor>>,
    {
        self.factors.pool_mut().insert_batch(model)
    }

    /// Retire an empty batch, invalidating its handle and releasing its model.
    /// Nonempty batches are rejected without removing any observations.
    pub fn remove_batch<B>(&mut self, batch: BatchKey<B>) -> Result<(), SolverError>
    where
        B: FactorBatch<S, Scalar = S::Scalar>,
        F: PoolAccess<BatchPool<B, B::Factor>>,
    {
        self.factors.pool_mut().remove_empty(batch)
    }

    /// Add a factor to a batch, validating the batch and all shared/local dependencies.
    /// On validation failure, storage is unchanged and the supplied payload is dropped.
    pub fn add_factor_to<B>(
        &mut self,
        batch: BatchKey<B>,
        factor: B::Factor,
    ) -> Result<FactorKey<B::Factor>, SolverError>
    where
        B: FactorBatch<S, Scalar = S::Scalar>,
        F: PoolAccess<BatchPool<B, B::Factor>>,
    {
        let model = self.factors.pool().model(batch)?;
        let mut dependencies = Dependencies {
            states: &mut self.states,
            result: Ok(()),
        };
        model.visit_variables(&factor, |id| dependencies.check(id));
        dependencies.result?;
        Ok(self.factors.pool_mut().insert_into(batch, factor)?)
    }

    /// Inspect an ordinary factor or batch payload together with current states.
    ///
    /// Borrows storage directly without cloning or allocating. Removed, unknown,
    /// or foreign keys return an error without invoking `inspect`. The callback
    /// may return references borrowing the graph; these prevent graph mutation
    /// until released. Callback errors are returned inside the outer `Result`.
    /// Shared references do not prevent user-defined interior mutation.
    pub fn with_factor<'a, T: 'a, R>(
        &'a self,
        key: FactorKey<T>,
        inspect: impl FnOnce(&'a T, &'a S) -> R,
    ) -> Result<R, KeyError>
    where
        F: FactorStore<S, T, Scalar = S::Scalar>,
    {
        Ok(inspect(self.factors.get_factor(key)?, &self.states))
    }

    /// Inspect a batch payload, its owning shared model, and current states.
    ///
    /// The payload key determines the model; no batch key is needed. Like
    /// [`Self::with_factor`], this borrows storage without cloning or allocating,
    /// rejects invalid keys before calling `inspect`, and allows borrowed results.
    /// Callback errors remain inside the outer `Result`. References are shared;
    /// user-defined interior mutation is still possible.
    pub fn with_batch_factor<'a, T: 'a, R>(
        &'a self,
        key: FactorKey<T>,
        inspect: impl FnOnce(&'a F::Model, &'a T, &'a S) -> R,
    ) -> Result<R, KeyError>
    where
        F: BatchFactorStore<S, T, Scalar = S::Scalar>,
        F::Model: 'a,
    {
        let (model, factor) = self.factors.get_batch_factor(key)?;
        Ok(inspect(model, factor, &self.states))
    }

    /// Evaluate one ordinary or batched factor's nonlinear cost without Jacobians.
    pub fn factor_cost<T>(&self, factor: FactorKey<T>) -> Result<S::Scalar, SolverError>
    where
        F: FactorStore<S, T, Scalar = S::Scalar>,
    {
        self.factors.factor_cost(&self.states, factor)
    }

    /// Discard one factor and its cached contribution, preserving sibling handles.
    pub fn remove_factor<T>(&mut self, factor: FactorKey<T>) -> Result<(), SolverError>
    where
        F: FactorStore<S, T, Scalar = S::Scalar>,
    {
        Ok(self.factors.remove_factor(factor)?)
    }

    /// Optimize with a reusable method and explicit stopping controls.
    ///
    /// The method owns its backend/workspace and can be reused after graph edits
    /// or on another graph. Invalid options are rejected before graph evaluation.
    ///
    /// ```no_run
    /// # use fagra::{Solver, GaussNewton, OptimizeOptions, SolverError};
    /// # fagra::states! { States {} }
    /// # fagra::factors! { Factors {} }
    /// # fn main() -> Result<(), SolverError> {
    /// let mut graph = Solver::<States, Factors>::new();
    /// let mut method = GaussNewton::default();
    /// let options = OptimizeOptions { max_iterations: 100, ..Default::default() };
    /// let report = graph.optimize_with(&mut method, &options)?;
    /// # let _ = report;
    /// # Ok(())
    /// # }
    /// ```
    pub(crate) fn optimize_with<O: Optimizer<S, F>>(
        &mut self,
        method: &mut O,
        options: &OptimizeOptions<S::Scalar>,
    ) -> Result<OptimizeReport<S::Scalar>, SolverError> {
        method.optimize(
            &mut self.states,
            &mut self.factors,
            &mut self.priors,
            options,
        )
    }

    /// Selected joint inverse-information covariance at the current estimates.
    ///
    /// Includes ordinary factors, batches, and internal marginal priors, without
    /// optimizer damping. Rows/columns concatenate each requested block's right
    /// tangent coordinates in request order, including all cross-correlations.
    /// Robust factors contribute their emitted IRLS information, not a sandwich
    /// estimator. Historical information is that retained by marginal priors.
    ///
    /// Evaluates a fresh model without changing estimates. All live coordinates
    /// must be observable, even those not selected; no pseudoinverse is used.
    /// Duplicate, stale, unknown, and foreign identities are rejected. An empty
    /// selection still validates the full information matrix.
    ///
    /// Uses dense O(n²) storage and O(n³) factorization, then O(n² k + n k²) work
    /// for k selected coordinates, never a full inverse. Buffers are retained;
    /// warmed calls within retained capacities allocate no library heap memory.
    /// The result borrows solver storage; use `.to_owned()` to retain a copy.
    pub fn joint_covariance(
        &mut self,
        blocks: &[BlockId],
    ) -> Result<faer::MatRef<'_, S::Scalar>, SolverError> {
        self.joint_covariance_with(blocks, &crate::CovarianceOptions::default())
    }

    /// Selected covariance with explicit numerical rank controls.
    pub fn joint_covariance_with(
        &mut self,
        blocks: &[BlockId],
        options: &crate::CovarianceOptions<S::Scalar>,
    ) -> Result<faer::MatRef<'_, S::Scalar>, SolverError> {
        options.validate()?;
        self.validate_covariance_blocks(blocks)?;
        self.covariance.selected.options = *options;
        self.covariance.evaluate(
            &mut self.states,
            &mut self.factors,
            &mut self.priors,
            blocks,
        )?;
        Ok(self.covariance.selected.matrix())
    }

    /// Selected covariance with typed block access and retained metadata storage.
    ///
    /// Shares the numerical and borrowing contract of [`Self::joint_covariance`].
    /// Metadata is prepared only on view paths; warmed calls reuse its allocation.
    pub fn joint_covariance_view(
        &mut self,
        blocks: &[BlockId],
    ) -> Result<crate::Covariance<'_, S::Scalar>, SolverError> {
        self.joint_covariance_view_with(blocks, &Default::default())
    }

    /// Selected covariance view with explicit numerical rank controls.
    pub fn joint_covariance_view_with(
        &mut self,
        blocks: &[BlockId],
        options: &crate::CovarianceOptions<S::Scalar>,
    ) -> Result<crate::Covariance<'_, S::Scalar>, SolverError> {
        self.joint_covariance_with(blocks, options)?;
        self.covariance_view(blocks)
    }

    fn covariance_view(
        &mut self,
        blocks: &[BlockId],
    ) -> Result<crate::Covariance<'_, S::Scalar>, SolverError> {
        crate::covariance::prepare_blocks(&mut self.states, blocks, &mut self.covariance_blocks)?;
        Ok(crate::Covariance {
            matrix: self.covariance.selected.matrix(),
            blocks: &self.covariance_blocks,
        })
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn optimize_with_covariance_view<O: crate::covariance::CovarianceOptimizer<S, F>>(
        &mut self,
        method: &mut O,
        options: &OptimizeOptions<S::Scalar>,
        blocks: &[BlockId],
        covariance_options: &crate::CovarianceOptions<S::Scalar>,
    ) -> Result<(OptimizeReport<S::Scalar>, crate::Covariance<'_, S::Scalar>), SolverError> {
        let (report, _) =
            self.optimize_with_covariance(method, options, blocks, covariance_options)?;
        Ok((report, self.covariance_view(blocks)?))
    }

    /// Optimize and extract selected covariance from the final undamped model.
    ///
    /// Reuses final Jacobians (and dense Cholesky when available) within this call.
    /// GN cost termination refreshes the model after its last accepted step. LM
    /// convergence already has a current model, so extraction does not reevaluate
    /// factors. LSMR/Schur assemble dense information from cached original-coordinate
    /// Jacobians once; this does not reuse a damped solve or preconditioner.
    /// DenseSchur instead supports retained selections only and factors the undamped
    /// reduced system, without reconstructing full landmark information.
    ///
    /// The covariance uses the dense storage/cost and borrowed-result contract of
    /// [`Self::joint_covariance`]. Selection and options are checked before optimization.
    /// If extraction fails, accepted optimized estimates remain visible. No numerical
    /// cache is reused across calls or graph edits. Evaluators must remain deterministic
    /// for fixed states throughout this call, as required during optimization.
    // Keep the report and borrowed matrix explicit rather than adding a wrapper type.
    #[allow(clippy::type_complexity)]
    pub(crate) fn optimize_with_covariance<O: crate::covariance::CovarianceOptimizer<S, F>>(
        &mut self,
        method: &mut O,
        options: &OptimizeOptions<S::Scalar>,
        blocks: &[BlockId],
        covariance_options: &crate::CovarianceOptions<S::Scalar>,
    ) -> Result<(OptimizeReport<S::Scalar>, faer::MatRef<'_, S::Scalar>), SolverError> {
        covariance_options.validate()?;
        self.validate_covariance_blocks(blocks)?;
        self.covariance.selected.options = *covariance_options;
        let report = method.optimize_covariance(
            &mut self.states,
            &mut self.factors,
            &mut self.priors,
            options,
            blocks,
            &mut self.covariance,
        )?;
        Ok((report, self.covariance.selected.matrix()))
    }

    fn validate_covariance_blocks(&mut self, blocks: &[BlockId]) -> Result<(), SolverError> {
        let mut dependencies = Dependencies {
            states: &mut self.states,
            result: Ok(()),
        };
        for (i, &block) in blocks.iter().enumerate() {
            if blocks[..i].contains(&block) {
                return Err(SolverError::DuplicateCovarianceBlock);
            }
            dependencies.check(block);
        }
        Ok(dependencies.result?)
    }

    /// Jointly eliminate the supplied states using square-root QR.
    ///
    /// The caller selects identities with [`StateKey::block_id`]; no age or budget
    /// policy is imposed. Duplicates are ignored and an empty selection is a no-op.
    /// Linearizes at the current estimates without optimizing first. Only incident
    /// factors (including individual batch payloads and old priors) are absorbed.
    /// Their replacement is exact for the linearized objective, up to rank tolerance.
    /// Robust factors contribute their emitted frozen-weight least-squares model.
    /// Its additive difference from the true robust cost is not inferred or retained;
    /// reported costs thereafter use the surrogate objective. Constants produced by
    /// QR elimination of the emitted rows are retained.
    ///
    /// Internal priors use fixed Lie-group references and need no schema registration.
    /// Removed state/factor handles become stale; surviving handles remain valid.
    /// Errors during validation or evaluation leave the graph unchanged. Empty
    /// batches remain reusable until explicitly retired with [`Self::remove_batch`].
    /// Planning, numerical, and replacement-prior buffers are reused. After
    /// warming the relevant shapes and simultaneous prior count, calls within
    /// retained capacities allocate no library heap memory. User geometry and
    /// evaluators must also avoid allocations for an end-to-end guarantee.
    pub fn marginalize(
        &mut self,
        variables: &[BlockId],
    ) -> Result<MarginalizationReport, SolverError> {
        self.marginalize_with(variables, &MarginalizationOptions::default())
    }

    /// Marginalize with explicit numerical rank controls.
    ///
    /// QR acts directly on Jacobian rows; no normal equations or damping are used.
    /// Rank deficiency is supported, but no nonlinear observability/FEJ policy is
    /// imposed. Use DenseQr or LSMR if normal equations must also be avoided
    /// in the nonlinear solve.
    pub fn marginalize_with(
        &mut self,
        variables: &[BlockId],
        options: &MarginalizationOptions<S::Scalar>,
    ) -> Result<MarginalizationReport, SolverError> {
        self.marginalizer.run(
            &mut self.states,
            &mut self.factors,
            &mut self.priors,
            variables,
            options,
        )
    }
}

impl<S, F> Default for Problem<S, F>
where
    S: StateSchema,
    F: FactorSchema<S, Scalar = S::Scalar>,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Compatibility graph owner with a retained default Gauss–Newton optimizer.
/// New code can use [`Problem`] and solver-side `solve_batch` instead.
pub struct Solver<S: StateSchema, F> {
    problem: Problem<S, F>,
    optimizer: GaussNewton<DenseNormalCholesky<S::Scalar>>,
}

impl<S: StateSchema, F: FactorSchema<S, Scalar = S::Scalar>> Solver<S, F> {
    /// Optimize with a reusable method. Prefer the method's `solve_batch` on a `Problem`.
    pub fn optimize_with<O: Optimizer<S, F>>(
        &mut self,
        method: &mut O,
        options: &OptimizeOptions<S::Scalar>,
    ) -> Result<OptimizeReport<S::Scalar>, SolverError> {
        self.problem.optimize_with(method, options)
    }

    /// Optimize and borrow selected covariance from the final undamped model.
    #[allow(clippy::type_complexity)]
    pub fn optimize_with_covariance<O: crate::covariance::CovarianceOptimizer<S, F>>(
        &mut self,
        method: &mut O,
        options: &OptimizeOptions<S::Scalar>,
        blocks: &[BlockId],
        covariance_options: &crate::CovarianceOptions<S::Scalar>,
    ) -> Result<(OptimizeReport<S::Scalar>, faer::MatRef<'_, S::Scalar>), SolverError> {
        self.problem
            .optimize_with_covariance(method, options, blocks, covariance_options)
    }

    /// Optimize and borrow typed covariance blocks from the final undamped model.
    #[allow(clippy::type_complexity)]
    pub fn optimize_with_covariance_view<O: crate::covariance::CovarianceOptimizer<S, F>>(
        &mut self,
        method: &mut O,
        options: &OptimizeOptions<S::Scalar>,
        blocks: &[BlockId],
        covariance_options: &crate::CovarianceOptions<S::Scalar>,
    ) -> Result<(OptimizeReport<S::Scalar>, crate::Covariance<'_, S::Scalar>), SolverError> {
        self.problem
            .optimize_with_covariance_view(method, options, blocks, covariance_options)
    }

    /// Construct an empty graph and its default optimizer.
    pub fn new() -> Self {
        Self {
            problem: Problem::new(),
            optimizer: GaussNewton::default(),
        }
    }

    /// Optimize with the retained default Gauss–Newton optimizer.
    pub fn optimize(&mut self) -> Result<OptimizeReport<S::Scalar>, SolverError> {
        self.optimizer
            .solve_batch(&mut self.problem, &OptimizeOptions::default())
    }
}

impl<S: StateSchema, F: FactorSchema<S, Scalar = S::Scalar>> Default for Solver<S, F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: StateSchema, F> std::ops::Deref for Solver<S, F> {
    type Target = Problem<S, F>;
    fn deref(&self) -> &Self::Target {
        &self.problem
    }
}

impl<S: StateSchema, F> std::ops::DerefMut for Solver<S, F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.problem
    }
}

struct Dependencies<'a, S> {
    states: &'a mut S,
    result: Result<(), KeyError>,
}

impl<S: StateSchema> Dependencies<'_, S> {
    fn check(&mut self, id: BlockId) {
        if self.result.is_err() {
            return;
        }
        struct Resolve {
            id: BlockId,
            result: Option<Result<(), KeyError>>,
        }

        impl<R: Real> StateVisitor<R> for Resolve {
            type Error = std::convert::Infallible;

            fn pool<T: Variable<Scalar = R>>(
                &mut self,
                pool: &mut StatePool<T>,
            ) -> Result<(), Self::Error> {
                if let Some(result) = pool.validate(self.id) {
                    self.result = Some(result);
                }
                Ok(())
            }
        }

        // O(families) per dependency on graph edits; add a pool-ID routing
        // index if measured insertion throughput warrants it. State reads stay O(1).
        let mut resolve = Resolve { id, result: None };
        self.states.visit(&mut resolve).unwrap();
        self.result = resolve.result.unwrap_or(Err(KeyError::ForeignSolver));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Value = crate::variable::test_support::Vector<1>;
    crate::states! { States { values: Value } }

    #[test]
    fn dependency_validation_distinguishes_stale_and_foreign_states() {
        let mut states = States::default();
        let stale = states.pool_mut().insert(Value::identity());
        states.pool_mut().remove(stale).unwrap();
        let live = states.pool_mut().insert(Value::identity());
        let foreign = States::default().pool_mut().insert(Value::identity());
        for (id, expected) in [
            (stale.block_id(), "stale"),
            (foreign.block_id(), "foreign"),
            (live.block_id(), "live"),
        ] {
            let mut dependencies = Dependencies {
                states: &mut states,
                result: Ok(()),
            };
            dependencies.check(id);
            match expected {
                "stale" => assert!(matches!(dependencies.result, Err(KeyError::Stale))),
                "foreign" => assert!(matches!(dependencies.result, Err(KeyError::ForeignSolver))),
                _ => dependencies.result.unwrap(),
            }
        }
    }
}
