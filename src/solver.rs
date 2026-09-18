use crate::{
    BatchKey, BlockId, DenseNormalCholesky, Factor, FactorBatch, FactorKey, GaussNewton, KeyError,
    OptimizeOptions, OptimizeReport, Real, SolverError, StateKey, StateStore, Variable,
    factors::{FactorSchema, FactorStore},
    marginalization::{MarginalizationOptions, MarginalizationReport, Marginalizer, Priors},
    optimization::Optimizer,
    states::{StateSchema, StateVisitor},
    storage::{BatchPool, FactorPool, PoolAccess, StatePool},
};

/// A graph over a declared state schema `S` and factor schema `F`.
///
/// Both schemas share one scalar type, inferred from `S::Scalar`. The retained
/// optimizer, factor costs, stopping controls, and reports use that same precision.
///
/// Created from [`states!`](crate::states!) and [`factors!`](crate::factors!)
/// declarations. Storage, checked graph insertion, factor costs, and removal work.
/// Dense full-step Gauss–Newton is available through [`optimize`](Self::optimize).
/// Bulk square-root marginalization replaces incident factors with internal priors.
pub struct Solver<S: StateSchema, F> {
    states: S,
    factors: F,
    optimizer: GaussNewton<DenseNormalCholesky<S::Scalar>>,
    priors: Priors<S::Scalar>,
    marginalizer: Marginalizer<S::Scalar>,
}

impl<S, F> Solver<S, F>
where
    S: StateSchema,
    F: FactorSchema<S, Scalar = S::Scalar>,
{
    /// Create an empty graph with the declared storage families.
    pub fn new() -> Self {
        Self {
            states: S::default(),
            factors: F::default(),
            optimizer: GaussNewton::default(),
            priors: Priors::default(),
            marginalizer: Marginalizer::default(),
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

    /// Optimize with full-step Gauss–Newton and dense normal-equation Cholesky.
    ///
    /// Uses [`OptimizeOptions::default`] and retains workspace across calls.
    /// All stored variables contribute coordinates, so unconstrained variables
    /// can make a required linear solve singular. Finite full steps are accepted
    /// even if cost increases; there is no damping or line search.
    ///
    /// Failed trial evaluation discards all trial values, retaining estimates
    /// accepted by earlier iterations. Iteration exhaustion returns `NoConvergence`.
    pub fn optimize(&mut self) -> Result<OptimizeReport<S::Scalar>, SolverError> {
        self.optimizer.optimize(
            &mut self.states,
            &mut self.factors,
            &mut self.priors,
            &OptimizeOptions::default(),
        )
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
    pub fn optimize_with<O: Optimizer<S, F>>(
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

    /// Jointly eliminate the supplied states using square-root QR.
    ///
    /// The caller selects identities with [`StateKey::block_id`]; no age or budget
    /// policy is imposed. Duplicates are ignored and an empty selection is a no-op.
    /// Linearizes at the current estimates without optimizing first. Only incident
    /// factors (including individual batch payloads and old priors) are absorbed.
    /// Their replacement is exact for the linearized objective, up to rank tolerance.
    ///
    /// Internal priors use fixed Lie-group references and need no schema registration.
    /// Removed state/factor handles become stale; surviving handles remain valid.
    /// Errors during validation or evaluation leave the graph unchanged. Empty
    /// batches remain reusable until explicitly retired with [`Self::remove_batch`].
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
    /// imposed. Use LSMR for optimization if normal equations must also be avoided
    /// in the nonlinear solve; [`Self::optimize`] still defaults to Cholesky.
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

impl<S, F> Default for Solver<S, F>
where
    S: StateSchema,
    F: FactorSchema<S, Scalar = S::Scalar>,
{
    fn default() -> Self {
        Self::new()
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
