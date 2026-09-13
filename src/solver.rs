use crate::{
    BatchKey, Factor, FactorBatch, FactorKey, KeyError, SolverError, StateKey, StateStore,
    Variable,
    factors::{FactorSchema, FactorStore},
    states::StateSchema,
    storage::{BatchPool, FactorPool, PoolAccess, StatePool},
};

/// A graph over a declared state schema `S` and factor schema `F`.
///
/// Created from [`states!`](crate::states!) and [`factors!`](crate::factors!)
/// declarations. All operations below are currently API stubs; see the
/// [crate status](crate#status).
pub struct Solver<S, F> {
    _states: S,
    _factors: F,
}

impl<S, F> Solver<S, F>
where
    S: StateSchema,
    F: FactorSchema<S>,
{
    /// Create an empty graph with the declared storage families.
    pub fn new() -> Self {
        todo!("API only: solver construction")
    }

    /// Insert an application-initialized variable into its registered pool.
    pub fn add<T: Variable>(&mut self, _value: T) -> StateKey<T>
    where
        S: PoolAccess<StatePool<T>>,
    {
        todo!("API only: variable insertion")
    }

    /// Borrow the current estimate; reject unknown, removed, or foreign keys.
    pub fn get<T: Variable>(&self, _key: StateKey<T>) -> Result<&T, KeyError>
    where
        S: StateStore<T>,
    {
        todo!("API only: current estimate access")
    }

    /// Insert an ordinary factor after validating its variable dependencies.
    pub fn add_factor<T>(&mut self, _factor: T) -> Result<FactorKey<T>, SolverError>
    where
        T: Factor<S>,
        F: PoolAccess<FactorPool<T>>,
    {
        todo!("API only: ordinary factor insertion")
    }

    /// Create an empty batch from its shared inputs and evaluator.
    ///
    /// Variable dependencies are validated when individual factors are added.
    pub fn add_batch<B>(&mut self, _model: B) -> BatchKey<B>
    where
        B: FactorBatch<S>,
        F: PoolAccess<BatchPool<B, B::Factor>>,
    {
        todo!("API only: batch insertion")
    }

    /// Add a factor to a batch, validating the batch and all shared/local dependencies.
    pub fn add_factor_to<B>(
        &mut self,
        _batch: BatchKey<B>,
        _factor: B::Factor,
    ) -> Result<FactorKey<B::Factor>, SolverError>
    where
        B: FactorBatch<S>,
        F: PoolAccess<BatchPool<B, B::Factor>>,
    {
        todo!("API only: batched factor insertion")
    }

    /// Evaluate one ordinary or batched factor's nonlinear cost without Jacobians.
    pub fn factor_cost<T>(&self, _factor: FactorKey<T>) -> Result<f64, SolverError>
    where
        F: FactorStore<S, T>,
    {
        todo!("API only: selected factor cost")
    }

    /// Discard one factor and its cached contribution, preserving sibling handles.
    pub fn remove_factor<T>(&mut self, _factor: FactorKey<T>) -> Result<(), SolverError>
    where
        F: FactorStore<S, T>,
    {
        todo!("API only: factor removal")
    }

    /// Optimize the current graph to convergence, grouping selected factors by batch.
    ///
    /// Rejected trial steps must preserve accepted estimates. Evaluation,
    /// linear-solve, and convergence failures are reported as errors.
    /// The intended implementation uses factor visitors for linearization and
    /// trial cost, and state visitors for staging, acceptance, and rejection.
    /// Failed staging or trial evaluation must discard all trial values before
    /// returning an error, retaining estimates accepted by earlier iterations.
    pub fn optimize(&mut self) -> Result<(), SolverError> {
        todo!("API only: nonlinear optimization")
    }

    /// Eliminate one variable and replace its incident factors with an internal prior.
    ///
    /// Only absorbed factors are removed, including within batches. Marginal
    /// priors require no user registration. The variable's old handle becomes stale.
    /// Manifold prior coordinates and heterogeneous bulk selection remain undesigned.
    pub fn marginalize<T: Variable>(&mut self, _variable: StateKey<T>) -> Result<(), SolverError>
    where
        S: PoolAccess<StatePool<T>>,
    {
        todo!("API only: variable marginalization")
    }
}

impl<S, F> Default for Solver<S, F>
where
    S: StateSchema,
    F: FactorSchema<S>,
{
    fn default() -> Self {
        todo!("API only: empty solver")
    }
}
