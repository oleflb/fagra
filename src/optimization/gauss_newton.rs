//! Full-step Gauss–Newton; shared graph mechanics live in the parent module.

use super::{
    Accept, LeastSquaresBackend, OptimizeOptions, OptimizeReport, Optimizer, Stage,
    TerminationReason, Trial, Workspace, cost, norm_inf,
};
use crate::{
    DenseNormalCholesky, EvaluationError, SolverError, factors::FactorSchema,
    marginalization::Priors, states::StateSchema, storage::checked_cost,
};
use faer::traits::math_utils::{max, one, zero};

/// Full-step Gauss–Newton with a replaceable least-squares backend.
///
/// The default backend is [`DenseNormalCholesky`]. Reuse this object across calls
/// to retain numerical and preparation buffers. Every call rebuilds ordering and
/// dependencies into retained storage, so edits and switching graphs are safe.
///
/// Finite full steps are accepted even if cost increases. There is no damping or
/// line search. A failed trial evaluation restores the last accepted estimates.
pub struct GaussNewton<B = DenseNormalCholesky> {
    work: Workspace<B>,
}

impl<B> GaussNewton<B> {
    /// Solve and extract covariance from the final undamped model.
    /// The returned matrix borrows the problem's retained query workspace.
    #[allow(clippy::type_complexity)]
    pub fn solve_batch_with_covariance<'a, S, F>(
        &mut self,
        problem: &'a mut impl crate::BatchProblem<S, F>,
        options: &OptimizeOptions<S::Scalar>,
        blocks: &[crate::BlockId],
        covariance_options: &crate::CovarianceOptions<S::Scalar>,
    ) -> Result<(OptimizeReport<S::Scalar>, faer::MatRef<'a, S::Scalar>), SolverError>
    where
        S: StateSchema,
        F: FactorSchema<S, Scalar = S::Scalar>,
        B: crate::covariance::CovarianceBackend<Scalar = S::Scalar>,
    {
        problem.run_batch_with_covariance(self, options, blocks, covariance_options)
    }

    /// Solve an ordinary or tracked problem, retaining this solver's workspace.
    pub fn solve_batch<S, F>(
        &mut self,
        problem: &mut impl crate::BatchProblem<S, F>,
        options: &OptimizeOptions<S::Scalar>,
    ) -> Result<OptimizeReport<S::Scalar>, SolverError>
    where
        S: StateSchema,
        F: FactorSchema<S, Scalar = S::Scalar>,
        B: LeastSquaresBackend<Scalar = S::Scalar>,
    {
        problem.run_batch(self, options)
    }

    /// Construct a reusable optimizer with the chosen backend.
    pub fn new(backend: B) -> Self {
        Self {
            work: Workspace::new(backend),
        }
    }

    /// Inspect backend diagnostics after an optimization call.
    pub fn backend(&self) -> &B {
        &self.work.backend
    }
}

impl<R: crate::Real> Default for GaussNewton<DenseNormalCholesky<R>> {
    fn default() -> Self {
        Self::new(DenseNormalCholesky::default())
    }
}

impl<S, F, B> Optimizer<S, F> for GaussNewton<B>
where
    S: StateSchema,
    F: FactorSchema<S, Scalar = S::Scalar>,
    B: LeastSquaresBackend<Scalar = S::Scalar>,
{
    fn optimize(
        &mut self,
        states: &mut S,
        factors: &mut F,
        priors: &mut Priors<S::Scalar>,
        options: &OptimizeOptions<S::Scalar>,
    ) -> Result<OptimizeReport<S::Scalar>, SolverError> {
        options.validate()?;
        self.work.prepare(states, factors)?;
        let initial_cost = cost(states, factors, priors)?;
        let constant = priors.constant;
        checked_cost(Ok(initial_cost + constant))?;
        let mut current_cost = initial_cost;
        let mut iterations = 0;
        let report = |termination, iterations, final_cost| OptimizeReport {
            initial_cost: initial_cost + constant,
            final_cost: final_cost + constant,
            iterations,
            termination,
        };
        if self.work.layout.dimension == 0 {
            return Ok(report(TerminationReason::NoVariables, 0, current_cost));
        }
        loop {
            self.work.linearize(states, factors, priors)?;
            if self.work.backend.gradient_norm()? <= options.gradient_tolerance {
                return Ok(report(
                    TerminationReason::GradientTolerance,
                    iterations,
                    current_cost,
                ));
            }
            if iterations == options.max_iterations {
                return Err(SolverError::NoConvergence);
            }
            let delta = self.work.backend.solve()?;
            if delta.len() != self.work.layout.dimension {
                return Err(SolverError::LinearSolveFailed);
            }
            let step_norm = norm_inf(delta).map_err(|_| SolverError::LinearSolveFailed)?;
            if step_norm <= options.step_tolerance {
                return Ok(report(
                    TerminationReason::StepTolerance,
                    iterations,
                    current_cost,
                ));
            }
            let trial_cost;
            {
                let trial = Trial { states };
                let mut stage = Stage { delta, position: 0 };
                trial.states.visit(&mut stage)?;
                if stage.position != delta.len() {
                    return Err(EvaluationError::DimensionMismatch.into());
                }
                trial_cost = cost(trial.states, factors, priors)?;
                checked_cost(Ok(trial_cost + constant))?;
                trial.states.visit(&mut Accept).unwrap();
                self.work.current = false;
                // Trial's drop now clears old accepted values after all buffers swap.
            }
            iterations += 1;
            let decrease = current_cost - trial_cost;
            let converged = trial_cost == zero()
                || (decrease >= zero()
                    && decrease / max(&current_cost, &one()) <= options.cost_tolerance);
            current_cost = trial_cost;
            if converged {
                return Ok(report(
                    TerminationReason::CostTolerance,
                    iterations,
                    current_cost,
                ));
            }
        }
    }
}

impl<S, F, B> crate::covariance::CovarianceOptimizer<S, F> for GaussNewton<B>
where
    S: StateSchema,
    F: FactorSchema<S, Scalar = S::Scalar>,
    B: crate::covariance::CovarianceBackend<Scalar = S::Scalar>,
{
    fn optimize_covariance(
        &mut self,
        states: &mut S,
        factors: &mut F,
        priors: &mut Priors<S::Scalar>,
        options: &OptimizeOptions<S::Scalar>,
        blocks: &[crate::BlockId],
        covariance: &mut crate::covariance::CovarianceWorkspace<S::Scalar>,
    ) -> Result<OptimizeReport<S::Scalar>, SolverError> {
        let report = self.optimize(states, factors, priors, options)?;
        self.work
            .covariance(states, factors, priors, blocks, covariance)?;
        Ok(report)
    }
}
