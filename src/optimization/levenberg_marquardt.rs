//! Gain-ratio LM with reusable linearizations and transactional trial states.

use faer::traits::ComplexField as _;
use faer_ext::nalgebra::{ComplexField as _, RealField as _};

use super::{
    Accept, DampedLeastSquaresBackend, OptimizeOptions, OptimizeReport, Optimizer, Stage,
    TerminationReason, Trial, Workspace, cost, norm_inf,
};
use crate::{
    EvaluationError, Lsmr, Real, SolverError, factors::FactorSchema, marginalization::Priors,
    states::StateSchema, storage::checked_cost,
};

/// Damping and retry controls for [`LevenbergMarquardt`].
#[derive(Debug, Clone, Copy)]
pub struct LmOptions<R: Real = f64> {
    /// Initial dimensionless damping, reset for each optimization call. Default: 1e-3.
    pub initial_damping: R,
    /// Positive lower damping bound. Default: 1e-12.
    pub min_damping: R,
    /// Finite upper damping bound. Default: 1e12.
    pub max_damping: R,
    /// Positive floor for Jacobian column norms in D. Default: 1e-6.
    /// Units depend on the application's whitened residuals and tangent coordinates.
    pub min_column_norm: R,
    /// Minimum actual/predicted reduction ratio, strictly between zero and one.
    /// Default: 1e-4. Every accepted step must also strictly decrease cost.
    pub acceptance_threshold: R,
    /// Maximum solve attempts per linearization, including the accepted attempt.
    /// Must be positive. Default: 16.
    pub max_trials: usize,
}
impl<R: Real> Default for LmOptions<R> {
    fn default() -> Self {
        Self {
            initial_damping: R::from_f64_impl(1e-3),
            min_damping: R::from_f64_impl(1e-12),
            max_damping: R::from_f64_impl(1e12),
            min_column_norm: R::from_f64_impl(1e-6),
            acceptance_threshold: R::from_f64_impl(1e-4),
            max_trials: 16,
        }
    }
}
impl<R: Real> LmOptions<R> {
    fn validate(&self) -> Result<(), SolverError> {
        if self.max_trials == 0
            || [
                self.initial_damping,
                self.min_damping,
                self.max_damping,
                self.min_column_norm,
                self.acceptance_threshold,
            ]
            .iter()
            .any(|x| !x.is_finite() || *x <= R::zero())
            || self.min_damping > self.initial_damping
            || self.initial_damping > self.max_damping
            || self.acceptance_threshold >= R::one()
        {
            return Err(SolverError::InvalidOptions);
        }
        Ok(())
    }
}

/// Reason for the most recently rejected LM attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialFailure {
    /// The backend could not produce a finite linear step.
    LinearSolve,
    /// The undamped model predicted no finite, positive improvement.
    Prediction,
    /// Retraction or nonlinear cost evaluation left the model's valid domain.
    InvalidEvaluation,
    /// Actual improvement was insufficient relative to the local prediction.
    PoorAgreement,
}

/// Diagnostics for the most recent LM call, available on success and error.
#[derive(Debug, Clone, Copy)]
pub struct LmStatistics<R: Real = f64> {
    /// Accepted-state linearizations, including final convergence checks.
    pub linearizations: usize,
    /// Linear solve attempts, including rejected and failed solves.
    pub attempts: usize,
    /// Rejected attempts; all leave accepted estimates intact.
    pub rejected_steps: usize,
    /// Successfully committed nonlinear steps.
    pub accepted_steps: usize,
    /// Cost evaluations, including the initial cost and invalid trial evaluations.
    pub cost_evaluations: usize,
    /// Damping used by the last attempt, or the initial damping before any attempt.
    pub damping: R,
    /// Latest accepted total objective, including constant marginalization cost.
    pub cost: Option<R>,
    /// Gradient infinity norm at the last linearization (may precede a final error).
    pub gradient_norm: Option<R>,
    /// Infinity norm of the last successfully solved step, accepted or rejected.
    pub step_norm: Option<R>,
    /// Most recent rejection reason, retained even if a later retry succeeds.
    pub last_rejection: Option<TrialFailure>,
}
impl<R: Real> Default for LmStatistics<R> {
    fn default() -> Self {
        Self {
            linearizations: 0,
            attempts: 0,
            rejected_steps: 0,
            accepted_steps: 0,
            cost_evaluations: 0,
            damping: R::zero(),
            cost: None,
            gradient_norm: None,
            step_norm: None,
            last_rejection: None,
        }
    }
}

/// Levenberg–Marquardt with scaled diagonal damping and gain-ratio acceptance.
///
/// Defaults to [`Lsmr`], using implicit augmented rows without normal equations.
/// Each accepted-state linearization is cached across retries; rejected trials
/// evaluate costs but never recompute Jacobians. Damping never enters the graph
/// or its marginal priors. Workspace is retained across calls and retries.
///
/// Only strictly decreasing steps with adequate model agreement are accepted.
/// Invalid trial geometry is retried; invalid keys/emissions and invalid geometry
/// at the accepted state are errors. Retraction/evaluator panics unwind through
/// the same rollback guard as GN, rather than being treated as rejected steps.
///
/// Gradient tolerance establishes convergence. Small accepted steps with small
/// cost changes trigger a fresh gradient check; if it fails, return `NoProgress`
/// rather than declaring an over-damped step converged. Set the shared step
/// tolerance to zero to disable stagnation detection; a zero cost tolerance
/// disables the additional cost-change condition.
pub struct LevenbergMarquardt<B: DampedLeastSquaresBackend = Lsmr> {
    /// Method-specific controls; validated before graph evaluation.
    pub options: LmOptions<B::Scalar>,
    /// Optional observer after each accepted step. Default: None.
    /// The cost is current; the gradient still describes the preceding linearization.
    pub on_accept: Option<fn(LmStatistics<B::Scalar>)>,
    work: Workspace<B>,
    statistics: LmStatistics<B::Scalar>,
}
impl<B: DampedLeastSquaresBackend> LevenbergMarquardt<B> {
    /// Construct an optimizer retaining the supplied backend and default controls.
    pub fn new(backend: B) -> Self {
        Self {
            options: LmOptions::default(),
            on_accept: None,
            work: Workspace::new(backend),
            statistics: LmStatistics::default(),
        }
    }
    /// Inspect the most recent call, including after an error.
    pub fn statistics(&self) -> LmStatistics<B::Scalar> {
        self.statistics
    }
    /// Inspect backend diagnostics, including linear iteration counts.
    pub fn backend(&self) -> &B {
        &self.work.backend
    }
}
impl<R: Real> Default for LevenbergMarquardt<Lsmr<R>> {
    fn default() -> Self {
        Self::new(Lsmr::default())
    }
}

impl<S, F, B> Optimizer<S, F> for LevenbergMarquardt<B>
where
    S: StateSchema,
    F: FactorSchema<S, Scalar = S::Scalar>,
    B: DampedLeastSquaresBackend<Scalar = S::Scalar>,
{
    fn optimize(
        &mut self,
        states: &mut S,
        factors: &mut F,
        priors: &mut Priors<S::Scalar>,
        options: &OptimizeOptions<S::Scalar>,
    ) -> Result<OptimizeReport<S::Scalar>, SolverError> {
        self.statistics = LmStatistics::default();
        options.validate()?;
        self.options.validate()?;
        let c = S::Scalar::from_f64_impl;
        let mut lambda = self.options.initial_damping;
        self.statistics.damping = lambda;
        self.work.prepare(states, factors)?;
        self.statistics.cost_evaluations += 1;
        let initial = cost(states, factors, priors)?;
        let constant = priors.constant;
        self.statistics.cost = Some(checked_cost(Ok(initial + constant))?);
        let mut current = initial;
        let mut stalled = false;
        loop {
            if self.work.layout.dimension == 0 {
                return Ok(OptimizeReport {
                    initial_cost: initial + constant,
                    final_cost: current + constant,
                    iterations: self.statistics.accepted_steps,
                    termination: TerminationReason::NoVariables,
                });
            }
            self.statistics.linearizations += 1;
            self.work.linearize(states, factors, priors)?;
            let gradient = self.work.backend.gradient_norm()?;
            self.statistics.gradient_norm = Some(gradient);
            if gradient <= options.gradient_tolerance {
                return Ok(OptimizeReport {
                    initial_cost: initial + constant,
                    final_cost: current + constant,
                    iterations: self.statistics.accepted_steps,
                    termination: TerminationReason::GradientTolerance,
                });
            }
            if stalled {
                return Err(SolverError::NoProgress);
            }
            if self.statistics.accepted_steps == options.max_iterations {
                return Err(SolverError::NoConvergence);
            }
            self.work
                .backend
                .prepare_damping(self.options.min_column_norm)?;
            let mut growth = c(2.0);
            let mut accepted = false;
            for _ in 0..self.options.max_trials {
                self.statistics.attempts += 1;
                self.statistics.damping = lambda;
                let rejection = match self.work.backend.solve_damped(lambda) {
                    Err(SolverError::LinearSolveFailed) => TrialFailure::LinearSolve,
                    Err(error) => return Err(error),
                    Ok(step) => {
                        if step.delta.len() != self.work.layout.dimension {
                            return Err(EvaluationError::DimensionMismatch.into());
                        }
                        let norm =
                            norm_inf(step.delta).map_err(|_| SolverError::LinearSolveFailed)?;
                        self.statistics.step_norm = Some(norm);
                        let predicted = step.predicted_reduction;
                        if !predicted.is_finite() || predicted <= c(0.0) {
                            TrialFailure::Prediction
                        } else {
                            let trial = Trial { states };
                            let mut stage = Stage {
                                delta: step.delta,
                                position: 0,
                            };
                            let value = (|| {
                                trial.states.visit(&mut stage)?;
                                if stage.position != self.work.layout.dimension {
                                    return Err(EvaluationError::DimensionMismatch.into());
                                }
                                self.statistics.cost_evaluations += 1;
                                let value = cost(trial.states, factors, priors)?;
                                checked_cost(Ok(value + constant))?;
                                Ok::<_, SolverError>(value)
                            })();
                            match value {
                                Err(SolverError::Evaluation(
                                    EvaluationError::InvalidEvaluation,
                                )) => TrialFailure::InvalidEvaluation,
                                Err(error) => return Err(error),
                                Ok(value) => {
                                    let actual = current - value;
                                    let ratio = (actual / predicted).min(c(2.0));
                                    if actual > c(0.0) && ratio >= self.options.acceptance_threshold
                                    {
                                        trial.states.visit(&mut Accept).unwrap();
                                        self.work.current = false;
                                        self.statistics.accepted_steps += 1;
                                        self.statistics.cost = Some(value + constant);
                                        stalled = options.step_tolerance > c(0.0)
                                            && norm <= options.step_tolerance
                                            && (options.cost_tolerance == c(0.0)
                                                || actual / current.max(c(1.0))
                                                    <= options.cost_tolerance);
                                        current = value;
                                        // Nielsen's LM update; clamp ratio before cubing to avoid overflow.
                                        let update = (c(1.0)
                                            - (c(2.0) * ratio.min(c(1.0)) - c(1.0)).powi(3))
                                        .max(c(1.0 / 3.0));
                                        lambda = (lambda * update)
                                            .max(self.options.min_damping)
                                            .min(self.options.max_damping);
                                        accepted = true;
                                        if let Some(observe) = self.on_accept {
                                            observe(self.statistics);
                                        }
                                        break;
                                    }
                                    TrialFailure::PoorAgreement
                                }
                            }
                        }
                    }
                };
                self.statistics.rejected_steps += 1;
                self.statistics.last_rejection = Some(rejection);
                if lambda == self.options.max_damping {
                    break;
                }
                lambda = (lambda * growth).min(self.options.max_damping);
                growth = (growth * c(2.0)).min(self.options.max_damping.max(c(2.0)));
            }
            if !accepted {
                return Err(SolverError::NoProgress);
            }
        }
    }
}

impl<S, F, B> crate::covariance::CovarianceOptimizer<S, F> for LevenbergMarquardt<B>
where
    S: StateSchema,
    F: FactorSchema<S, Scalar = S::Scalar>,
    B: DampedLeastSquaresBackend<Scalar = S::Scalar> + crate::covariance::CovarianceBackend,
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
