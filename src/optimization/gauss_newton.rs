//! Full-step Gauss–Newton; shared graph mechanics live in the parent module.

use super::{
    Accept, CheckedSink, Layout, LeastSquaresBackend, Linearize, OptimizeOptions, OptimizeReport,
    Optimizer, Stage, TerminationReason, Trial, cost, norm_inf,
};
use crate::{
    DenseNormalCholesky, EvaluationError, SolverError, factors::FactorSchema, states::StateSchema,
};

/// Full-step Gauss–Newton with a replaceable least-squares backend.
///
/// The default backend is [`DenseNormalCholesky`]. Reuse this object across calls
/// to retain numerical and preparation buffers. Every call rebuilds ordering and
/// dependencies into retained storage, so edits and switching graphs are safe.
///
/// Finite full steps are accepted even if cost increases. There is no damping or
/// line search. A failed trial evaluation restores the last accepted estimates.
pub struct GaussNewton<B = DenseNormalCholesky> {
    backend: B,
    layout: Layout,
    seen: Vec<bool>,
    block_marks: Vec<usize>,
    columns: Vec<usize>,
}

impl<B> GaussNewton<B> {
    /// Construct a reusable optimizer with the chosen backend.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            layout: Layout::default(),
            seen: Vec::new(),
            block_marks: Vec::new(),
            columns: Vec::new(),
        }
    }
}

impl Default for GaussNewton<DenseNormalCholesky> {
    fn default() -> Self {
        Self::new(DenseNormalCholesky::default())
    }
}

impl<S: StateSchema, F: FactorSchema<S>, B: LeastSquaresBackend> Optimizer<S, F>
    for GaussNewton<B>
{
    fn optimize(
        &mut self,
        states: &mut S,
        factors: &mut F,
        options: &OptimizeOptions,
    ) -> Result<OptimizeReport, SolverError> {
        options.validate()?;
        self.layout.clear();
        states.visit(&mut self.layout)?;
        factors.visit(&mut self.layout)?;
        let initial_cost = cost(states, factors)?;
        let mut current_cost = initial_cost;
        let mut iterations = 0;
        let report = |termination, iterations, final_cost| OptimizeReport {
            initial_cost,
            final_cost,
            iterations,
            termination,
        };
        if self.layout.dimension == 0 {
            return Ok(report(TerminationReason::NoVariables, 0, current_cost));
        }
        self.backend.prepare(self.layout.dimension)?;
        self.seen.resize(self.layout.factors.len(), false);
        self.block_marks.resize(self.layout.blocks.len(), 0);
        self.columns.clear();
        self.columns.reserve(
            self.layout
                .factors
                .iter()
                .map(|p| p.dependencies.len())
                .max()
                .unwrap_or(0),
        );

        loop {
            self.backend.clear();
            self.seen.fill(false);
            self.block_marks.fill(0);
            {
                let mut sink = CheckedSink {
                    backend: &mut self.backend,
                    layout: &self.layout,
                    seen: &mut self.seen,
                    block_marks: &mut self.block_marks,
                    columns: &mut self.columns,
                    emission: 0,
                    allowed: 0..0,
                    next_expected: 0,
                    active: None,
                    failed: false,
                };
                factors.visit(&mut Linearize {
                    states,
                    sink: &mut sink,
                    position: 0,
                })?;
                if sink.failed || sink.seen.iter().any(|seen| !seen) {
                    return Err(EvaluationError::InvalidEmission.into());
                }
            }
            if self.backend.gradient_norm()? <= options.gradient_tolerance {
                return Ok(report(
                    TerminationReason::GradientTolerance,
                    iterations,
                    current_cost,
                ));
            }
            if iterations == options.max_iterations {
                return Err(SolverError::NoConvergence);
            }
            let delta = self.backend.solve()?;
            if delta.len() != self.layout.dimension {
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
                trial_cost = cost(trial.states, factors)?;
                trial.states.visit(&mut Accept).unwrap();
                // Trial's drop now clears old accepted values after all buffers swap.
            }
            iterations += 1;
            let decrease = current_cost - trial_cost;
            let converged = trial_cost == 0.0
                || (decrease >= 0.0 && decrease / current_cost.max(1.0) <= options.cost_tolerance);
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
