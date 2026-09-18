//! Exact scalar marginalization, checked against an analytical solution.
//!
//! Run with `cargo run --example marginalization`.
//! E(x,y) = 0.5(x-1)^2 + 0.5(y-x-2)^2 + 0.5(y-6)^2.
//! Eliminating x leaves E(y) = 0.25(y-3)^2 + 0.5(y-6)^2.
//! Both problems have y=5 and cost=1.5; the full problem has x=2.

#[allow(dead_code)]
#[path = "scalar_prior.rs"]
mod scalar;
pub use scalar::{Prior, Scalar};

use faer_ext::nalgebra::{RealField, SMatrix, SVector};
use fagra::{
    BlockId, EvaluationError, Factor, GaussNewton, JacobianBlock, KeyError, LinearizationSink,
    Lsmr, OptimizeOptions, Solver, SolverError, StateKey, StateStore,
};

pub struct Difference<R: RealField + Copy = f64> {
    pub x: StateKey<Scalar<R>>,
    pub y: StateKey<Scalar<R>>,
    pub measurement: R,
}

impl<R: RealField + Copy, S: StateStore<Scalar<R>>> Factor<S> for Difference<R> {
    type Scalar = R;

    fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
        visit(self.x.block_id());
        visit(self.y.block_id());
    }

    fn cost(&self, states: &S) -> Result<R, EvaluationError> {
        let r = states.get(self.y)?.0 - states.get(self.x)?.0 - self.measurement;
        Ok(R::from_f64(0.5).unwrap() * r * r)
    }

    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let residual =
            SVector::<R, 1>::new(states.get(self.y)?.0 - states.get(self.x)?.0 - self.measurement);
        let minus = SMatrix::<R, 1, 1>::new(-R::one());
        let plus = -minus;
        if self.x == self.y {
            let zero = SMatrix::<R, 1, 1>::zeros();
            sink.residual(&residual, &[JacobianBlock::new(self.x, &zero)])
        } else {
            sink.residual(
                &residual,
                &[
                    JacobianBlock::new(self.x, &minus),
                    JacobianBlock::new(self.y, &plus),
                ],
            )
        }
    }
}

fagra::states! { pub States<R> { scalars: Scalar<R> } }
fagra::factors! { pub Factors<R> { priors: Prior<R>, differences: Difference<R> } }

pub fn run() -> Result<(), SolverError> {
    let mut estimates = [0.0; 2];
    for (index, marginalize) in [false, true].into_iter().enumerate() {
        let mut solver = Solver::<States<f64>, Factors<f64>>::new();
        let x = solver.add(Scalar(0.0));
        let y = solver.add(Scalar(0.0));
        let x_prior = solver.add_factor(Prior {
            variable: x,
            measurement: 1.0,
        })?;
        let difference = solver.add_factor(Difference {
            x,
            y,
            measurement: 2.0,
        })?;
        let y_prior = solver.add_factor(Prior {
            variable: y,
            measurement: 6.0,
        })?;

        if marginalize {
            // The application supplies the keys. Eliminate before optimization:
            // the new prior must carry the right residual as well as information.
            let report = solver.marginalize(&[x.block_id()])?;
            assert_eq!(report.removed_dof, 1);
            assert_eq!(report.absorbed_factors, 2);
            assert_eq!(report.separator_dof, 1);
            assert_eq!(report.eliminated_rank, 1);
            assert_eq!(report.prior_rows, 1);
            assert!(matches!(solver.get(x), Err(KeyError::Stale)));
            assert!(matches!(
                solver.factor_cost(x_prior),
                Err(SolverError::Key(KeyError::Stale))
            ));
            assert!(matches!(
                solver.factor_cost(difference),
                Err(SolverError::Key(KeyError::Stale))
            ));
            assert_eq!(solver.get(y)?.0, 0.0);
            assert_eq!(solver.factor_cost(y_prior)?, 18.0);
        }

        // QR marginalization and LSMR optimization both avoid normal equations.
        let mut method = GaussNewton::new(Lsmr::default());
        let report = solver.optimize_with(&mut method, &OptimizeOptions::default())?;
        estimates[index] = solver.get(y)?.0;
        assert!((estimates[index] - 5.0).abs() < 1e-10);
        assert!((report.final_cost - 1.5).abs() < 1e-10);
        assert!((solver.factor_cost(y_prior)? - 0.5).abs() < 1e-10);
        if !marginalize {
            assert!((solver.get(x)?.0 - 2.0).abs() < 1e-10);
        }
        println!(
            "{}: y = {:.12}, cost = {:.12}",
            if marginalize {
                "marginalized"
            } else {
                "full graph"
            },
            estimates[index],
            report.final_cost
        );
    }
    assert!((estimates[0] - estimates[1]).abs() < 1e-10);
    Ok(())
}

fn main() -> Result<(), SolverError> {
    run()
}
