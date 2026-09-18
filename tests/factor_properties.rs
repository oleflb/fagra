//! Runnable batch fixture: shared nonlinear preparation, multiple residual
//! blocks, and payloads that can alias the shared state.
#![cfg(feature = "test-support")]

use faer_ext::nalgebra::{RealField, SMatrix, SVector};
use fagra::testing::{TestFactorBatch, TestStates, proptest::prelude::*};
use fagra::{
    BlockId, EvaluationError, FactorBatch, FactorSelection, JacobianBlock, LinearizationSink,
    StateKey, StateStore,
};

#[allow(dead_code)]
#[path = "../examples/scalar_prior.rs"]
mod scalar;
use scalar::Scalar;

struct SharedSquare<R: RealField + Copy> {
    shared: StateKey<Scalar<R>>,
}

struct Observation<R: RealField + Copy> {
    value: StateKey<Scalar<R>>,
    measurement: R,
}

impl<R: RealField + Copy, S: StateStore<Scalar<R>>> FactorBatch<S> for SharedSquare<R> {
    type Scalar = R;
    type Factor = Observation<R>;

    fn visit_variables(&self, factor: &Self::Factor, mut visitor: impl FnMut(BlockId)) {
        visitor(self.shared.block_id());
        visitor(factor.value.block_id());
    }

    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
    ) -> Result<R, EvaluationError> {
        if factors.is_empty() {
            return Ok(R::zero());
        }
        let x = states.get(self.shared)?.0;
        let square = x * x;
        let mut cost = R::zero();
        for (_, factor) in factors {
            let residual = square - states.get(factor.value)?.0 - factor.measurement;
            // Two identical residual blocks contribute 2 * 0.5 * r².
            cost += residual * residual;
        }
        Ok(cost)
    }

    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        if factors.is_empty() {
            return Ok(());
        }
        let singleton = factors.len() == 1;
        let x = states.get(self.shared)?.0;
        let square = x * x;
        let derivative = x + x;
        for (id, factor) in factors {
            let residual =
                SVector::<R, 1>::new(square - states.get(factor.value)?.0 - factor.measurement);
            let shared = SMatrix::<R, 1, 1>::new(derivative);
            let value = SMatrix::<R, 1, 1>::new(-R::one());
            let combined = shared + value;
            sink.factor(id, |sink| {
                // Equivalent row order, different block boundaries across selections.
                if singleton {
                    let residual = SVector::<R, 2>::repeat(residual[0]);
                    let shared = SMatrix::<R, 2, 1>::repeat(derivative);
                    let value = SMatrix::<R, 2, 1>::repeat(-R::one());
                    let combined = shared + value;
                    return if self.shared == factor.value {
                        sink.residual(&residual, &[JacobianBlock::new(self.shared, &combined)])
                    } else {
                        sink.residual(
                            &residual,
                            &[
                                JacobianBlock::new(self.shared, &shared),
                                JacobianBlock::new(factor.value, &value),
                            ],
                        )
                    };
                }
                for _ in 0..2 {
                    if self.shared == factor.value {
                        sink.residual(&residual, &[JacobianBlock::new(self.shared, &combined)])?;
                    } else {
                        sink.residual(
                            &residual,
                            &[
                                JacobianBlock::new(self.shared, &shared),
                                JacobianBlock::new(factor.value, &value),
                            ],
                        )?;
                    }
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct BatchCase {
    shared: f64,
    observations: Vec<(f64, f64, bool)>,
}

impl TestFactorBatch for BatchCase {
    type Batch<R: RealField + Copy> = SharedSquare<R>;

    fn cases() -> impl Strategy<Value = Self> {
        (
            -3.0..3.0,
            proptest::collection::vec((-3.0..3.0, -3.0..3.0, any::<bool>()), 0..5),
        )
            .prop_map(|(shared, observations)| Self {
                shared,
                observations,
            })
    }

    fn build<R: RealField + Copy>(
        &self,
        states: &mut TestStates<R>,
    ) -> (SharedSquare<R>, Vec<Observation<R>>) {
        let shared = states.insert(Scalar(R::from_f64(self.shared).unwrap()));
        let factors = self
            .observations
            .iter()
            .map(|&(value, measurement, alias)| Observation {
                value: if alias {
                    shared
                } else {
                    states.insert(Scalar(R::from_f64(value).unwrap()))
                },
                measurement: R::from_f64(measurement).unwrap(),
            })
            .collect();
        (SharedSquare { shared }, factors)
    }
}

fagra::factor_batch_tests!(double, BatchCase, f64);
fagra::factor_batch_tests!(single, BatchCase, f32);
