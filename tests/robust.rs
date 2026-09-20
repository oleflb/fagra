//! User-defined block Huber: no built-in loss or graph-level wrapper required.
use faer_ext::nalgebra::{RealField, SMatrix, SVector};
use fagra::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorSelection, JacobianBlock,
    LinearizationSink, Solver, StateKey, StateStore,
};
#[allow(dead_code)]
#[path = "../examples/marginalization.rs"]
mod example;
use example::{Difference, Prior, Scalar, States};

struct Pixel<R: RealField + Copy> {
    key: StateKey<Scalar<R>>,
    target: [R; 2],
    robust: bool,
}
impl<R: RealField + Copy> Pixel<R> {
    fn residual<S: StateStore<Scalar<R>>>(
        &self,
        states: &S,
    ) -> Result<SVector<R, 2>, EvaluationError> {
        let x = states.get(self.key)?.0;
        Ok(SVector::<R, 2>::new(
            x - self.target[0],
            x + x - self.target[1],
        ))
    }
    fn cost_and_scale(&self, r: &SVector<R, 2>) -> (R, R) {
        let squared = r.norm_squared();
        let half = R::from_f64(0.5).unwrap();
        // Avoid differentiating sqrt(0) in the quadratic branch.
        if !self.robust || squared <= R::one() {
            (half * squared, R::one())
        } else {
            let norm = squared.sqrt();
            (norm - half, (R::one() / norm).sqrt())
        }
    }
}
impl<R: RealField + Copy, S: StateStore<Scalar<R>>> Factor<S> for Pixel<R> {
    type Scalar = R;
    fn visit_variables(&self, mut v: impl FnMut(BlockId)) {
        v(self.key.block_id());
    }
    fn cost(&self, s: &S) -> Result<R, EvaluationError> {
        Ok(self.cost_and_scale(&self.residual(s)?).0)
    }
    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        s: &S,
        out: &mut L,
    ) -> Result<(), EvaluationError> {
        let r = self.residual(s)?;
        let scale = self.cost_and_scale(&r).1;
        let j = SMatrix::<R, 2, 1>::new(scale, scale + scale);
        out.residual(&(r * scale), &[JacobianBlock::new(self.key, &j)])
    }
}
struct Pixels<R>(std::marker::PhantomData<R>);
struct Payload<R: RealField + Copy>(Pixel<R>);
impl<R: RealField + Copy, S: StateStore<Scalar<R>>> FactorBatch<S> for Pixels<R> {
    type Scalar = R;
    type Factor = Payload<R>;
    fn visit_variables(&self, f: &Payload<R>, v: impl FnMut(BlockId)) {
        <Pixel<R> as Factor<S>>::visit_variables(&f.0, v);
    }
    fn cost(
        &self,
        s: &S,
        mut factors: FactorSelection<'_, Payload<R>>,
    ) -> Result<R, EvaluationError> {
        factors.try_fold(R::zero(), |sum, (_, f)| Ok(sum + f.0.cost(s)?))
    }
    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        s: &S,
        factors: FactorSelection<'_, Payload<R>>,
        out: &mut L,
    ) -> Result<(), EvaluationError> {
        for (id, f) in factors {
            out.factor(id, |out| f.0.linearize(s, out))?;
        }
        Ok(())
    }
}
fagra::factors! { Factors<R> { pixels: Pixel<R>, batches: Batch<Pixels<R>, Payload<R>>, edges: Difference<R>, priors: Prior<R> } }

#[test]
fn blockwise_huber_curvature_and_marginal_surrogate_cost() {
    for batched in [false, true] {
        let mut graph = Solver::<States<f64>, Factors<f64>>::new();
        let x = graph.add(Scalar(0.));
        let pixel = Pixel {
            key: x,
            target: [3., 4.],
            robust: true,
        };
        if batched {
            let batch = graph.add_batch(Pixels(std::marker::PhantomData));
            graph.add_factor_to(batch, Payload(pixel)).unwrap();
        } else {
            graph.add_factor(pixel).unwrap();
        }
        // ||r||=5; one weight 1/5 for the whole pixel, H=(1²+2²)/5=1.
        assert!((graph.joint_covariance(&[x.block_id()]).unwrap()[(0, 0)] - 1.).abs() < 1e-12);
        let y = graph.add(Scalar(-2.2));
        graph
            .add_factor(Difference {
                x,
                y,
                measurement: 0.,
            })
            .unwrap();
        let before = graph.joint_covariance(&[y.block_id()]).unwrap()[(0, 0)];
        graph.marginalize(&[x.block_id()]).unwrap();
        assert!((graph.joint_covariance(&[y.block_id()]).unwrap()[(0, 0)] - before).abs() < 1e-12);
        // x=0 minimizes the frozen model conditional on y=-2.2. The retained
        // surrogate starts at 2.5 + 0.5*2.2², omitting the robust offset of 2.
        // Its minimum of 0.08 is an irreducible QR constant and must survive.
        let report = graph.optimize().unwrap();
        assert!((report.initial_cost - 4.92).abs() < 1e-12);
        assert!((graph.get(y).unwrap().0 - 2.2).abs() < 1e-12);
        assert!((report.final_cost - 0.08).abs() < 1e-12);
        graph.set(y, Scalar(100.)).unwrap();
        assert!((graph.optimize().unwrap().final_cost - 0.08).abs() < 1e-10);
        graph.marginalize(&[y.block_id()]).unwrap();
        assert!((graph.optimize().unwrap().final_cost - 0.08).abs() < 1e-10);
    }
}

#[test]
fn batch_weights_are_per_observation_and_lm_uses_true_cost() {
    let mut graph = Solver::<States<f64>, Factors<f64>>::new();
    let x = graph.add(Scalar(0.));
    let b = graph.add_batch(Pixels(std::marker::PhantomData));
    for target in [[3., 4.], [0., 0.]] {
        graph
            .add_factor_to(
                b,
                Payload(Pixel {
                    key: x,
                    target,
                    robust: true,
                }),
            )
            .unwrap();
    }
    assert!((graph.joint_covariance(&[x.block_id()]).unwrap()[(0, 0)] - 1. / 6.).abs() < 1e-12);
    let (report, _) = graph
        .optimize_with_covariance(
            &mut fagra::LevenbergMarquardt::default(),
            &Default::default(),
            &[x.block_id()],
            &Default::default(),
        )
        .unwrap();
    assert_eq!(report.initial_cost, 4.5);
    assert!(report.final_cost < report.initial_cost);
}

#[test]
fn mixed_cost_scales_preserve_the_weighted_marginal_model() {
    fn check<R: fagra::Real>() {
        let c = R::from_f64_impl;
        for batched in [false, true] {
            let mut graph = Solver::<States<R>, Factors<R>>::new();
            let large = graph.add(Scalar(c(2000.)));
            graph
                .add_factor(Prior {
                    variable: large,
                    measurement: c(0.),
                })
                .unwrap();
            let x = graph.add(Scalar(c(0.)));
            let y = graph.add(Scalar(c(-2.2)));
            let pixel = Pixel {
                key: x,
                target: [c(3.), c(4.)],
                robust: true,
            };
            if batched {
                let b = graph.add_batch(Pixels(std::marker::PhantomData));
                graph.add_factor_to(b, Payload(pixel)).unwrap();
            } else {
                graph.add_factor(pixel).unwrap();
            }
            graph
                .add_factor(Difference {
                    x,
                    y,
                    measurement: c(0.),
                })
                .unwrap();
            graph
                .marginalize(&[large.block_id(), x.block_id()])
                .unwrap();
            let report = graph.optimize().unwrap();
            // QR also transforms the 2000-sized residual; allow its roundoff.
            // Costs describe only the emitted model, without the robust offset.
            let tolerance = c(4096.) * R::epsilon_impl();
            assert!(
                (report.initial_cost - c(4.92)).abs() < tolerance,
                "initial {:?}",
                report.initial_cost
            );
            assert!(
                (report.final_cost - c(0.08)).abs() < tolerance,
                "final {:?}",
                report.final_cost
            );
            graph.marginalize(&[y.block_id()]).unwrap();
            assert!((graph.optimize().unwrap().final_cost - c(0.08)).abs() < tolerance);
        }
    }
    check::<f32>();
    check::<f64>();
}

struct RepeatedPriors<R: RealField + Copy> {
    key: StateKey<Scalar<R>>,
}
impl<R: RealField + Copy, S: StateStore<Scalar<R>>> FactorBatch<S> for RepeatedPriors<R> {
    type Scalar = R;
    type Factor = ();
    fn visit_variables(&self, _: &(), mut visit: impl FnMut(BlockId)) {
        visit(self.key.block_id());
    }
    fn cost(&self, states: &S, factors: FactorSelection<'_, ()>) -> Result<R, EvaluationError> {
        let r = states.get(self.key)?.0;
        let row_cost = (R::from_f64(0.5).unwrap() * r) * r;
        // Deliberately sum sequentially. Batch-cost rounding must never be
        // interpreted as an additive constant in the marginalized model.
        Ok(factors.fold(R::zero(), |sum, _| sum + row_cost))
    }
    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        factors: FactorSelection<'_, ()>,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let r = SVector::<R, 1>::new(states.get(self.key)?.0);
        let j = SMatrix::<R, 1, 1>::identity();
        for (id, _) in factors {
            sink.factor(id, |out| {
                out.residual(&r, &[JacobianBlock::new(self.key, &j)])
            })?;
        }
        Ok(())
    }
}
fagra::factors! { RepeatedFactors<R> { batches: Batch<RepeatedPriors<R>, ()> } }

#[test]
fn ordinary_batch_cost_rounding_does_not_change_the_marginal_model() {
    fn check<R: fagra::Real>() {
        for residual in [0.1, 0.03] {
            let mut graph = Solver::<States<R>, RepeatedFactors<R>>::new();
            let x = graph.add(Scalar(R::from_f64_impl(residual)));
            let b = graph.add_batch(RepeatedPriors { key: x });
            for _ in 0..10_000 {
                graph.add_factor_to(b, ()).unwrap();
            }
            graph.marginalize(&[x.block_id()]).unwrap();
            let cost = graph.optimize().unwrap().final_cost;
            assert!(cost >= R::zero());
            assert!(cost < R::from_f64_impl(64.) * R::epsilon_impl());
        }
    }
    check::<f32>();
    check::<f64>();
}

#[cfg(feature = "test-support")]
mod properties {
    use super::*;
    use fagra::testing::{
        FactorProperty, TestFactor, TestFactorBatch, TestStates, check_factor, check_factor_batch,
        proptest::prelude::*,
    };
    #[derive(Debug, Clone)]
    struct Case {
        x: f64,
        target: [f64; 2],
    }
    impl TestFactor for Case {
        type Factor<R: RealField + Copy> = Pixel<R>;
        fn cases() -> impl Strategy<Value = Self> {
            prop_oneof![
                Just(Self {
                    x: 0.,
                    target: [0., 0.]
                }),
                Just(Self {
                    x: 0.,
                    target: [1., 0.]
                }),
                (-3.0..3.0, -5.0..5.0, -5.0..5.0).prop_map(|(x, a, b)| Self { x, target: [a, b] })
            ]
        }
        fn build<R: RealField + Copy>(&self, states: &mut TestStates<R>) -> Pixel<R> {
            Pixel {
                key: states.insert(Scalar(R::from_f64(self.x).unwrap())),
                target: self.target.map(|x| R::from_f64(x).unwrap()),
                robust: true,
            }
        }
    }
    impl TestFactorBatch for Case {
        type Batch<R: RealField + Copy> = Pixels<R>;
        fn cases() -> impl Strategy<Value = Self> {
            <Self as TestFactor>::cases()
        }
        fn build<R: RealField + Copy>(
            &self,
            states: &mut TestStates<R>,
        ) -> (Pixels<R>, Vec<Payload<R>>) {
            let p = <Self as TestFactor>::build(self, states);
            let second = Pixel {
                key: p.key,
                target: [R::zero(); 2],
                robust: true,
            };
            (
                Pixels(std::marker::PhantomData),
                vec![Payload(p), Payload(second)],
            )
        }
    }
    #[derive(Debug)]
    struct Raw(Case);
    impl TestFactor for Raw {
        type Factor<R: RealField + Copy> = Pixel<R>;
        fn cases() -> impl Strategy<Value = Self> {
            <Case as TestFactor>::cases().prop_map(Self)
        }
        fn build<R: RealField + Copy>(&self, states: &mut TestStates<R>) -> Pixel<R> {
            let mut pixel = <Case as TestFactor>::build(&self.0, states);
            pixel.robust = false;
            pixel
        }
    }
    #[test]
    fn raw_derivatives_and_robust_local_models() {
        check_factor::<Raw, f64>(FactorProperty::Jacobians);
        check_factor::<Raw, f32>(FactorProperty::Jacobians);
        check_factor::<Case, f64>(FactorProperty::LocalModel);
        check_factor::<Case, f32>(FactorProperty::LocalModel);
        check_factor_batch::<Case, f64>(FactorProperty::LocalModel);
        check_factor_batch::<Case, f32>(FactorProperty::LocalModel);
    }

    // All three models have the correct gradient. Only MODE=2 assigns constants
    // independently to each observation and therefore obeys batch additivity.
    struct OffsetBatch<R, const MODE: u8>(Pixels<R>);
    impl<R: RealField + Copy, S: StateStore<Scalar<R>>, const MODE: u8> FactorBatch<S>
        for OffsetBatch<R, MODE>
    {
        type Scalar = R;
        type Factor = Payload<R>;
        fn visit_variables(&self, factor: &Payload<R>, visit: impl FnMut(BlockId)) {
            <Pixels<R> as FactorBatch<S>>::visit_variables(&self.0, factor, visit);
        }
        fn cost(
            &self,
            states: &S,
            factors: FactorSelection<'_, Payload<R>>,
        ) -> Result<R, EvaluationError> {
            let count = factors.len();
            let extra = match MODE {
                0 => usize::from(count == 0),
                1 => usize::from(count > 0),
                _ => count,
            };
            Ok(self.0.cost(states, factors)? + R::from_f64(extra as f64).unwrap())
        }
        fn linearize<L: LinearizationSink<Scalar = R>>(
            &self,
            states: &S,
            factors: FactorSelection<'_, Payload<R>>,
            sink: &mut L,
        ) -> Result<(), EvaluationError> {
            self.0.linearize(states, factors, sink)
        }
    }
    #[derive(Debug, Clone)]
    struct OffsetCase<const MODE: u8>;
    impl<const MODE: u8> TestFactorBatch for OffsetCase<MODE> {
        type Batch<R: RealField + Copy> = OffsetBatch<R, MODE>;
        fn cases() -> impl Strategy<Value = Self> {
            Just(Self)
        }
        fn build<R: RealField + Copy>(
            &self,
            states: &mut TestStates<R>,
        ) -> (OffsetBatch<R, MODE>, Vec<Payload<R>>) {
            let (model, factors) = <Case as TestFactorBatch>::build(
                &Case {
                    x: 0.,
                    target: [3., 4.],
                },
                states,
            );
            (OffsetBatch(model), factors)
        }
        fn config() -> proptest::test_runner::Config {
            proptest::test_runner::Config {
                cases: 1,
                failure_persistence: None,
                ..Default::default()
            }
        }
    }
    #[test]
    fn batch_cost_checker_rejects_nonadditive_constants() {
        fn rejected<R: fagra::testing::TestScalar, const MODE: u8>(expected: &str) {
            let failure = std::panic::catch_unwind(|| {
                check_factor_batch::<OffsetCase<MODE>, R>(FactorProperty::LocalModel)
            })
            .expect_err("nonadditive batch passed");
            let message = failure
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| failure.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(message.contains(expected), "unexpected failure: {message}");
        }
        rejected::<f32, 0>("empty batch cost");
        rejected::<f64, 0>("empty batch cost");
        rejected::<f32, 1>("full vs summed singleton costs");
        rejected::<f64, 1>("full vs summed singleton costs");
        check_factor_batch::<OffsetCase<2>, f32>(FactorProperty::LocalModel);
        check_factor_batch::<OffsetCase<2>, f64>(FactorProperty::LocalModel);
    }
}
