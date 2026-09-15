use std::{cell::Cell, rc::Rc};

use faer_ext::{
    IntoFaer,
    nalgebra::{Const, DefaultAllocator, SMatrix, SVector},
};
use fagra::{
    __private::LeastSquaresBackend, BlockId, DenseNormalCholesky, EvaluationError, Factor,
    FactorBatch, FactorSelection, GaussNewton, Jacobian, JacobianBlock, LinearizationSink, Lsmr,
    OptimizeOptions, OptimizeReport, Real, Solver, SolverError, StateKey, StateStore, Tangent,
    TerminationReason, Variable,
};

use super::scalar::{Prior, Scalar};

pub(super) struct Square<R: Real>(pub(super) StateKey<Scalar<R>>);
impl<R: Real, S: StateStore<Scalar<R>>> Factor<S> for Square<R> {
    type Scalar = R;

    fn visit_variables(&self, mut visit: impl FnMut(BlockId)) {
        visit(self.0.block_id());
    }

    fn cost(&self, states: &S) -> Result<R, EvaluationError> {
        let residual = states.get(self.0)?.0.powi(2) - R::one();
        Ok(residual * residual * R::from_f64_impl(0.5))
    }

    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let x = states.get(self.0)?.0;
        sink.residual(
            &SVector::<R, 1>::new(x * x - R::one()),
            &[JacobianBlock::new(self.0, &SMatrix::<R, 1, 1>::new(x + x))],
        )
    }
}

struct Pair<R: Real>(SVector<R, 2>);
impl<R: Real> Variable for Pair<R> {
    type Scalar = R;
    type Dim = Const<2>;
    type Allocator = DefaultAllocator;

    fn identity() -> Self {
        Self(SVector::zeros())
    }
    fn compose(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }
    fn inverse(&self) -> Self {
        Self(-self.0)
    }
    fn exp(delta: &Tangent<Self>) -> Self {
        Self(*delta)
    }
    fn log(&self) -> Tangent<Self> {
        self.0
    }
    fn adjoint(&self) -> Jacobian<Self> {
        SMatrix::identity()
    }
    fn right_jacobian(_: &Tangent<Self>) -> Jacobian<Self> {
        SMatrix::identity()
    }
    fn right_jacobian_inverse(_: &Tangent<Self>) -> Jacobian<Self> {
        SMatrix::identity()
    }
}

struct Observation<R: Real> {
    scalar: StateKey<Scalar<R>>,
    pair: StateKey<Pair<R>>,
}

struct Model<R: Real> {
    target: Rc<Cell<R>>,
    limit: Rc<Cell<R>>,
    invalid_jacobian: Rc<Cell<bool>>,
}

impl<R: Real> Model<R> {
    fn residual<S: StateStore<Scalar<R>> + StateStore<Pair<R>>>(
        &self,
        states: &S,
        factor: &Observation<R>,
    ) -> Result<SVector<R, 3>, EvaluationError> {
        let x = states.get(factor.scalar)?.0;
        let y = states.get(factor.pair)?.0[0];
        let z = states.get(factor.pair)?.0[1];
        if x > self.limit.get() {
            return Err(EvaluationError::InvalidEvaluation);
        }
        let t = self.target.get();
        Ok(SVector::<R, 3>::new(
            x + y - R::from_f64_impl(3.0) * t,
            y + z - R::from_f64_impl(5.0) * t,
            x + R::from_f64_impl(2.0) * z - R::from_f64_impl(7.0) * t,
        ))
    }
}

impl<R: Real, S: StateStore<Scalar<R>> + StateStore<Pair<R>>> FactorBatch<S> for Model<R> {
    type Scalar = R;
    type Factor = Observation<R>;

    fn visit_variables(&self, factor: &Self::Factor, mut visit: impl FnMut(BlockId)) {
        visit(factor.scalar.block_id());
        visit(factor.pair.block_id());
    }

    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
    ) -> Result<R, EvaluationError> {
        let mut cost = R::zero();
        for (_, factor) in factors {
            cost += self.residual(states, factor)?.norm_squared() * R::from_f64_impl(0.5);
        }
        Ok(cost)
    }

    fn linearize<L: LinearizationSink<Scalar = R>>(
        &self,
        states: &S,
        factors: FactorSelection<'_, Self::Factor>,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let jx = SMatrix::<R, 3, 1>::new(R::one(), R::zero(), R::one());
        let mut workspace = SMatrix::<R, 7, 5>::zeros();
        // Both strides are noncontiguous; this storage is dropped after linearization.
        let mut view = workspace.fixed_view_with_steps_mut::<3, 2>((1, 1), (1, 1));
        view.copy_from(&SMatrix::<R, 3, 2>::from_row_slice(
            &[1.0, 0.0, 1.0, 1.0, 0.0, 2.0].map(R::from_f64_impl),
        ));
        if self.invalid_jacobian.get() {
            view[(0, 0)] = R::nan_impl();
        }
        for (id, factor) in factors {
            let jy = JacobianBlock::new(factor.pair, &view);
            let borrowed: faer::MatRef<'_, R> = jy.jacobian().into_faer();
            assert_eq!(borrowed.as_ptr(), view.as_ptr());
            assert_eq!((borrowed.row_stride(), borrowed.col_stride()), (2, 14));
            // Reverse dependency order to exercise cross-block assembly in both orientations.
            sink.factor(id, |sink| {
                sink.residual(
                    &self.residual(states, factor)?,
                    &[jy, JacobianBlock::new(factor.scalar, &jx)],
                )
            })?;
        }
        Ok(())
    }
}

// Exercise different generic names, multiple variable families, and batch-first syntax.
fagra::states! { States<V> { pairs: Pair<V>, scalars: Scalar<V> } }
fagra::factors! { Factors<S> { observations: Batch<Model<S>, Observation<S>>, priors: Prior<S>, squares: Square<S> } }

fn check<R: Real, B: LeastSquaresBackend<Scalar = R>>(backend: B) {
    let mut graph = Solver::<States<R>, Factors<R>>::new();
    assert_eq!(
        graph.optimize().unwrap().termination,
        TerminationReason::NoVariables
    );
    let target = Rc::new(Cell::new(R::one()));
    let limit = Rc::new(Cell::new(R::infinity_impl()));
    let invalid_jacobian = Rc::new(Cell::new(false));
    let x = graph.add(Scalar(R::zero()));
    let y = graph.add(Pair(SVector::zeros()));
    let anchor = graph.add(Scalar(R::zero()));
    let prior = graph
        .add_factor(Prior {
            variable: anchor,
            measurement: R::from_f64_impl(3.0),
        })
        .unwrap();
    let batch = graph.add_batch(Model {
        target: target.clone(),
        limit: limit.clone(),
        invalid_jacobian: invalid_jacobian.clone(),
    });
    let first = graph
        .add_factor_to(batch, Observation { scalar: x, pair: y })
        .unwrap();
    let second = graph
        .add_factor_to(batch, Observation { scalar: x, pair: y })
        .unwrap();
    let mut method = GaussNewton::new(backend);
    let options = OptimizeOptions::<R>::default();
    let tolerance = R::from_f64_impl(1e-10).max(R::epsilon_impl() * R::from_f64_impl(256.0));
    let report: OptimizeReport<R> = graph.optimize_with(&mut method, &options).unwrap();
    assert!(report.iterations > 0);
    assert!(report.final_cost < tolerance);
    assert!((graph.get(x).unwrap().0 - R::one()).abs() < tolerance);
    assert!((graph.get(y).unwrap().0[0] - R::from_f64_impl(2.0)).abs() < tolerance);
    assert!((graph.get(y).unwrap().0[1] - R::from_f64_impl(3.0)).abs() < tolerance);
    assert!(graph.factor_cost(prior).unwrap() < tolerance);
    assert_eq!(
        graph.factor_cost(first).unwrap(),
        graph.factor_cost(second).unwrap()
    );

    target.set(R::from_f64_impl(2.0));
    let (result, count) = super::allocations(|| graph.optimize_with(&mut method, &options));
    assert!(result.unwrap().iterations > 0);
    assert_eq!(count, 0, "warmed generic backend allocated");
    assert!((graph.get(x).unwrap().0 - target.get()).abs() < tolerance);

    // Reject a trial after staging both variable families, preserving accepted values.
    let accepted = (graph.get(x).unwrap().0, graph.get(y).unwrap().0);
    limit.set(R::from_f64_impl(3.0));
    target.set(R::from_f64_impl(4.0));
    assert!(matches!(
        graph.optimize_with(&mut method, &options),
        Err(SolverError::Evaluation(EvaluationError::InvalidEvaluation))
    ));
    assert_eq!((graph.get(x).unwrap().0, graph.get(y).unwrap().0), accepted);
    limit.set(R::infinity_impl());

    target.set(R::nan_impl());
    assert!(matches!(
        graph.factor_cost(first),
        Err(SolverError::Evaluation(EvaluationError::InvalidEvaluation))
    ));
    assert!(matches!(
        graph.optimize_with(&mut method, &options),
        Err(SolverError::Evaluation(EvaluationError::InvalidEvaluation))
    ));
    target.set(R::from_f64_impl(4.0));
    invalid_jacobian.set(true);
    assert!(matches!(
        graph.optimize_with(&mut method, &options),
        Err(SolverError::Evaluation(EvaluationError::InvalidEvaluation))
    ));
    assert_eq!((graph.get(x).unwrap().0, graph.get(y).unwrap().0), accepted);
    invalid_jacobian.set(false);
    graph.remove_factor(first).unwrap();
    graph.optimize_with(&mut method, &options).unwrap();
    assert!((graph.get(x).unwrap().0 - target.get()).abs() < tolerance);

    // Also exercise the retained default optimizer with an actual warmed step.
    target.set(R::from_f64_impl(5.0));
    graph.optimize().unwrap();
    target.set(R::from_f64_impl(6.0));
    let (result, count) = super::allocations(|| graph.optimize());
    assert!(result.unwrap().iterations > 0);
    assert_eq!(count, 0, "warmed default optimizer allocated");
    assert!((graph.get(x).unwrap().0 - target.get()).abs() < tolerance);
}

#[test]
fn both_precisions_and_backends_support_the_complete_graph_path() {
    check::<f32, _>(DenseNormalCholesky::default());
    check::<f32, _>(Lsmr::default());
    check::<f64, _>(DenseNormalCholesky::default());
    check::<f64, _>(Lsmr::default());
}

fn nonlinear<R: Real, B: LeastSquaresBackend<Scalar = R>>(backend: B) {
    let mut graph = Solver::<States<R>, Factors<R>>::new();
    let x = graph.add(Scalar(R::from_f64_impl(2.0)));
    graph.add_factor(Square(x)).unwrap();
    let report = graph
        .optimize_with(&mut GaussNewton::new(backend), &OptimizeOptions::default())
        .unwrap();
    assert!(report.iterations > 1);
    let tolerance = R::from_f64_impl(1e-8).max(R::epsilon_impl() * R::from_f64_impl(16.0));
    assert!((graph.get(x).unwrap().0 - R::one()).abs() < tolerance);
}

#[test]
fn nonlinear_convergence_in_both_precisions() {
    nonlinear::<f32, _>(DenseNormalCholesky::default());
    nonlinear::<f32, _>(Lsmr::default());
    nonlinear::<f64, _>(DenseNormalCholesky::default());
    nonlinear::<f64, _>(Lsmr::default());
}

#[test]
fn empty_generic_schemas_and_precision_aware_defaults() {
    fagra::states! { EmptyStates<R> {} }
    fagra::factors! { EmptyFactors<R> {} }
    assert_eq!(
        Solver::<EmptyStates<f32>, EmptyFactors<f32>>::new()
            .optimize()
            .unwrap()
            .final_cost,
        0.0_f32
    );
    assert_eq!(
        Solver::<EmptyStates<f64>, EmptyFactors<f64>>::new()
            .optimize()
            .unwrap()
            .final_cost,
        0.0_f64
    );
    let single = OptimizeOptions::<f32>::default();
    assert!(single.gradient_tolerance >= f32::EPSILON);
    assert!(single.step_tolerance >= f32::EPSILON);
    assert!(single.cost_tolerance >= f32::EPSILON);
    assert!(Lsmr::<f32>::default().relative_tolerance >= f32::EPSILON);
    let double = OptimizeOptions::<f64>::default();
    assert_eq!(
        (
            double.gradient_tolerance,
            double.step_tolerance,
            double.cost_tolerance
        ),
        (1e-8, 1e-10, 1e-12)
    );
    assert_eq!(Lsmr::<f64>::default().relative_tolerance, 1e-10);
}
