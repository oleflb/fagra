use std::{any::Any, fmt::Debug};

use faer_ext::nalgebra::{DMatrix, DimName, Matrix, RealField, Storage, U1, storage::IsContiguous};
use proptest::{
    prelude::*,
    test_runner::{Config, TestCaseError, TestCaseResult},
};

use super::{TestScalar, Tolerance, near, runner};
use crate::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorId, FactorSelection, JacobianBlock,
    KeyError, LinearizationSink, StateKey, StateStore, Variable,
    key::{LocalKey, PoolId, RawKey},
};

/// Generated input data for an ordinary factor's property tests.
///
/// Implement this on a `Debug` fixture containing state values and measurements,
/// not on the evaluator itself. The same fixture builds real and dual evaluators.
pub trait TestFactor: Debug + Sized {
    /// The production evaluator, generic over geometry scalars.
    type Factor<R: RealField + Copy>: Factor<TestStates<R>, Scalar = R>;

    /// Generate valid, finite cases in a smooth residual domain. Shrinking must
    /// preserve that domain. Include zero-residual and nonzero-residual cases.
    fn cases() -> impl Strategy<Value = Self>;

    /// Insert states and construct the factor from their keys and measurements.
    /// Insert the same states in the same order for every scalar. Measurements
    /// must be constants, not derived from the store's possibly perturbed states.
    fn build<R: RealField + Copy>(&self, states: &mut TestStates<R>) -> Self::Factor<R>;

    /// Override case count, seed, shrinking, or persistence.
    fn config() -> Config {
        Config::default()
    }

    /// Override precision-dependent comparison tolerances.
    fn tolerance<R: TestScalar>() -> Tolerance {
        Tolerance::for_scalar::<R>()
    }
}

/// Generated input data for a batch's property tests.
///
/// Like [`TestFactor`], but constructs shared model data and independently
/// identified payloads. Include multi-payload cases to exercise shared work.
pub trait TestFactorBatch: Debug + Sized {
    /// The production shared evaluator.
    type Batch<R: RealField + Copy>: FactorBatch<TestStates<R>, Scalar = R>;

    /// Generate finite, smooth cases, preserving validity during shrinking.
    fn cases() -> impl Strategy<Value = Self>;

    /// Insert each shared state once and reuse its key in the model/payloads.
    /// State and payload order must be identical across scalar types. Keep
    /// measurements independent of the store's possibly perturbed states.
    // Keep the payload projection here so fixtures need no duplicate associated type.
    #[allow(clippy::type_complexity)]
    fn build<R: RealField + Copy>(
        &self,
        states: &mut TestStates<R>,
    ) -> (
        Self::Batch<R>,
        Vec<<Self::Batch<R> as FactorBatch<TestStates<R>>>::Factor>,
    );

    /// Override case count, seed, shrinking, or persistence.
    fn config() -> Config {
        Config::default()
    }

    /// Override precision-dependent comparison tolerances.
    fn tolerance<R: TestScalar>() -> Tolerance {
        Tolerance::for_scalar::<R>()
    }
}

/// Small heterogeneous state store supplied to test fixture builders.
///
/// The runner owns construction and derivative seeds. Insertion assigns stable
/// identities across real/dual builds and applies a seeded right retraction to
/// one variable. This supports factors generic over [`StateStore`], without a
/// solver or a dual-compatible solver schema.
pub struct TestStates<R> {
    pool: PoolId,
    values: Vec<Box<dyn Any>>,
    widths: Vec<usize>,
    seed: Option<(usize, usize, R)>,
}

impl<R: RealField + Copy> TestStates<R> {
    /// Insert a state and return its typed key. To represent aliased inputs,
    /// insert once and pass the same key to multiple factor inputs.
    pub fn insert<T: Variable<Scalar = R> + 'static>(&mut self, mut value: T) -> StateKey<T> {
        let index = self.values.len();
        if let Some((variable, column, seed)) = self.seed
            && variable == index
        {
            assert!(
                column < T::Dim::DIM,
                "state dimensions changed across builds"
            );
            let mut delta = vec![R::zero(); T::Dim::DIM];
            delta[column] = seed;
            value = value.retract(&T::tangent_from_slice(&delta));
        }
        self.widths.push(T::Dim::DIM);
        self.values.push(Box::new(value));
        StateKey::from_raw(raw(self.pool, index))
    }
}

impl<R, T: Variable<Scalar = R> + 'static> StateStore<T> for TestStates<R> {
    fn get(&self, key: StateKey<T>) -> Result<&T, KeyError> {
        if key.raw.pool != self.pool {
            return Err(KeyError::ForeignSolver);
        }
        if key.raw.local.generation != 0 {
            return Err(KeyError::Stale);
        }
        self.values
            .get(key.raw.local.slot as usize)
            .and_then(|value| value.downcast_ref())
            .ok_or(KeyError::Unknown)
    }
}

fn raw(pool: PoolId, index: usize) -> RawKey {
    RawKey {
        pool,
        local: LocalKey {
            slot: index.try_into().expect("too many test entries"),
            generation: 0,
        },
    }
}

/// Separately runnable factor properties, shared by ordinary and batch suites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorProperty {
    /// Finite, nonnegative cost matching the emitted least-squares objective.
    Cost,
    /// Emitted Jacobians versus dual residual derivatives in right coordinates.
    Jacobians,
    /// True nonlinear cost gradient versus the emitted `Jᵀr`, including IRLS.
    ///
    /// Checks finite nonnegative costs, real/dual values, batch selections, and
    /// first-order consistency. Batch costs must be zero for empty selections and
    /// additive across singletons. Does not differentiate emitted weighted residuals
    /// or assert that their squared norm equals cost. Test raw residual derivatives
    /// and the prescribed weighting/curvature independently.
    LocalModel,
}

struct Residual<R> {
    values: Vec<R>,
    jacobians: Vec<(BlockId, DMatrix<R>)>,
}

impl<R: TestScalar> Residual<R> {
    fn derivative(&self, id: BlockId, row: usize, column: usize) -> f64 {
        self.jacobians
            .iter()
            .find(|(variable, _)| *variable == id)
            .map_or(0.0, |(_, j)| j[(row, column)].test_value())
    }
}

struct Scope<R> {
    id: FactorId,
    dependencies: Vec<BlockId>,
    residuals: Vec<Residual<R>>,
    seen: bool,
}

impl<R> Scope<R> {
    fn rows(&self) -> impl Iterator<Item = (&Residual<R>, usize)> + Clone {
        self.residuals
            .iter()
            .flat_map(|r| (0..r.values.len()).map(move |row| (r, row)))
    }
}

struct Capture<R> {
    pool: PoolId,
    widths: Vec<usize>,
    scopes: Vec<Scope<R>>,
    active: Option<usize>,
}

impl<R: RealField + Copy> Capture<R> {
    fn new(states: &TestStates<R>) -> Self {
        Self {
            pool: states.pool,
            widths: states.widths.clone(),
            scopes: Vec::new(),
            active: None,
        }
    }

    fn register(&mut self, id: FactorId, visit: impl FnOnce(&mut Vec<BlockId>)) {
        let mut dependencies = Vec::new();
        visit(&mut dependencies);
        for &id in &dependencies {
            self.width(id);
        }
        self.scopes.push(Scope {
            id,
            dependencies,
            residuals: Vec::new(),
            seen: false,
        });
    }

    fn width(&self, id: BlockId) -> usize {
        assert_eq!(id.0.pool, self.pool, "foreign variable: {id:?}");
        assert_eq!(id.0.local.generation, 0, "stale variable: {id:?}");
        *self
            .widths
            .get(id.0.local.slot as usize)
            .expect("unknown variable")
    }

    fn finish(self, cost: R) -> (R, Self) {
        assert!(
            self.scopes.iter().all(|scope| scope.seen),
            "missing factor scope"
        );
        (cost, self)
    }
}

impl<R: RealField + Copy> LinearizationSink for Capture<R> {
    type Scalar = R;

    fn factor(
        &mut self,
        id: FactorId,
        emit: impl FnOnce(&mut Self) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        assert!(self.active.is_none(), "nested factor scope");
        let index = self
            .scopes
            .iter()
            .position(|s| s.id == id)
            .expect("unknown factor");
        assert!(!self.scopes[index].seen, "duplicate factor scope: {id:?}");
        self.scopes[index].seen = true;
        self.active = Some(index);
        emit(self).expect("factor scope evaluation failed");
        self.active = None;
        Ok(())
    }

    fn residual<Rows: DimName, S>(
        &mut self,
        residual: &Matrix<R, Rows, U1, S>,
        jacobians: &[JacobianBlock<'_, R>],
    ) -> Result<(), EvaluationError>
    where
        S: Storage<R, Rows, U1> + IsContiguous,
    {
        let index = self.active.expect("residual outside factor scope");
        assert!(residual.iter().all(|v| v.is_finite()), "nonfinite residual");
        let mut blocks = Vec::new();
        for block in jacobians {
            let id = block.variable();
            let matrix = block.jacobian();
            assert!(
                self.scopes[index].dependencies.contains(&id),
                "undeclared variable: {id:?}"
            );
            assert!(
                !blocks.iter().any(|(other, _)| *other == id),
                "duplicate Jacobian: {id:?}"
            );
            assert_eq!(
                matrix.shape(),
                (Rows::DIM, self.width(id)),
                "Jacobian dimensions"
            );
            assert!(matrix.iter().all(|v| v.is_finite()), "nonfinite Jacobian");
            blocks.push((id, matrix.into_owned()));
        }
        self.scopes[index].residuals.push(Residual {
            values: residual.as_slice().to_vec(),
            jacobians: blocks,
        });
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum Selection {
    All,
    Empty,
    One(usize),
}

fn ordinary<T: TestFactor, R: RealField + Copy>(
    case: &T,
    states: &mut TestStates<R>,
    _: Selection,
) -> Result<(R, Capture<R>), EvaluationError> {
    let factor = case.build(states);
    let mut sink = Capture::new(states);
    let id = FactorId(raw(states.pool, 0));
    sink.register(id, |ids| factor.visit_variables(|id| ids.push(id)));
    let cost = factor.cost(states)?;
    sink.factor(id, |sink| factor.linearize(states, sink))?;
    Ok(sink.finish(cost))
}

fn batch<T: TestFactorBatch, R: RealField + Copy>(
    case: &T,
    states: &mut TestStates<R>,
    selection: Selection,
) -> Result<(R, Capture<R>), EvaluationError> {
    let (model, payloads) = case.build(states);
    let keys: Vec<_> = (0..payloads.len())
        .map(|i| raw(states.pool, i).local)
        .collect();
    let range = match selection {
        Selection::All => 0..payloads.len(),
        Selection::Empty => 0..0,
        Selection::One(i) => i..i + 1,
    };
    let selected = payloads
        .get(range.clone())
        .ok_or(EvaluationError::InvalidEmission)?;
    let select = || FactorSelection::contiguous(states.pool, &keys[range.clone()], selected);
    let mut sink = Capture::new(states);
    for (id, payload) in select() {
        sink.register(id, |ids| model.visit_variables(payload, |id| ids.push(id)));
    }
    let cost = model.cost(states, select())?;
    model.linearize(states, select(), &mut sink)?;
    Ok(sink.finish(cost))
}

fn cost<R: TestScalar>(
    value: R,
    capture: &Capture<R>,
    t: Tolerance,
    local: bool,
) -> TestCaseResult {
    let value = value.test_value();
    prop_assert!(value.is_finite() && value >= 0.0, "invalid cost: {value:?}");
    if local {
        return Ok(());
    }
    let expected = capture
        .scopes
        .iter()
        .flat_map(|s| &s.residuals)
        .flat_map(|r| &r.values)
        // Scale first: r² can overflow even when 0.5 * r² is finite.
        .map(|v| (0.5 * v.test_value()) * v.test_value())
        .sum();
    near(t, "cost vs emitted residuals", value, expected)
}

fn dual<R: TestScalar>(
    real_cost: R,
    real: &Capture<R>,
    dual_cost: R::Dual,
    dual: &Capture<R::Dual>,
    seed: Option<(usize, usize)>,
    t: Tolerance,
    local: bool,
) -> TestCaseResult {
    prop_assert_eq!(
        &real.widths,
        &dual.widths,
        "state layout changed across builds"
    );
    prop_assert_eq!(real.scopes.len(), dual.scopes.len());
    let mut cost_derivative = 0.0;
    for (scope, other) in real.scopes.iter().zip(&dual.scopes) {
        prop_assert_eq!(scope.id, other.id);
        let rows = scope.rows();
        let other_rows = other.rows();
        prop_assert_eq!(
            rows.clone().count(),
            other_rows.clone().count(),
            "residual dimension changed"
        );
        for (index, ((r, row), (d, other_row))) in rows.zip(other_rows).enumerate() {
            let value = r.values[row].test_value();
            let (primal, slope) = R::parts(d.values[other_row]);
            let expected = seed.map_or(0.0, |(variable, column)| {
                r.derivative(BlockId(raw(real.pool, variable)), row, column)
            });
            let label = format!("factor {:?}, row {index}, seed {seed:?}", scope.id);
            near(t, &format!("{label} value"), primal, value)?;
            if !local {
                near(t, &format!("{label} derivative"), slope, expected)?;
            }
            cost_derivative += value * expected;
        }
    }
    let (primal, slope) = R::parts(dual_cost);
    near(t, "dual cost value", primal, real_cost.test_value())?;
    near(t, "dual cost derivative", slope, cost_derivative)
}

// Full and singleton paths must describe the same per-factor linearizations.
fn selection<R: TestScalar>(full: &Capture<R>, part: &Capture<R>, t: Tolerance) -> TestCaseResult {
    prop_assert_eq!(
        &full.widths,
        &part.widths,
        "state layout changed across builds"
    );
    for scope in &part.scopes {
        let original = full
            .scopes
            .iter()
            .find(|s| s.id == scope.id)
            .ok_or_else(|| TestCaseError::fail("selection changed factor identities"))?;
        let rows = original.rows();
        let other_rows = scope.rows();
        prop_assert_eq!(
            rows.clone().count(),
            other_rows.clone().count(),
            "residual dimension changed"
        );
        for ((a, row), (b, other_row)) in rows.zip(other_rows) {
            near(
                t,
                "full vs singleton residual",
                a.values[row].test_value(),
                b.values[other_row].test_value(),
            )?;
            for (variable, &width) in full.widths.iter().enumerate() {
                let id = BlockId(raw(full.pool, variable));
                for column in 0..width {
                    near(
                        t,
                        "full vs singleton Jacobian",
                        a.derivative(id, row, column),
                        b.derivative(id, other_row, column),
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[track_caller]
fn run<T: Debug, R: TestScalar>(
    cases: impl Strategy<Value = T>,
    config: Config,
    t: Tolerance,
    property: FactorProperty,
    batched: bool,
    real: impl Fn(&T, &mut TestStates<R>, Selection) -> Result<(R, Capture<R>), EvaluationError>,
    dual_evaluate: impl Fn(
        &T,
        &mut TestStates<R::Dual>,
        Selection,
    ) -> Result<(R::Dual, Capture<R::Dual>), EvaluationError>,
) {
    runner(config, t)
        .run(&cases, |case| {
            let pool = PoolId::new();
            let new_states = || TestStates {
                pool,
                values: Vec::new(),
                widths: Vec::new(),
                seed: None,
            };
            let failure = |e| TestCaseError::fail(format!("{property:?} evaluation: {e}"));
            let (full_cost, full) =
                real(&case, &mut new_states(), Selection::All).map_err(failure)?;
            let local = property == FactorProperty::LocalModel;
            cost(full_cost, &full, t, local)?;
            let mut selections = vec![Selection::All];
            if batched {
                selections.push(Selection::Empty);
                selections.extend((0..full.scopes.len()).map(Selection::One));
            }
            let mut singleton_cost = 0.0;
            for selected in selections {
                let partial;
                let (real_cost, capture) = if matches!(selected, Selection::All) {
                    (full_cost, &full)
                } else {
                    let (value, sink) =
                        real(&case, &mut new_states(), selected).map_err(failure)?;
                    cost(value, &sink, t, local)?;
                    selection(&full, &sink, t)?;
                    partial = sink;
                    (value, &partial)
                };
                match selected {
                    Selection::Empty => near(t, "empty batch cost", real_cost.test_value(), 0.0)?,
                    Selection::One(_) => singleton_cost += real_cost.test_value(),
                    Selection::All => {}
                }
                if property != FactorProperty::Cost {
                    let seeds = std::iter::once(None).chain(
                        capture
                            .widths
                            .iter()
                            .enumerate()
                            .flat_map(|(i, &width)| (0..width).map(move |j| Some((i, j)))),
                    );
                    for seed in seeds {
                        let mut states = TestStates {
                            pool,
                            values: Vec::new(),
                            widths: Vec::new(),
                            seed: seed.map(|(i, j)| (i, j, R::from_test_value(0.0).dual(1.0))),
                        };
                        let (value, sink) =
                            dual_evaluate(&case, &mut states, selected).map_err(failure)?;
                        dual(real_cost, capture, value, &sink, seed, t, local)?;
                    }
                }
            }
            if batched {
                near(
                    t,
                    "full vs summed singleton costs",
                    full_cost.test_value(),
                    singleton_cost,
                )?;
            }
            Ok(())
        })
        .unwrap_or_else(|e| panic!("{}::{property:?}: {e}", std::any::type_name::<T>()));
}

/// Run one ordinary-factor property with shrinking and failure persistence.
#[track_caller]
pub fn check_factor<T: TestFactor, R: TestScalar>(property: FactorProperty) {
    run(
        T::cases(),
        T::config(),
        T::tolerance::<R>(),
        property,
        false,
        ordinary::<T, R>,
        ordinary::<T, R::Dual>,
    );
}

/// Run one batch property on full, empty, and every singleton selection.
#[track_caller]
pub fn check_factor_batch<T: TestFactorBatch, R: TestScalar>(property: FactorProperty) {
    run(
        T::cases(),
        T::config(),
        T::tolerance::<R>(),
        property,
        true,
        batch::<T, R>,
        batch::<T, R::Dual>,
    );
}

/// Register `cost` and `jacobians` tests for a [`TestFactor`] fixture and precision.
///
/// Requires `test-support`; the generated module is guarded by the consumer's
/// `cfg(test)`. Example: `fagra::factor_tests!(prior, PriorCase, f64);`.
#[macro_export]
macro_rules! factor_tests {
    ($name:ident, $case:ty, $scalar:ty $(,)?) => {
        $crate::factor_tests!(@register $name, $case, $scalar, check_factor);
    };
    (@register $name:ident, $case:ty, $scalar:ty, $check:ident) => {
        #[cfg(test)]
        mod $name {
            #[allow(unused_imports)]
            use super::*;
            #[test]
            fn cost() {
                $crate::testing::$check::<$case, $scalar>($crate::testing::FactorProperty::Cost);
            }
            #[test]
            fn jacobians() {
                $crate::testing::$check::<$case, $scalar>($crate::testing::FactorProperty::Jacobians);
            }
        }
    };
}

/// Register `cost` and `jacobians` tests for a [`TestFactorBatch`] fixture.
///
/// Example: `fagra::factor_batch_tests!(pixels, ReprojectionCase, f64);`.
#[macro_export]
macro_rules! factor_batch_tests {
    ($name:ident, $case:ty, $scalar:ty $(,)?) => {
        $crate::factor_tests!(@register $name, $case, $scalar, check_factor_batch);
    };
}
