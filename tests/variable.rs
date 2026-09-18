//! Check the documented geometry convention and its use in a dimension-generic factor.

use faer_ext::nalgebra::{DefaultAllocator, SVector, Vector3, VectorView, allocator::Allocator};
use fagra::{
    BlockId, EvaluationError, Factor, GaussNewton, Jacobian, JacobianBlock, LinearizationSink,
    Lsmr, OptimizeOptions, Real, Solver, StateKey, StateStore, Tangent, Variable,
};

#[allow(dead_code)]
#[path = "../examples/scalar_prior.rs"]
mod scalar;
#[allow(dead_code)]
#[path = "../examples/slam.rs"]
mod slam;

fn check_geometry<R: Real, T: Variable<Scalar = R, Allocator = DefaultAllocator>>(
    a: &[f64],
    d: &[f64],
    h: f64,
    tolerance: f64,
) where
    DefaultAllocator: Allocator<T::Dim> + Allocator<T::Dim, T::Dim>,
{
    let coords = |values: &[f64]| {
        T::tangent_from_slice(
            &values
                .iter()
                .copied()
                .map(R::from_f64_impl)
                .collect::<Vec<_>>(),
        )
    };
    let a = coords(a);
    let delta = coords(d);
    let x = T::exp(&a);
    let y = x.retract(&delta);
    let z = T::exp(&(&a * R::from_f64_impl(-0.3)));
    let tolerance = R::from_f64_impl(tolerance);
    let h = R::from_f64_impl(h);
    let zero = Tangent::<T>::zeros();
    let identity = Jacobian::<T>::identity();
    let near = |actual: &Tangent<T>, expected: &Tangent<T>| {
        assert!(
            (actual - expected).amax() <= tolerance,
            "actual={actual:?}, expected={expected:?}"
        );
    };
    let matrix_near = |actual: &Jacobian<T>, expected: &Jacobian<T>| {
        assert!(
            (actual - expected).amax() <= tolerance,
            "actual={actual:?}, expected={expected:?}"
        );
    };
    let derivative = |f: &dyn Fn(&Tangent<T>) -> Tangent<T>| {
        let mut result = Jacobian::<T>::zeros();
        for i in 0..delta.len() {
            let mut epsilon = zero.clone();
            epsilon[i] = h;
            let column = (f(&epsilon) - f(&(-&epsilon))) / (h + h);
            result.set_column(i, &column);
        }
        result
    };

    near(&T::identity().log(), &zero);
    near(&T::exp(&zero).log(), &zero);
    near(&x.log(), &a);
    near(&T::exp(&delta).log(), &delta);
    near(&x.local(&x), &zero);
    near(&x.local(&x.retract(&zero)), &zero);
    near(&x.local(&y), &delta);
    near(&y.local(&x.retract(&x.local(&y))), &zero);
    near(&x.local(&x.compose(&T::identity())), &zero);
    near(&x.local(&T::identity().compose(&x)), &zero);
    near(&x.compose(&x.inverse()).log(), &zero);
    near(&x.inverse().compose(&x).log(), &zero);
    near(
        &x.compose(&y).compose(&z).local(&x.compose(&y.compose(&z))),
        &zero,
    );
    near(
        &x.compose(&T::exp(&delta))
            .compose(&x.inverse())
            .local(&T::exp(&(x.adjoint() * &delta))),
        &zero,
    );
    matrix_near(&T::right_jacobian(&zero), &identity);
    matrix_near(&T::right_jacobian_inverse(&zero), &identity);
    matrix_near(
        &(T::right_jacobian(&delta) * T::right_jacobian_inverse(&delta)),
        &identity,
    );

    let exp = T::exp(&delta);
    matrix_near(
        &derivative(&|e| exp.local(&T::exp(&(&delta + e)))),
        &T::right_jacobian(&delta),
    );
    matrix_near(
        &derivative(&|e| exp.retract(e).log()),
        &T::right_jacobian_inverse(&delta),
    );
    let xy = x.compose(&y);
    matrix_near(
        &derivative(&|e| xy.local(&x.retract(e).compose(&y))),
        &y.inverse().adjoint(),
    );
    matrix_near(
        &derivative(&|e| xy.local(&x.compose(&y.retract(e)))),
        &identity,
    );
    matrix_near(
        &derivative(&|e| x.inverse().local(&x.retract(e).inverse())),
        &(-x.adjoint()),
    );
    matrix_near(
        &derivative(&|e| y.local(&x.retract(e).retract(&delta))),
        &exp.inverse().adjoint(),
    );
    matrix_near(
        &derivative(&|e| y.local(&x.retract(&(&delta + e)))),
        &T::right_jacobian(&delta),
    );
    matrix_near(
        &derivative(&|e| x.retract(e).local(&y)),
        &(-T::right_jacobian_inverse(&(-&delta))),
    );
    matrix_near(
        &derivative(&|e| x.local(&y.retract(e))),
        &T::right_jacobian_inverse(&delta),
    );
}

#[test]
fn tangent_from_slice_rejects_short_and_long_inputs() {
    assert_eq!(
        slam::Landmark::tangent_from_slice(&[1.0, 2.0, 3.0]).as_slice(),
        &[1.0, 2.0, 3.0]
    );
    for coords in [&[1.0, 2.0][..], &[1.0, 2.0, 3.0, 4.0][..]] {
        assert!(std::panic::catch_unwind(|| slam::Landmark::tangent_from_slice(coords)).is_err());
    }
}

#[test]
fn right_perturbation_geometry_matches_finite_differences() {
    check_geometry::<f64, scalar::Scalar>(&[0.7], &[-0.2], 1e-6, 1e-7);
    check_geometry::<f32, scalar::Scalar<f32>>(&[0.7], &[-0.2], 1e-2, 2e-4);
    check_geometry::<f64, slam::Landmark>(&[0.1, 0.5, 0.2], &[-0.7, 0.2, 0.8], 1e-6, 1e-7);
    for delta in [
        [0.0; 6],
        [0.5, -0.2, 0.3, 0.0, 0.0, 0.0],
        [0.5, -0.2, 0.3, 1e-9, -2e-9, 3e-9],
        [0.5, -0.2, 0.3, 0.7, -0.9, 0.4],
        [0.5, -0.2, 0.3, 0.0, 0.0, std::f64::consts::PI - 1e-3],
    ] {
        check_geometry::<f64, slam::Pose>(&[0.4, -0.1, 0.7, 0.2, -0.3, 0.1], &delta, 1e-6, 1e-7);
    }
    // An independent SE(3) reference checks coordinate order and translation/rotation coupling.
    let pose = slam::Pose::exp(&SVector::<f64, 6>::new(
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        std::f64::consts::FRAC_PI_2,
    ));
    let radius = 2.0 / std::f64::consts::PI;
    assert!((pose.translation - Vector3::new(radius, radius, 0.0)).norm() < 1e-12);
    assert!((pose.rotation * Vector3::x() - Vector3::y()).norm() < 1e-12);
}

struct Prior<T> {
    variable: StateKey<T>,
    measurement: T,
}

impl<T, S> Factor<S> for Prior<T>
where
    T: Variable<Scalar = f64>,
    S: StateStore<T>,
{
    type Scalar = f64;
    fn visit_variables(&self, mut visitor: impl FnMut(BlockId)) {
        visitor(self.variable.block_id());
    }
    fn cost(&self, states: &S) -> Result<f64, EvaluationError> {
        Ok(0.5
            * self
                .measurement
                .local(states.get(self.variable)?)
                .norm_squared())
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        states: &S,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let residual = self.measurement.local(states.get(self.variable)?);
        let jacobian = T::right_jacobian_inverse(&residual);
        // Exercise borrowed contiguous storage through both solver backends.
        let view = VectorView::<f64, T::Dim>::from_slice(residual.as_slice());
        sink.residual(&view, &[JacobianBlock::new(self.variable, &jacobian)])
    }
}

fagra::states! { States { poses: slam::Pose, scalars: scalar::Scalar } }
fagra::factors! { Factors { poses: Prior<slam::Pose>, scalars: Prior<scalar::Scalar> } }

#[test]
fn generic_prior_emits_associated_dimension_residuals_to_both_backends() {
    for iterative in [false, true] {
        let mut solver = Solver::<States, Factors>::new();
        let target = SVector::<f64, 6>::new(0.5, -0.2, 0.3, 0.4, -0.3, 0.2);
        let pose = solver.add(slam::Pose::identity());
        let scalar = solver.add(scalar::Scalar(0.0));
        solver
            .add_factor(Prior {
                variable: pose,
                measurement: slam::Pose::exp(&target),
            })
            .unwrap();
        solver
            .add_factor(Prior {
                variable: scalar,
                measurement: scalar::Scalar(3.0),
            })
            .unwrap();
        let report = if iterative {
            solver.optimize_with(
                &mut GaussNewton::new(Lsmr::default()),
                &OptimizeOptions::default(),
            )
        } else {
            solver.optimize()
        }
        .unwrap();
        assert!(report.iterations > 0);
        assert!(report.final_cost < 1e-16);
        assert!((solver.get(pose).unwrap().log() - target).norm() < 1e-8);
        assert!((solver.get(scalar).unwrap().0 - 3.0).abs() < 1e-8);
    }
}
