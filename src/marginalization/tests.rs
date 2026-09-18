use super::*;
use crate::{Tangent, storage::PoolAccess};
use faer_ext::nalgebra::{DMatrix, DVector};

impl<R: Real> Elimination<R> {
    fn eliminate(
        &mut self,
        removed: usize,
        options: &MarginalizationOptions<R>,
    ) -> Result<(DMatrix<R>, DVector<R>, R, usize), SolverError> {
        let mut prior = Prior::default();
        let (constant, rank) = self.reduce(removed, options, &mut prior)?;
        Ok((
            prior.a.as_ref().into_nalgebra().into_owned(),
            DVector::from_vec(prior.b),
            constant,
            rank,
        ))
    }
}

#[allow(dead_code)]
#[path = "../../examples/slam/geometry.rs"]
mod geometry;
use geometry::Pose;

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "actual={actual}, expected={expected}"
    );
}

#[test]
fn reduced_objective_matches_svd_in_both_precisions() {
    fn check<R: Real + Into<f64>>(tolerance: f64) {
        // Dependent eliminated columns, rectangular systems, zero columns, and
        // full-rank elimination. The SVD operates on the original rounded input.
        let cases: &[(usize, usize, usize, &[f64])] = &[
            (
                4,
                4,
                2,
                &[
                    1., 2., 1., 0., 3., 2., 4., 0., 2., -1., 0., 0., 1., 1., 2., 3., 6., -1., 0.,
                    1.,
                ],
            ),
            (2, 4, 3, &[1., 0., 2., 3., 4., 0., 1., 1., 2., 3.]),
            (3, 2, 1, &[0., 1., 2., 0., 2., 3., 0., -1., 1.]),
            (
                4,
                3,
                1,
                &[
                    1., 2., 0., 1., 2., -1., 1., 3., -1., 0., 2., 4., 0., 1., -1., 2.,
                ],
            ),
            (3, 2, 0, &[1., 0., 2., 0., 1., 3., 0., 0., 4.]),
            (3, 1, 1, &[1., 1., 2., 3., -1., 2.]),
        ];
        for &(m, n, removed, values) in cases {
            let mut work = Elimination::<R>::default();
            work.rows.prepare(n).unwrap();
            work.rows.values = values.iter().map(|&v| R::from_f64_impl(v)).collect();
            let original =
                DMatrix::from_row_slice(m, n + 1, &work.rows.values).map(Into::<f64>::into);
            let (a, b, constant, _) = work.eliminate(removed, &Default::default()).unwrap();
            for shift in [-2.0, 0.0, 0.5, 3.0] {
                let delta = DVector::from_fn(n - removed, |i, _| shift + i as f64 * 0.3);
                let rhs = original.columns(removed, n - removed) * &delta + original.column(n);
                let residual = if removed == 0 {
                    rhs
                } else {
                    let eliminated = original.columns(0, removed).into_owned();
                    let step = eliminated
                        .clone()
                        .svd(true, true)
                        .solve(&(-&rhs), 1e-10)
                        .unwrap();
                    eliminated * step + rhs
                };
                let expected = 0.5 * residual.norm_squared();
                let reduced = a.map(Into::<f64>::into) * delta + b.map(Into::<f64>::into);
                near(
                    0.5 * reduced.norm_squared() + constant.into(),
                    expected,
                    tolerance,
                );
            }
        }
    }
    check::<f64>(1e-11);
    check::<f32>(2e-5);
}

#[test]
fn qr_keeps_directions_lost_by_normal_equations() {
    fn check<R: Real + Into<f64>>(epsilon: f64, tolerance: f64) {
        let mut work = Elimination::<R>::default();
        work.rows.prepare(3).unwrap();
        // Jm's columns are almost parallel. Squaring their condition number loses
        // the second direction, but their column space is exactly the first two rows.
        work.rows.values = [1., 1., 1., 2., epsilon, -epsilon, 2., -1., 0., 0., 3., 4.]
            .map(R::from_f64_impl)
            .to_vec();
        let (a, b, constant, rank) = work.eliminate(2, &Default::default()).unwrap();
        assert_eq!(rank, 2);
        for y in [-2.0_f64, 0., 1., 5.] {
            let r = a[(0, 0)].into() * y + b[0].into();
            near(
                0.5 * r * r + constant.into(),
                0.5 * (3.0 * y + 4.0).powi(2),
                tolerance,
            );
        }
    }
    check::<f64>(1e-10, 1e-11);
    check::<f32>(1e-4, 1e-5);
}

crate::states! { PoseStates { poses: Pose } }

#[test]
fn manifold_prior_uses_the_current_chart_derivative() {
    let mut states = PoseStates::default();
    let pool: &mut StatePool<Pose> = states.pool_mut();
    let reference = Pose::exp(&Tangent::<Pose>::from_column_slice(&[
        0.2, -0.3, 0.1, 0.1, 0.2, -0.1,
    ]));
    let inverse = reference.inverse();
    let key = pool.insert(reference);
    let values = [1., 2., -1., 0.3, 0.2, 0.5, -0.5, 1., 0.2, 2., -1., 0.7];
    let a = Mat::from_fn(2, 6, |i, j| values[i * 6 + j]);
    let b = vec![0.7, -0.4];
    let mut priors = Priors::default();
    let prior = priors.entries.insert(Prior {
        blocks: vec![(key.block_id(), 0..6)],
        residual: b.clone(),
        jacobian: a.clone(),
        a: a,
        b,
    });
    pool.anchors.push(Anchor {
        prior,
        state: key,
        column: 0,
        inverse,
    });
    pool.stage(&[0.4, -0.2, 0.3, 0.2, -0.3, 0.4]).unwrap();
    pool.accept_trial();
    priors.update(&mut states, true, None).unwrap();
    let analytical = priors.entries.get(prior).unwrap().jacobian.clone();
    for column in 0..6 {
        let mut sample = |step| {
            let mut delta = [0.0; 6];
            delta[column] = step;
            let pool: &mut StatePool<Pose> = states.pool_mut();
            pool.stage(&delta).unwrap();
            priors.update(&mut states, false, None).unwrap();
            let value = DVector::from_column_slice(&priors.entries.get(prior).unwrap().residual);
            let pool: &mut StatePool<Pose> = states.pool_mut();
            pool.reject_trial();
            value
        };
        let derivative = (sample(1e-6) - sample(-1e-6)) / 2e-6;
        for row in 0..2 {
            near(derivative[row], analytical[(row, column)], 1e-8);
        }
    }
}
