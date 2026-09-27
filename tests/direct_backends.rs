use faer_ext::nalgebra::{DMatrix, DVector};
use fagra::{__private::LeastSquaresBackend, DenseNormalCholesky, DenseQr, Real, SolverError};

fn check<R: Real + Into<f64>, B: LeastSquaresBackend<Scalar = R> + Default>() {
    let c = R::from_f64_impl;
    let mut backend = B::default();
    // Reuse across changing dimensions; include underdetermined and empty models.
    for (m, n) in [(7, 3), (43, 35), (1, 3), (0, 2), (0, 0), (7, 3)] {
        let j = DMatrix::from_fn(m, n, |i, k| {
            if m == 1 {
                (k + 1) as f64
            } else {
                ((i * 7 + k * 13) % 17) as f64 / 5. - 1. + if i == k { 4. } else { 0. }
            }
        });
        let r = DVector::from_fn(m, |i, _| i as f64 / 3. - 0.7);
        let jr = j.map(c);
        let rr = r.map(c);
        backend.prepare(n).unwrap();
        backend.clear();
        assert!(matches!(
            backend.solve_damped(c(1.0)),
            Err(SolverError::InvalidOptions)
        ));
        // Shuffled block order and strided Jacobian views.
        for row in 0..m {
            backend
                .accumulate(
                    rr.rows(row, 1),
                    (0..n)
                        .rev()
                        .map(|col| (col, jr.view_with_steps((row, col), (1, 1), (0, 0)))),
                )
                .unwrap();
        }
        let h = j.transpose() * &j;
        let g = j.transpose() * &r;
        if m >= n {
            let expected = if n == 0 {
                DVector::zeros(0)
            } else {
                h.clone().lu().solve(&(-&g)).unwrap()
            };
            let actual = backend.solve().unwrap();
            for (a, b) in actual.iter().zip(expected.iter()) {
                let a: f64 = (*a).into();
                assert!((a - b).abs() < 2e-5);
            }
            // Destructive GN implementations need a fresh model before damping.
            backend.clear();
            backend
                .accumulate(rr.rows(0, m), std::iter::once((0, jr.as_view())))
                .unwrap();
        }
        for floor in [0., -1., f64::NAN, f64::INFINITY] {
            assert!(matches!(
                backend.prepare_damping(c(floor)),
                Err(SolverError::InvalidOptions)
            ));
        }
        backend.prepare_damping(c(0.25)).unwrap();
        for lambda in [-1., f64::NAN, f64::INFINITY] {
            assert!(matches!(
                backend.solve_damped(c(lambda)),
                Err(SolverError::InvalidOptions)
            ));
        }
        // A failed singular attempt must not poison the cached model.
        if m < n {
            assert!(matches!(
                backend.solve_damped(c(0.)),
                Err(SolverError::LinearSolveFailed)
            ));
        }
        for lambda in [0.1, 10., 0.01, 0.1] {
            let mut damped = h.clone();
            for i in 0..n {
                damped[(i, i)] += lambda * h[(i, i)].max(0.25_f64.powi(2));
            }
            let expected = if n == 0 {
                DVector::zeros(0)
            } else {
                damped.lu().solve(&(-&g)).unwrap()
            };
            let actual = backend.solve_damped(c(lambda)).unwrap();
            for (a, b) in actual.delta.iter().zip(expected.iter()) {
                let a: f64 = (*a).into();
                assert!(
                    (a - b).abs() < 2e-4,
                    "{} lambda={lambda}",
                    std::any::type_name::<B>()
                );
            }
            let d = DVector::from_iterator(n, actual.delta.iter().map(|&v| v.into()));
            let prediction = -g.dot(&d) - 0.5 * (&j * &d).norm_squared();
            let predicted: f64 = actual.predicted_reduction.into();
            assert!((predicted - prediction).abs() < 2e-4 * (1. + prediction.abs()));
        }
        backend.clear();
        assert!(matches!(
            backend.solve_damped(c(1.)),
            Err(SolverError::InvalidOptions)
        ));
    }
}

#[test]
fn direct_steps_and_predictions_match_reference_in_both_precisions() {
    check::<f64, DenseQr<f64>>();
    check::<f32, DenseQr<f32>>();
    check::<f64, DenseNormalCholesky<f64>>();
    check::<f32, DenseNormalCholesky<f32>>();
}

#[test]
fn qr_damping_handles_unused_coordinates_and_mixed_units() {
    fn check<R: Real>() {
        let c = R::from_f64_impl;
        let mut qr = DenseQr::<R>::default();
        let options = fagra::LmOptions::<R>::default();
        // Different column norms and pivot order exercise original-coordinate recovery.
        let j = DMatrix::from_row_slice(2, 3, &[c(0.), c(1e9), c(0.), c(0.), c(0.), c(1e-3)]);
        let r = DVector::from_element(2, -R::one());
        qr.prepare(3).unwrap();
        qr.accumulate(r.rows(0, 2), std::iter::once((0, j.as_view())))
            .unwrap();
        qr.prepare_damping(options.min_column_norm).unwrap();
        for lambda in [
            options.initial_damping,
            R::one(),
            c(10.),
            options.initial_damping,
        ] {
            let step = qr.solve_damped(lambda).unwrap();
            let t = R::one() / (R::one() + lambda);
            assert_eq!(step.delta[0], R::zero());
            for (index, scale) in [(1, c(1e9)), (2, c(1e-3))] {
                assert!(
                    (step.delta[index] * scale / t - R::one()).abs() < c(1e-5),
                    "{} lambda={lambda:?}, index={index}, actual={:?}, expected={:?}",
                    std::any::type_name::<R>(),
                    step.delta[index] * scale,
                    t
                );
            }
            let prediction = c(2.) * t - t * t;
            assert!((step.predicted_reduction / prediction - R::one()).abs() < c(1e-5));
        }
    }
    check::<f32>();
    check::<f64>();
}

#[test]
fn qr_preserves_a_direction_lost_by_normal_equations() {
    let j = DMatrix::from_row_slice(3, 2, &[1., 1., 1e-10, 0., 0., 1e-10]);
    let r = DVector::from_row_slice(&[-3., -1e-10, -2e-10]);
    let mut qr = DenseQr::<f64>::default();
    let mut normal = DenseNormalCholesky::<f64>::default();
    qr.prepare(2).unwrap();
    normal.prepare(2).unwrap();
    qr.accumulate(r.rows(0, 3), std::iter::once((0, j.as_view())))
        .unwrap();
    normal
        .accumulate(r.rows(0, 3), std::iter::once((0, j.as_view())))
        .unwrap();
    let step = qr.solve().unwrap();
    assert!((step[0] - 1.).abs() < 1e-6 && (step[1] - 2.).abs() < 1e-6);
    assert!(matches!(
        normal.solve(),
        Err(SolverError::LinearSolveFailed)
    ));

    qr.relative_rank_tolerance = 1e-8;
    assert!(matches!(qr.solve(), Err(SolverError::LinearSolveFailed)));
    qr.prepare_damping(0.1).unwrap();
    assert!(qr.solve_damped(0.01).is_ok());
}
