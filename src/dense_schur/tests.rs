use super::*;
use crate::{DenseNormalCholesky, Variable};
use faer_ext::nalgebra::{DMatrix, DVector, DVectorView};

mod sizes;

#[test]
fn mixed_blocks_and_coupled_retained_system_match_dense_damping_retries() {
    fn check<R: Real + Into<f64>>(tolerance: f64) {
        let c = R::from_f64_impl;
        // Retained widths 2+1+3, landmarks widths 2+3. Retained blocks are
        // deliberately emitted out of coordinate order, with strided views.
        let mut schur = DenseSchur::<R>::new(6, [2, 3]);
        let mut dense = DenseNormalCholesky::<R>::default();
        schur.prepare(11).unwrap();
        dense.prepare(11).unwrap();
        schur.clear();
        dense.clear();
        for (count, columns) in [
            (9, vec![(3, 3), (6, 2), (0, 2), (2, 1)]),
            (10, vec![(8, 3), (2, 1), (0, 2), (3, 3)]),
            (8, vec![(2, 1), (3, 3), (0, 2)]),
        ] {
            let storage = DMatrix::from_fn(count * 2, 11, |i, j| {
                c(if i / 2 == j {
                    3.
                } else {
                    ((i * 7 + j * 3) % 13) as f64 / 11.
                })
            });
            let r = DVector::from_fn(count, |i, _| c(i as f64 * 0.17 - 0.5));
            let blocks = columns.iter().map(|&(offset, width)| {
                (
                    offset,
                    storage.view_with_steps((0, offset), (count, width), (1, 0)),
                )
            });
            schur.accumulate(r.column(0), blocks.clone()).unwrap();
            dense.accumulate(r.column(0), blocks).unwrap();
        }
        schur.prepare_damping(c(1e-6)).unwrap();
        dense.prepare_damping(c(1e-6)).unwrap();
        for lambda in [0., 0.001, 10., 0.01, 0.] {
            let actual = schur.solve_damped(c(lambda)).unwrap();
            let expected = dense.solve_damped(c(lambda)).unwrap();
            for (&a, &b) in actual.delta.iter().zip(expected.delta) {
                assert!((Into::<f64>::into(a) - Into::<f64>::into(b)).abs() < tolerance);
            }
            assert!(
                (Into::<f64>::into(actual.predicted_reduction)
                    - Into::<f64>::into(expected.predicted_reduction))
                .abs()
                    < tolerance * 10.
            );
        }
    }
    check::<f64>(1e-11);
    check::<f32>(2e-5);
}

#[test]
fn mixed_covariance_matches_dense_and_never_prepares_full_information() {
    use crate::{
        Problem,
        covariance::SelectedCovariance,
        optimization::{Block, Layout},
        variable::test_support::Vector,
    };
    crate::states! { States { two: Vector<2>, one: Vector<1> } }
    crate::factors! { Factors {} }
    let mut graph = Problem::<States, Factors>::new();
    let two = graph.add(Vector::<2>::identity());
    let one = graph.add(Vector::<1>::identity());
    let mut layout = Layout {
        dimension: 8,
        ..Default::default()
    };
    for (id, offset, width) in [(two.block_id(), 0, 2), (one.block_id(), 2, 1)] {
        layout.block_index.insert(id, layout.blocks.len());
        layout.blocks.push(Block { id, offset, width });
    }
    let mut schur = DenseSchur::<f64>::new(3, [2, 3]);
    let mut dense = DenseNormalCholesky::default();
    schur.prepare(8).unwrap();
    dense.prepare(8).unwrap();
    schur.clear();
    dense.clear();
    for (landmark, width) in [(3, 2), (5, 3)] {
        let j = DMatrix::from_fn(9, 8, |row, col| {
            if row == col {
                2. + col as f64
            } else {
                ((row * 7 + col * 11) % 17) as f64 / 13.
            }
        });
        let residual = DVector::zeros(9);
        let blocks = [(2, 1), (landmark, width), (0, 2)]
            .map(|(col, width)| (col, j.view_with_steps((0, col), (9, width), (0, 0))));
        schur
            .accumulate(residual.column(0), blocks.into_iter())
            .unwrap();
        dense
            .accumulate(residual.column(0), blocks.into_iter())
            .unwrap();
    }
    let mut actual = SelectedCovariance::default();
    let mut expected = SelectedCovariance::default();
    let selected = [one.block_id(), two.block_id()];
    actual.prepare(&layout, &selected).unwrap();
    expected.prepare(&layout, &selected).unwrap();
    let mut unused_full_normal = DenseNormalCholesky::default();
    schur
        .covariance(&mut unused_full_normal, &mut actual)
        .unwrap();
    expected.compute(&mut dense).unwrap();
    for col in 0..3 {
        for row in 0..3 {
            assert!((actual.matrix()[(row, col)] - expected.matrix()[(row, col)]).abs() < 1e-12);
        }
    }
    // Passed as the full-normal fallback, but left dimension-zero by DenseSchur.
    assert_eq!(unused_full_normal.solve().unwrap().len(), 0);
}

#[test]
fn invalid_topology_is_rejected_before_caching_and_singular_landmarks_fail() {
    assert!(
        DenseSchur::<f64>::new(0, [usize::MAX])
            .prepare(usize::MAX)
            .is_err()
    );
    let mut backend = DenseSchur::<f64>::new(3, [2, 1]);
    assert!(backend.prepare(5).is_err());
    backend.prepare(6).unwrap();
    backend.clear();
    // Empty emissions consume no row offset, but still must not cross-couple
    // successive landmarks when the cached Jacobian is assembled.
    let empty = DMatrix::zeros(0, 2);
    backend
        .accumulate(
            DVectorView::from_slice(&[], 0),
            [(3, empty.as_view())].into_iter(),
        )
        .unwrap();
    let empty = DMatrix::zeros(0, 1);
    backend
        .accumulate(
            DVectorView::from_slice(&[], 0),
            [(5, empty.as_view())].into_iter(),
        )
        .unwrap();
    backend.prepare_damping(1e-6).unwrap();
    assert!(backend.solve_damped(0.1).is_ok());
    backend.clear();
    let j = DMatrix::from_element(1, 2, 1.);
    let k = DMatrix::from_element(1, 1, 1.);
    let r = DVector::from_element(1, 1.);
    for blocks in [
        vec![(3, j.as_view()), (5, k.as_view())],
        vec![(4, k.as_view())],
        vec![(2, j.as_view())],
    ] {
        assert!(matches!(
            backend.accumulate(r.column(0), blocks.into_iter()),
            Err(EvaluationError::UnsupportedStructure)
        ));
        assert!(backend.model.jacobian.blocks.is_empty());
    }
    backend
        .accumulate(r.column(0), [(3, j.as_view())].into_iter())
        .unwrap();
    backend.prepare_damping(1e-6).unwrap();
    assert!(backend.solve_damped(0.).is_err());
    assert!(backend.solve_damped(0.1).is_ok());
    assert!(backend.solve_damped(f64::NAN).is_err());
}

#[test]
fn empty_partitions_and_unobserved_coordinates_have_explicit_rank_behavior() {
    for (retained, widths, dimension) in [(0, vec![], 0), (2, vec![], 2), (0, vec![2], 2)] {
        let mut backend = DenseSchur::<f64>::new(retained, widths);
        backend.prepare(dimension).unwrap();
        backend.clear();
        if dimension > 0 {
            let j = DMatrix::identity(dimension, dimension);
            let r = DVector::repeat(dimension, 0.5);
            backend
                .accumulate(r.column(0), [(0, j.as_view())].into_iter())
                .unwrap();
        }
        assert!(backend.solve().is_ok());
    }
    let mut backend = DenseSchur::<f64>::new(1, [1]);
    backend.prepare(2).unwrap();
    backend.clear();
    backend
        .accumulate(DVectorView::from_slice(&[0.], 1), std::iter::empty())
        .unwrap();
    backend.prepare_damping(1e-6).unwrap();
    assert!(backend.solve_damped(0.).is_err());
    assert!(backend.solve_damped(1.).is_ok());
}

#[test]
fn covariance_rank_screen_checks_landmarks_and_retained_gauge_even_without_selection() {
    for (retained, widths) in [(0, vec![2]), (2, vec![])] {
        let mut backend = DenseSchur::<f64>::new(retained, widths);
        backend.prepare(2).unwrap();
        backend.clear();
        let j = DMatrix::from_row_slice(2, 2, &[1., 1., 0., 1e-4]);
        let r = DVector::zeros(2);
        backend
            .accumulate(r.column(0), [(0, j.as_view())].into_iter())
            .unwrap();
        let mut selected = SelectedCovariance::default();
        selected.options.relative_pivot_tolerance = 1e-6;
        let mut unused = DenseNormalCholesky::default();
        assert!(matches!(
            backend.covariance(&mut unused, &mut selected),
            Err(SolverError::SingularInformation)
        ));
        selected.options.relative_pivot_tolerance = 1e-12;
        assert!(backend.covariance(&mut unused, &mut selected).is_ok());
    }
    // An entirely unobserved landmark is not an ignorable minimum-norm nuisance.
    let mut backend = DenseSchur::<f64>::new(1, [2]);
    backend.prepare(3).unwrap();
    backend.clear();
    let j = DMatrix::identity(1, 1);
    let r = DVector::zeros(1);
    backend
        .accumulate(r.column(0), [(0, j.as_view())].into_iter())
        .unwrap();
    assert!(matches!(
        backend.covariance(
            &mut DenseNormalCholesky::default(),
            &mut SelectedCovariance::default()
        ),
        Err(SolverError::SingularInformation)
    ));
    backend.model.gradient[0] = f64::NAN;
    assert!(matches!(
        backend.covariance(
            &mut DenseNormalCholesky::default(),
            &mut SelectedCovariance::default()
        ),
        Err(SolverError::InvalidInformation)
    ));
}
