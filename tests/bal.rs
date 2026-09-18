#[allow(dead_code)]
#[path = "../examples/bal/model.rs"]
mod model;
use faer_ext::nalgebra::{SVector, Vector3};
use model::*;

#[test]
fn batched_bal_graph_reduces_reprojection_error() {
    use fagra::{LevenbergMarquardt, Solver};
    fagra::states! { States { cameras: Camera, points: Point } }
    fagra::factors! { Factors { frames: Batch<Frame, Reprojection> } }
    let mut graph = Solver::<States, Factors>::new();
    let camera = Camera(SVector::from_row_slice(&[
        0., 0., 0., 0., 0., 0., 1., 0., 0.,
    ]));
    let projection = Projection::new(&camera);
    let camera = graph.add(camera);
    let frame = graph.add_batch(Frame { camera });
    let mut factors = Vec::new();
    for i in 0..12 {
        let truth = Point(Vector3::new(
            (i % 4) as f64 * 0.2 - 0.3,
            (i / 4) as f64 * 0.3 - 0.2,
            -3. - i as f64 * 0.1,
        ));
        let pixel = projection.value(&truth).unwrap();
        let point = graph.add(Point(truth.0 + Vector3::new(0.02, -0.03, 0.01)));
        factors.push(
            graph
                .add_factor_to(frame, Reprojection { point, pixel })
                .unwrap(),
        );
    }
    let initial: f64 = factors.iter().map(|&f| graph.factor_cost(f).unwrap()).sum();
    let mut lm = LevenbergMarquardt::default();
    let report = graph.optimize_with(&mut lm, &Default::default()).unwrap();
    let final_cost: f64 = factors.iter().map(|&f| graph.factor_cost(f).unwrap()).sum();
    assert!(final_cost < initial * 1e-6);
    assert!((report.final_cost - final_cost).abs() < 1e-12);
    assert!(lm.statistics().accepted_steps > 0);
}

#[test]
fn parser_rejects_invalid_input() {
    let valid = "1 1 1 0 0 0 0 0 0 0 0 0 0 100 0 0 0 0 -5";
    let p = Problem::parse(valid).unwrap();
    assert_eq!(p.cameras.len(), 1);
    assert_eq!(p.points[0].0.z, -5.);
    for text in [
        "",
        "0 1 1",
        "1 1 1 1 0 0 0",
        "1 1 1 0 1 0 0",
        "1 1 1 0 0 NaN 0",
    ] {
        assert!(Problem::parse(text).is_err());
    }
    assert!(Problem::parse(&format!("{valid} 0")).is_err());
    assert!(Problem::parse(&valid.replace("100", "inf")).is_err());
    assert!(Problem::parse(valid.rsplit_once(' ').unwrap().0).is_err());
}

#[test]
fn bal_projection_and_additive_jacobians() {
    for rotation in [
        Vector3::zeros(),
        Vector3::new(1e-10, -2e-10, 3e-10),
        Vector3::new(0.4, -0.2, 0.1),
        Vector3::new(3.1, 0.1, -0.1),
    ] {
        let c = Camera(SVector::from_row_slice(&[
            rotation.x, rotation.y, rotation.z, 0.2, -0.1, 0.3, 500., 0.01, -0.001,
        ]));
        let p = Point(Vector3::new(0.8, -0.4, -5.));
        let projection = Projection::new(&c);
        let (jc, jp) = projection.jacobians(&p);
        for k in 0..12 {
            let mut plus_c = c.clone();
            let mut minus_c = c.clone();
            let mut plus_p = p.clone();
            let mut minus_p = p.clone();
            let h = 1e-6;
            if k < 9 {
                plus_c.0[k] += h;
                minus_c.0[k] -= h;
            } else {
                plus_p.0[k - 9] += h;
                minus_p.0[k - 9] -= h;
            }
            let numerical = (Projection::new(&plus_c).value(&plus_p).unwrap()
                - Projection::new(&minus_c).value(&minus_p).unwrap())
                / (2. * h);
            let analytical = if k < 9 {
                jc.column(k)
            } else {
                jp.column(k - 9)
            };
            assert!(
                (numerical - analytical).norm() < 1e-5 * analytical.norm().max(1.),
                "column {k}"
            );
        }
    }
    let c = Camera(SVector::from_row_slice(&[
        0., 0., 0., 0., 0., 0., 100., 0.1, 0.01,
    ]));
    let pixel = Projection::new(&c)
        .value(&Point(Vector3::new(1., 2., -2.)))
        .unwrap();
    assert!((pixel.x - 50. * (1. + 0.1 * 1.25 + 0.01 * 1.25 * 1.25)).abs() < 1e-12);
    assert_eq!(pixel.y, 2. * pixel.x);
    assert!(Projection::new(&c).value(&Point(Vector3::zeros())).is_err());
}
