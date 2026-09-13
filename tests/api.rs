//! API contracts and generated schema traversal; no operational stub is executed.

use std::{fs, process::Command};

#[allow(dead_code)]
#[path = "../examples/slam.rs"]
mod slam;

use fagra as fg;
use fg::__private::{
    BatchPool, FactorPool, FactorSchema, FactorStore, FactorVisitor, PoolAccess, StatePool,
    StateSchema, StateVisitor,
};
use fg::{
    BatchKey, EvaluationError, FactorBatch, FactorKey, KeyError, Solver, SolverError, StateKey,
};
use slam::{CameraIntrinsics, FrameReprojections, Landmark, Pose, Reprojection, SlamFactors};

fg::states! {
    OtherStates {
        points: Landmark,
        calibration: CameraIntrinsics,
        trajectory: Pose
    }
}

fg::states! { EmptyStates {} }
fg::factors! { EmptyFactors {} }
fg::factors! {
    /// Exercise batch-first declarations and an omitted trailing comma.
    pub(crate) ReverseFactors {
        reprojections: Batch<FrameReprojections, Reprojection>,
        priors: slam::PosePrior
    }
}

#[test]
fn schema_macros_ignore_caller_result_and_ok_names() {
    type Result<T> = ::core::result::Result<T, ()>;
    struct Ok;

    fg::states! { States { poses: Pose } }
    fg::factors! {
        Factors {
            priors: slam::PosePrior,
            reprojections: Batch<FrameReprojections, Reprojection>,
        }
    }

    let _ = States::default();
    let _ = Factors::default();
    let _: Result<()> = ::core::result::Result::Ok(());
    let _ = Ok;
}

#[test]
fn typed_pool_access_and_factor_routes_are_generated() {
    fn access<S: PoolAccess<P>, P>(schema: &mut S) {
        let shared = <S as PoolAccess<P>>::pool(schema) as *const P;
        let mutable = <S as PoolAccess<P>>::pool_mut(schema) as *const P;
        assert_eq!(shared, mutable);
    }
    fn readable<S: fg::StateStore<T>, T: fg::Variable>() {}
    fn routed<S, F: FactorStore<S, P>, P>() {}

    // None of these application values or models implement Default.
    let mut states = OtherStates::default();
    let mut factors = SlamFactors::default();
    access::<_, StatePool<Pose>>(&mut states);
    access::<_, StatePool<Landmark>>(&mut states);
    access::<_, StatePool<CameraIntrinsics>>(&mut states);
    access::<_, FactorPool<slam::PosePrior>>(&mut factors);
    access::<_, BatchPool<FrameReprojections, Reprojection>>(&mut factors);
    readable::<OtherStates, Pose>();
    readable::<OtherStates, Landmark>();
    routed::<OtherStates, SlamFactors, slam::PosePrior>();
    routed::<OtherStates, SlamFactors, Reprojection>();
    routed::<slam::SlamStates, SlamFactors, Reprojection>();
}

#[test]
fn visitors_preserve_order_and_stop_at_the_first_error() {
    struct Trace {
        families: Vec<&'static str>,
        stop_after: usize,
    }

    impl Trace {
        fn record<T>(&mut self) -> Result<(), &'static str> {
            self.families.push(std::any::type_name::<T>());
            if self.families.len() == self.stop_after {
                Err("stop")
            } else {
                Ok(())
            }
        }
    }

    impl StateVisitor for Trace {
        type Error = &'static str;

        fn pool<T: fg::Variable>(&mut self, _: &mut StatePool<T>) -> Result<(), Self::Error> {
            self.record::<T>()
        }
    }

    impl<S> FactorVisitor<S> for Trace {
        type Error = &'static str;

        fn standalone<T: fg::Factor<S>>(
            &mut self,
            _: &mut FactorPool<T>,
        ) -> Result<(), Self::Error> {
            self.record::<T>()
        }

        fn batch<B: FactorBatch<S>>(
            &mut self,
            _: &mut BatchPool<B, B::Factor>,
        ) -> Result<(), Self::Error> {
            self.record::<B>()
        }
    }

    let state_order = [
        std::any::type_name::<Landmark>(),
        std::any::type_name::<CameraIntrinsics>(),
        std::any::type_name::<Pose>(),
    ];
    let factor_order = [
        std::any::type_name::<slam::PosePrior>(),
        std::any::type_name::<FrameReprojections>(),
    ];
    let mut states = OtherStates::default();
    let mut factors = SlamFactors::default();

    for stop_after in [usize::MAX, 1, 2] {
        let mut trace = Trace {
            families: Vec::new(),
            stop_after,
        };
        let result = states.visit(&mut trace);
        let count = stop_after.min(state_order.len());
        assert_eq!(trace.families, state_order[..count]);
        assert_eq!(
            result,
            if stop_after <= state_order.len() {
                Err("stop")
            } else {
                Ok(())
            }
        );

        trace.families.clear();
        let result = <SlamFactors as FactorSchema<OtherStates>>::visit(&mut factors, &mut trace);
        let count = stop_after.min(factor_order.len());
        assert_eq!(trace.families, factor_order[..count]);
        assert_eq!(
            result,
            if stop_after <= factor_order.len() {
                Err("stop")
            } else {
                Ok(())
            }
        );
    }

    let mut trace = Trace {
        families: Vec::new(),
        stop_after: 1,
    };
    EmptyStates::default().visit(&mut trace).unwrap();
    <EmptyFactors as FactorSchema<EmptyStates>>::visit(&mut EmptyFactors::default(), &mut trace)
        .unwrap();
    assert!(trace.families.is_empty());

    trace.stop_after = usize::MAX;
    <ReverseFactors as FactorSchema<OtherStates>>::visit(
        &mut ReverseFactors::default(),
        &mut trace,
    )
    .unwrap();
    assert_eq!(trace.families, [factor_order[1], factor_order[0]]);
}

#[test]
fn infallible_visitors_support_both_schema_kinds() {
    use std::convert::Infallible;

    struct Count(usize);

    impl StateVisitor for Count {
        type Error = Infallible;

        fn pool<T: fg::Variable>(&mut self, _: &mut StatePool<T>) -> Result<(), Infallible> {
            self.0 += 1;
            Ok(())
        }
    }

    impl<S> FactorVisitor<S> for Count {
        type Error = Infallible;

        fn standalone<T: fg::Factor<S>>(
            &mut self,
            _: &mut FactorPool<T>,
        ) -> Result<(), Infallible> {
            self.0 += 1;
            Ok(())
        }

        fn batch<B: FactorBatch<S>>(
            &mut self,
            _: &mut BatchPool<B, B::Factor>,
        ) -> Result<(), Infallible> {
            self.0 += 1;
            Ok(())
        }
    }

    let mut count = Count(0);
    OtherStates::default().visit(&mut count).unwrap();
    assert_eq!(count.0, 3);
    <SlamFactors as FactorSchema<OtherStates>>::visit(&mut SlamFactors::default(), &mut count)
        .unwrap();
    assert_eq!(count.0, 5);
}

#[test]
fn handles_do_not_require_copy_payloads() {
    fn copy<T: Copy>() {}
    copy::<StateKey<Pose>>();
    copy::<FactorKey<Reprojection>>();
    copy::<BatchKey<FrameReprojections>>();
}

#[test]
fn batches_and_factor_storage_support_multiple_state_schemas() {
    fn evaluator<S, B: FactorBatch<S, Factor = Reprojection>>() {}
    evaluator::<slam::SlamStates, FrameReprojections>();
    evaluator::<OtherStates, FrameReprojections>();

    let _: fn() -> Solver<slam::SlamStates, SlamFactors> = Solver::new;
    let _: fn() -> Solver<OtherStates, SlamFactors> = Solver::new;
    let _: fn() -> Solver<EmptyStates, EmptyFactors> = Solver::new;
}

#[test]
fn batched_payload_is_inferred_from_the_model() {
    // Type-check the complete handle path without calling its API-only methods.
    let _workflow = |solver: &mut Solver<OtherStates, SlamFactors>,
                     model: FrameReprojections,
                     payload: Reprojection|
     -> Result<(), SolverError> {
        let batch: BatchKey<FrameReprojections> = solver.add_batch(model);
        let factor: FactorKey<Reprojection> = solver.add_factor_to(batch, payload)?;
        solver.optimize()?;
        let _: f64 = solver.factor_cost(factor)?;
        solver.remove_factor(factor)
    };
}

#[test]
fn error_traits_support_question_mark_propagation() {
    fn error<E: std::error::Error>() {}
    error::<KeyError>();
    error::<EvaluationError>();
    error::<SolverError>();

    let _: fn(KeyError) -> EvaluationError = EvaluationError::from;
    let _: fn(KeyError) -> SolverError = SolverError::from;
    let _: fn(EvaluationError) -> SolverError = SolverError::from;
}

#[test]
fn nalgebra_views_convert_to_faer_without_copying() {
    use faer_ext::{
        IntoFaer,
        nalgebra::{DMatrix, DMatrixView, Dyn, SMatrix},
    };

    fn check(view: DMatrixView<'_, f64, Dyn, Dyn>) {
        let matrix: faer::MatRef<'_, f64> = view.into_faer();
        assert_eq!(matrix.as_ptr(), view.as_ptr());
        assert_eq!((matrix.nrows(), matrix.ncols()), view.shape());
        assert_eq!(
            (matrix.row_stride(), matrix.col_stride()),
            (view.strides().0 as isize, view.strides().1 as isize),
        );
        for col in 0..view.ncols() {
            for row in 0..view.nrows() {
                assert_eq!(matrix[(row, col)], view[(row, col)]);
            }
        }
    }

    let owned = SMatrix::<f64, 2, 3>::from_fn(|row, col| (row + 10 * col) as f64);
    check(owned.as_view());

    let mut workspace = DMatrix::<f64>::from_fn(6, 8, |row, col| (row + 10 * col) as f64);
    check(workspace.fixed_view::<2, 3>(1, 1).as_view());
    check(
        workspace
            .fixed_view_with_steps::<2, 3>((1, 1), (1, 1))
            .as_view(),
    );
    check(
        workspace
            .fixed_view_with_steps_mut::<2, 3>((1, 1), (1, 1))
            .as_view(),
    );
    check(DMatrix::<f64>::zeros(0, 3).as_view());
}

#[test]
fn jacobian_storage_and_columns_are_checked_during_codegen() {
    // Metadata-only checks may skip generic const assertions. Build a tiny
    // consumer with both a valid width and an invalid width instead.
    let directory = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("jacobian-width-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    fs::write(
        directory.join("Cargo.toml"),
        format!(
            r#"[package]
name = "jacobian-width-check"
version = "0.0.0"
edition = "2024"

[workspace]

[dependencies]
fagra = {{ path = {:?} }}
faer-ext = {{ version = "0.8", features = ["nalgebra"] }}

[features]
wrong-width = []

[[bin]]
name = "jacobian-width-check"
path = "main.rs"
"#,
            env!("CARGO_MANIFEST_DIR")
        ),
    )
    .unwrap();
    fs::write(
        directory.join("main.rs"),
        r#"use fagra::{JacobianBlock, StateKey, Variable};
use faer_ext::nalgebra::{Const, DMatrix, DMatrixView, Dyn, Matrix, SMatrix, Storage};

struct Pose;
impl Variable for Pose {
    type Tangent = [f64; 6];
    const DOF: usize = 6;
    fn tangent_from_slice(_: &[f64]) -> Self::Tangent { todo!() }
    fn retract(&self, _: &Self::Tangent) -> Self { todo!() }
}

const C: usize = if cfg!(feature = "wrong-width") { 3 } else { 6 };

fn check_storage<'a, S: Storage<f64, Const<2>, Const<C>>>(
    _: &'a Matrix<f64, Const<2>, Const<C>, S>,
) {
    // Taking a function pointer forces monomorphization without executing stubs.
    let constructor: fn(StateKey<Pose>, &'a Matrix<f64, Const<2>, Const<C>, S>) -> JacobianBlock<'a> =
        JacobianBlock::new::<Pose, 2, C, S>;
    std::hint::black_box(constructor);
}

fn main() {
    check_storage(&SMatrix::<f64, 2, C>::zeros());
    let mut workspace = DMatrix::<f64>::zeros(6, 2 * C + 2);
    check_storage(&workspace.fixed_view::<2, C>(1, 1));
    check_storage(&workspace.fixed_view_with_steps::<2, C>((1, 1), (1, 1)));
    check_storage(&workspace.fixed_view_mut::<2, C>(1, 1));
    check_storage(&workspace.fixed_view_with_steps_mut::<2, C>((1, 1), (1, 1)));

    // The getter must preserve dynamically represented row and column strides.
    let _: fn(&JacobianBlock<'static>) -> DMatrixView<'static, f64, Dyn, Dyn> =
        JacobianBlock::jacobian;
}
"#,
    )
    .unwrap();

    let build = |wrong_width| {
        let mut command = Command::new(env!("CARGO"));
        command
            .args(["build", "--offline", "--manifest-path"])
            .arg(directory.join("Cargo.toml"))
            .arg("--target-dir")
            .arg(directory.join("target"));
        if wrong_width {
            command.args(["--features", "wrong-width"]);
        }
        command.output().unwrap()
    };

    let valid = build(false);
    let invalid = build(true);
    fs::remove_dir_all(&directory).unwrap();

    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );
    assert!(
        !invalid.status.success(),
        "mismatched Jacobian width compiled"
    );
    let diagnostic = String::from_utf8_lossy(&invalid.stderr);
    assert!(diagnostic.contains("E0080"), "{diagnostic}");
    assert!(
        diagnostic.contains("Jacobian columns must match state DOF"),
        "{diagnostic}"
    );
}
