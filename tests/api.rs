//! Compile-time API contracts; no operational stub is executed.

use std::{fs, process::Command};

#[allow(dead_code)]
#[path = "../examples/slam.rs"]
mod slam;

use fagra as fg;
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
        let _: f64 = solver.factor_error(factor)?;
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
fn jacobian_columns_are_checked_during_codegen() {
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

struct Pose;
impl Variable for Pose {
    type Tangent = [f64; 6];
    const DOF: usize = 6;
    fn tangent_from_slice(_: &[f64]) -> Self::Tangent { todo!() }
    fn retract(&self, _: &Self::Tangent) -> Self { todo!() }
}

const C: usize = if cfg!(feature = "wrong-width") { 3 } else { 6 };

fn main() {
    // Taking a function pointer forces monomorphization without executing stubs.
    let constructor: fn(StateKey<Pose>, &'static _) -> JacobianBlock<'static> =
        JacobianBlock::new::<Pose, 2, C>;
    std::hint::black_box(constructor);
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
