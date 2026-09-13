#[test]
fn compiletest() -> ui_test::color_eyre::Result<()> {
    let mut config = ui_test::Config::rustc("tests/compile-fail");
    // Check diagnostic annotations without snapshots of compiler internals.
    config.output_conflict_handling = ui_test::ignore_output_conflict;
    config.comment_defaults.base().add_custom(
        "dependencies",
        ui_test::dependencies::DependencyBuilder::default(),
    );
    ui_test::run_tests(config)
}
