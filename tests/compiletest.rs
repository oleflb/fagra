#[test]
fn compiletest() -> ui_test::color_eyre::Result<()> {
    let mut config = ui_test::Config::rustc("tests/compile-fail");
    // Check diagnostic annotations without snapshots of compiler internals.
    config.output_conflict_handling = ui_test::ignore_output_conflict;
    let mut dependencies = ui_test::dependencies::DependencyBuilder::default();
    // ui_test 0.30 expects artifacts for every normal dependency, even optional
    // ones. Build the exported testing API as well as the ordinary graph API.
    dependencies
        .program
        .args
        .extend(["--features".into(), "test-support".into()]);
    config
        .comment_defaults
        .base()
        .add_custom("dependencies", dependencies);
    ui_test::run_tests(config)
}
