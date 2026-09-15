//@ revisions: normal test
//@ edition: 2024
//@[normal] check-pass
//@[test] check-pass
//@[test] compile-flags: --test --cfg 'feature="test-support"'

use fagra as renamed;

// This invocation must compile in a normal downstream build even though its
// variable and TestVariable implementation exist only in the test build.
renamed::variable_tests!(properties, OnlyInTests);

#[cfg(test)]
#[path = "../../examples/scalar_prior.rs"]
mod scalar;

#[cfg(test)]
type OnlyInTests = scalar::Scalar<f64>;

fn main() {}
