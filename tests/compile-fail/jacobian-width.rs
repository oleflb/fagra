//@ revisions: valid invalid
//@ edition: 2024
//@[valid] check-pass

// Column dimensions are checked during type checking. Valid dimensions support
// owned matrices and views with contiguous or dynamic row strides.

use faer_ext::nalgebra::{Const, DMatrix, DMatrixView, Dyn, Matrix, SMatrix, Storage};
use fagra::{JacobianBlock, StateKey};

#[allow(dead_code)]
#[path = "../../examples/slam.rs"]
mod slam;
use slam::Pose;

const C: usize = if cfg!(invalid) { 3 } else { 6 };

fn check_storage<'a, S: Storage<f64, Const<2>, Const<C>>>(
    _: &'a Matrix<f64, Const<2>, Const<C>, S>,
) {
    let constructor: fn(
        StateKey<Pose>,
        &'a Matrix<f64, Const<2>, Const<C>, S>,
    ) -> JacobianBlock<'a> = JacobianBlock::new::<Pose, Const<2>, S>;
    //~[invalid]^ ERROR: trait bound
    //~[invalid]| ERROR: mismatched types
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
