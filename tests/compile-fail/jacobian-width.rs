//@ revisions: valid invalid
//@ edition: 2024
//@ compile-flags: --emit=link
//@[valid] check-pass
//@[invalid] error-in-other-file: Jacobian columns must match state DOF
//@[invalid] error-in-other-file: Jacobian columns must match state DOF
//@[invalid] error-in-other-file: Jacobian columns must match state DOF
//@[invalid] error-in-other-file: Jacobian columns must match state DOF
//@[invalid] error-in-other-file: Jacobian columns must match state DOF

// One const-evaluation error per storage type: owned, views, and mutable views,
// with contiguous or dynamic row strides.

use faer_ext::nalgebra::{Const, DMatrix, DMatrixView, Dyn, Matrix, SMatrix, Storage};
use fagra::{JacobianBlock, StateKey, Variable};

struct Pose;
impl Variable for Pose {
    type Scalar = f64;
    type Tangent = [f64; 6];
    const DOF: usize = 6;
    fn tangent_from_slice(_: &[f64]) -> Self::Tangent {
        todo!()
    }
    fn retract(&self, _: &Self::Tangent) -> Self {
        todo!()
    }
}

const C: usize = if cfg!(invalid) { 3 } else { 6 };

fn check_storage<'a, S: Storage<f64, Const<2>, Const<C>>>(
    _: &'a Matrix<f64, Const<2>, Const<C>, S>,
) {
    // Force monomorphization: metadata-only checks can miss the const assertion.
    let constructor: fn(
        StateKey<Pose>,
        &'a Matrix<f64, Const<2>, Const<C>, S>,
    ) -> JacobianBlock<'a> = JacobianBlock::new::<Pose, 2, C, S>;
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
