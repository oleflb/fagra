/// An optimizable value with its own tangent representation.
///
/// Stored size need not equal [`DOF`](Self::DOF): a quaternion and translation
/// may use a six-dimensional tangent. Initialization belongs to the application.
pub trait Variable {
    /// Scalar used for optimization coordinates and factor Jacobians.
    type Scalar: crate::Real;

    /// A tangent-space increment, typically a fixed-size vector of scalars.
    type Tangent;

    /// Number of scalar optimization coordinates.
    const DOF: usize;

    /// Convert exactly [`DOF`](Self::DOF) scalar coordinates into a tangent.
    ///
    /// The solver guarantees the slice length. Implementations may panic on a
    /// different length and should avoid allocating.
    fn tangent_from_slice(delta: &[Self::Scalar]) -> Self::Tangent;

    /// Apply an increment in the same tangent convention used by factor Jacobians.
    fn retract(&self, delta: &Self::Tangent) -> Self;
}
