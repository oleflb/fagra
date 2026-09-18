use faer_ext::nalgebra::{DimName, Matrix, RealField, U1, allocator::Allocator};

type Buffer<T, Rows, Cols> =
    <<T as Variable>::Allocator as Allocator<Rows, Cols>>::Buffer<<T as Variable>::Scalar>;

/// A variable's column vector of local increment coordinates.
///
/// Its scalar, compile-time dimension, and storage come from [`Variable`]. With
/// `Scalar = f64`, `Dim = Const<6>`, and `Allocator = DefaultAllocator`, this is
/// `SVector<f64, 6>`.
pub type Tangent<T> =
    Matrix<<T as Variable>::Scalar, <T as Variable>::Dim, U1, Buffer<T, <T as Variable>::Dim, U1>>;

/// A square matrix mapping between a variable's tangent coordinates.
///
/// For `Dim = Const<6>` this has six rows and six columns. Measurement Jacobians
/// may have a different row count: a two-coordinate pixel residual has two rows.
pub type Jacobian<T> = Matrix<
    <T as Variable>::Scalar,
    <T as Variable>::Dim,
    <T as Variable>::Dim,
    Buffer<T, <T as Variable>::Dim, <T as Variable>::Dim>,
>;

/// An optimizable Lie-group value, with **right-side** tangent increments.
///
/// A Lie group is a space with a smooth composition operation and inverse. For
/// poses, composition chains rigid transforms; for ordinary vectors, it is
/// addition. Not every manifold is a Lie group (a unit direction on a sphere,
/// for example, does not satisfy this contract).
///
/// # Coordinates and notation
///
/// In the formulas below, `x * y` means `x.compose(y)`, `Exp` means [`exp`](Self::exp),
/// and `Log` means [`log`](Self::log). For transforms acting on column vectors,
/// `x * y` applies `y` first, then `x`. All tangent vectors are column vectors;
/// matrices multiply them on the left.
///
/// Implementors must document their tangent coordinate order and units. Every
/// operation and Jacobian must use that same order. A pose might use three
/// translation coordinates followed by three rotation coordinates; its stored
/// quaternion and translation still have seven coefficients, not six.
///
/// # Derivative convention
///
/// Perturb a state as `x * Exp(epsilon)`, where `epsilon` is a small tangent
/// vector. For a state-valued function `f`, its Jacobian `J` is defined by
/// `f(x * Exp(epsilon)) ≈ f(x) * Exp(J * epsilon)`. Thus output changes are
/// measured in local coordinates at `f(x)`, not in stored quaternion or matrix
/// coefficients. For a vector-valued residual, use ordinary addition instead:
/// `r(x * Exp(epsilon)) ≈ r(x) + J * epsilon`.
///
/// The approximations specify first-order derivatives at `epsilon = 0`.
/// Factors combine these geometry derivatives with their own measurement
/// derivatives. This trait does not differentiate an arbitrary factor for them.
///
/// # Domains and implementation requirements
///
/// `Exp` and `Log` are locally inverse, not necessarily globally inverse. An
/// implementation must document its log branch and Jacobian singularities.
/// Derivative identities involving `Log` apply only on a smooth branch.
/// These operations assume finite inputs in their documented domains; factors
/// must report invalid measurements/evaluations through [`crate::EvaluationError`].
///
/// Use a fixed nalgebra dimension such as `Const<3>` and select `DefaultAllocator`
/// as the associated [`Allocator`](Self::Allocator) for array-backed geometry.
/// Generic graph access and geometry method calls need only `T: Variable`.
/// Generic code constructing nalgebra-owned arithmetic results may additionally
/// need `T: Variable<Allocator = DefaultAllocator>` and explicit nalgebra allocator
/// bounds. Keep those requirements local to the numerical code that needs them.
pub trait Variable: Sized {
    /// Scalar shared by the state, its tangent coordinates, and its Jacobians.
    ///
    /// Geometry only requires nalgebra's real-number operations, allowing the
    /// same implementation to run on dual numbers for derivative tests. Solver
    /// schemas and numerical backends separately require [`crate::Real`].
    type Scalar: RealField + Copy;

    /// Compile-time tangent dimension, e.g. `faer_ext::nalgebra::Const<6>`.
    ///
    /// This determines both vector length and square Jacobian dimensions.
    /// The number of solver coordinates is `Self::Dim::DIM`.
    type Dim: DimName;

    /// Selects owned storage for this variable's tangent vectors and Jacobians.
    ///
    /// Normally `faer_ext::nalgebra::DefaultAllocator`: with `Dim = Const<N>`,
    /// it selects fixed-size arrays, with no heap allocation. This is a type-level
    /// storage choice, not a runtime allocator argument.
    ///
    /// The bounds live on this associated type so `T: Variable` is sufficient to
    /// construct, return, and borrow its geometry values. Graph storage and state
    /// access do not need to repeat bounds on nalgebra's `DefaultAllocator`.
    type Allocator: Allocator<Self::Dim> + Allocator<Self::Dim, Self::Dim>;

    /// Convert a coordinate slice into an owned tangent vector.
    ///
    /// The solver supplies exactly `Self::Dim::DIM` coordinates. Panics on a
    /// different length. Coordinate readback is available through `as_slice()`
    /// on the returned nalgebra vector.
    fn tangent_from_slice(coords: &[Self::Scalar]) -> Tangent<Self> {
        assert_eq!(
            coords.len(),
            Self::Dim::DIM,
            "tangent coordinate count must match Dim"
        );
        Matrix::from_data(Self::Allocator::allocate_from_iterator(
            Self::Dim::name(),
            U1::name(),
            coords.iter().copied(),
        ))
    }

    /// The composition identity: `identity * x = x * identity = x`.
    ///
    /// This is zero for an additive vector, and the identity transform for a pose.
    fn identity() -> Self;

    /// Group product `self * other` (apply `other` first for rigid transforms).
    ///
    /// Under right perturbations, the tangent-space Jacobians of `x * y` are
    /// `Ad(y.inverse())` with respect to `x`, and identity with respect to `y`:
    /// `(x * Exp(epsilon)) * y ≈ (x * y) * Exp(Ad(y.inverse()) * epsilon)`.
    fn compose(&self, other: &Self) -> Self;

    /// Group inverse, satisfying `x * x.inverse() = x.inverse() * x = identity`.
    ///
    /// Its right-perturbation Jacobian is `-Ad(x)`:
    /// `(x * Exp(epsilon)).inverse() ≈ x.inverse() * Exp(-Ad(x) * epsilon)`.
    fn inverse(&self) -> Self;

    /// Map tangent coordinates at the identity into the group.
    ///
    /// `Exp(0) = identity`. For additive vectors this just wraps the coordinates.
    /// For a rigid pose, the SE(3) exponential couples translation and rotation;
    /// it is not independent translation addition and quaternion construction.
    /// Its output-local derivative is [`right_jacobian`](Self::right_jacobian).
    fn exp(delta: &Tangent<Self>) -> Self;

    /// Map this group value into tangent coordinates at the identity.
    ///
    /// `Exp(Log(x)) = x` on the represented domain, while `Log(Exp(delta)) = delta`
    /// only within the chosen log branch. For example, principal rotation logs
    /// cannot recover arbitrary rotations beyond pi radians without wrapping.
    ///
    /// At a smooth branch point, the derivative with respect to a right increment
    /// on `x` is `Jr_inverse(Log(x))`; see [`right_jacobian_inverse`](Self::right_jacobian_inverse).
    fn log(&self) -> Tangent<Self>;

    /// Convert a right-side increment into an equivalent left-side increment.
    ///
    /// `Ad(x)` is the square matrix defined by the conjugation identity:
    ///
    /// ```text
    /// x * Exp(epsilon) * x.inverse() = Exp(Ad(x) * epsilon)
    /// x * Exp(epsilon) = Exp(Ad(x) * epsilon) * x
    /// ```
    ///
    /// It changes the frame in which an increment is expressed, not the stored
    /// representation of `x`. For additive vectors it is identity. For SE(3)
    /// it includes rotation and translation-dependent cross terms.
    fn adjoint(&self) -> Jacobian<Self>;

    /// Derivative of `Exp` with its output change measured on the **right**.
    ///
    /// For a fixed `delta` and small additive input change `epsilon`, `Jr(delta)`
    /// is defined by:
    ///
    /// ```text
    /// Exp(delta + epsilon) ≈ Exp(delta) * Exp(Jr(delta) * epsilon)
    /// ```
    ///
    /// Equivalently, it differentiates
    /// `Log(Exp(delta).inverse() * Exp(delta + epsilon))` at `epsilon = 0`.
    /// Column `i` is the output-local change caused by changing input coordinate
    /// `i`. This is a square tangent-space matrix, not a derivative of the stored
    /// quaternion coefficients. At zero it is identity.
    ///
    /// The left-side version is `Jl(delta) = Jr(-delta)`; implementations do not
    /// need a second formula. For the retraction `x * Exp(delta)`, the derivative
    /// with respect to `delta` is this matrix; the derivative with respect to a
    /// right increment on the base `x` is `Ad(Exp(-delta))`.
    fn right_jacobian(delta: &Tangent<Self>) -> Jacobian<Self>;

    /// Inverse of [`right_jacobian`](Self::right_jacobian), where it is nonsingular.
    ///
    /// On a smooth log branch with `Log(Exp(delta)) = delta`:
    ///
    /// ```text
    /// Log(Exp(delta) * Exp(epsilon)) ≈ delta + Jr_inverse(delta) * epsilon
    /// ```
    ///
    /// It maps a right-side group increment into an additive change of log
    /// coordinates. At zero it is identity. It is **not** generally `Jr(-delta)`.
    /// Implementors must document where it is undefined and use a numerically
    /// stable expression near zero rather than dividing small numbers naively.
    fn right_jacobian_inverse(delta: &Tangent<Self>) -> Jacobian<Self>;

    /// Apply a right-side increment: `self * Exp(delta)`.
    ///
    /// The optimizer uses this update, and all factor Jacobians must differentiate
    /// this convention. Overrides may optimize evaluation but must preserve it.
    fn retract(&self, delta: &Tangent<Self>) -> Self {
        self.compose(&Self::exp(delta))
    }

    /// Coordinates of `other` relative to `self`: `Log(self.inverse() * other)`.
    ///
    /// Locally, `x.local(&x.retract(delta)) = delta` and
    /// `x.retract(&x.local(y)) = y`. The first identity requires `delta` to be in
    /// the selected log branch. For `r = x.local(y)`, the derivatives are
    /// `-Jr_inverse(-r)` with respect to a right increment on `x`, and
    /// `Jr_inverse(r)` with respect to a right increment on `y`.
    fn local(&self, other: &Self) -> Tangent<Self> {
        self.inverse().compose(other).log()
    }
}

// Real additive variables for storage/backend unit tests, shared across modules.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use faer_ext::nalgebra::{Const, DefaultAllocator, SMatrix, SVector};

    pub(crate) struct Vector<const N: usize>(SVector<f64, N>);

    impl<const N: usize> Variable for Vector<N> {
        type Scalar = f64;
        type Dim = Const<N>;
        type Allocator = DefaultAllocator;
        fn identity() -> Self {
            Self(SVector::zeros())
        }
        fn compose(&self, other: &Self) -> Self {
            Self(self.0 + other.0)
        }
        fn inverse(&self) -> Self {
            Self(-self.0)
        }
        fn exp(delta: &Tangent<Self>) -> Self {
            Self(*delta)
        }
        fn log(&self) -> Tangent<Self> {
            self.0
        }
        fn adjoint(&self) -> Jacobian<Self> {
            SMatrix::identity()
        }
        fn right_jacobian(_: &Tangent<Self>) -> Jacobian<Self> {
            SMatrix::identity()
        }
        fn right_jacobian_inverse(_: &Tangent<Self>) -> Jacobian<Self> {
            SMatrix::identity()
        }
    }
}
