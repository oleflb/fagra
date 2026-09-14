/// A real scalar supported by both the geometry and numerical backends.
///
/// `f32` and `f64` implement this trait. Arithmetic remains statically dispatched.
pub trait Real: faer::traits::RealField + faer_ext::nalgebra::RealField + Copy {}

impl<T> Real for T where T: faer::traits::RealField + faer_ext::nalgebra::RealField + Copy {}
