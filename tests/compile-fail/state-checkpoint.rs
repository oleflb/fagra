//@ revisions: valid concrete generic
//@ edition: 2024
//@[valid] check-pass

#![allow(dead_code, unused_variables, unused_mut)]

#[path = "../../examples/scalar_prior.rs"]
mod scalar;
use scalar::Scalar;

// Only one scalar specialization supports copying. The generic schema itself
// must remain usable with either specialization.
impl Clone for Scalar<f32> {
    fn clone(&self) -> Self { Self(self.0) }
}

fagra::states! { pub Concrete { values: Scalar<f64> } }
fagra::states! { pub Generic<R> { values: Scalar<R> } }
fagra::factors! { Empty<R> {} }

fn main() {
    let mut concrete = fagra::Problem::<Concrete, Empty<f64>>::new();
    let key = concrete.add(Scalar(1_f64));
    concrete.set(key, Scalar(2.)).unwrap();
    let mut generic = fagra::Problem::<Generic<f64>, Empty<f64>>::new();
    generic.add(Scalar(1_f64));

    #[cfg(valid)]
    {
        let mut cloneable = fagra::Problem::<Generic<f32>, Empty<f32>>::new();
        cloneable.add(Scalar(1_f32));
        cloneable.restore_states(cloneable.checkpoint_states()).unwrap();
    }

    #[cfg(concrete)]
    concrete.checkpoint_states();
    //~[concrete]^ ERROR: trait bound

    #[cfg(generic)]
    generic.checkpoint_states();
    //~[generic]^ ERROR: trait bound
}
