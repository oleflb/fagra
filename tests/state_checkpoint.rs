#[allow(dead_code)]
#[path = "../examples/marginalization.rs"]
mod example;

use example::{Difference, Factors, Prior, Scalar};
use faer_ext::nalgebra::{Const, DefaultAllocator, SMatrix, SVector};
use fagra::{
    Jacobian, Problem, ProblemChange, Real, StateCheckpointError, Tangent, TrackedProblem, Variable,
};

// Opt in only this fixture; other integration tests keep the original non-Clone type.
impl<R: Real> Clone for Scalar<R> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Tagged<const FAMILY: u8 = 0> {
    value: f64,
    label: String,
}

impl<const FAMILY: u8> Variable for Tagged<FAMILY> {
    type Scalar = f64;
    type Dim = Const<1>;
    type Allocator = DefaultAllocator;

    fn identity() -> Self {
        Self {
            value: 0.,
            label: String::new(),
        }
    }
    fn compose(&self, other: &Self) -> Self {
        Self {
            value: self.value + other.value,
            label: String::new(),
        }
    }
    fn inverse(&self) -> Self {
        Self {
            value: -self.value,
            label: String::new(),
        }
    }
    fn exp(delta: &Tangent<Self>) -> Self {
        Self {
            value: delta[0],
            label: String::new(),
        }
    }
    fn log(&self) -> Tangent<Self> {
        SVector::<f64, 1>::new(self.value)
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

fagra::states! { pub Mixed { scalars: Scalar, tagged: Tagged } }
fagra::states! { pub Generic<R> { scalars: Scalar<R> } }
fagra::states! { pub Empty<R> {} }
fagra::factors! { NoFactors<R> {} }

#[test]
fn restores_exact_values_across_families_and_preserves_factor_edits() {
    let mut graph = Problem::<Mixed, Factors<f64>>::new();
    let x = graph.add(Scalar(-0.0));
    let tagged = graph.add(Tagged {
        value: 3.,
        label: "not a tangent coordinate".into(),
    });
    let checkpoint = graph.checkpoint_states();
    graph.set(x, Scalar(12.)).unwrap();
    graph.set(tagged, Tagged::identity()).unwrap();
    let factor = graph
        .add_factor(Prior {
            variable: x,
            measurement: 2.,
        })
        .unwrap();
    let x_id = x.block_id();
    let tagged_id = tagged.block_id();

    // Moving the graph does not change checkpoint ownership or state identities.
    let mut moved = graph;
    moved.restore_states(checkpoint).unwrap();
    assert_eq!(moved.get(x).unwrap().0.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(
        moved.get(tagged).unwrap(),
        &Tagged {
            value: 3.,
            label: "not a tangent coordinate".into(),
        }
    );
    assert_eq!(x.block_id(), x_id);
    assert_eq!(tagged.block_id(), tagged_id);
    assert_eq!(moved.factor_cost(factor).unwrap(), 2.);

    let checkpoint = moved.checkpoint_states();
    moved.remove_factor(factor).unwrap();
    moved.restore_states(checkpoint).unwrap();
    assert!(moved.factor_cost(factor).is_err());
}

#[test]
fn changed_later_family_rejects_before_restoring_earlier_family() {
    let mut graph = Problem::<Mixed, Factors<f64>>::new();
    let x = graph.add(Scalar(1.));
    let checkpoint = graph.checkpoint_states();
    graph.set(x, Scalar(9.)).unwrap();
    let added = graph.add(Tagged::identity());
    assert_eq!(
        graph.restore_states(checkpoint),
        Err(StateCheckpointError::MembershipChanged)
    );
    assert_eq!(graph.get(x).unwrap().0, 9.);
    assert!(graph.get(added).is_ok());
}

#[test]
fn removed_and_reused_slots_are_rejected_even_at_same_length() {
    let mut graph = Problem::<Generic<f64>, Factors<f64>>::new();
    let removed = graph.add(Scalar(1.));
    let survivor = graph.add(Scalar(2.));
    let checkpoint = graph.checkpoint_states();
    graph.marginalize(&[removed.block_id()]).unwrap();
    let replacement = graph.add(Scalar(3.));
    graph.set(survivor, Scalar(8.)).unwrap();
    assert_eq!(
        graph.restore_states(checkpoint),
        Err(StateCheckpointError::MembershipChanged)
    );
    assert!(graph.get(removed).is_err());
    assert_ne!(removed, replacement);
    assert_eq!(graph.get(survivor).unwrap().0, 8.);
    assert_eq!(graph.get(replacement).unwrap().0, 3.);
}

#[test]
fn rejects_foreign_checkpoints_including_empty_schemas() {
    let mut first = Problem::<Generic<f64>, Factors<f64>>::new();
    let mut second = Problem::<Generic<f64>, Factors<f64>>::new();
    first.add(Scalar(1.));
    let x = second.add(Scalar(9.));
    assert_eq!(
        second.restore_states(first.checkpoint_states()),
        Err(StateCheckpointError::ForeignGraph)
    );
    assert_eq!(second.get(x).unwrap().0, 9.);

    let first = Problem::<Empty<f32>, NoFactors<f32>>::new();
    let mut second = Problem::<Empty<f32>, NoFactors<f32>>::new();
    let own = second.checkpoint_states();
    assert_eq!(
        second.restore_states(first.checkpoint_states()),
        Err(StateCheckpointError::ForeignGraph)
    );
    second.restore_states(own).unwrap();
}

#[test]
fn tracking_invalidates_only_after_successful_restore() {
    let mut graph = TrackedProblem::new(Problem::<Generic<f32>, NoFactors<f32>>::new());
    let x = graph.add(Scalar(1_f32));
    let before = graph.changes().to_vec();
    let checkpoint = graph.checkpoint_states();
    assert_eq!(graph.changes(), before);
    graph.set(x, Scalar(9.)).unwrap();
    graph.restore_states(checkpoint).unwrap();
    assert_eq!(graph.get(x).unwrap().0, 1.);
    assert_eq!(graph.changes(), &[ProblemChange::Rebuild]);

    let checkpoint = graph.checkpoint_states();
    graph.add(Scalar(3.));
    let before = graph.changes().to_vec();
    assert_eq!(
        graph.restore_states(checkpoint),
        Err(StateCheckpointError::MembershipChanged)
    );
    assert_eq!(graph.changes(), before);
    let foreign = Problem::<Generic<f32>, NoFactors<f32>>::new();
    assert_eq!(
        graph.restore_states(foreign.checkpoint_states()),
        Err(StateCheckpointError::ForeignGraph)
    );
    assert_eq!(graph.changes(), before);
}

#[test]
fn restore_preserves_historical_prior_and_anchor() {
    let mut graph = Problem::<Generic<f64>, Factors<f64>>::new();
    let x = graph.add(Scalar(0.));
    let y = graph.add(Scalar(0.));
    graph
        .add_factor(Prior {
            variable: x,
            measurement: 1.,
        })
        .unwrap();
    graph
        .add_factor(Difference {
            x,
            y,
            measurement: 2.,
        })
        .unwrap();
    graph.marginalize(&[x.block_id()]).unwrap();
    graph.set(y, Scalar(7.)).unwrap();
    let checkpoint = graph.checkpoint_states();
    graph.set(y, Scalar(20.)).unwrap();
    graph.restore_states(checkpoint).unwrap();
    let report = fagra::GaussNewton::default()
        .solve_batch(&mut graph, &Default::default())
        .unwrap();
    assert!((report.initial_cost - 4.).abs() < 1e-12);
    assert!((graph.get(y).unwrap().0 - 3.).abs() < 1e-12);
    assert!((graph.joint_covariance(&[y.block_id()]).unwrap()[(0, 0)] - 2.).abs() < 1e-12);
}

#[test]
fn snapshot_macro_is_hygienic_with_concrete_and_empty_schemas() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Problem<Mixed, Factors<f64>>>();
    type Result<T> = ::core::result::Result<T, ()>;
    struct Ok;
    struct Err;
    fagra::states! { Concrete { scalars: Scalar } }
    fagra::states! { EmptyConcrete {} }
    let mut graph = Problem::<Concrete, Factors<f64>>::new();
    graph.restore_states(graph.checkpoint_states()).unwrap();
    let mut empty = Problem::<EmptyConcrete, NoFactors<f64>>::new();
    empty.restore_states(empty.checkpoint_states()).unwrap();
    let _: Result<()> = ::core::result::Result::Ok(());
    let _ = (Ok, Err);
}

#[test]
fn conditional_capability_handles_many_families_and_colliding_field_names() {
    fagra::states! {
        #[allow(non_snake_case)]
        pub Many<__Pools> {
            __Pools: Scalar<__Pools>,
        }
    }
    let mut generic = Problem::<Many<f32>, NoFactors<f32>>::new();
    let x = generic.add(Scalar(1_f32));
    let checkpoint = generic.checkpoint_states();
    generic.set(x, Scalar(3.)).unwrap();
    generic.restore_states(checkpoint).unwrap();
    assert_eq!(generic.get(x).unwrap().0, 1.);

    fagra::states! {
        #[allow(non_snake_case)]
        pub Large {
            __Scalar: Tagged<0>, __Pools: Tagged<1>, pools: Tagged<2>,
            pools_mut: Tagged<3>, snapshot: Tagged<4>, values: Tagged<5>,
            identity: Tagged<6>, scalar: Tagged<7>, a: Tagged<8>,
            b: Tagged<9>, c: Tagged<10>, d: Tagged<11>, e: Tagged<12>,
            f: Tagged<13>, g: Tagged<14>, h: Tagged<15>,
        }
    }
    let mut graph = Problem::<Large, NoFactors<f64>>::new();
    let last = graph.add(Tagged::<15> {
        value: 7.,
        label: "last family".into(),
    });
    let checkpoint = graph.checkpoint_states();
    graph.set(last, Tagged::identity()).unwrap();
    graph.restore_states(checkpoint).unwrap();
    assert_eq!(graph.get(last).unwrap().label, "last family");
}
