# Fagra

A typed Rust factor-graph API with homogeneous storage and static dispatch.

**API scaffold:** operational methods are `todo!()` stubs. Optimization, handle
management, and marginalization are not implemented. Examples currently type-check only.

## Declare storage

```rust
use fagra::{Batch, Solver};

fagra::states! {
    pub SlamStates {
        poses: Pose,
        cameras: CameraIntrinsics,
        landmarks: Landmark,
    }
}

fagra::factors! {
    pub SlamFactors {
        priors: PosePrior,
        reprojections: Batch<FrameReprojections, Reprojection>,
    }
}

type SlamSolver = Solver<SlamStates, SlamFactors>;
```

- `Variable` defines tangent conversion and retraction.
- `StateKey<T>` identifies a state; `FactorKey<T>` and `BatchKey<T>` identify graph contributions and evaluators.
- `Factor<S>` evaluates an ordinary constraint.
- `FactorBatch<S>` shares work across independently identified factor payloads.
- `Batch<B, F>` stores model `B` and contiguous payloads `F`; no second batch trait is needed.
- `LinearizationSink` receives factor-tagged residuals and borrowed Jacobian blocks.

`JacobianBlock::new(state_key, &jacobian)` derives the numerical identity from
the typed state key and checks the column count against `T::DOF` during code generation.

Each payload type selects one registered family. Each batch instance owns its
own factor buffer. The solver controls storage mutation and stable handles.

## Use the graph

```rust
let mut solver = SlamSolver::new();
let camera = solver.add(initial_intrinsics);
let landmark = solver.add(initial_landmark);

let frame = solver.add_batch(FrameReprojections { trajectory, camera });
let observation = solver.add_factor_to(frame, Reprojection { landmark, pixel })?;

solver.update()?;
let estimate = solver.get(landmark)?;
let cost = solver.factor_error(observation)?;
solver.remove_factor(observation)?;
```

See [examples/slam.rs](examples/slam.rs) for the compile-checked declarations,
storage-generic implementations, six-variable reprojection emissions, and marginalization API.
Application geometry is deliberately left as placeholders.

The intended hot path reuses workspace after structure preparation. Graph edits
may allocate; user evaluators must also avoid allocations. Marginalization's
manifold-coordinate contract and heterogeneous bulk API are still open.

```sh
cargo check --all-targets
cargo test
cargo doc --no-deps
```
