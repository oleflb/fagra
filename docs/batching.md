# Shared evaluation with batches

Start with the [ordinary scalar-prior example](../examples/scalar_prior.rs).
Use `FactorBatch<S>` when many independent factors can share expensive work,
such as interpolating a camera trajectory once for a frame's observations.

## Model, payload, and handles

| Concept | Role | SLAM example |
| --- | --- | --- |
| Model implementing `FactorBatch<S>` | Shared inputs and evaluation | `FrameReprojections` |
| Payload (`FactorBatch::Factor`) | One factor's local measurement | `Reprojection` |
| `BatchKey<Model>` | One model instance and its payload storage | A frame |
| `FactorKey<Payload>` | One independently removable factor | An observation |
| `FactorSelection<Payload>` | Selected payloads paired with their `FactorId`s | Observations needed by this pass |

The payload does not implement `Factor` itself. The model's `visit_variables`
must report each payload's complete dependencies, including shared inputs.
Its `cost` sums only the selected factors' costs, using the same objective as
`Factor::cost`. Empty selections do no work.

## Register and use a batch

Using the types in [examples/slam.rs](../examples/slam.rs):

```rust
fagra::factors! {
    SlamFactors {
        priors: PosePrior,
        reprojections: Batch<FrameReprojections, Reprojection>,
    }
}
```

`Batch<Model, Payload>` is literal macro syntax. There is no public `Batch` type
to import or construct; create the model and payload values instead.
Each payload type selects one registered family. That family can contain many
batch instances, each with its own model and contiguous payload buffer.

```rust
let frame = solver.add_batch(FrameReprojections { trajectory, camera });
let observation = solver.add_factor_to(frame, Reprojection { landmark, pixel })?;

solver.optimize()?;
let cost = solver.factor_cost(observation)?;
solver.remove_factor(observation)?;
```

Removing one observation preserves sibling factor handles. The graph operations
above remain API stubs; the SLAM example type-checks the complete workflow.

## Ordinary versus batched factor scopes

Both evaluators emit through `LinearizationSink`, but they receive it in different
states. A scope associates all residual blocks with one graph factor.

| Evaluator | Who opens the scope? | What the evaluator emits |
| --- | --- | --- |
| `Factor::linearize` | The solver, before calling the factor | Residuals directly |
| `FactorBatch::linearize` | The batch evaluator, once per selected `FactorId` | Each factor's residuals inside its scope |

**Ordinary factor — emit directly:**

```rust
sink.residual(&residual, &jacobians)?;
```

**Batch evaluator — prepare shared work once, then scope each factor:**

```rust
for (id, payload) in factors {
    // Compute this payload's residual and Jacobians using shared intermediates.
    sink.factor(id, |out| out.residual(&residual, &jacobians))?;
}
```

Place all residual blocks for a factor inside the same scope. Do not call
`sink.factor` from an ordinary factor (that would nest scopes), or emit unscoped
residuals from a batch (that would omit the factor identity). Propagate evaluation
errors rather than silently skipping measurements.

The `LinearizationSink` rustdoc contains compile-checked versions of both emission
patterns. The SLAM example shows the full batch computation and identity flow.
