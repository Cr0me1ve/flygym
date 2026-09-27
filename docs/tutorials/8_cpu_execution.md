# Avoiding repeated work in CPU simulations

These APIs preserve the physics timestep. Select them according to the cadence
of the controller and observations; skipping controller updates changes a task.

## Held controls and batched steps

`sim.step(nstep=5)` calls MuJoCo's native multi-step binding. It is equivalent to
five `sim.step()` calls if controls and other Python-side inputs are held over
that interval. MuJoCo callbacks still run on each native step.
`sim.step_with_profile(nstep=5)` counts all five physical steps.

## Ommatidia readouts

`sim.get_ommatidia_readouts("fly")` pools directly from raw camera frames with a
cached fisheye lookup; it does not allocate corrected RGB images. The existing
`get_raw_vision()` API still returns corrected images.

Use `sim.warmup_vision("fly")` before timing to initialize rendering/JIT without
stepping physics. Forward the model first if its transforms are not current.
Warmup discards its frame, so the first scheduled observation remains fresh.

An optional simulation-time cadence avoids repeated rendering:

```python
frame = sim.get_ommatidia_readouts("fly", refresh_interval=0.02)
```

The first call renders immediately. Repeated queries hold the frame until the
next 20 ms deadline. Each fly has its own cache; returned arrays are independent.
The default (`refresh_interval=None`) always renders. This is an explicit sensory
sampling choice, not an equivalent replacement for vision on every physics step.
`sim.reset()` clears all caches. Call `sim.clear_vision_cache()` after restoring a
checkpoint or directly editing physics state, cameras, retinal parameters, model
appearance or scene options, including edits at the same simulation timestamp.
The cache is not a serialized checkpoint.

## Isolated observations and shadow physics

MuJoCo advances qpos at the end of a step; derived arrays can describe the
preceding force evaluation. Forward a persistent snapshot to observe current
integration state without changing live warmstart/contact data:

```python
from flygym.utils.physics import PhysicsSnapshot

snapshot = PhysicsSnapshot(sim.mj_model)  # allocate once, outside the loop
observed = snapshot.update(sim.mj_data)  # integration-state transfer + forward
positions = observed.xpos.copy()        # retain only data the caller needs
```

`observed` is borrowed storage and is overwritten by the next update. For a
shadow rollout, `snapshot.update(sim.mj_data, forward=False)` uses `mj_copyData`
to preserve lagged derived fields exactly before stepping `snapshot.data`.
The model is shared, not cloned. Both data objects must belong to that exact
model. Callbacks/plugins with external effects are not isolated by copying data.

FlyGym already keeps raw ground-contact sensors (`get_ground_contact_info`)
separate from detailed contact traversal (`get_bodysegment_contact_forces`);
request the detailed diagnostics only when needed. The latter reuses fixed-size
wrench/force scratch, while each returned force array remains independent.
