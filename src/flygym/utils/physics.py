"""Reusable MuJoCo data for observations and shadow rollouts."""

import mujoco as mj
import numpy as np


class PhysicsSnapshot:
    """Own a persistent data object and integration-state transfer buffer.

    ``update(source)`` forwards an isolated copy of the integration state so
    observations describe current qpos/qvel without changing the live solver's
    warmstart or lagged contacts. ``update(source, forward=False)`` instead uses
    mj_copyData, retaining derived arrays exactly for a shadow rollout.

    Source data must belong to the exact model passed at construction. Model
    parameters are shared, not copied; this is not a serialized checkpoint or
    a mechanism for transferring state between independently compiled models.
    User callbacks/plugins may have external side effects when forwarded; those
    are the caller's responsibility. The instance must not be used concurrently.
    """

    def __init__(self, model: mj.MjModel) -> None:
        self.model = model
        self.data = mj.MjData(model)
        self._state = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))

    def update(self, source: mj.MjData, *, forward: bool = True) -> mj.MjData:
        """Refresh and return borrowed storage, overwritten on the next update."""
        if source.model is not self.model:
            raise ValueError("source must belong to the snapshot model")
        if source is self.data:
            raise ValueError("source must differ from the snapshot data")
        if forward:
            mj.mj_getState(
                self.model, source, self._state, mj.mjtState.mjSTATE_INTEGRATION
            )
            mj.mj_setState(
                self.model, self.data, self._state, mj.mjtState.mjSTATE_INTEGRATION
            )
            mj.mj_forward(self.model, self.data)
        else:
            mj.mj_copyData(self.data, self.model, source)
        return self.data
