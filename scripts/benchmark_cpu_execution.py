"""Small-model execution microbenchmarks; no full-fly throughput claim.

Run with: PYTHONPATH=src python scripts/benchmark_cpu_execution.py
"""

import json
import platform
import statistics
import time

import mujoco as mj
import numpy as np

from flygym.simulation import Simulation
from flygym.utils.physics import PhysicsSnapshot


def main():
    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body pos="0 0 .11"><freejoint/>
      <geom type="sphere" size=".1" mass="1"/></body>
      </worldbody></mujoco>""")
    sim = object.__new__(Simulation)
    sim.mj_model, sim.mj_data = model, mj.MjData(model)
    serial = mj.MjData(model)
    for _ in range(1000):
        mj.mj_step(model, serial)
    sim.step(1000)
    np.testing.assert_array_equal(sim.mj_data.qpos, serial.qpos)
    np.testing.assert_array_equal(sim.mj_data.qvel, serial.qvel)
    reference = mj.MjData(model)
    snapshot = PhysicsSnapshot(model)

    def serial_step():
        for _ in range(1000):
            mj.mj_step(model, sim.mj_data)

    def batch_step():
        sim.step(1000)

    def copy_forward():
        for _ in range(100):
            mj.mj_copyData(reference, model, serial)
            mj.mj_forward(model, reference)

    def compact_forward():
        for _ in range(100):
            snapshot.update(serial)

    copy_forward()
    compact_forward()
    for field in ("xpos", "qvel", "qacc", "qacc_warmstart"):
        np.testing.assert_array_equal(
            getattr(reference, field), getattr(snapshot.data, field)
        )
    functions = [
        ("serial_1000_steps", serial_step),
        ("batch_1000_steps", batch_step),
        ("copy_forward_100", copy_forward),
        ("compact_forward_100", compact_forward),
    ]
    samples = {name: [] for name, _ in functions}
    for repeat in range(20):
        for name, function in functions[:: 1 if repeat % 2 == 0 else -1]:
            mj.mj_copyData(sim.mj_data, model, serial)
            start = time.perf_counter()
            function()
            samples[name].append(time.perf_counter() - start)
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "mujoco": mj.__version__,
                "exact": True,
                "model_nq": model.nq,
                "model_nv": model.nv,
                "median_seconds": {k: statistics.median(v) for k, v in samples.items()},
                "samples_seconds": samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
