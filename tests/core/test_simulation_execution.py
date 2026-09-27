"""Execution equivalence with real MuJoCo and no downloadable mesh fixtures."""

from types import SimpleNamespace
from unittest.mock import Mock

import mujoco as mj
import numpy as np
import pytest

from flygym.simulation import Simulation
from flygym.utils.physics import PhysicsSnapshot
from flygym.vision.retina import Retina


XML = """<mujoco><option timestep="0.001"/>
<worldbody><geom type="plane" size="2 2 .1"/>
<body pos="0 0 .12"><freejoint/><geom type="sphere" size=".1" mass="1"/>
<site name="sensor"/></body></worldbody>
<sensor><accelerometer site="sensor"/></sensor>
<keyframe><key name="neutral"/></keyframe></mujoco>"""


def simulation(model=None):
    sim = object.__new__(Simulation)
    sim.mj_model = model if model is not None else mj.MjModel.from_xml_string(XML)
    sim.mj_data = mj.MjData(sim.mj_model)
    sim._neutral_keyframe_id = 0
    sim.renderer = None
    sim._vision_cache = {}
    sim._curr_step = sim._frames_rendered = 0
    sim._total_physics_time_ns = sim._total_render_time_ns = 0
    return sim


@pytest.mark.parametrize("integrator", [0, 1, 2, 3])
def test_native_batch_exact(integrator):
    sim = simulation()
    sim.mj_model.opt.integrator = integrator
    reference = mj.MjData(sim.mj_model)
    for data in [sim.mj_data, reference]:
        data.qvel[0] = 0.15
        data.qfrc_applied[0] = 0.2
    sim.step_with_profile(250)
    for _ in range(250):
        mj.mj_step(sim.mj_model, reference)
    for field in ["qpos", "qvel", "qacc_warmstart", "sensordata", "qfrc_constraint"]:
        np.testing.assert_array_equal(
            getattr(sim.mj_data, field), getattr(reference, field)
        )
    assert sim.mj_data.time == reference.time
    assert sim._curr_step == 250
    assert sim._total_physics_time_ns > 0


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_invalid_nstep_does_not_step(bad):
    sim = simulation()
    with pytest.raises((ValueError, TypeError)):
        sim.step(bad)
    assert sim.mj_data.time == 0


def test_snapshot_isolation_and_exact_shadow_continuation():
    sim = simulation()
    sim.step(250)
    snap = PhysicsSnapshot(sim.mj_model)
    before = sim.mj_data.qacc_warmstart.copy()
    live_pos = sim.mj_data.xpos.copy()
    ref = mj.MjData(sim.mj_model)
    mj.mj_copyData(ref, sim.mj_model, sim.mj_data)
    mj.mj_forward(sim.mj_model, ref)
    observed = snap.update(sim.mj_data)
    for field in ["qpos", "qvel", "sensordata", "xpos", "qfrc_constraint"]:
        np.testing.assert_array_equal(getattr(observed, field), getattr(ref, field))
    np.testing.assert_array_equal(sim.mj_data.qacc_warmstart, before)
    np.testing.assert_array_equal(sim.mj_data.xpos, live_pos)
    assert snap.update(sim.mj_data) is observed
    shadow = snap.update(sim.mj_data, forward=False)
    mj.mj_step(sim.mj_model, shadow, nstep=100)
    sim.step(100)
    for field in ["qpos", "qvel", "qacc_warmstart", "sensordata"]:
        np.testing.assert_array_equal(
            getattr(shadow, field), getattr(sim.mj_data, field)
        )
    with pytest.raises(ValueError):
        snap.update(shadow)
    with pytest.raises(ValueError, match="model"):
        snap.update(simulation().mj_data)


def vision_sim():
    sim = simulation()
    sim.retina = Retina()
    sim._intern_eye_camera_ids_by_fly = {"a": [0, 1], "b": [2, 3]}
    sim.eye_renderer_scene_option = mj.MjvOption()
    sim.eye_renderer = Mock()
    sim.eye_renderer.render.return_value = np.full((512, 450, 3), 100, dtype=np.uint8)
    return sim


def test_vision_cadence_reset_warmup_and_independent_fly_caches():
    sim = vision_sim()
    sim.warmup_vision("a")
    assert sim.mj_data.time == 0
    assert not sim._vision_cache
    sim.eye_renderer.render.reset_mock()
    first = sim.get_ommatidia_readouts("a", refresh_interval=0.02)
    first[:] = 0
    assert sim.get_ommatidia_readouts("a", refresh_interval=0.02).max() > 0
    assert sim.eye_renderer.render.call_count == 2
    sim.mj_data.time = 0.019
    sim.get_ommatidia_readouts("a", refresh_interval=0.02)
    assert sim.eye_renderer.render.call_count == 2
    sim.mj_data.time = 0.02
    sim.get_ommatidia_readouts("a", refresh_interval=0.02)
    assert sim.eye_renderer.render.call_count == 4
    sim.get_ommatidia_readouts("b", refresh_interval=0.02)
    assert sim.eye_renderer.render.call_count == 6
    sim.clear_vision_cache("a")
    assert "b" in sim._vision_cache
    sim.get_ommatidia_readouts("a", refresh_interval=0.02)
    assert sim.eye_renderer.render.call_count == 8
    sim.reset()
    assert not sim._vision_cache


def test_vision_rewind_interval_change_and_unconditional_refresh():
    sim = vision_sim()
    sim.mj_data.time = 0.1
    sim.get_ommatidia_readouts("a", refresh_interval=0.02)
    sim.mj_data.time = 0.05
    sim.get_ommatidia_readouts("a", refresh_interval=0.02)
    sim.get_ommatidia_readouts("a", refresh_interval=0.03)
    sim.get_ommatidia_readouts("a")
    sim.get_ommatidia_readouts("a")
    assert sim.eye_renderer.render.call_count == 10
    assert "a" not in sim._vision_cache


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_invalid_vision_interval(bad):
    sim = vision_sim()
    with pytest.raises(ValueError):
        sim.get_ommatidia_readouts("a", refresh_interval=bad)
    sim.eye_renderer.render.assert_not_called()


def test_contact_force_scratch_reused_without_aliasing():
    from flygym.anatomy import BodySegment

    sim = simulation()
    sim.step(500)
    segment = BodySegment("c_thorax")
    sim._internal_geomid_by_bodyseg_by_fly = {"a": {segment: 1}}
    sim._internal_ground_geom_ids = np.array([0], dtype=np.int32)
    expected = np.zeros(3)
    wrench = np.zeros(6)
    for i, contact in enumerate(sim.mj_data.contact):
        if contact.exclude:
            continue
        mj.mj_contactForce(sim.mj_model, sim.mj_data, i, wrench)
        force = contact.frame.reshape(3, 3).T @ wrench[:3]
        expected += force if contact.geom2 == 1 else -force
    result = sim.get_bodysegment_contact_forces("a", [segment])
    np.testing.assert_array_equal(result[0], expected)
    scratch = sim._contact_force_scratch
    again = sim.get_bodysegment_contact_forces("a", [segment])
    assert sim._contact_force_scratch is scratch
    again[:] = 0
    np.testing.assert_array_equal(result[0], expected)
    assert np.linalg.norm(expected) > 0
