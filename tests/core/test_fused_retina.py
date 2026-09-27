"""Exact equivalence of fused and two-stage retinal sampling."""

from unittest.mock import Mock

import numpy as np
import pytest

from flygym.simulation import Simulation
from flygym.vision.retina import Retina


def assert_reference(retina, image):
    expected = retina.raw_image_to_hex_pxls(retina.correct_fisheye(image))
    actual = retina.fisheye_image_to_hex_pxls(image)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("zoom,distortion", [(1.0, 0.0), (0.7, 0.9), (2.0, -0.5)])
@pytest.mark.parametrize(
    "kind", ["random", "zero", "white", "green", "blue", "strided"]
)
def test_custom_retina(zoom, distortion, kind):
    ids = np.zeros((13, 17), dtype=np.int16)
    ids[1:6, 2:9] = 1
    ids[6:12, 9:16] = 2
    retina = Retina(ids, np.array([0, 1]), distortion, zoom, 13, 17)
    image = np.random.default_rng(42).integers(0, 256, (13, 17, 3), dtype=np.uint8)
    if kind == "zero":
        image[:] = 0
    elif kind == "white":
        image[:] = 255
    elif kind in ("green", "blue"):
        image[:] = 0
        image[..., 1 if kind == "green" else 2] = 255
    elif kind == "strided":
        image = image[:, ::-1]
    assert_reference(retina, image)


def test_default_retina_and_mutation():
    retina = Retina()
    image = np.random.default_rng(123).integers(
        0, 256, (retina.nrows, retina.ncols, 3), dtype=np.uint8
    )
    assert_reference(retina, image)
    lookup = retina._fisheye_lookup
    assert_reference(retina, image)
    assert retina._fisheye_lookup is lookup
    retina.zoom *= 0.8
    assert_reference(retina, image)
    assert retina._fisheye_lookup is not lookup
    retina.distortion_coefficient *= 0.5
    assert_reference(retina, image)
    retina.pale_type_mask[:] = 1 - retina.pale_type_mask
    retina.ommatidia_id_map[:] = retina.ommatidia_id_map[:, ::-1].copy()
    assert_reference(retina, image)


@pytest.mark.parametrize(
    "bad",
    [np.zeros((2, 3, 3), dtype=np.uint8), np.zeros((512, 450, 3), dtype=np.float32)],
)
def test_invalid_image(bad):
    with pytest.raises(ValueError):
        Retina().fisheye_image_to_hex_pxls(bad)


def test_simulation_paths():
    # Exercise the real rendering iterator with a fake renderer (no GL required).
    sim = object.__new__(Simulation)
    sim.retina = Retina()
    sim._intern_eye_camera_ids_by_fly = {"fly": [3, 7]}
    sim.mj_data = object()
    sim.eye_renderer_scene_option = object()
    sim.eye_renderer = Mock()
    rng = np.random.default_rng(5)
    frames = [
        rng.integers(0, 256, (sim.retina.nrows, sim.retina.ncols, 3), dtype=np.uint8)
        for _ in range(2)
    ]
    expected_raw = np.array([sim.retina.correct_fisheye(f) for f in frames])
    expected = np.array(
        [sim.retina.raw_image_to_hex_pxls(f) for f in expected_raw], dtype=np.float32
    )
    sim.eye_renderer.render.side_effect = frames
    np.testing.assert_array_equal(sim.get_raw_vision("fly"), expected_raw)
    sim.eye_renderer.render.side_effect = frames
    np.testing.assert_array_equal(sim.get_ommatidia_readouts("fly"), expected)
    assert sim.eye_renderer.update_scene.call_args_list[-1].args[1] == 7
    with pytest.raises(ValueError, match="does not have any eye cameras"):
        sim.get_ommatidia_readouts("missing")


@pytest.mark.parametrize("method", ["get_raw_vision", "get_ommatidia_readouts"])
def test_lazy_renderer_initialization(monkeypatch, method):
    import mujoco as mj

    sim = object.__new__(Simulation)
    sim.retina = None
    sim.eye_renderer = None
    sim.mj_model = object()
    sim.mj_data = object()
    sim._intern_eye_camera_ids_by_fly = {"fly": [0, 1]}
    renderer = Mock()
    retina = Retina()
    renderer.render.return_value = np.zeros(
        (retina.nrows, retina.ncols, 3), dtype=np.uint8
    )
    constructor = Mock(return_value=renderer)
    monkeypatch.setattr(mj, "Renderer", constructor)
    result = getattr(sim, method)("fly")
    assert result.shape[0] == 2
    np.testing.assert_array_equal(result, 0)
    constructor.assert_called_once_with(
        sim.mj_model, height=retina.nrows, width=retina.ncols
    )
    assert renderer.render.call_count == 2
    assert sim.eye_renderer_scene_option.geomgroup[1] == 0
    assert sim.eye_renderer_scene_option.geomgroup[2] == 0
