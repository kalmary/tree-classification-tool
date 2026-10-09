import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from tree_classification.rendering.depth_map import (
    VIEW_NAMES,
    DepthMapRenderer,
)


def _box_cloud() -> NDArray[np.float64]:
    return np.array([
        [0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0],
        [0, 0, 10], [10, 0, 10], [0, 10, 10], [10, 10, 10],
        [5, 5, 5]  # Center
    ], dtype=np.float64)


def test_depth_map_renderer():
    from tree_classification.rendering.depth_map import DepthMapRenderer
    
    renderer = DepthMapRenderer(device="cpu", resolution=64)
    
    # Create a simple cubic point cloud (corners of a box)
    points = np.array([
        [0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0],
        [0, 0, 10], [10, 0, 10], [0, 10, 10], [10, 10, 10],
        [5, 5, 5] # Center
    ], dtype=np.float64)
    
    views = renderer.render(points, rgb=None, resolution=64)
    
    assert len(views) == 5, "Should generate 5 views (top, front, back, left, right)"
    
    for i, view in enumerate(views):
        assert view.shape == (64, 64), f"View {i} has incorrect shape"
        assert view.dtype == np.float32, f"View {i} has incorrect dtype"
        assert np.max(view) > 0, f"View {i} is completely empty"


def test_render_uses_configured_resolution_by_default():
    renderer = DepthMapRenderer(device="cpu", resolution=32)

    for view in renderer.render(_box_cloud()):
        assert view.shape == (32, 32)


def test_grayscale_views_span_background_to_nearest_surface():
    renderer = DepthMapRenderer(device="cpu", resolution=16)

    for name, view in zip(VIEW_NAMES, renderer.render(_box_cloud())):
        assert view.min() == 0.0, f"View {name} has no empty pixels"
        assert view.max() == pytest.approx(1.0), f"View {name} has no nearest surface"


def test_rgb_views_take_the_nearest_point_colour():
    renderer = DepthMapRenderer(device="cpu", resolution=8)
    # Both points fall on one pixel of the top view, which sees the upper one.
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    rgb = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

    top = renderer.render(points, rgb=rgb)[0]

    assert top.shape == (8, 8, 3)
    assert top.dtype == np.float32
    coloured = top[top.any(axis=-1)]
    assert len(coloured) == 1
    assert coloured[0] == pytest.approx([0.0, 1.0, 0.0])


def test_front_and_back_views_are_mirrored():
    renderer = DepthMapRenderer(device="cpu", resolution=8)
    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])

    _, front, back, _, _ = renderer.render(points)

    assert np.array_equal(front, np.fliplr(back))


def test_left_and_right_views_are_mirrored():
    renderer = DepthMapRenderer(device="cpu", resolution=8)
    points = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

    *_, left, right = renderer.render(points)

    assert np.array_equal(left, np.fliplr(right))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_render_matches_cpu():
    rng = np.random.default_rng(0)
    points = rng.uniform(-5.0, 5.0, size=(2000, 3))
    rgb = rng.integers(0, 256, size=(2000, 3), dtype=np.uint8)
    cpu = DepthMapRenderer(device="cpu", resolution=32)
    gpu = DepthMapRenderer(device="cuda", resolution=32)

    for colours in (None, rgb):
        cpu_views = cpu.render(points, rgb=colours)
        gpu_views = gpu.render(points, rgb=colours)

        for name, cpu_view, gpu_view in zip(VIEW_NAMES, cpu_views, gpu_views):
            assert isinstance(gpu_view, np.ndarray), f"View {name} is not on the host"
            assert gpu_view.dtype == np.float32
            np.testing.assert_allclose(gpu_view, cpu_view, err_msg=f"View {name} differs")
