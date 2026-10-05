import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from tree_classification.core.protocols import Renderer

# Order of the views returned by DepthMapRenderer.render.
VIEW_NAMES = ("top", "front", "back", "left", "right")


class DepthMapRenderer(Renderer):
    """Projects a tree point cloud onto the 5 orthographic views in VIEW_NAMES.

    Points are fitted into a cube first, so all views share one scale and the
    tree keeps its proportions.

    A grayscale view is a (resolution, resolution) depth map in which 1.0 is the
    surface nearest the camera and 1/255 the farthest, leaving 0.0 to mean "no
    point projected onto this pixel". Given rgb, the nearest point's colour
    replaces that depth shade and views become (resolution, resolution, 3).
    """

    def __init__(self, device: str = "cpu", resolution: int = 256,
                 margin_ratio: float = 0.05):
        self.device = torch.device(device)
        self.resolution = resolution
        self.margin_ratio = margin_ratio

    def render(self, points: NDArray[np.float64],
               rgb: NDArray[np.uint8] | None = None,
               resolution: int | None = None) -> list[NDArray[np.float32]]:
        """Render the 5 views. resolution defaults to the configured one."""
        resolution_xy = self.resolution if resolution is None else resolution

        points_t = torch.as_tensor(points, dtype=torch.float64, device=self.device)
        colours = None
        if rgb is not None:
            colours = torch.as_tensor(rgb, dtype=torch.float64, device=self.device) / 255.0

        min_xyz = points_t.min(dim=0).values
        max_xyz = points_t.max(dim=0).values

        center = (min_xyz + max_xyz) / 2
        max_range = (max_xyz - min_xyz).max()
        cube_half = max_range / 2 * (1 + 2 * self.margin_ratio)

        cube_min = center - cube_half
        cube_max = center + cube_half

        def to_grid(values, min_val, max_val):
            return torch.clamp(
                ((values - min_val) / (max_val - min_val + 1e-8) * (resolution_xy - 1)).long(),
                0, resolution_xy - 1,
            )

        x, y, z = points_t[:, 0], points_t[:, 1], points_t[:, 2]

        gx = to_grid(x, cube_min[0], cube_max[0])
        gy = to_grid(y, cube_min[1], cube_max[1])
        gz = to_grid(z, cube_min[2], cube_max[2])

        def build_view(row_idx, col_idx, distances, flip_y=False, flip_x=False):
            if flip_y:
                row_idx = resolution_xy - 1 - row_idx
            if flip_x:
                col_idx = resolution_xy - 1 - col_idx

            flat_indices = row_idx * resolution_xy + col_idx
            nearest = torch.full((resolution_xy * resolution_xy,), float("inf"),
                                 dtype=torch.float64, device=self.device)
            nearest = torch.scatter_reduce(nearest, 0, flat_indices, distances,
                                           reduce="amin", include_self=True)

            if colours is None:
                img = torch.zeros(resolution_xy * resolution_xy, dtype=torch.float64,
                                  device=self.device)
                occupied = torch.isfinite(nearest)
                if torch.any(occupied):
                    depths = nearest[occupied]
                    shade = (depths.max() - depths) / (depths.max() - depths.min() + 1e-8)
                    img[occupied] = shade * (1.0 - 1.0 / 255.0) + 1.0 / 255.0
                img = img.view(resolution_xy, resolution_xy)
            else:
                img = torch.zeros((resolution_xy * resolution_xy, 3), dtype=torch.float64,
                                  device=self.device)
                # A point is visible where its distance is its pixel's minimum.
                # Points coinciding at that distance resolve arbitrarily.
                visible = distances == nearest[flat_indices]
                img[flat_indices[visible]] = colours[visible]
                img = img.view(resolution_xy, resolution_xy, 3)

            return img.type(torch.float32).cpu().numpy()

        return [
            build_view(gy, gx, cube_max[2] - z),
            build_view(gz, gx, cube_max[1] - y, flip_y=True),
            build_view(gz, gx, y - cube_min[1], flip_y=True, flip_x=True),
            build_view(gz, gy, cube_max[0] - x, flip_y=True),
            build_view(gz, gy, x - cube_min[0], flip_y=True, flip_x=True),
        ]


# --- Tests ---

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
