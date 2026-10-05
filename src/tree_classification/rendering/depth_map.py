import numpy as np
import pytest

# --- Tests ---
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
