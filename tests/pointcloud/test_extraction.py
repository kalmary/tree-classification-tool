import numpy as np

from tree_classification.core.models import PointCloud
from tree_classification.pointcloud.extraction import (
    extract_trees,
)


def test_extract_trees():
    # Setup dummy data
    points = np.array([
        [0, 0, 0], [0, 0, 10],  # Tree 1, height 10
        [100, 100, 5], [100, 100, 25] # Tree 2, height 20
    ], dtype=np.float64)
    tree_ids = np.array([1, 1, 2, 2], dtype=np.int64)
    
    pc = PointCloud(points=points, tree_ids=tree_ids, rgb=None, crs_wkt=None)
    trees = extract_trees(pc, "testfile")
    
    assert len(trees) == 2
    assert trees[0].tree_id == 1
    assert trees[0].height == 10.0
    assert trees[0].latitude == 0.0 # fallback
    
    assert trees[1].tree_id == 2
    assert trees[1].height == 20.0
    assert trees[1].latitude == 100.0 # fallback
