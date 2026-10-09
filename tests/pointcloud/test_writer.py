import laspy
import numpy as np

from tree_classification.core.models import TreeData
from tree_classification.pointcloud.writer import (
    LaspyWriter,
)


def test_writer(tmp_path):

    
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    rgb = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    
    tree = TreeData(
        tree_id=42,
        points=points,
        rgb=rgb,
        latitude=50.0,
        longitude=20.0,
        height=3.0,
        source_filename="test_source"
    )
    
    out_file = tmp_path / "test_tree.las"
    writer = LaspyWriter()
    writer.write(tree, out_file)
    
    assert out_file.exists()
    
    las = laspy.read(out_file)
    assert len(las.points) == 2
    assert np.allclose(las.x, points[:, 0])
    assert np.allclose(las.y, points[:, 1])
    assert np.allclose(las.z, points[:, 2])
    
    assert las.red[0] == 255 * 256
    assert las.green[1] == 255 * 256
