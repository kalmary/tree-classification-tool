from pathlib import Path

import laspy
import numpy as np

from tree_classification.core.models import TreeData
from tree_classification.core.protocols import PointCloudWriter


class LaspyWriter(PointCloudWriter):
    """Writes a single TreeData back to a .las/.laz file."""
    
    def write(self, tree: TreeData, output_path: Path, source_format: str = "") -> None:
        # Create a new las file. Format 3 supports RGB.
        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)
        
        las.x = tree.points[:, 0]
        las.y = tree.points[:, 1]
        las.z = tree.points[:, 2]
        
        if tree.rgb is not None:
            las.red = (tree.rgb[:, 0].astype(np.uint16) * 256)
            las.green = (tree.rgb[:, 1].astype(np.uint16) * 256)
            las.blue = (tree.rgb[:, 2].astype(np.uint16) * 256)
            
        las.write(output_path)

# --- Tests ---
def test_writer(tmp_path):
    import laspy
    import numpy as np

    from tree_classification.core.models import TreeData
    from tree_classification.pointcloud.writer import LaspyWriter
    
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
