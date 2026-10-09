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
