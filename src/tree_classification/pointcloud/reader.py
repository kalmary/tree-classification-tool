from pathlib import Path
import laspy
import numpy as np

from tree_classification.core.models import PointCloud
from tree_classification.core.protocols import PointCloudReader

class LaspyReader(PointCloudReader):
    """Reads a .laz/.las file and extracts required features."""
    
    def read(self, path: Path) -> PointCloud:
        las = laspy.read(path)
        
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        # Extract tree_ids
        if hasattr(las, "tree_ids"):
            tree_ids = np.array(las.tree_ids)
        elif hasattr(las, "tree_id"):
            tree_ids = np.array(las.tree_id)
        else:
            raise ValueError(f"File {path} does not contain a 'tree_ids' or 'tree_id' field.")
            
        # Extract RGB if available (convert 16-bit to 8-bit)
        rgb = None
        if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
            r = np.array(las.red)
            if len(r) > 0:
                if r.max() > 255:
                    r = (r / 256).astype(np.uint8)
                    g = (np.array(las.green) / 256).astype(np.uint8)
                    b = (np.array(las.blue) / 256).astype(np.uint8)
                else:
                    r = r.astype(np.uint8)
                    g = np.array(las.green).astype(np.uint8)
                    b = np.array(las.blue).astype(np.uint8)
                rgb = np.vstack((r, g, b)).transpose()
            
        # Extract CRS WKT
        crs_wkt = None
        for vlr in las.vlrs:
            if vlr.record_id == 2112 and vlr.user_id == "LASF_Projection":
                if hasattr(vlr, 'string'):
                    crs_wkt = vlr.string
                elif hasattr(vlr, 'wkt'):
                    crs_wkt = vlr.wkt
                break
                
        return PointCloud(
            points=points,
            tree_ids=tree_ids,
            rgb=rgb,
            crs_wkt=crs_wkt
        )

# --- Tests ---
def test_reader(tmp_path):
    import numpy as np
    import laspy
    from tree_classification.pointcloud.reader import LaspyReader
    
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = [10.0, 20.0]
    las.y = [10.0, 20.0]
    las.z = [5.0, 15.0]
    las.red = [65535, 0]
    las.green = [0, 65535]
    las.blue = [0, 0]
    
    las.add_extra_dim(laspy.ExtraBytesParams(name="tree_ids", type=np.int32))
    las.tree_ids = [1, 2]
    
    test_file = tmp_path / "test_read.las"
    las.write(test_file)
    
    reader = LaspyReader()
    pc = reader.read(test_file)
    
    assert len(pc.points) == 2
    assert np.array_equal(pc.tree_ids, [1, 2])
    
    assert pc.rgb[0][0] == 255
    assert pc.rgb[1][1] == 255
