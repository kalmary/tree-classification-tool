import laspy
import numpy as np

from tree_classification.pointcloud.reader import (
    LaspyReader,
)


def test_reader(tmp_path):
    
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
