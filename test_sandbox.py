import numpy as np
import laspy
header = laspy.LasHeader(point_format=3, version="1.2")
las = laspy.LasData(header)
las.x = [10.0, 20.0]
las.add_extra_dim(laspy.ExtraBytesParams(name="tree_ids", type=np.int32))
las.tree_ids = [1, 2]
las.write("test_sandbox.las")
