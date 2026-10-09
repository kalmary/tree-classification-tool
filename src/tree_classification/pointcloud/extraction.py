import numpy as np
from pyproj import CRS, Transformer

from tree_classification.core.models import PointCloud, TreeData

def extract_trees(point_cloud: PointCloud, source_filename: str) -> list[TreeData]:
    """Splits a full PointCloud into individual TreeData objects."""
    trees = []
    unique_ids = np.unique(point_cloud.tree_ids)
    
    # Setup transformer if CRS is available
    transformer = None
    if point_cloud.crs_wkt:
        try:
            crs = CRS.from_wkt(point_cloud.crs_wkt)
            transformer = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
        except Exception:
            pass # Invalid WKT or unsupported, fallback to raw XY

    for tree_id in unique_ids:
        mask = point_cloud.tree_ids == tree_id
        tree_points = point_cloud.points[mask]
        
        if len(tree_points) == 0:
            continue
            
        tree_rgb = point_cloud.rgb[mask] if point_cloud.rgb is not None else None
        
        # Calculate centroid
        centroid_x = np.mean(tree_points[:, 0])
        centroid_y = np.mean(tree_points[:, 1])
        
        if transformer:
            # always_xy=True -> (lon, lat)
            lon, lat = transformer.transform(centroid_x, centroid_y)
        else:
            lon, lat = centroid_x, centroid_y
            
        height = float(np.max(tree_points[:, 2]) - np.min(tree_points[:, 2]))
        
        trees.append(TreeData(
            tree_id=int(tree_id),
            points=tree_points,
            rgb=tree_rgb,
            latitude=float(lat),
            longitude=float(lon),
            height=height,
            source_filename=source_filename
        ))
        
    return trees
