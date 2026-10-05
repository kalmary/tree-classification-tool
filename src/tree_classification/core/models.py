from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PointCloud:
    """Raw point cloud data as read from a .laz/.las file."""
    points: NDArray[np.float64]
    tree_ids: NDArray[np.int64]
    rgb: NDArray[np.uint8] | None
    crs_wkt: str | None


@dataclass(frozen=True)
class TreeData:
    """Single extracted tree with computed metadata."""
    tree_id: int
    points: NDArray[np.float64]
    rgb: NDArray[np.uint8] | None
    latitude: float
    longitude: float
    height: float
    source_filename: str


@dataclass
class TreeLabel:
    """One row in the CSV label store."""
    tree_id: int
    latitude: float
    longitude: float
    height: float
    source_tree_id: str
    label: int


@dataclass(frozen=True)
class TreeReport:
    """A tree with its rendered views, ready for PDF composition."""
    tree: TreeData
    views: "list[NDArray[np.float32]]"
