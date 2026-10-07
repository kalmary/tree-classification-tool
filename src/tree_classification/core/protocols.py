from typing import Protocol, runtime_checkable
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from tree_classification.core.models import PointCloud, TreeData, TreeLabel, TreeReport

@runtime_checkable
class PointCloudReader(Protocol):
    def read(self, path: Path) -> PointCloud: ...

@runtime_checkable
class PointCloudWriter(Protocol):
    def write(self, tree: TreeData, output_path: Path, source_format: str) -> None: ...

@runtime_checkable
class Renderer(Protocol):
    def render(self, points: NDArray[np.float64],
               rgb: NDArray[np.uint8] | None = None,
               resolution: int = 256) -> list[NDArray[np.float32]]: ...

@runtime_checkable
class ReportWriter(Protocol):
    def write_report(self, reports: list[TreeReport], output_path: Path) -> None: ...

@runtime_checkable
class LabelStore(Protocol):
    def read_labels(self, path: Path) -> list[TreeLabel]: ...
    def write_labels(self, labels: list[TreeLabel], path: Path) -> None: ...
    def update_label(self, path: Path, index: int, label: int) -> None: ...

@runtime_checkable
class PageReader(Protocol):
    def page_count(self, path: Path) -> int: ...
    def render_page(self, path: Path, index: int, dpi: int = 100) -> NDArray[np.uint8]: ...
