from pathlib import Path

import numpy as np
import pymupdf
from numpy.typing import NDArray

from tree_classification.core.protocols import PageReader


class PymupdfPageReader(PageReader):
    """Rasterises PDF pages to RGB arrays. The document is opened per call, so no file handle outlives a call."""

    def page_count(self, path: Path) -> int:
        with pymupdf.open(path) as doc:
            return doc.page_count

    def render_page(self, path: Path, index: int, dpi: int = 100) -> NDArray[np.uint8]:
        """Returns page `index` as an (H, W, 3) uint8 RGB array."""
        with pymupdf.open(path) as doc:
            if not 0 <= index < doc.page_count:
                raise IndexError(f"{path}: page index {index} out of range (0..{doc.page_count - 1})")
            pixmap = doc[index].get_pixmap(dpi=dpi, alpha=False)
            return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3).copy()
