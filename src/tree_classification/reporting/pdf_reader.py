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


# --- Tests ---
def _write_report(path: Path, tree_ids: list[int]) -> None:
    from tree_classification.core.models import TreeData, TreeReport
    from tree_classification.reporting.pdf_writer import PdfReportWriter

    rng = np.random.default_rng(0)
    reports = [
        TreeReport(
            tree=TreeData(tree_id, np.zeros((1, 3)), None, 52.0, 21.0, 10.0, "cloud"),
            views=[rng.random((32, 32), dtype=np.float32) for _ in range(5)],
        )
        for tree_id in tree_ids
    ]
    PdfReportWriter().write_report(reports, path)


def test_page_count_matches_written_pages(tmp_path):
    path = tmp_path / "report.pdf"
    _write_report(path, [1, 2, 3])

    assert PymupdfPageReader().page_count(path) == 3


def test_render_page_returns_landscape_rgb_image(tmp_path):
    path = tmp_path / "report.pdf"
    _write_report(path, [1])

    image = PymupdfPageReader().render_page(path, 0, dpi=50)

    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.shape[1] > image.shape[0]
    assert image.min() < 255


def test_render_page_scales_with_dpi(tmp_path):
    path = tmp_path / "report.pdf"
    _write_report(path, [1])
    reader = PymupdfPageReader()

    small, large = reader.render_page(path, 0, dpi=36), reader.render_page(path, 0, dpi=72)

    assert large.shape[1] == 2 * small.shape[1]


def test_render_page_rejects_out_of_range_index(tmp_path):
    import pytest

    path = tmp_path / "report.pdf"
    _write_report(path, [1, 2])

    for index in (-1, 2):
        with pytest.raises(IndexError):
            PymupdfPageReader().render_page(path, index)
