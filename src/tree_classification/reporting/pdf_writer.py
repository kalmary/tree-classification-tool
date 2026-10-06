import os
import tempfile
from pathlib import Path

import numpy as np
from fpdf import FPDF
from numpy.typing import NDArray
from PIL import Image

from tree_classification.core.models import TreeData, TreeReport
from tree_classification.core.protocols import ReportWriter

# A4 landscape (mm): fits a screen better than portrait when the GUI displays the page.
PAGE_WIDTH = 297
PAGE_HEIGHT = 210
MARGIN = 12
TILE_SIZE = 79
TILE_GAP = 8
CAPTION_HEIGHT = 6
TITLE_HEIGHT = 10
ROW_GAP = 6

VIEW_NAMES = ("Top", "Front", "Back", "Left", "Right")

FONT = "Helvetica"  


def _view_to_image(view: NDArray[np.floating]) -> Image.Image:
    """Converts a [0, 1] depth map (H, W) or colour view (H, W, 3) to an 8-bit image."""
    if not np.issubdtype(view.dtype, np.floating):
        raise ValueError(f"view must be a float array, got dtype {view.dtype}")
    if not (view.ndim == 2 or (view.ndim == 3 and view.shape[2] == 3)):
        raise ValueError(f"view must have shape (H, W) or (H, W, 3), got {view.shape}")
    if not np.all(np.isfinite(view)) or view.min() < 0.0 or view.max() > 1.0:
        raise ValueError("view values must be finite and within [0, 1]")
    return Image.fromarray(np.round(view * 255).astype(np.uint8))


def _source_tree_id(tree: TreeData) -> str:
    return f"{tree.source_filename}_{tree.tree_id}"


def _metadata_lines(tree: TreeData) -> list[str]:
    return [
        f"Lat: {tree.latitude:.6f}",
        f"Lon: {tree.longitude:.6f}",
        f"Height: {tree.height:.2f} m",
    ]


def _tile_position(index: int) -> tuple[float, float]:
    row, column = divmod(index, 3)
    x = MARGIN + column * (TILE_SIZE + TILE_GAP)
    y = MARGIN + TITLE_HEIGHT + row * (TILE_SIZE + CAPTION_HEIGHT + ROW_GAP)
    return x, y


def _add_tree_page(pdf: FPDF, report: TreeReport) -> None:
    pdf.add_page()

    pdf.set_font(FONT, style="B", size=14)
    pdf.text(MARGIN, MARGIN + 6, _source_tree_id(report.tree))

    pdf.set_font(FONT, size=10)
    for index, (name, view) in enumerate(zip(VIEW_NAMES, report.views)):
        x, y = _tile_position(index)
        pdf.image(_view_to_image(view), x=x, y=y, w=TILE_SIZE, h=TILE_SIZE)
        pdf.text(x, y + TILE_SIZE + 4, name)

    x, y = _tile_position(len(VIEW_NAMES))
    pdf.set_font(FONT, size=12)
    for line_number, line in enumerate(_metadata_lines(report.tree)):
        pdf.text(x, y + 6 + line_number * 8, line)


class PdfReportWriter(ReportWriter):
    """Writes one PDF page per tree. Page order must match CSV row order."""

    def write_report(self, reports: list[TreeReport], output_path: Path) -> None:
        if not reports:
            raise ValueError("reports must not be empty")
        for report in reports:
            if len(report.views) != len(VIEW_NAMES):
                raise ValueError(
                    f"tree {report.tree.tree_id}: expected {len(VIEW_NAMES)} views, got {len(report.views)}"
                )

        pdf = FPDF(orientation="landscape", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=False)
        for report in reports:
            _add_tree_page(pdf, report)
        data = pdf.output()

        output_path = Path(output_path)
        fd, tmp_name = tempfile.mkstemp(dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_name, output_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise


def _make_report(tree_id: int, source_filename: str = "cloud_01", rgb: bool = False, n_views: int = 5) -> TreeReport:
    rng = np.random.default_rng(tree_id)
    shape = (64, 64, 3) if rgb else (64, 64)
    # Distinct random views: fpdf2 stores identical images once, which would hide missing tiles.
    views = [rng.random(shape, dtype=np.float32) for _ in range(n_views)]
    tree = TreeData(
        tree_id=tree_id,
        points=np.zeros((1, 3)),
        rgb=None,
        latitude=52.229701234,
        longitude=21.012228765,
        height=14.354,
        source_filename=source_filename,
    )
    return TreeReport(tree=tree, views=views)


def test_one_page_per_tree_in_order(tmp_path):
    import pymupdf

    path = tmp_path / "report.pdf"
    PdfReportWriter().write_report([_make_report(i) for i in (3, 1, 7)], path)

    with pymupdf.open(path) as doc:
        assert doc.page_count == 3
        titles = [page.get_text().splitlines()[0] for page in doc]
    assert titles == ["cloud_01_3", "cloud_01_1", "cloud_01_7"]


def test_page_contains_metadata_and_view_captions(tmp_path):
    import pymupdf

    path = tmp_path / "report.pdf"
    PdfReportWriter().write_report([_make_report(12)], path)

    with pymupdf.open(path) as doc:
        text = doc[0].get_text()
    for expected in ("cloud_01_12", "Lat: 52.229701", "Lon: 21.012229", "Height: 14.35 m", *VIEW_NAMES):
        assert expected in text


def test_each_page_has_five_images(tmp_path):
    import pymupdf

    path = tmp_path / "report.pdf"
    PdfReportWriter().write_report([_make_report(1), _make_report(2)], path)

    with pymupdf.open(path) as doc:
        assert [len(page.get_images()) for page in doc] == [5, 5]


def test_accepts_grayscale_and_rgb_views(tmp_path):
    path = tmp_path / "report.pdf"
    PdfReportWriter().write_report([_make_report(1, rgb=False), _make_report(2, rgb=True)], path)

    assert path.stat().st_size > 0


def test_view_to_image_modes():
    assert _view_to_image(np.zeros((4, 4), dtype=np.float32)).mode == "L"
    assert _view_to_image(np.ones((4, 4, 3), dtype=np.float32)).mode == "RGB"


def test_view_to_image_rejects_invalid_views():
    import pytest

    invalid = [
        np.full((4, 4), 1.5, dtype=np.float32),
        np.full((4, 4), -0.1, dtype=np.float32),
        np.full((4, 4), np.nan, dtype=np.float32),
        np.zeros((4, 4, 4), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        np.zeros((4, 4), dtype=np.uint8),
    ]
    for view in invalid:
        with pytest.raises(ValueError):
            _view_to_image(view)


def test_rejects_empty_list_and_wrong_view_count(tmp_path):
    import pytest

    path = tmp_path / "report.pdf"
    with pytest.raises(ValueError, match="empty"):
        PdfReportWriter().write_report([], path)
    with pytest.raises(ValueError, match="expected 5 views"):
        PdfReportWriter().write_report([_make_report(1, n_views=4)], path)
    assert not path.exists()


def test_non_latin1_filename_fails_loudly_without_output(tmp_path):
    import pytest
    from fpdf.errors import FPDFUnicodeEncodingException

    path = tmp_path / "report.pdf"
    with pytest.raises(FPDFUnicodeEncodingException):
        PdfReportWriter().write_report([_make_report(1, source_filename="łódź")], path)
    assert list(tmp_path.iterdir()) == []


def test_overwrites_and_leaves_no_temp_files(tmp_path):
    import pymupdf

    path = tmp_path / "report.pdf"
    PdfReportWriter().write_report([_make_report(1), _make_report(2)], path)
    PdfReportWriter().write_report([_make_report(3)], path)

    with pymupdf.open(path) as doc:
        assert doc.page_count == 1
    assert [p.name for p in tmp_path.iterdir()] == ["report.pdf"]