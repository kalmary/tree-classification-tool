import numpy as np

from tree_classification.core.models import TreeData, TreeReport
from tree_classification.reporting.pdf_writer import (
    VIEW_NAMES,
    PdfReportWriter,
    _view_to_image,
)


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
