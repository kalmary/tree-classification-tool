from pathlib import Path

import numpy as np

from tree_classification.core.models import TreeLabel
from tree_classification.services.classification import (
    REPORT_SUFFIX,
    ClassificationService,
    LabelMatch,
    LabelName,
    ReportPair,
    find_report_pairs,
    load_label_names,
)

LABELS_JSON = Path(__file__).resolve().parents[2] / "labels.json"


class _StubPageReader:
    def __init__(self, pages: int):
        self.pages = pages

    def page_count(self, path):
        return self.pages

    def render_page(self, path, index, dpi=100):
        return np.full((2, 3, 3), index, dtype=np.uint8)


def _make_service(tmp_path, labels: list[int], pages: int | None = None) -> ClassificationService:
    from tree_classification.reporting.csv_store import CsvLabelStore

    pair = ReportPair(tmp_path / f"a{REPORT_SUFFIX}.pdf", tmp_path / f"a{REPORT_SUFFIX}.csv")
    rows = [TreeLabel(i, 52.0, 21.0, 10.0, f"a_{i}", label) for i, label in enumerate(labels)]
    CsvLabelStore().write_labels(rows, pair.csv_path)
    return ClassificationService(
        _StubPageReader(len(labels) if pages is None else pages),
        CsvLabelStore(),
        pair,
        load_label_names(LABELS_JSON),
    )


def _csv_labels(service: ClassificationService) -> list[int]:
    return [tree.label for tree in service.label_store.read_labels(service.pair.csv_path)]


def test_load_label_names_reads_repo_labels():
    names = load_label_names(LABELS_JSON)

    assert names[-2] == LabelName("Unclassified", "Nieklasyfikowane")
    assert names[1] == LabelName("Picea", "świerk")


def test_starts_at_first_unclassified(tmp_path):
    service = _make_service(tmp_path, [3, 5, -2, -2])

    assert service.current_index == 2
    assert not service.is_complete


def test_starts_at_zero_when_all_classified(tmp_path):
    service = _make_service(tmp_path, [3, 5])

    assert service.current_index == 0
    assert service.is_complete


def test_rejects_page_and_row_count_mismatch(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="pages"):
        _make_service(tmp_path, [-2, -2], pages=3)


def test_resolve_label_accepts_code_names_and_option_text(tmp_path):
    service = _make_service(tmp_path, [-2])

    assert service.resolve_label(" 1 ") == 1
    assert service.resolve_label("01") == 1
    assert service.resolve_label("picea") == 1
    assert service.resolve_label("Świerk") == 1
    assert service.resolve_label("DĄB") == 5
    assert service.resolve_label("incorrect segmentation") == 15
    assert service.resolve_label(service.label_options()[2]) == 1


def test_resolve_label_rejects_unknown_input(tmp_path):
    import pytest

    service = _make_service(tmp_path, [-2])

    for text in ("51", "sosnaa", "", "1.5"):
        with pytest.raises(ValueError):
            service.resolve_label(text)


def test_match_label_by_code(tmp_path):
    service = _make_service(tmp_path, [-2])

    assert service.match_label("1") == LabelMatch(selected=1, candidates=[1, 10, 11, 12, 13, 14, 15, 16, 17])
    assert service.match_label(" 12 ").selected == 12
    assert service.match_label("01").selected == 1
    assert service.match_label("-2").selected == -2
    assert service.match_label("99") == LabelMatch(selected=None, candidates=[])


def test_match_label_by_name_ignores_case_and_polish_characters(tmp_path):
    service = _make_service(tmp_path, [-2])

    assert service.match_label("dab").selected == 5
    assert service.match_label("QUERC").selected == 5
    assert service.match_label("glog").selected == 14
    assert service.match_label("bledna").selected == 15
    assert service.match_label("swierk").selected == 1


def test_match_label_ambiguous_or_unknown_selects_nothing(tmp_path):
    service = _make_service(tmp_path, [-2])

    ambiguous = service.match_label("ab")
    assert ambiguous.selected is None and sorted(ambiguous.candidates) == [2, 5, 9]  # Abies, dąb, grab
    assert service.match_label("palma") == LabelMatch(selected=None, candidates=[])
    assert service.match_label("") == LabelMatch(selected=None, candidates=[])


def test_classified_count(tmp_path):
    service = _make_service(tmp_path, [3, -2, -2])
    assert service.classified_count() == 1

    service.next_page(4)
    assert service.classified_count() == 2


def test_save_label_rejects_unknown_code(tmp_path):
    import pytest

    service = _make_service(tmp_path, [-2])

    with pytest.raises(ValueError):
        service.save_label(51)
    assert _csv_labels(service) == [-2]


def test_next_page_saves_current_row_and_advances(tmp_path):
    service = _make_service(tmp_path, [-2, -2, -2])

    assert service.next_page(4) is True

    assert service.current_index == 1
    assert _csv_labels(service) == [4, -2, -2]
    assert service.first_unclassified_index() == 1


def test_next_page_on_last_page_saves_and_stays(tmp_path):
    service = _make_service(tmp_path, [1, -2])

    assert service.next_page(2) is False

    assert service.current_index == 1
    assert _csv_labels(service) == [1, 2]
    assert service.is_complete


def test_previous_page_moves_back_without_saving(tmp_path):
    service = _make_service(tmp_path, [1, -2])
    before = service.pair.csv_path.read_bytes()

    assert service.previous_page() is True
    assert service.current_index == 0
    assert service.previous_page() is False
    assert service.pair.csv_path.read_bytes() == before


def test_current_page_image_uses_current_index(tmp_path):
    service = _make_service(tmp_path, [1, -2])

    assert service.current_page_image()[0, 0, 0] == 1


def test_current_map_url_follows_current_tree(tmp_path):
    from tree_classification.reporting.csv_store import CsvLabelStore

    pair = ReportPair(tmp_path / f"a{REPORT_SUFFIX}.pdf", tmp_path / f"a{REPORT_SUFFIX}.csv")
    rows = [TreeLabel(0, 52.0, 21.0, 10.0, "a_0", -2), TreeLabel(1, 54.0, 23.0, 10.0, "a_1", -2)]
    CsvLabelStore().write_labels(rows, pair.csv_path)
    service = ClassificationService(_StubPageReader(2), CsvLabelStore(), pair, load_label_names(LABELS_JSON))

    assert service.current_map_url == "https://www.google.com/maps/search/?api=1&query=52.000000,21.000000"
    service.next_page(1)
    assert service.current_map_url == "https://www.google.com/maps/search/?api=1&query=54.000000,23.000000"
    assert service.current_bdl_url == "https://www.bdl.lasy.gov.pl/portal/mapy?location=23.000000,54.000000"


def test_find_report_pairs_in_directory_searches_by_csv(tmp_path):
    for stem in ("b", "a"):
        (tmp_path / stem).mkdir()
        (tmp_path / stem / f"{stem}{REPORT_SUFFIX}.pdf").touch()
        (tmp_path / stem / f"{stem}{REPORT_SUFFIX}.csv").touch()
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / f"c{REPORT_SUFFIX}.pdf").touch()

    pairs = find_report_pairs(tmp_path)

    assert [pair.csv_path.name for pair in pairs] == [f"a{REPORT_SUFFIX}.csv", f"b{REPORT_SUFFIX}.csv"]
    assert all(pair.pdf_path == pair.csv_path.with_suffix(".pdf") for pair in pairs)


def test_find_report_pairs_for_single_file(tmp_path):
    pdf_path, csv_path = tmp_path / "r.pdf", tmp_path / "r.csv"
    pdf_path.touch()
    csv_path.touch()

    assert find_report_pairs(pdf_path) == [ReportPair(pdf_path, csv_path)]
    assert find_report_pairs(csv_path) == [ReportPair(pdf_path, csv_path)]


def test_find_report_pairs_rejects_missing_partner(tmp_path):
    import pytest

    (tmp_path / "r.pdf").touch()
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / f"x{REPORT_SUFFIX}.csv").touch()

    with pytest.raises(ValueError, match="r.csv"):
        find_report_pairs(tmp_path / "r.pdf")
    with pytest.raises(ValueError, match="pdf"):
        find_report_pairs(tmp_path / "x")
