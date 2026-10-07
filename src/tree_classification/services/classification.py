import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tree_classification.core.models import TreeLabel
from tree_classification.core.protocols import LabelStore, PageReader
from tree_classification.services.preprocessing import UNCLASSIFIED_LABEL

REPORT_SUFFIX = "_trees_report"
GOOGLE_MAPS_URL = "https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"


@dataclass(frozen=True)
class LabelName:
    latin: str
    polish: str


@dataclass(frozen=True)
class ReportPair:
    pdf_path: Path
    csv_path: Path


def load_label_names(path: Path) -> dict[int, LabelName]:
    """Reads labels.json: {"<int code>": {"latin": ..., "polish": ...}}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(code): LabelName(latin=names["latin"], polish=names["polish"]) for code, names in data.items()}


def format_label(value: int, name: LabelName) -> str:
    """Single text form of a label, used for dropdown entries and accepted back by `resolve_label`."""
    return f"{value} - {name.latin} / {name.polish}"


def _pair_for(path: Path) -> ReportPair:
    pair = ReportPair(pdf_path=path.with_suffix(".pdf"), csv_path=path.with_suffix(".csv"))
    for required in (pair.pdf_path, pair.csv_path):
        if not required.is_file():
            raise ValueError(f"{path}: matching file {required.name} not found")
    return pair


def find_report_pairs(path: Path) -> list[ReportPair]:
    """Finds PDF+CSV report pairs for a single .pdf/.csv file, or recursively in a directory.

    A directory is searched by CSV, because preprocessing writes the CSV last:
    a PDF without a CSV is an interrupted run, not a report to classify.
    """
    path = Path(path)
    if path.is_dir():
        return [_pair_for(csv_path) for csv_path in sorted(path.rglob(f"*{REPORT_SUFFIX}.csv"))]
    if path.suffix.lower() not in (".pdf", ".csv"):
        raise ValueError(f"{path}: expected a directory, .pdf or .csv file")
    return [_pair_for(path)]


class ClassificationService:
    """Navigation and label persistence for one report pair. PDF page i is CSV row i.

    Labels are saved only when passed to `save_label`/`next_page`; moving back saves nothing.
    """

    def __init__(self, page_reader: PageReader,
                 label_store: LabelStore,
                 pair: ReportPair,
                 label_names: dict[int, LabelName]):
        self.page_reader = page_reader
        self.label_store = label_store
        self.pair = pair
        self.label_names = label_names
        self._trees = label_store.read_labels(pair.csv_path)

        page_count = page_reader.page_count(pair.pdf_path)
        if page_count != len(self._trees):
            raise ValueError(
                f"{pair.pdf_path} has {page_count} pages but {pair.csv_path} has {len(self._trees)} rows"
            )

        self._lookup: dict[str, int] = {}
        for value, name in label_names.items():
            for text in (str(value), name.latin, name.polish, format_label(value, name)):
                key = text.strip().casefold()
                if self._lookup.get(key, value) != value:
                    raise ValueError(f"label text {text!r} is ambiguous")
                self._lookup[key] = value

        first_unclassified = self.first_unclassified_index()
        self._index = 0 if first_unclassified is None else first_unclassified

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def total_pages(self) -> int:
        return len(self._trees)

    @property
    def current_tree(self) -> TreeLabel:
        return self._trees[self._index]

    @property
    def current_map_url(self) -> str:
        tree = self.current_tree
        return GOOGLE_MAPS_URL.format(lat=tree.latitude, lon=tree.longitude)

    @property
    def is_complete(self) -> bool:
        return self.first_unclassified_index() is None

    def first_unclassified_index(self) -> int | None:
        return next((i for i, tree in enumerate(self._trees) if tree.label == UNCLASSIFIED_LABEL), None)

    def current_page_image(self, dpi: int = 100) -> NDArray[np.uint8]:
        return self.page_reader.render_page(self.pair.pdf_path, self._index, dpi)

    def label_options(self) -> list[str]:
        return [format_label(value, name) for value, name in self.label_names.items()]

    def resolve_label(self, text: str) -> int:
        """Maps user input (code, Latin name, Polish name or dropdown text; any letter case) to a label code."""
        key = text.strip().casefold()
        try:
            key = str(int(key))
        except ValueError:
            pass
        if key not in self._lookup:
            raise ValueError(f"unknown label {text!r}")
        return self._lookup[key]

    def save_label(self, label: int) -> None:
        if label not in self.label_names:
            raise ValueError(f"unknown label code {label}")
        self.label_store.update_label(self.pair.csv_path, self._index, label)
        self._trees[self._index].label = label

    def next_page(self, label: int) -> bool:
        """Saves the label, then advances. Returns False (staying put) on the last page."""
        self.save_label(label)
        if self._index + 1 >= len(self._trees):
            return False
        self._index += 1
        return True

    def previous_page(self) -> bool:
        """Goes back without saving. Returns False on the first page."""
        if self._index == 0:
            return False
        self._index -= 1
        return True


# --- Tests ---
LABELS_JSON = Path(__file__).resolve().parents[3] / "labels.json"


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
