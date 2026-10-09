import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tree_classification.core.models import TreeLabel
from tree_classification.core.protocols import LabelStore, PageReader
from tree_classification.services.preprocessing import UNCLASSIFIED_LABEL

REPORT_SUFFIX = "_trees_report"
GOOGLE_MAPS_URL = "https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"
# Same format as the portal's "Udostępnij lokalizację" link; note longitude comes first.
BDL_MAP_URL = "https://www.bdl.lasy.gov.pl/portal/mapy?location={lon:.6f},{lat:.6f}"


@dataclass(frozen=True)
class LabelName:
    latin: str
    polish: str


@dataclass(frozen=True)
class LabelMatch:
    """`selected`: the code the typed text unambiguously means, if any. `candidates`: codes it may still mean."""
    selected: int | None
    candidates: list[int]


def _fold(text: str) -> str:
    """Lower case without Polish diacritics; "ł" has no Unicode decomposition, so it is mapped by hand."""
    text = text.strip().casefold().replace("ł", "l")
    return "".join(char for char in unicodedata.normalize("NFD", text) if not unicodedata.combining(char))


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
    def current_bdl_url(self) -> str:
        tree = self.current_tree
        return BDL_MAP_URL.format(lat=tree.latitude, lon=tree.longitude)

    @property
    def is_complete(self) -> bool:
        return self.first_unclassified_index() is None

    def classified_count(self) -> int:
        return sum(tree.label != UNCLASSIFIED_LABEL for tree in self._trees)

    def match_label(self, text: str) -> LabelMatch:
        """Interprets text typed in the label field while it is being typed.

        Digits match a code exactly ("1" is code 1 even when 10-17 exist) and list codes starting
        with them as candidates. Other text matches part of a Latin or Polish name, ignoring case
        and Polish diacritics ("dab" finds "dąb"); a single match, or a single exact name, is selected.
        """
        query = _fold(text)
        if not query:
            return LabelMatch(selected=None, candidates=[])
        if re.fullmatch(r"-?\d+", query):
            query = str(int(query))
            candidates = [code for code in self.label_names if str(code).startswith(query)]
            return LabelMatch(selected=int(query) if int(query) in self.label_names else None, candidates=candidates)

        candidates = [
            code for code, name in self.label_names.items()
            if query in _fold(name.latin) or query in _fold(name.polish)
        ]
        exact = [code for code in candidates if query in (_fold(self.label_names[code].latin),
                                                          _fold(self.label_names[code].polish))]
        if len(exact) == 1:
            selected = exact[0]
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            selected = None
        return LabelMatch(selected=selected, candidates=candidates)

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
