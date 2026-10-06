import os
import csv
import tempfile
from pathlib import Path
 
from tree_classification.core.models import TreeLabel
from tree_classification.core.protocols import LabelStore
 
COLUMNS = ("tree_id", "lat", "long", "height", "original_filename_tree_id", "label")
 
def _to_row(label: TreeLabel) -> list[str]:
    return [
        str(label.tree_id),
        repr(float(label.latitude)),
        repr(float(label.longitude)),
        repr(float(label.height)),
        label.source_tree_id,
        str(label.label),
    ]
 
def _from_row(row: list[str]) -> TreeLabel:
    tree_id, lat, long, height, source_tree_id, label = row
    return TreeLabel(
        tree_id=int(tree_id),
        latitude=float(lat),
        longitude=float(long),
        height=float(height),
        source_tree_id=source_tree_id,
        label=int(label),
    )
 
 
class CsvLabelStore(LabelStore):
    """Persists tree labels as CSV. Row order must match PDF page order."""
 
    def read_labels(self, path: Path) -> list[TreeLabel]:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != list(COLUMNS):
                raise ValueError(f"{path}: expected header {list(COLUMNS)}, got {header}")
 
            labels = []
            for row in reader:
                if len(row) != len(COLUMNS):
                    raise ValueError(f"{path}:{reader.line_num}: expected {len(COLUMNS)} fields, got {len(row)}")
                try:
                    labels.append(_from_row(row))
                except ValueError as err:
                    raise ValueError(f"{path}:{reader.line_num}: {err}") from err
            return labels
 
    def write_labels(self, labels: list[TreeLabel], path: Path) -> None:
        """Overwrites the file atomically, so an interrupted write never leaves a partial CSV."""
        path = Path(path)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(COLUMNS)
                writer.writerows(_to_row(label) for label in labels)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
 
    def update_label(self, path: Path, index: int, label: int) -> None:
        labels = self.read_labels(path)
        if not 0 <= index < len(labels):
            raise IndexError(f"{path}: row index {index} out of range (0..{len(labels) - 1})")
        labels[index].label = label
        self.write_labels(labels, path)
 

def _sample_labels() -> list[TreeLabel]:
    return [
        TreeLabel(1, 52.229701234567, 21.012228765432, 14.35, "test_01_1", -2),
        TreeLabel(2, 52.2298, 21.0123, 8.1, "test_2", -2),
        TreeLabel(3, 52.2299, 21.0124, 22.0, "test_01_3", 5),
    ]
 
 
def test_round_trip_preserves_values(tmp_path):
    store = CsvLabelStore()
    path = tmp_path / "labels.csv"
    labels = _sample_labels()
 
    store.write_labels(labels, path)
 
    assert store.read_labels(path) == labels
 
 
def test_header_matches_spec(tmp_path):
    path = tmp_path / "labels.csv"
    CsvLabelStore().write_labels(_sample_labels(), path)
 
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "tree_id,lat,long,height,original_filename_tree_id,label"
 
 
def test_numpy_floats_are_written_as_plain_numbers(tmp_path):
    import numpy as np
 
    path = tmp_path / "labels.csv"
    label = TreeLabel(1, np.float64(52.5), np.float64(21.5), np.float64(10.0), "a_1", -2)
    CsvLabelStore().write_labels([label], path)
 
    assert path.read_text(encoding="utf-8").splitlines()[1] == "1,52.5,21.5,10.0,a_1,-2"
 
 
def test_update_label_changes_only_target_row(tmp_path):
    store = CsvLabelStore()
    path = tmp_path / "labels.csv"
    store.write_labels(_sample_labels(), path)
    before = path.read_text(encoding="utf-8").splitlines()
 
    store.update_label(path, 1, 7)
 
    after = path.read_text(encoding="utf-8").splitlines()
    assert after[2] == before[2].rsplit(",", 1)[0] + ",7"
    assert [line for i, line in enumerate(after) if i != 2] == [line for i, line in enumerate(before) if i != 2]
 
 
def test_update_label_rejects_out_of_range_index(tmp_path):
    import pytest
 
    store = CsvLabelStore()
    path = tmp_path / "labels.csv"
    store.write_labels(_sample_labels(), path)
 
    for index in (-1, 3):
        with pytest.raises(IndexError):
            store.update_label(path, index, 1)
 
 
def test_read_rejects_wrong_header(tmp_path):
    import pytest
 
    path = tmp_path / "labels.csv"
    path.write_text("tree_id,lat,longt,height,original_filename_tree_id,label\n", encoding="utf-8")
 
    with pytest.raises(ValueError, match="header"):
        CsvLabelStore().read_labels(path)
 
 
def test_read_rejects_malformed_row(tmp_path):
    import pytest
 
    path = tmp_path / "labels.csv"
    path.write_text(
        "tree_id,lat,long,height,original_filename_tree_id,label\n"
        "1,52.0,21.0,10.0,a_1,-2\n"
        "2,52.0,21.0,10.0,a_2,sosna\n",
        encoding="utf-8",
    )
 
    with pytest.raises(ValueError, match=":3:"):
        CsvLabelStore().read_labels(path)
 
 
def test_read_accepts_utf8_bom(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "tree_id,lat,long,height,original_filename_tree_id,label\n1,52.0,21.0,10.0,a_1,-2\n",
        encoding="utf-8-sig",
    )
 
    assert CsvLabelStore().read_labels(path)[0].tree_id == 1
 
 
def test_write_overwrites_and_leaves_no_temp_files(tmp_path):
    store = CsvLabelStore()
    path = tmp_path / "labels.csv"
 
    store.write_labels(_sample_labels(), path)
    store.write_labels(_sample_labels()[:1], path)
 
    assert len(store.read_labels(path)) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["labels.csv"]
 