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
 

