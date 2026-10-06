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
 
 
