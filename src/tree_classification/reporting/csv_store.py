import csv
import os
import tempfile
from pathlib import Path
 
from tree_classification.core.models import TreeLabel
from tree_classification.core.protocols import LabelStore
 
COLUMNS = ("tree_id", "lat", "long", "height", "original_filename_tree_id", "label")
 
 