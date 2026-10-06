import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path

from tree_classification.core.models import TreeLabel, TreeReport
from tree_classification.core.protocols import (
    LabelStore,
    PointCloudReader,
    PointCloudWriter,
    Renderer,
    ReportWriter,
)
from tree_classification.pointcloud.extraction import extract_trees
from tree_classification.tracking.file_tracker import FileTracker

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".las", ".laz")
UNCLASSIFIED_LABEL = -2


class FileStatus(enum.Enum):
    PROCESSED = "processed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class ProcessingSummary:
    processed: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    failed: list[Path] = field(default_factory=list)


def find_point_cloud_files(directory: Path) -> list[Path]:
    """Lists .las/.laz files (any letter case) directly inside directory, sorted by name."""
    return sorted(
        path for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


class PreprocessingService:
    """Turns .las/.laz files into per-tree point clouds, a PDF report and a label CSV.

    Output layout for input `{stem}.laz`:
        {output_dir}/{stem}/pcds/{stem}_{tree_id}.laz
        {output_dir}/{stem}/{stem}_trees_report.pdf
        {output_dir}/{stem}/{stem}_trees_report.csv
    PDF page i and CSV row i describe the same tree.
    """

    def __init__(self, reader: PointCloudReader,
                 writer: PointCloudWriter,
                 renderer: Renderer,
                 report_writer: ReportWriter,
                 label_store: LabelStore,
                 tracker: FileTracker,
                 output_dir: Path):
        self.reader = reader
        self.writer = writer
        self.renderer = renderer
        self.report_writer = report_writer
        self.label_store = label_store
        self.tracker = tracker
        self.output_dir = Path(output_dir)

    def process_file(self, input_path: Path) -> FileStatus:
        """Processes one file. A failure is logged and recorded, never raised, so a batch can continue."""
        input_path = Path(input_path)
        if self.tracker.is_processed(input_path):
            logger.info("Skipping %s: already processed", input_path)
            return FileStatus.SKIPPED

        try:
            tree_count = self._write_outputs(input_path)
        except Exception:
            logger.exception("Failed to process %s", input_path)
            self.tracker.log_error(input_path)
            return FileStatus.FAILED

        self.tracker.mark_processed(input_path)
        logger.info("Processed %s: %d trees", input_path, tree_count)
        return FileStatus.PROCESSED

    def process_files(self, input_paths: list[Path]) -> ProcessingSummary:
        summary = ProcessingSummary()
        for path in input_paths:
            status = self.process_file(path)
            getattr(summary, status.value).append(Path(path))
        return summary

    def process_directory(self, input_dir: Path) -> ProcessingSummary:
        return self.process_files(find_point_cloud_files(input_dir))

    def _write_outputs(self, input_path: Path) -> int:
        stem = input_path.stem
        suffix = input_path.suffix.lower()

        trees = extract_trees(self.reader.read(input_path), stem)
        if not trees:
            raise ValueError(f"{input_path}: no trees found")

        # Render everything before writing anything, so a rendering error leaves no new outputs.
        reports = [TreeReport(tree=tree, views=self.renderer.render(tree.points, tree.rgb)) for tree in trees]
        labels = [
            TreeLabel(
                tree_id=tree.tree_id,
                latitude=tree.latitude,
                longitude=tree.longitude,
                height=tree.height,
                source_tree_id=f"{tree.source_filename}_{tree.tree_id}",
                label=UNCLASSIFIED_LABEL,
            )
            for tree in trees
        ]

        file_dir = self.output_dir / stem
        pcd_dir = file_dir / "pcds"
        pcd_dir.mkdir(parents=True, exist_ok=True)
        for tree, label in zip(trees, labels):
            self.writer.write(tree, pcd_dir / f"{label.source_tree_id}{suffix}", suffix.lstrip("."))

        # CSV last: its existence implies the matching PDF is complete.
        self.report_writer.write_report(reports, file_dir / f"{stem}_trees_report.pdf")
        self.label_store.write_labels(labels, file_dir / f"{stem}_trees_report.csv")
        return len(trees)


# --- Tests ---
def test_find_point_cloud_files_filters_and_sorts(tmp_path):
    for name in ("b.laz", "a.LAS", "c.LAZ", "notes.txt", "d.las.bak"):
        (tmp_path / name).touch()
    (tmp_path / "nested.laz").mkdir()

    assert [p.name for p in find_point_cloud_files(tmp_path)] == ["a.LAS", "b.laz", "c.LAZ"]


def test_process_files_groups_results_by_status(tmp_path):
    class StubService(PreprocessingService):
        def __init__(self):
            pass

        def process_file(self, input_path):
            return {"a": FileStatus.PROCESSED, "b": FileStatus.SKIPPED, "c": FileStatus.FAILED}[Path(input_path).stem]

    summary = StubService().process_files([tmp_path / "a.laz", tmp_path / "b.laz", tmp_path / "c.laz"])

    assert [p.stem for p in summary.processed] == ["a"]
    assert [p.stem for p in summary.skipped] == ["b"]
    assert [p.stem for p in summary.failed] == ["c"]