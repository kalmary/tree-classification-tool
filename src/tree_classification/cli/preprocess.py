import argparse
import logging
import sys
from pathlib import Path

import torch

from tree_classification.pointcloud.reader import LaspyReader
from tree_classification.pointcloud.writer import LaspyWriter
from tree_classification.rendering.depth_map import DepthMapRenderer
from tree_classification.reporting.csv_store import CsvLabelStore
from tree_classification.reporting.pdf_writer import PdfReportWriter
from tree_classification.services.preprocessing import (
    SUPPORTED_SUFFIXES,
    PreprocessingService,
    find_point_cloud_files,
)
from tree_classification.tracking.file_tracker import FileTracker

logger = logging.getLogger(__name__)


def argparser(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse and validate command-line arguments for the LiDAR tree preprocessing pipeline.
    Returns parsed arguments: input_path, output_dir, device
    """

    parser = argparse.ArgumentParser(
        description="Script for preprocessing .LAS/.LAZ files with segmented trees (tree_ids field). For every input file it creates:\n"
        "1. pcds/{file}_{tree_id}.laz|.las - point cloud of each tree, same format as the input\n"
        "2. {file}_trees_report.pdf - one page per tree: 5 depth map views + lat/lon/height\n"
        "3. {file}_trees_report.csv - one row per tree, label initialised to -2 (unclassified)\n"
        "Processed files are listed in processed.txt and skipped on the next run; failed ones go to error_files.txt and are retried.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--input_path',
        type=Path,
        required=True,
        help=(
            "Path to a single .las/.laz file or a directory of them (not searched recursively)."
        )
    )

    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help=(
            "Destination dir for all outputs, processed.txt and error_files.txt."
        )
    )

    parser.add_argument(
        '--device',
        choices=["cpu", "cuda"],
        default="cpu",
        help=(
            "Device used for depth map rendering."
        )
    )

    args = parser.parse_args(argv)

    if not args.input_path.exists():
        parser.error(f"input path does not exist: {args.input_path}")
    if args.input_path.is_file() and args.input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        parser.error(f"input file must be .las or .laz: {args.input_path}")
    # Checked up front: otherwise every file would fail separately and land in error_files.txt.
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is not available")

    return args


def main(argv: list[str] | None = None) -> int:
    """Returns 0 when every file was processed or skipped, 1 when any file failed."""
    args = argparser(argv)
    if args.input_path.is_dir():
        input_paths = find_point_cloud_files(args.input_path)
    else:
        input_paths = [args.input_path]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    service = PreprocessingService(
        reader=LaspyReader(),
        writer=LaspyWriter(),
        renderer=DepthMapRenderer(device=args.device),
        report_writer=PdfReportWriter(),
        label_store=CsvLabelStore(),
        tracker=FileTracker(args.output_dir),
        output_dir=args.output_dir,
    )
    summary = service.process_files(input_paths)

    logger.info(
        "Done: %d processed, %d skipped, %d failed",
        len(summary.processed), len(summary.skipped), len(summary.failed),
    )
    for path in summary.failed:
        logger.error("Failed: %s", path)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())