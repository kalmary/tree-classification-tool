import argparse
import logging
import sys
from pathlib import Path

from tree_classification.reporting.csv_store import CsvLabelStore
from tree_classification.reporting.pdf_reader import PymupdfPageReader
from tree_classification.services.classification import (
    ClassificationService,
    ReportPair,
    find_report_pairs,
    load_label_names,
)

logger = logging.getLogger(__name__)

DEFAULT_LABELS = Path(__file__).resolve().parents[3] / "labels.json"


def argparser(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[ReportPair]]:
    """
    Parse and validate command-line arguments for the manual classification GUI.
    Returns parsed arguments (input_path, labels) and the PDF+CSV report pairs found.
    """

    parser = argparse.ArgumentParser(
        description="GUI for labelling trees from {file}_trees_report.pdf + .csv pairs created by preprocessing.\n"
        "Each PDF page is labelled with a species button; Next saves the label to the CSV.\n"
        "Files with all trees labelled are skipped; the rest resume at the first unlabelled (-2) tree.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--input_path',
        type=Path,
        required=True,
        help=(
            "Directory searched recursively for report pairs, or a single report .pdf/.csv file."
        )
    )

    parser.add_argument(
        '--labels',
        type=Path,
        default=DEFAULT_LABELS,
        help=(
            "JSON mapping of label codes to latin/polish names."
        )
    )

    args = parser.parse_args(argv)

    if not args.input_path.exists():
        parser.error(f"input path does not exist: {args.input_path}")
    if not args.labels.is_file():
        parser.error(f"labels file does not exist: {args.labels}")
    try:
        pairs = find_report_pairs(args.input_path)
    except ValueError as err:
        parser.error(str(err))

    return args, pairs


def main(argv: list[str] | None = None) -> int:
    args, pairs = argparser(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    label_names = load_label_names(args.labels)
    services = [
        ClassificationService(PymupdfPageReader(), CsvLabelStore(), pair, label_names)
        for pair in pairs
    ]
    pending = [service for service in services if not service.is_complete]
    logger.info("Found %d report(s), %d with unlabelled trees", len(services), len(pending))
    if not pending:
        logger.info("Nothing to classify")
        return 0

    # Imported late so the checks above work without a display.
    from tree_classification.gui.app import run_app

    return run_app(pending)


if __name__ == "__main__":
    sys.exit(main())
