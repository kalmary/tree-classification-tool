from pathlib import Path

import numpy as np
import pytest

from test_preprocessing_service import make_service as make_preprocessing_service
from test_preprocessing_service import write_point_cloud
from tree_classification.cli.classify import main
from tree_classification.reporting.csv_store import CsvLabelStore
from tree_classification.reporting.pdf_reader import PymupdfPageReader
from tree_classification.services.classification import (
    ClassificationService,
    find_report_pairs,
    load_label_names,
)

LABELS_JSON = Path(__file__).resolve().parents[1] / "labels.json"


def make_service(pair) -> ClassificationService:
    return ClassificationService(PymupdfPageReader(), CsvLabelStore(), pair, load_label_names(LABELS_JSON))


def test_classify_preprocessed_report_and_resume(tmp_path):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    input_dir.mkdir()
    write_point_cloud(input_dir / "plot_a.laz", tree_ids=[5, 2, 9])
    make_preprocessing_service(output_dir).process_directory(input_dir)

    [pair] = find_report_pairs(output_dir)
    service = make_service(pair)
    assert (service.current_index, service.total_pages) == (0, 3)
    assert service.current_tree.source_tree_id == "plot_a_2"

    image = service.current_page_image(dpi=50)
    assert image.dtype == np.uint8 and image.shape[2] == 3

    assert service.next_page(service.resolve_label("sosna")) is True

    resumed = make_service(pair)
    assert resumed.current_index == 1
    assert resumed.current_tree.source_tree_id == "plot_a_5"
    assert [tree.label for tree in CsvLabelStore().read_labels(pair.csv_path)] == [0, -2, -2]


def preprocess(tmp_path, files: dict[str, list[int]]) -> Path:
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    input_dir.mkdir()
    for stem, tree_ids in files.items():
        write_point_cloud(input_dir / f"{stem}.laz", tree_ids=tree_ids)
    make_preprocessing_service(output_dir).process_directory(input_dir)
    return output_dir


def test_cli_rejects_missing_input_path(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--input_path", str(tmp_path / "missing")])
    assert exc.value.code == 2


def test_cli_skips_gui_when_everything_is_classified(tmp_path, monkeypatch):
    import tree_classification.gui.app as app

    output_dir = preprocess(tmp_path, {"plot_a": [1, 2]})
    service = make_service(find_report_pairs(output_dir)[0])
    while service.next_page(3):
        pass
    monkeypatch.setattr(app, "run_app", lambda services: pytest.fail("GUI must not open"))

    assert main(["--input_path", str(output_dir)]) == 0


def test_cli_opens_gui_with_unclassified_reports_only(tmp_path, monkeypatch):
    import tree_classification.gui.app as app

    output_dir = preprocess(tmp_path, {"plot_a": [1], "plot_b": [1, 2]})
    make_service(find_report_pairs(output_dir)[0]).next_page(3)
    opened = []
    monkeypatch.setattr(app, "run_app", lambda services: opened.extend(services) or 0)

    assert main(["--input_path", str(output_dir)]) == 0
    assert [service.pair.csv_path.name for service in opened] == ["plot_b_trees_report.csv"]
