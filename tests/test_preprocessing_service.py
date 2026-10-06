from pathlib import Path

import laspy
import numpy as np
import pymupdf
import pytest

from tree_classification.cli.preprocess import main
from tree_classification.pointcloud.reader import LaspyReader
from tree_classification.pointcloud.writer import LaspyWriter
from tree_classification.rendering.depth_map import DepthMapRenderer
from tree_classification.reporting.csv_store import CsvLabelStore
from tree_classification.reporting.pdf_writer import PdfReportWriter
from tree_classification.services.preprocessing import FileStatus, PreprocessingService
from tree_classification.tracking.file_tracker import FileTracker


def write_point_cloud(path: Path, tree_ids: list[int], points_per_tree: int = 50, rgb: bool = True) -> None:
    rng = np.random.default_rng(len(tree_ids))
    header = laspy.LasHeader(point_format=3 if rgb else 0, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="tree_ids", type=np.int32))
    las = laspy.LasData(header)

    centers = np.array([[i * 20.0, i * 10.0, 0.0] for i in range(len(tree_ids))])
    xyz = np.concatenate([c + rng.uniform([-2, -2, 0], [2, 2, 15], size=(points_per_tree, 3)) for c in centers])
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.tree_ids = np.repeat(tree_ids, points_per_tree)
    if rgb:
        colours = rng.integers(0, 65536, size=(len(xyz), 3))
        las.red, las.green, las.blue = colours[:, 0], colours[:, 1], colours[:, 2]
    las.write(path)


def make_service(output_dir: Path, renderer=None) -> PreprocessingService:
    return PreprocessingService(
        reader=LaspyReader(),
        writer=LaspyWriter(),
        renderer=renderer or DepthMapRenderer(device="cpu", resolution=32),
        report_writer=PdfReportWriter(),
        label_store=CsvLabelStore(),
        tracker=FileTracker(output_dir),
        output_dir=output_dir,
    )


@pytest.fixture
def dirs(tmp_path):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    input_dir.mkdir()
    return input_dir, output_dir


def test_process_file_writes_pcds_pdf_and_csv(dirs):
    input_dir, output_dir = dirs
    source = input_dir / "plot_a.laz"
    write_point_cloud(source, tree_ids=[5, 2, 9])

    assert make_service(output_dir).process_file(source) is FileStatus.PROCESSED

    file_dir = output_dir / "plot_a"
    assert sorted(p.name for p in (file_dir / "pcds").iterdir()) == ["plot_a_2.laz", "plot_a_5.laz", "plot_a_9.laz"]
    labels = CsvLabelStore().read_labels(file_dir / "plot_a_trees_report.csv")
    assert [label.tree_id for label in labels] == [2, 5, 9]
    assert {label.label for label in labels} == {-2}
    with pymupdf.open(file_dir / "plot_a_trees_report.pdf") as doc:
        page_titles = [page.get_text().splitlines()[0] for page in doc]
    assert page_titles == [label.source_tree_id for label in labels]

    tree_cloud = laspy.read(file_dir / "pcds" / "plot_a_5.laz")
    assert len(tree_cloud.points) == 50
    assert tree_cloud.header.point_format.id == 3


def test_las_input_produces_uncompressed_las_pcds(dirs):
    input_dir, output_dir = dirs
    source = input_dir / "plot_b.las"
    write_point_cloud(source, tree_ids=[1], rgb=False)

    make_service(output_dir).process_file(source)

    pcd = output_dir / "plot_b" / "pcds" / "plot_b_1.las"
    with laspy.open(pcd) as reader:
        assert not reader.header.are_points_compressed


def test_processed_file_is_skipped_on_rerun(dirs):
    input_dir, output_dir = dirs
    source = input_dir / "plot_a.laz"
    write_point_cloud(source, tree_ids=[1, 2])

    assert make_service(output_dir).process_file(source) is FileStatus.PROCESSED
    assert make_service(output_dir).process_file(source) is FileStatus.SKIPPED


def test_corrupt_file_is_logged_and_batch_continues(dirs):
    input_dir, output_dir = dirs
    (input_dir / "broken.laz").write_bytes(b"not a point cloud")
    write_point_cloud(input_dir / "good.laz", tree_ids=[1])

    summary = make_service(output_dir).process_directory(input_dir)

    assert [p.name for p in summary.failed] == ["broken.laz"]
    assert [p.name for p in summary.processed] == ["good.laz"]
    assert FileTracker(output_dir).get_errors() == [(input_dir / "broken.laz").resolve()]
    assert not (output_dir / "broken").exists()


def test_failed_file_is_retried_on_next_run(dirs):
    input_dir, output_dir = dirs
    source = input_dir / "plot_a.laz"
    source.write_bytes(b"truncated")
    assert make_service(output_dir).process_file(source) is FileStatus.FAILED

    write_point_cloud(source, tree_ids=[1])

    assert make_service(output_dir).process_file(source) is FileStatus.PROCESSED


def test_rendering_failure_leaves_no_outputs(dirs):
    class FailingRenderer:
        def render(self, points, rgb=None, resolution=None):
            raise RuntimeError("renderer exploded")

    input_dir, output_dir = dirs
    source = input_dir / "plot_a.laz"
    write_point_cloud(source, tree_ids=[1, 2])

    assert make_service(output_dir, renderer=FailingRenderer()).process_file(source) is FileStatus.FAILED
    assert not (output_dir / "plot_a").exists()
    assert not FileTracker(output_dir).is_processed(source)


def test_cli_returns_zero_on_success(dirs):
    input_dir, output_dir = dirs
    write_point_cloud(input_dir / "plot_a.laz", tree_ids=[1, 2])

    assert main(["--input_path", str(input_dir / "plot_a.laz"), "--output_dir", str(output_dir)]) == 0
    assert (output_dir / "plot_a" / "plot_a_trees_report.csv").exists()


def test_cli_returns_one_when_any_file_fails(dirs):
    input_dir, output_dir = dirs
    write_point_cloud(input_dir / "good.laz", tree_ids=[1])
    (input_dir / "broken.laz").write_bytes(b"garbage")

    assert main(["--input_path", str(input_dir), "--output_dir", str(output_dir)]) == 1
    assert (output_dir / "good" / "good_trees_report.pdf").exists()


@pytest.mark.parametrize("make_input", [
    lambda d: d / "missing.laz",
    lambda d: (d / "notes.txt").write_text("x") and d / "notes.txt",
])
def test_cli_rejects_invalid_input_path(dirs, make_input):
    input_dir, output_dir = dirs

    with pytest.raises(SystemExit) as exc:
        main(["--input_path", str(make_input(input_dir)), "--output_dir", str(output_dir)])
    assert exc.value.code == 2


def test_cli_requires_input_and_output_paths(dirs):
    input_dir, _ = dirs

    with pytest.raises(SystemExit) as exc:
        main(["--input_path", str(input_dir)])
    assert exc.value.code == 2


def test_cli_rejects_cuda_when_unavailable(dirs, monkeypatch):
    import torch

    input_dir, output_dir = dirs
    write_point_cloud(input_dir / "plot_a.laz", tree_ids=[1])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit) as exc:
        main(["--input_path", str(input_dir), "--output_dir", str(output_dir), "--device", "cuda"])
    assert exc.value.code == 2
    assert not output_dir.exists()