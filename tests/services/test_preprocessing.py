from pathlib import Path

from tree_classification.services.preprocessing import (
    FileStatus,
    PreprocessingService,
    find_point_cloud_files,
)


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
