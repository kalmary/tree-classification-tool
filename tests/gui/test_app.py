from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
)

from tree_classification.gui.app import (
    ClassificationWindow,
)
from tree_classification.services.classification import ClassificationService


def _qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _make_window(tmp_path, *files: list[int]) -> ClassificationWindow:
    from pathlib import Path

    import numpy as np

    from tree_classification.core.models import TreeLabel
    from tree_classification.reporting.csv_store import CsvLabelStore
    from tree_classification.services.classification import ReportPair, load_label_names

    class StubPageReader:
        def __init__(self, pages):
            self.pages = pages

        def page_count(self, path):
            return self.pages

        def render_page(self, path, index, dpi=100):
            return np.zeros((20, 30, 3), dtype=np.uint8)

    _qapp()
    label_names = load_label_names(Path(__file__).resolve().parents[2] / "labels.json")
    services = []
    for file_number, labels in enumerate(files):
        pair = ReportPair(tmp_path / f"f{file_number}.pdf", tmp_path / f"f{file_number}.csv")
        rows = [TreeLabel(i, 52.0 + file_number, 21.0 + i, 10.0, f"f{file_number}_{i}", label) for i, label in enumerate(labels)]
        CsvLabelStore().write_labels(rows, pair.csv_path)
        services.append(ClassificationService(StubPageReader(len(labels)), CsvLabelStore(), pair, label_names))
    return ClassificationWindow(services)


def _csv_labels(window: ClassificationWindow, file_index: int = 0) -> list[int]:
    service = window.services[file_index]
    return [tree.label for tree in service.label_store.read_labels(service.pair.csv_path)]


def test_next_requires_selection_then_saves_and_advances(tmp_path):
    window = _make_window(tmp_path, [-2, -2])
    assert not window.next_button.isEnabled()
    assert not window.previous_button.isEnabled()

    window.label_buttons.button(5).click()
    assert window.next_button.isEnabled()
    window.next_button.click()

    assert _csv_labels(window) == [5, -2]
    assert window.service.current_index == 1
    assert window.label_buttons.selected() is None
    assert window.name_label.text() == "f0_1"
    assert window.position_label.text() == "Drzewo 2 / 2 · plik 1 / 1"


def test_progress_counts_classified_trees(tmp_path):
    window = _make_window(tmp_path, [3, -2, -2])
    assert (window.progress.value(), window.progress.maximum()) == (1, 3)

    window.label_buttons.button(4).click()
    window.next_button.click()

    assert window.progress.value() == 2


def test_map_link_points_to_current_tree(tmp_path):
    window = _make_window(tmp_path, [-2, -2], [-2])
    assert "query=52.000000,21.000000" in window.map_link.text()

    window.label_buttons.button(1).click()
    window.next_button.click()
    assert "query=52.000000,22.000000" in window.map_link.text()
    assert "location=22.000000,52.000000" in window.map_link.text()

    window.label_buttons.button(1).click()
    window.next_button.click()
    assert "query=53.000000,21.000000" in window.map_link.text()


def test_previous_discards_selection_and_shows_saved_label(tmp_path):
    window = _make_window(tmp_path, [3, -2])
    assert window.service.current_index == 1

    window.label_buttons.button(7).click()
    window.previous_button.click()

    assert _csv_labels(window) == [3, -2]
    assert window.label_buttons.selected() == 3
    assert window.label_buttons.is_marked(3, "saved")
    assert "(zapisana)" in window.feedback.text()


def test_next_on_last_page_opens_next_file(tmp_path):
    window = _make_window(tmp_path, [-2], [-2, -2])

    window.label_buttons.button(1).click()
    window.next_button.click()

    assert _csv_labels(window, 0) == [1]
    assert window.service is window.services[1]
    assert window.service.current_index == 0


def test_next_on_last_page_of_last_file_reports_done_and_closes(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: shown.append(args[2]))
    window = _make_window(tmp_path, [-2])
    window.show()

    window.label_buttons.button(-2).click()
    window.next_button.click()

    assert shown == ["Wszystkie drzewa sklasyfikowane"]
    assert not window.isVisible()


def test_save_and_quit_saves_selection_and_closes(tmp_path):
    window = _make_window(tmp_path, [-2, -2])
    window.show()

    window.label_buttons.button(12).click()
    window.save_quit_button.click()

    assert _csv_labels(window) == [12, -2]
    assert not window.isVisible()


def test_save_and_quit_without_selection_writes_nothing(tmp_path):
    window = _make_window(tmp_path, [-2])
    before = window.service.pair.csv_path.read_bytes()
    window.show()

    window.save_quit_button.click()

    assert window.service.pair.csv_path.read_bytes() == before
    assert not window.isVisible()


def test_clicking_selected_label_twice_saves_and_advances(tmp_path):
    window = _make_window(tmp_path, [-2, -2])

    window.label_buttons.button(4).click()
    window.label_buttons.button(4).click()

    assert _csv_labels(window) == [4, -2]
    assert window.service.current_index == 1


def test_typing_code_selects_label_and_outlines_candidates(tmp_path):
    window = _make_window(tmp_path, [-2])

    window.class_input.setText("1")
    assert window.label_buttons.selected() == 1
    assert window.label_buttons.is_marked(12, "candidate")
    assert not window.label_buttons.is_marked(5, "candidate")

    window.class_input.setText("12")
    assert window.label_buttons.selected() == 12
    assert window.next_button.isEnabled()

    window.class_input.setText("99")
    assert window.label_buttons.selected() is None
    assert not window.next_button.isEnabled()
    assert window.feedback.text() == "Brak etykiety „99”"


def test_typing_name_selects_only_unambiguous_label(tmp_path):
    window = _make_window(tmp_path, [-2])

    window.class_input.setText("dab")
    assert window.label_buttons.selected() == 5
    assert "niezapisana" in window.feedback.text()

    window.class_input.setText("ab")
    assert window.label_buttons.selected() is None
    assert window.feedback.text().startswith("Pasuje 3 etykiet")


def test_enter_advances_only_with_selection_and_clears_input(tmp_path):
    from PySide6.QtTest import QTest

    window = _make_window(tmp_path, [-2, -2])

    QTest.keyClick(window.class_input, Qt.Key.Key_Return)
    assert window.service.current_index == 0

    QTest.keyClicks(window.class_input, "sosna")
    QTest.keyClick(window.class_input, Qt.Key.Key_Return)

    assert _csv_labels(window) == [0, -2]
    assert window.service.current_index == 1
    assert window.class_input.text() == ""
