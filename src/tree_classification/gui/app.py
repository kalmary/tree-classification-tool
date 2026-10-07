import sys

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from tree_classification.gui.widgets import LabelButtons, LabelCodeValidator, PageViewer
from tree_classification.services.classification import ClassificationService
from tree_classification.services.preprocessing import UNCLASSIFIED_LABEL


class ClassificationWindow(QMainWindow):
    """Shows one report page at a time. All label handling is delegated to ClassificationService."""

    def __init__(self, services: list[ClassificationService]):
        super().__init__()
        if not services:
            raise ValueError("services must not be empty")
        self.services = services
        self._file_index = 0
        self.setWindowTitle("Tree classification")

        self.viewer = PageViewer()
        self.label_buttons = LabelButtons(services[0].label_names)
        self.class_input = QLineEdit()
        self.class_input.setValidator(LabelCodeValidator(services[0].label_names, self))
        self.class_input.setPlaceholderText("nr")
        self.class_input.setFixedWidth(60)
        self.status = QLabel()
        self.map_link = QLabel()
        self.map_link.setOpenExternalLinks(True)
        self.map_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        # Keeps keyboard focus in the class number input.
        self.map_link.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.save_quit_button = QPushButton("Save and Quit")

        navigation = QHBoxLayout()
        navigation.addWidget(QLabel("Class:"))
        navigation.addWidget(self.class_input)
        navigation.addWidget(self.status, stretch=1)
        navigation.addWidget(self.map_link)
        for button in (self.previous_button, self.next_button, self.save_quit_button):
            # Keeps keyboard focus in the class number input.
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            navigation.addWidget(button)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.viewer, stretch=1)
        layout.addWidget(self.label_buttons)
        layout.addLayout(navigation)
        self.setCentralWidget(central)

        self.label_buttons.changed.connect(self._update_controls)
        self.label_buttons.confirmed.connect(self._on_next_if_enabled)
        self.class_input.textChanged.connect(self._on_class_typed)
        self.previous_button.clicked.connect(self._on_previous)
        self.next_button.clicked.connect(self._on_next)
        self.save_quit_button.clicked.connect(self._on_save_quit)
        self._show_page()

    @property
    def service(self) -> ClassificationService:
        return self.services[self._file_index]

    def _show_page(self) -> None:
        tree = self.service.current_tree
        self.viewer.set_image(self.service.current_page_image())
        self.label_buttons.select(None if tree.label == UNCLASSIFIED_LABEL else tree.label)
        with QSignalBlocker(self.class_input):
            self.class_input.clear()
        self.class_input.setFocus()
        self.status.setText(
            f"{tree.source_tree_id} · tree {self.service.current_index + 1}/{self.service.total_pages}"
            f" · file {self._file_index + 1}/{len(self.services)}"
        )
        self.map_link.setText(f'<a href="{self.service.current_map_url}">Google Maps</a>')
        self._update_controls()

    def _update_controls(self) -> None:
        self.next_button.setEnabled(self.label_buttons.selected() is not None)
        self.previous_button.setEnabled(self.service.current_index > 0)

    def _on_class_typed(self, text: str) -> None:
        try:
            code = int(text)
        except ValueError:
            return
        if self.label_buttons.button(code) is not None:
            self.label_buttons.select(code)
            self._update_controls()

    def keyPressEvent(self, event) -> None:
        # Children such as QLineEdit ignore Enter, so it reaches the window from anywhere.
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_next_if_enabled()
        else:
            super().keyPressEvent(event)

    def _on_next_if_enabled(self) -> None:
        if self.next_button.isEnabled():
            self._on_next()

    def _on_next(self) -> None:
        if self.service.next_page(self.label_buttons.selected()):
            self._show_page()
        elif self._file_index + 1 < len(self.services):
            self._file_index += 1
            self._show_page()
        else:
            QMessageBox.information(self, "Tree classification", "Wszystkie drzewa sklasyfikowane")
            self.close()

    def _on_previous(self) -> None:
        self.service.previous_page()
        self._show_page()

    def _on_save_quit(self) -> None:
        label = self.label_buttons.selected()
        if label is not None:
            self.service.save_label(label)
        self.close()


def run_app(services: list[ClassificationService]) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = ClassificationWindow(services)
    window.showMaximized()
    return app.exec()


# --- Tests ---
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
    label_names = load_label_names(Path(__file__).resolve().parents[3] / "labels.json")
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
    assert "f0_1 · tree 2/2 · file 1/1" == window.status.text()


def test_previous_discards_selection_and_shows_saved_label(tmp_path):
    window = _make_window(tmp_path, [3, -2])
    assert window.service.current_index == 1

    window.label_buttons.button(7).click()
    window.previous_button.click()

    assert _csv_labels(window) == [3, -2]
    assert window.label_buttons.selected() == 3


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


def test_typing_class_number_selects_known_label_only(tmp_path):
    from PySide6.QtTest import QTest

    window = _make_window(tmp_path, [-2])

    window.class_input.setText("12")
    assert window.label_buttons.selected() == 12
    assert window.next_button.isEnabled()
    window.class_input.setText("")
    QTest.keyClicks(window.class_input, "99")
    assert window.class_input.text() == "9"
    assert window.label_buttons.selected() == 9


def test_enter_advances_only_with_selection_and_clears_input(tmp_path):
    from PySide6.QtTest import QTest

    window = _make_window(tmp_path, [-2, -2])

    QTest.keyClick(window.class_input, Qt.Key.Key_Return)
    assert window.service.current_index == 0

    QTest.keyClicks(window.class_input, "3")
    QTest.keyClick(window.class_input, Qt.Key.Key_Return)

    assert _csv_labels(window) == [3, -2]
    assert window.service.current_index == 1
    assert window.class_input.text() == ""


def test_map_link_points_to_current_tree(tmp_path):
    window = _make_window(tmp_path, [-2, -2], [-2])
    assert "query=52.000000,21.000000" in window.map_link.text()

    window.label_buttons.button(1).click()
    window.next_button.click()
    assert "query=52.000000,22.000000" in window.map_link.text()

    window.label_buttons.button(1).click()
    window.next_button.click()
    assert "query=53.000000,21.000000" in window.map_link.text()
