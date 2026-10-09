import sys

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from tree_classification.gui.widgets import LabelButtons, PageViewer
from tree_classification.services.classification import ClassificationService
from tree_classification.services.preprocessing import UNCLASSIFIED_LABEL

WINDOW_TITLE = "Klasyfikacja drzew"
PAGE_DPI = 150
PANEL_WIDTH = 500

FEEDBACK_COLORS = {"ok": "#43a047", "warn": "#f9a825", "error": "#e53935", "hint": "palette(mid)"}

# Font sizes in pt, not px, so they follow the screen's DPI scaling.
WINDOW_STYLE = """
QPushButton#next { background-color: #2e7d32; color: white; font-weight: bold; border: 1px solid #2e7d32; }
QPushButton#next:disabled {
    background-color: rgba(128, 128, 128, 0.2); color: rgba(128, 128, 128, 0.9); border: 1px solid rgba(128, 128, 128, 0.6);
}
QPushButton#navigation { background-color: rgba(128, 128, 128, 0.12); border: 1px solid rgba(128, 128, 128, 0.6); }
QPushButton#navigation:hover { border-color: palette(highlight); }
QPushButton#navigation:disabled { color: rgba(128, 128, 128, 0.7); }
QPushButton#navigation, QPushButton#next { min-height: 48px; padding: 0 16px; border-radius: 6px; font-size: 14pt; }
QProgressBar {
    min-height: 32px; border: 1px solid rgba(128, 128, 128, 0.6); border-radius: 6px;
    background-color: rgba(128, 128, 128, 0.15); text-align: center; font-size: 13pt; font-weight: bold;
}
QProgressBar::chunk { background-color: #2e7d32; border-radius: 5px; }
QLabel#title { font-size: 18pt; font-weight: bold; }
QLabel#subtitle { font-size: 12pt; }
QLabel#links { font-size: 14pt; }
QLabel#panelTitle { font-size: 14pt; font-weight: bold; }
QLabel#feedback { font-size: 12pt; }
QLineEdit#labelInput { font-size: 16pt; min-height: 48px; padding: 0 8px; }
"""


class ClassificationWindow(QMainWindow):
    """Shows one report page at a time. All label handling is delegated to ClassificationService.

    Left: the page. Right: label input, label list and navigation, so the eyes and hands stay in one place.
    """

    def __init__(self, services: list[ClassificationService]):
        super().__init__()
        if not services:
            raise ValueError("services must not be empty")
        self.services = services
        self._file_index = 0
        self.setWindowTitle(WINDOW_TITLE)
        self.setStyleSheet(WINDOW_STYLE)

        self.name_label = QLabel()
        self.name_label.setObjectName("title")
        self.name_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.position_label = QLabel()
        self.position_label.setObjectName("subtitle")
        self.map_link = QLabel()
        self.map_link.setObjectName("links")
        self.map_link.setOpenExternalLinks(True)
        self.map_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        # Keeps keyboard focus in the label input.
        self.map_link.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(460)
        self.progress.setFormat("sklasyfikowano %v / %m")
        self.viewer = PageViewer()

        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("numer lub nazwa, np. 5 / dąb / quercus")
        self.class_input.setObjectName("labelInput")
        self.feedback = QLabel()
        self.feedback.setObjectName("feedback")
        self.feedback.setWordWrap(True)
        self.label_buttons = LabelButtons(services[0].label_names)
        self.previous_button = QPushButton("← Poprzednie")
        self.next_button = QPushButton("Dalej  ⏎")
        self.save_quit_button = QPushButton("Zapisz i zakończ")
        self.previous_button.setObjectName("navigation")
        self.save_quit_button.setObjectName("navigation")
        self.next_button.setObjectName("next")
        self.next_button.setToolTip("Zapisuje etykietę i przechodzi dalej (Enter albo drugi klik na etykiecie)")
        self.previous_button.setToolTip("Wraca do poprzedniego drzewa bez zapisywania")
        for button in (self.previous_button, self.next_button, self.save_quit_button):
            # Keeps keyboard focus in the label input.
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.setCentralWidget(self._build_layout())

        self.label_buttons.changed.connect(self._refresh_controls)
        self.label_buttons.confirmed.connect(self._on_next_if_enabled)
        self.class_input.textChanged.connect(self._on_class_typed)
        self.previous_button.clicked.connect(self._on_previous)
        self.next_button.clicked.connect(self._on_next)
        self.save_quit_button.clicked.connect(self._on_save_quit)
        self._show_page()

    def _build_layout(self) -> QWidget:
        title = QVBoxLayout()
        title.setSpacing(2)
        title.addWidget(self.name_label)
        title.addWidget(self.position_label)
        title_widget = QWidget()
        title_widget.setLayout(title)
        links = QHBoxLayout()
        links.addStretch(1)
        links.addWidget(self.map_link)
        links_widget = QWidget()
        links_widget.setLayout(links)
        # Equal stretch on both sides keeps the progress bar in the middle of the window.
        header = QHBoxLayout()
        header.addWidget(title_widget, stretch=1)
        header.addWidget(self.progress)
        header.addWidget(links_widget, stretch=1)

        labels_scroll = QScrollArea()
        labels_scroll.setWidget(self.label_buttons)
        labels_scroll.setWidgetResizable(True)
        labels_scroll.setFrameShape(QFrame.Shape.NoFrame)

        navigation = QHBoxLayout()
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.next_button, stretch=1)

        panel = QVBoxLayout()
        panel_title = QLabel("Etykieta")
        panel_title.setObjectName("panelTitle")
        panel.addWidget(panel_title)
        panel.addWidget(self.class_input)
        panel.addWidget(self.feedback)
        panel.addWidget(labels_scroll, stretch=1)
        panel.addLayout(navigation)
        panel.addWidget(self.save_quit_button)
        panel.setContentsMargins(0, 0, 0, 0)
        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(PANEL_WIDTH)

        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self.viewer, stretch=1)
        body.addWidget(panel_widget)

        root = QVBoxLayout()
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        root.addLayout(header)
        root.addLayout(body, stretch=1)
        central = QWidget()
        central.setLayout(root)
        return central

    @property
    def service(self) -> ClassificationService:
        return self.services[self._file_index]

    def _saved_label(self) -> int | None:
        label = self.service.current_tree.label
        return None if label == UNCLASSIFIED_LABEL else label

    def _show_page(self) -> None:
        tree = self.service.current_tree
        self.viewer.set_image(self.service.current_page_image(dpi=PAGE_DPI))
        self.name_label.setText(tree.source_tree_id)
        self.position_label.setText(
            f"Drzewo {self.service.current_index + 1} / {self.service.total_pages}"
            f" · plik {self._file_index + 1} / {len(self.services)}"
        )
        self.map_link.setText(
            f'<a href="{self.service.current_map_url}">Google Maps</a>'
            f' · <a href="{self.service.current_bdl_url}">BDL</a>'
        )
        self.progress.setMaximum(self.service.total_pages)
        self.progress.setValue(self.service.classified_count())

        self.label_buttons.set_saved(self._saved_label())
        self.label_buttons.select(self._saved_label())
        with QSignalBlocker(self.class_input):
            self.class_input.clear()
        self.label_buttons.set_candidates([])
        self.class_input.setFocus()
        self._refresh_controls()

    def _on_class_typed(self, text: str) -> None:
        match = self.service.match_label(text)
        self.label_buttons.set_candidates(match.candidates)
        if text.strip():
            self.label_buttons.select(match.selected)
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        selected = self.label_buttons.selected()
        self.next_button.setEnabled(selected is not None)
        self.previous_button.setEnabled(self.service.current_index > 0)
        self._show_feedback(selected)

    def _show_feedback(self, selected: int | None) -> None:
        text = self.class_input.text().strip()
        if selected is not None:
            name = self.service.label_names[selected]
            state = "zapisana" if selected == self._saved_label() else "niezapisana – Enter zapisuje"
            message, kind = f"→ {selected}: {name.polish} · {name.latin} ({state})", "ok"
        elif text:
            candidates = self.service.match_label(text).candidates
            if candidates:
                message, kind = f"Pasuje {len(candidates)} etykiet – doprecyzuj lub kliknij", "warn"
            else:
                message, kind = f"Brak etykiety „{text}”", "error"
        else:
            message, kind = "Wpisz numer lub nazwę albo kliknij etykietę na liście", "hint"
        self.feedback.setText(message)
        self.feedback.setStyleSheet(f"color: {FEEDBACK_COLORS[kind]};")

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
            QMessageBox.information(self, WINDOW_TITLE, "Wszystkie drzewa sklasyfikowane")
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
