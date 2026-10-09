import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QButtonGroup, QGridLayout, QLabel, QPushButton, QSizePolicy, QWidget

from tree_classification.services.classification import LabelName

# Pixels lighter than this on every channel count as paper when trimming page margins.
PAPER_THRESHOLD = 245
TRIM_PADDING = 12
LABEL_COLUMNS = 2

LABEL_BUTTON_STYLE = """
QPushButton {
    text-align: left;
    padding: 6px 12px;
    font-size: 13pt;
    border: 1px solid rgba(128, 128, 128, 0.6);
    border-radius: 6px;
    background-color: rgba(128, 128, 128, 0.12);
}
QPushButton:hover { border-color: palette(highlight); }
QPushButton[candidate="true"] { border: 2px solid #7cb342; }
QPushButton[saved="true"] { border-left: 6px solid #43a047; }
QPushButton:checked {
    background-color: #2e7d32;
    border-color: #2e7d32;
    color: white;
    font-weight: bold;
}
"""


def trim_margins(image: NDArray[np.uint8], padding: int = TRIM_PADDING) -> NDArray[np.uint8]:
    """Crops the white paper around the page content, keeping `padding` pixels of it."""
    content = np.any(image < PAPER_THRESHOLD, axis=2)
    rows, cols = np.flatnonzero(content.any(axis=1)), np.flatnonzero(content.any(axis=0))
    if rows.size == 0:
        return image
    top, bottom = max(rows[0] - padding, 0), min(rows[-1] + padding + 1, image.shape[0])
    left, right = max(cols[0] - padding, 0), min(cols[-1] + padding + 1, image.shape[1])
    return image[top:bottom, left:right]


class PageViewer(QLabel):
    """Shows an RGB page image without its white margins, scaled to the widget, keeping its aspect ratio."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap = QPixmap()

    def set_image(self, image: NDArray[np.uint8]) -> None:
        """Takes an (H, W, 3) uint8 RGB array."""
        image = np.ascontiguousarray(trim_margins(image))
        height, width, _ = image.shape
        # copy() detaches the QImage from the numpy buffer, which may be freed afterwards.
        qimage = QImage(image.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self._rescale()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if not self._pixmap.isNull():
            self.setPixmap(self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))


class LabelButtons(QWidget):
    """One checkable button per label: code and Polish name, Latin name below. At most one is selected.

    Labels fill LABEL_COLUMNS columns top to bottom, so codes read in order down each column.
    Negative codes (-2, unclassified) go last, away from the labels used most.
    Clicking a different label emits `changed`; clicking the already selected one emits `confirmed`.
    """

    changed = Signal()
    confirmed = Signal()

    def __init__(self, label_names: dict[int, LabelName]):
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.setStyleSheet(LABEL_BUTTON_STYLE)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        ordered = sorted(label_names.items(), key=lambda item: (item[0] < 0, item[0]))
        rows = -(-len(ordered) // LABEL_COLUMNS)
        for position, (code, name) in enumerate(ordered):
            button = QPushButton(f"{code}: {name.polish}\n{name.latin}")
            button.setCheckable(True)
            button.setMinimumHeight(54)
            # Keeps keyboard focus in the label input of the window.
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._group.addButton(button, code)
            layout.addWidget(button, position % rows, position // rows)
        layout.setRowStretch(rows, 1)
        # Selection before the latest click; an exclusive group gives no way to tell a re-click apart.
        self._last: int | None = None
        self._group.idClicked.connect(self._on_clicked)

    def _on_clicked(self, code: int) -> None:
        if code == self._last:
            self.confirmed.emit()
        else:
            self._last = code
            self.changed.emit()

    def button(self, code: int) -> QPushButton:
        return self._group.button(code)

    def selected(self) -> int | None:
        button = self._group.checkedButton()
        return None if button is None else self._group.id(button)

    def select(self, code: int | None) -> None:
        if code is None:
            # An exclusive group refuses to uncheck its last checked button.
            self._group.setExclusive(False)
            for button in self._group.buttons():
                button.setChecked(False)
            self._group.setExclusive(True)
        else:
            self._group.button(code).setChecked(True)
        self._last = code

    def set_candidates(self, codes: list[int]) -> None:
        """Outlines the labels the typed text may still mean."""
        self._mark("candidate", set(codes))

    def set_saved(self, code: int | None) -> None:
        """Marks the label already stored in the CSV for the current tree."""
        self._mark("saved", set() if code is None else {code})

    def is_marked(self, code: int, mark: str) -> bool:
        return bool(self._group.button(code).property(mark))

    def _mark(self, mark: str, codes: set[int]) -> None:
        for button in self._group.buttons():
            button.setProperty(mark, self._group.id(button) in codes)
            # Qt applies property-based style rules only after the style is re-polished.
            button.style().unpolish(button)
            button.style().polish(button)


# --- Tests ---
def _qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


_NAMES = {-2: LabelName("Unclassified", "Nieklasyfikowane"), 0: LabelName("Pinus", "sosna"), 1: LabelName("Picea", "świerk")}


def test_label_buttons_show_code_polish_and_latin_names_with_negative_codes_last():
    _qapp()
    buttons = LabelButtons(_NAMES)

    assert {code: buttons.button(code).text() for code in _NAMES} == {
        -2: "-2: Nieklasyfikowane\nUnclassified",
        0: "0: sosna\nPinus",
        1: "1: świerk\nPicea",
    }
    layout = buttons.layout()
    order = [layout.itemAt(i).widget().text().split(":")[0] for i in range(len(_NAMES))]
    assert order == ["0", "1", "-2"]
    assert buttons.selected() is None


def test_label_buttons_select_and_clear():
    _qapp()
    buttons = LabelButtons(_NAMES)

    buttons.select(1)
    assert buttons.selected() == 1
    buttons.select(-2)
    assert buttons.selected() == -2
    buttons.select(None)
    assert buttons.selected() is None


def test_label_button_click_selects_and_emits_changed():
    _qapp()
    buttons = LabelButtons(_NAMES)
    emitted = []
    buttons.changed.connect(lambda: emitted.append(buttons.selected()))

    buttons.button(0).click()
    buttons.button(1).click()

    assert emitted == [0, 1]
    assert not buttons.button(0).isChecked()


def test_clicking_selected_button_emits_confirmed():
    _qapp()
    buttons = LabelButtons(_NAMES)
    events = []
    buttons.changed.connect(lambda: events.append("changed"))
    buttons.confirmed.connect(lambda: events.append("confirmed"))

    buttons.button(0).click()
    buttons.button(0).click()
    buttons.button(1).click()
    buttons.select(0)
    buttons.button(0).click()

    assert events == ["changed", "confirmed", "changed", "confirmed"]
    assert buttons.selected() == 0


def test_candidate_and_saved_marks_replace_previous_ones():
    _qapp()
    buttons = LabelButtons(_NAMES)

    buttons.set_candidates([0, 1])
    buttons.set_candidates([1])
    buttons.set_saved(0)

    assert [buttons.is_marked(code, "candidate") for code in (0, 1)] == [False, True]
    assert buttons.is_marked(0, "saved") and not buttons.is_marked(1, "saved")
    buttons.set_saved(None)
    assert not buttons.is_marked(0, "saved")


def test_trim_margins_keeps_content_and_padding():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    image[40:60, 50:150] = 0

    trimmed = trim_margins(image, padding=5)

    assert trimmed.shape == (30, 110, 3)
    assert trim_margins(np.full((10, 10, 3), 255, dtype=np.uint8)).shape == (10, 10, 3)


def test_page_viewer_scales_trimmed_image_to_widget():
    _qapp()
    viewer = PageViewer()
    viewer.resize(400, 400)

    viewer.set_image(np.zeros((100, 200, 3), dtype=np.uint8))

    assert viewer.pixmap().width() == 400
    assert viewer.pixmap().height() == 200
