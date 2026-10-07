import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QValidator
from PySide6.QtWidgets import QButtonGroup, QGridLayout, QLabel, QPushButton, QSizePolicy, QWidget

from tree_classification.services.classification import LabelName

BUTTONS_PER_ROW = 6


class PageViewer(QLabel):
    """Shows an RGB page image scaled to the widget size, keeping its aspect ratio."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap = QPixmap()

    def set_image(self, image: NDArray[np.uint8]) -> None:
        """Takes an (H, W, 3) uint8 RGB array."""
        image = np.ascontiguousarray(image)
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


class LabelCodeValidator(QValidator):
    """Accepts only label codes; text that can still grow into one (e.g. "-", "1") is intermediate."""

    def __init__(self, codes, parent=None):
        super().__init__(parent)
        self._codes = {str(code) for code in codes}

    def validate(self, text: str, pos: int):
        if text in self._codes:
            return QValidator.State.Acceptable, text, pos
        if text == "" or any(code.startswith(text) for code in self._codes):
            return QValidator.State.Intermediate, text, pos
        return QValidator.State.Invalid, text, pos


class LabelButtons(QWidget):
    """One checkable button per label, showing its code, Polish and Latin name. At most one is selected.

    Clicking a different label emits `changed`; clicking the already selected one emits `confirmed`.
    """

    changed = Signal()
    confirmed = Signal()

    def __init__(self, label_names: dict[int, LabelName]):
        super().__init__()
        layout = QGridLayout(self)
        # Most styles mark a checked button too faintly to see which label is selected.
        self.setStyleSheet("QPushButton:checked { background-color: #2e7d32; color: white; font-weight: bold; }")
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for position, (code, name) in enumerate(label_names.items()):
            button = QPushButton(f"{code}: {name.polish}\n{name.latin}")
            button.setCheckable(True)
            # Keeps keyboard focus in the class number input of the window.
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._group.addButton(button, code)
            layout.addWidget(button, *divmod(position, BUTTONS_PER_ROW))
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


# --- Tests ---
def _qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


_NAMES = {-2: LabelName("Unclassified", "Nieklasyfikowane"), 0: LabelName("Pinus", "sosna"), 1: LabelName("Picea", "świerk")}


def test_label_buttons_show_code_polish_and_latin_names():
    _qapp()
    buttons = LabelButtons(_NAMES)

    assert {code: buttons.button(code).text() for code in _NAMES} == {
        -2: "-2: Nieklasyfikowane\nUnclassified",
        0: "0: sosna\nPinus",
        1: "1: świerk\nPicea",
    }
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


def test_label_code_validator_rejects_out_of_range_text():
    _qapp()
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit

    typed = {}
    for keys in ["17", "-2", "0", "18", "99", "-3", "-1", "100"]:
        line = QLineEdit()
        line.setValidator(LabelCodeValidator([-2, *range(18)]))
        QTest.keyClicks(line, keys)
        typed[keys] = line.text()

    assert typed == {"17": "17", "-2": "-2", "0": "0", "18": "1", "99": "9", "-3": "-", "-1": "-", "100": "10"}


def test_page_viewer_scales_image_to_widget():
    _qapp()
    viewer = PageViewer()
    viewer.resize(400, 400)

    viewer.set_image(np.zeros((100, 200, 3), dtype=np.uint8))

    assert viewer.pixmap().width() == 400
    assert viewer.pixmap().height() == 200
