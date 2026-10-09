import numpy as np

from tree_classification.gui.widgets import (
    LabelButtons,
    PageViewer,
    trim_margins,
)
from tree_classification.services.classification import LabelName


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
