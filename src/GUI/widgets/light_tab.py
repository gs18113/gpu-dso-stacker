"""
widgets/light_tab.py — LightTab

The Light Frames tab. Extends FrameTableWidget with a leading "Ref" column
containing QRadioButtons so the user can designate exactly one frame as the
alignment reference. A QButtonGroup enforces single-selection; the first
frame added becomes the default reference.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from widgets.frame_table import FrameTableWidget, COL_FILENAME, COL_PATH, COL_SIZE, COL_DIMENSIONS

# The "Ref" column we prepend is at index 0; shift all others right by 1.
COL_REF      = 0
COL_FILENAME_ = COL_FILENAME + 1
COL_PATH_    = COL_PATH + 1
COL_SIZE_    = COL_SIZE + 1
COL_DIM_     = COL_DIMENSIONS + 1

_N_BASE_COLS = 4  # original FrameTableWidget columns


class LightTab(QWidget):
    """Light frames tab with reference-frame radio buttons."""

    files_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._radio_buttons: list[QRadioButton] = []

        self._frame_widget = FrameTableWidget(self)
        self._table = self._frame_widget.table()

        # Insert the "Ref" column at position 0 in the underlying table.
        self._table.insertColumn(COL_REF)
        self._table.setHorizontalHeaderItem(
            COL_REF,
            __import__("PyQt6.QtWidgets", fromlist=["QTableWidgetItem"]).QTableWidgetItem("Ref"),
        )
        from PyQt6.QtWidgets import QHeaderView
        self._table.horizontalHeader().setSectionResizeMode(
            COL_REF, QHeaderView.ResizeMode.ResizeToContents
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._frame_widget)

        # Forward files_changed
        self._frame_widget.files_changed.connect(self._on_files_changed)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def reference_index(self) -> int:
        """Row of the selected reference frame, or -1 if no frames loaded."""
        checked = self._btn_group.checkedButton()
        if checked is None:
            return -1
        return self._radio_buttons.index(checked)

    def get_file_paths(self) -> list[str]:
        return self._frame_widget.get_file_paths()

    def add_files(self, paths: list[str]) -> None:
        self._frame_widget.add_files(paths)

    def clear_all(self) -> None:
        self._frame_widget.clear_all()
        # Remove all radio buttons from the group
        for rb in self._radio_buttons:
            self._btn_group.removeButton(rb)
        self._radio_buttons.clear()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _on_files_changed(self) -> None:
        """Sync radio buttons with the current table row count."""
        n_rows   = self._table.rowCount()
        n_radios = len(self._radio_buttons)

        if n_rows > n_radios:
            # Rows were added — append radio buttons.
            for row in range(n_radios, n_rows):
                rb = QRadioButton()
                rb.setStyleSheet("QRadioButton { margin-left: 6px; }")
                container = QWidget()
                h = QHBoxLayout(container)
                h.setContentsMargins(4, 0, 4, 0)
                h.setAlignment(Qt.AlignmentFlag.AlignCenter)
                h.addWidget(rb)
                self._table.setCellWidget(row, COL_REF, container)
                self._btn_group.addButton(rb, row)
                self._radio_buttons.append(rb)
                # Default: select first frame.
                if row == 0:
                    rb.setChecked(True)
        elif n_rows < n_radios:
            # Rows were removed — rebuild radio list from scratch.
            for rb in self._radio_buttons:
                self._btn_group.removeButton(rb)
            self._radio_buttons.clear()
            for row in range(n_rows):
                rb = QRadioButton()
                rb.setStyleSheet("QRadioButton { margin-left: 6px; }")
                container = QWidget()
                h = QHBoxLayout(container)
                h.setContentsMargins(4, 0, 4, 0)
                h.setAlignment(Qt.AlignmentFlag.AlignCenter)
                h.addWidget(rb)
                self._table.setCellWidget(row, COL_REF, container)
                self._btn_group.addButton(rb, row)
                self._radio_buttons.append(rb)
            if self._radio_buttons:
                self._radio_buttons[0].setChecked(True)

        self.files_changed.emit()
