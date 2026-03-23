"""
widgets/calib_tab.py — CalibTab

Generic calibration-frame tab (Dark, Flat, Bias, Darkflat).
Wraps FrameTableWidget and adds a stacking-method combo box below the
table so the user can choose between "winsorized-mean" and "median"
for master generation.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Signal

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from widgets.frame_table import FrameTableWidget


_METHODS = ["winsorized-mean", "median"]


class CalibTab(QWidget):
    """Calibration frame tab: file list + stacking method selector."""

    files_changed = Signal()

    def __init__(self, tab_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tab_name = tab_name
        self._setup_ui()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def file_paths(self) -> list[str]:
        return self._frame_widget.get_file_paths()

    @property
    def method(self) -> str:
        return self._method_combo.currentText()

    @method.setter
    def method(self, value: str) -> None:
        idx = self._method_combo.findText(value)
        if idx >= 0:
            self._method_combo.setCurrentIndex(idx)

    def clear_all(self) -> None:
        self._frame_widget.clear_all()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._frame_widget = FrameTableWidget(self)
        self._frame_widget.files_changed.connect(self.files_changed)
        layout.addWidget(self._frame_widget)

        # Stacking method row
        method_row = QHBoxLayout()
        method_row.setSpacing(8)
        method_row.addWidget(QLabel("Stacking method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(_METHODS)
        self._method_combo.setFixedWidth(180)
        method_row.addWidget(self._method_combo)
        method_row.addStretch()
        layout.addLayout(method_row)
