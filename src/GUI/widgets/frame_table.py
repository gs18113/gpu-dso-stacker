"""
widgets/frame_table.py — FrameTableWidget

A QTableWidget subclass that acts as the shared base for all five frame-list
tabs (Light, Dark, Flat, Bias, Darkflat).

Features
--------
- Four columns: Filename | Full Path | Size | Dimensions (W×H)
- Drag-and-drop of .fit/.fits/.fts files from the OS file manager
- "Add Files…" button opening QFileDialog
- "Remove Selected" button
- Background FITS-header reading via FitsMetaWorker / QThreadPool
- Deduplication: files already in the list are silently ignored
- files_changed signal emitted whenever the list is modified
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))

from fits_meta import FitsMetaWorker
from utils import format_size

_FITS_EXTS = {".fit", ".fits", ".fts"}

# Column indices
COL_FILENAME   = 0
COL_PATH       = 1
COL_SIZE       = 2
COL_DIMENSIONS = 3

_HEADERS = ["Filename", "Full Path", "Size", "Dimensions"]


def _is_fits(url_path: str) -> bool:
    return Path(url_path).suffix.lower() in _FITS_EXTS


class FrameTableWidget(QWidget):
    """Frame list with drag-and-drop, toolbar, and async FITS metadata loading."""

    files_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loaded_paths: set[str] = set()
        self._pool = QThreadPool.globalInstance()
        self._setup_ui()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self._btn_add = QPushButton("Add Files…")
        self._btn_add.setFixedHeight(32)
        self._btn_remove = QPushButton("Remove Selected")
        self._btn_remove.setFixedHeight(32)
        toolbar.addWidget(self._btn_add)
        toolbar.addWidget(self._btn_remove)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Table
        self._table = _InnerTable(self)
        self._table.setColumnCount(len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(COL_FILENAME,   QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(COL_PATH,        QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(COL_SIZE,        QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(COL_DIMENSIONS,  QHeaderView.ResizeMode.ResizeToContents)
        self._table.setStyleSheet("QTableWidget { gridline-color: #3a3a3a; }")
        layout.addWidget(self._table)

        # Connect signals
        self._btn_add.clicked.connect(self._on_add_clicked)
        self._btn_remove.clicked.connect(self.remove_selected)
        self._table.files_dropped.connect(self.add_files)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add_files(self, paths: list[str]) -> None:
        """Add files to the list, skipping duplicates."""
        new_paths = [p for p in paths if p not in self._loaded_paths]
        if not new_paths:
            return
        for path in new_paths:
            self._add_row(path)
        self.files_changed.emit()

    def remove_selected(self) -> None:
        """Remove all currently selected rows."""
        rows = sorted(
            {idx.row() for idx in self._table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            path_item = self._table.item(row, COL_PATH)
            if path_item:
                self._loaded_paths.discard(path_item.text())
            self._table.removeRow(row)
        if rows:
            self.files_changed.emit()

    def get_file_paths(self) -> list[str]:
        """Return file paths in current row order."""
        result = []
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_PATH)
            if item:
                result.append(item.text())
        return result

    def clear_all(self) -> None:
        """Remove every row from the table."""
        self._table.setRowCount(0)
        self._loaded_paths.clear()
        self.files_changed.emit()

    def table(self) -> QTableWidget:
        """Return the underlying QTableWidget (needed by LightTab to insert columns)."""
        return self._table

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _add_row(self, path: str) -> None:
        self._loaded_paths.add(path)
        row = self._table.rowCount()
        self._table.insertRow(row)

        # Filename
        self._table.setItem(row, COL_FILENAME, _make_item(Path(path).name))
        # Full path (tooltip shows it in full when column is too narrow)
        path_item = _make_item(path)
        path_item.setToolTip(path)
        self._table.setItem(row, COL_PATH, path_item)
        # Size
        try:
            sz = format_size(os.path.getsize(path))
        except OSError:
            sz = "?"
        self._table.setItem(row, COL_SIZE, _make_item(sz))
        # Dimensions (filled asynchronously)
        self._table.setItem(row, COL_DIMENSIONS, _make_item("…"))

        # Launch background header read
        worker = FitsMetaWorker(row, path)
        worker.signals.result.connect(self._on_meta_loaded)
        self._pool.start(worker)

    def _on_meta_loaded(self, row: int, meta: dict) -> None:
        """Update the Dimensions cell when the background worker finishes."""
        if row >= self._table.rowCount():
            return
        w, h = meta.get("width"), meta.get("height")
        if w is not None and h is not None:
            text = f"{w}×{h}"
        else:
            text = "N/A"
        item = self._table.item(row, COL_DIMENSIONS)
        if item:
            item.setText(text)

    def _on_add_clicked(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select FITS frames",
            "",
            "FITS files (*.fit *.fits *.fts);;All files (*)",
        )
        if paths:
            self.add_files(paths)


# ------------------------------------------------------------------ #
# Inner table subclass that handles OS drag-and-drop                 #
# ------------------------------------------------------------------ #

class _InnerTable(QTableWidget):
    """QTableWidget that accepts FITS-file drops from the OS file manager."""

    files_dropped = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)

    # Override drag events on the table itself; Qt routes viewport events here.
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._has_fits(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if self._has_fits(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        paths = [
            u.toLocalFile()
            for u in urls
            if u.isLocalFile() and _is_fits(u.toLocalFile())
        ]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    @staticmethod
    def _has_fits(event) -> bool:
        if not event.mimeData().hasUrls():
            return False
        return any(
            u.isLocalFile() and _is_fits(u.toLocalFile())
            for u in event.mimeData().urls()
        )


def _make_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    return item
