"""
main_window.py — MainWindow

Top-level application window.  Contains:
  • A QTabWidget with six tabs:
      Light, Dark, Flat, Bias, Darkflat, Stacking Options
  • A bottom toolbar with Run / Abort buttons and a status label
  • A collapsible log panel (QPlainTextEdit) showing subprocess output
  • A menu bar: File (New/Open/Save/Save As) and Help (About)

Bias / Darkflat mutual exclusion is enforced here by disabling the
opposing tab whenever one of them contains frames.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QPalette
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import sys
sys.path.insert(0, os.path.dirname(__file__))

from project import ProjectState
from runner import SubprocessRunner
from utils import build_command, write_calib_list, write_csv
from widgets.calib_tab import CalibTab
from widgets.light_tab import LightTab
from widgets.stacking_options_tab import StackingOptionsTab


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self._project    = ProjectState()
        self._save_path: Optional[str] = None
        self._runner:    Optional[SubprocessRunner] = None
        self._tmpdir:    Optional[tempfile.TemporaryDirectory] = None

        self.setWindowTitle("DSO Stacker")
        self.setMinimumSize(960, 680)
        self._setup_ui()
        self._setup_menu()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ---- Tab widget ----
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._light_tab   = LightTab()
        self._dark_tab    = CalibTab("Dark")
        self._flat_tab    = CalibTab("Flat")
        self._bias_tab    = CalibTab("Bias")
        self._darkflat_tab = CalibTab("Darkflat")
        self._options_tab = StackingOptionsTab()

        self._tabs.addTab(self._light_tab,    "Light")
        self._tabs.addTab(self._dark_tab,     "Dark")
        self._tabs.addTab(self._flat_tab,     "Flat")
        self._tabs.addTab(self._bias_tab,     "Bias")
        self._tabs.addTab(self._darkflat_tab, "Darkflat")
        self._tabs.addTab(self._options_tab,  "Stacking Options")

        # Bias ↔ Darkflat mutual exclusion
        self._bias_tab.files_changed.connect(self._update_calib_exclusion)
        self._darkflat_tab.files_changed.connect(self._update_calib_exclusion)

        # Global calibration method → push to all CalibTabs
        self._options_tab.calib_method_changed.connect(self._on_global_calib_method)

        # ---- Bottom toolbar ----
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self._run_btn   = QPushButton("▶  Stack")
        self._run_btn.setFixedHeight(34)
        self._run_btn.setMinimumWidth(120)
        self._run_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #2e7d32; color: white; "
            "border-radius: 4px; padding: 0 16px; }"
            "QPushButton:hover { background-color: #388e3c; }"
            "QPushButton:disabled { background-color: #555; color: #aaa; }"
        )

        self._abort_btn = QPushButton("■  Abort")
        self._abort_btn.setFixedHeight(34)
        self._abort_btn.setMinimumWidth(100)
        self._abort_btn.setEnabled(False)
        self._abort_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #c62828; color: white; "
            "border-radius: 4px; padding: 0 16px; }"
            "QPushButton:hover { background-color: #d32f2f; }"
            "QPushButton:disabled { background-color: #555; color: #aaa; }"
        )

        self._log_toggle_btn = QPushButton("▼  Log")
        self._log_toggle_btn.setFixedHeight(34)
        self._log_toggle_btn.setCheckable(True)
        self._log_toggle_btn.setChecked(True)
        self._log_toggle_btn.setStyleSheet(
            "QPushButton { border-radius: 4px; padding: 0 12px; }"
        )

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        toolbar.addWidget(self._run_btn)
        toolbar.addWidget(self._abort_btn)
        toolbar.addStretch()
        toolbar.addWidget(self._status_lbl)
        toolbar.addStretch()
        toolbar.addWidget(self._log_toggle_btn)

        # ---- Log panel ----
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(5000)
        self._log.setMinimumHeight(120)
        self._log.setMaximumHeight(300)
        self._log.setStyleSheet(
            "QPlainTextEdit { font-family: monospace; font-size: 12px; "
            "background: #1e1e1e; color: #d4d4d4; border: 1px solid #444; "
            "border-radius: 4px; }"
        )

        # ---- Splitter: tabs / log ----
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._tabs)
        splitter.addWidget(self._log)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)
        root.addLayout(toolbar)

        # Connect signals
        self._run_btn.clicked.connect(self._on_run)
        self._abort_btn.clicked.connect(self._on_abort)
        self._log_toggle_btn.toggled.connect(self._log.setVisible)
        self._options_tab.output_path_changed.connect(self._on_output_path_changed)

    def _setup_menu(self) -> None:
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")

        act_new = QAction("&New Project", self)
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self._on_new)
        file_menu.addAction(act_new)

        act_open = QAction("&Open Project…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open)
        file_menu.addAction(act_open)

        file_menu.addSeparator()

        act_save = QAction("&Save Project", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._on_save)
        file_menu.addAction(act_save)

        act_save_as = QAction("Save Project &As…", self)
        act_save_as.setShortcut("Ctrl+Shift+S")
        act_save_as.triggered.connect(self._on_save_as)
        file_menu.addAction(act_save_as)

        # Help
        help_menu = mb.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

    # ------------------------------------------------------------------ #
    # Bias / Darkflat mutual exclusion                                     #
    # ------------------------------------------------------------------ #

    def _update_calib_exclusion(self) -> None:
        bias_idx     = self._tabs.indexOf(self._bias_tab)
        darkflat_idx = self._tabs.indexOf(self._darkflat_tab)
        has_bias     = len(self._bias_tab.file_paths) > 0
        has_darkflat = len(self._darkflat_tab.file_paths) > 0

        if has_darkflat and not has_bias:
            self._tabs.setTabEnabled(bias_idx, False)
            self._tabs.setTabToolTip(
                bias_idx,
                "Disabled: Darkflat frames are loaded.\n"
                "Bias and Darkflat are mutually exclusive.",
            )
        else:
            self._tabs.setTabEnabled(bias_idx, True)
            self._tabs.setTabToolTip(bias_idx, "")

        if has_bias and not has_darkflat:
            self._tabs.setTabEnabled(darkflat_idx, False)
            self._tabs.setTabToolTip(
                darkflat_idx,
                "Disabled: Bias frames are loaded.\n"
                "Bias and Darkflat are mutually exclusive.",
            )
        else:
            self._tabs.setTabEnabled(darkflat_idx, True)
            self._tabs.setTabToolTip(darkflat_idx, "")

    def _on_global_calib_method(self, method: str) -> None:
        """Push the global calibration method to all CalibTabs."""
        for tab in (self._dark_tab, self._flat_tab,
                    self._bias_tab, self._darkflat_tab):
            tab.method = method

    # ------------------------------------------------------------------ #
    # Run / Abort                                                          #
    # ------------------------------------------------------------------ #

    def _on_run(self) -> None:
        # --- Gather state ---
        light_paths = self._light_tab.get_file_paths()
        ref_idx     = self._light_tab.reference_index

        # --- Validation ---
        if not light_paths:
            QMessageBox.critical(self, "Cannot Stack", "No light frames loaded.")
            return
        if ref_idx < 0:
            QMessageBox.critical(self, "Cannot Stack", "No reference frame selected.")
            return
        bias_paths     = self._bias_tab.file_paths
        darkflat_paths = self._darkflat_tab.file_paths
        if bias_paths and darkflat_paths:
            QMessageBox.critical(
                self,
                "Cannot Stack",
                "Bias and Darkflat frames are mutually exclusive — "
                "remove one set before stacking.",
            )
            return

        # --- Sync project state from UI ---
        self._sync_project_from_ui()

        # --- Collect options ---
        opts = self._project.options

        # --- Validate binary exists ---
        try:
            from utils import _binary_path
            _binary_path()
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Binary not found", str(exc))
            return

        # --- Create temp directory ---
        self._tmpdir = tempfile.TemporaryDirectory(prefix="dso_stacker_")
        tmpdir = self._tmpdir.name

        csv_path = os.path.join(tmpdir, "frames.csv")
        write_csv(light_paths, ref_idx, csv_path)

        calib_paths: dict[str, str] = {}
        for key, paths in (
            ("dark",     self._dark_tab.file_paths),
            ("flat",     self._flat_tab.file_paths),
            ("bias",     bias_paths),
            ("darkflat", darkflat_paths),
        ):
            if not paths:
                continue
            if len(paths) == 1:
                calib_paths[key] = paths[0]
            else:
                list_file = os.path.join(tmpdir, f"{key}_list.txt")
                write_calib_list(paths, list_file)
                calib_paths[key] = list_file

        # Merge per-tab methods back into options so build_command can use them
        opts["dark_method"]     = self._dark_tab.method
        opts["flat_method"]     = self._flat_tab.method
        opts["bias_method"]     = self._bias_tab.method
        opts["darkflat_method"] = self._darkflat_tab.method

        # --- Build command ---
        try:
            argv = build_command(self._project, csv_path, calib_paths)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Binary not found", str(exc))
            self._tmpdir.cleanup()
            self._tmpdir = None
            return

        # --- Start runner ---
        self._log.clear()
        self._log.appendPlainText("$ " + " ".join(argv))
        self._log.appendPlainText("")
        self._run_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)
        self._status_lbl.setText("Running…")

        self._runner = SubprocessRunner(argv, self)
        self._runner.line_ready.connect(self._log.appendPlainText)
        self._runner.finished.connect(self._on_run_finished)
        self._runner.start()

    def _on_abort(self) -> None:
        if self._runner:
            self._runner.abort()
            self._status_lbl.setText("Aborting…")
            self._abort_btn.setEnabled(False)

    def _on_run_finished(self, code: int) -> None:
        self._run_btn.setEnabled(True)
        self._abort_btn.setEnabled(False)

        if code == 0:
            self._status_lbl.setText("Done  ✓")
            self._log.appendPlainText("\n[Finished — exit code 0]")
        else:
            self._status_lbl.setText(f"Failed  ✗  (exit {code})")
            self._log.appendPlainText(f"\n[Finished — exit code {code}]")

        # Clean up temp directory
        if self._tmpdir:
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass
            self._tmpdir = None

    def _on_output_path_changed(self, _text: str) -> None:
        pass  # placeholder for future status-bar sync

    # ------------------------------------------------------------------ #
    # File menu actions                                                    #
    # ------------------------------------------------------------------ #

    def _on_new(self) -> None:
        reply = QMessageBox.question(
            self,
            "New Project",
            "Clear all frames and reset options?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._project.reset()
        self._load_project_into_ui()
        self._save_path = None
        self.setWindowTitle("DSO Stacker")

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "DSO Stacker project (*.yaml *.yml);;All files (*)"
        )
        if not path:
            return
        try:
            self._project = ProjectState.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", f"Could not load project:\n{exc}")
            return
        self._save_path = path
        self._load_project_into_ui()
        self.setWindowTitle(f"DSO Stacker — {Path(path).name}")

    def _on_save(self) -> None:
        if self._save_path:
            self._do_save(self._save_path)
        else:
            self._on_save_as()

    def _on_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "DSO Stacker project (*.yaml);;All files (*)"
        )
        if not path:
            return
        if not path.endswith((".yaml", ".yml")):
            path += ".yaml"
        self._save_path = path
        self._do_save(path)
        self.setWindowTitle(f"DSO Stacker — {Path(path).name}")

    def _do_save(self, path: str) -> None:
        self._sync_project_from_ui()
        try:
            self._project.save(path)
            self._status_lbl.setText(f"Saved: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save project:\n{exc}")

    # ------------------------------------------------------------------ #
    # Help                                                                 #
    # ------------------------------------------------------------------ #

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About DSO Stacker",
            "<b>DSO Stacker GUI</b><br>"
            "A graphical front-end for the <tt>dso_stacker</tt> CLI.<br><br>"
            "GPU-accelerated deep-sky image stacking with star detection, "
            "triangle matching alignment, Lanczos warping, and kappa-sigma integration.<br><br>"
            "Stack frames, save YAML projects, and view live output in the log panel.",
        )

    # ------------------------------------------------------------------ #
    # Project ↔ UI sync                                                   #
    # ------------------------------------------------------------------ #

    def _sync_project_from_ui(self) -> None:
        """Copy current UI state into self._project."""
        self._project.light_files           = self._light_tab.get_file_paths()
        self._project.light_reference_index = max(0, self._light_tab.reference_index)
        self._project.dark_files            = self._dark_tab.file_paths
        self._project.flat_files            = self._flat_tab.file_paths
        self._project.bias_files            = self._bias_tab.file_paths
        self._project.darkflat_files        = self._darkflat_tab.file_paths
        opts = self._options_tab.get_options()
        opts["dark_method"]     = self._dark_tab.method
        opts["flat_method"]     = self._flat_tab.method
        opts["bias_method"]     = self._bias_tab.method
        opts["darkflat_method"] = self._darkflat_tab.method
        self._project.options = opts

    def _load_project_into_ui(self) -> None:
        """Populate UI from self._project (after New or Open)."""
        p = self._project

        # Clear and reload light frames
        self._light_tab.clear_all()
        if p.light_files:
            self._light_tab.add_files(p.light_files)
            # Restore reference index: find the radio button for that row and check it.
            ref = p.light_reference_index
            if 0 <= ref < len(self._light_tab._radio_buttons):
                self._light_tab._radio_buttons[ref].setChecked(True)

        # Calibration tabs
        for tab, files in (
            (self._dark_tab,     p.dark_files),
            (self._flat_tab,     p.flat_files),
            (self._bias_tab,     p.bias_files),
            (self._darkflat_tab, p.darkflat_files),
        ):
            tab.clear_all()
            if files:
                tab._frame_widget.add_files(files)

        # Per-tab methods
        self._dark_tab.method     = p.options.get("dark_method",     "winsorized-mean")
        self._flat_tab.method     = p.options.get("flat_method",     "winsorized-mean")
        self._bias_tab.method     = p.options.get("bias_method",     "winsorized-mean")
        self._darkflat_tab.method = p.options.get("darkflat_method", "winsorized-mean")

        # Stacking options
        self._options_tab.set_options(p.options)

        # Re-evaluate mutual exclusion
        self._update_calib_exclusion()
