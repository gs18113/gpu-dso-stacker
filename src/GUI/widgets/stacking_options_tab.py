"""
widgets/stacking_options_tab.py — StackingOptionsTab

All dso_stacker CLI parameters grouped into labelled QGroupBox sections,
displayed inside a QScrollArea. Conditional visibility is managed by a
single _update_visibility() slot connected to every relevant widget's
change signal.

Sections
--------
  I/O              output path + format info
  Execution        CPU mode, batch size
  Integration      method, kappa, iterations
  Star Detection   star_sigma, moffat_alpha/beta, top_stars, min_stars
  Triangle Matching  triangle_iters, triangle_thresh, match_radius, match_device
  Sensor           Bayer pattern override
  Output Format    bit depth, TIFF compression, stretch bounds
  Calibration      save_master_dir, wsor_clip, calib_kappa, calib_iterations
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import detect_output_format, query_backends


class StackingOptionsTab(QWidget):
    """Stacking-options tab with all CLI parameters and conditional visibility."""

    # Emitted whenever the output path text changes (so MainWindow can sync).
    output_path_changed = Signal(str)
    # Emitted when the global calibration method changes (so MainWindow can
    # push the selection to all CalibTabs).
    calib_method_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()
        self._update_visibility()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_options(self) -> dict:
        """Return the current widget values as a dict matching ProjectState.options."""
        opts = {
            "output_path":      self._output_edit.text().strip() or "output.fits",
            "backend":          self._backend_combo.currentText(),
            "integration":      self._integration_combo.currentText(),
            "kappa":            self._kappa_spin.value(),
            "iterations":       self._iterations_spin.value(),
            "batch_size":       self._batch_spin.value(),
            "star_sigma":       self._star_sigma_spin.value(),
            "moffat_alpha":     self._moffat_alpha_spin.value(),
            "moffat_beta":      self._moffat_beta_spin.value(),
            "top_stars":        self._top_stars_spin.value(),
            "min_stars":        self._min_stars_spin.value(),
            "min_quality":      self._min_quality_spin.value(),
            "lm_centroid":      self._lm_centroid_check.isChecked(),
            "lm_radius":        self._lm_radius_spin.value(),
            "lm_iterations":    self._lm_iterations_spin.value(),
            "min_inliers":      self._min_inliers_spin.value(),
            "triangle_iters":   self._triangle_iters_spin.value(),
            "triangle_thresh":  self._triangle_thresh_spin.value(),
            "match_radius":     self._match_radius_spin.value(),
            "match_device":     self._match_device_combo.currentText(),
            "bayer":            self._bayer_combo.currentText(),
            "bit_depth":        self._bit_depth_combo.currentText(),
            "tiff_compression": self._tiff_compress_combo.currentText(),
            "stretch_min":      self._parse_stretch(self._stretch_min_edit.text()),
            "stretch_max":      self._parse_stretch(self._stretch_max_edit.text()),
            "save_master_dir":  self._save_master_edit.text().strip() or "./master",
            "calib_method":     self._calib_method_combo.currentText(),
            "wsor_clip":        self._wsor_clip_spin.value(),
            "calib_kappa":      self._calib_kappa_spin.value(),
            "calib_iterations": self._calib_iterations_spin.value(),
            "bg_calibration":   self._bg_calib_combo.currentText(),
            # Per-tab methods are managed by CalibTab; these keys may also
            # live here for completeness (overridden by CalibTab on run).
            "dark_method":      "kappa-sigma",
            "bias_method":      "kappa-sigma",
            "flat_method":      "kappa-sigma",
            "darkflat_method":  "kappa-sigma",
            "white_balance":    self._wb_mode_combo.currentText(),
            "wb_red":           self._wb_red_spin.value(),
            "wb_green":         self._wb_green_spin.value(),
            "wb_blue":          self._wb_blue_spin.value(),
        }
        return opts

    def set_options(self, opts: dict) -> None:
        """Restore widget values from a dict (e.g. loaded from YAML)."""
        self._set_text(self._output_edit,      opts.get("output_path", "output.fits"))
        # Backward compat: use_cpu → backend
        backend = opts.get("backend", "cpu" if opts.get("use_cpu") else "auto")
        self._set_combo(self._backend_combo,   backend)
        self._set_combo(self._integration_combo, opts.get("integration", "kappa-sigma"))
        self._kappa_spin.setValue(             opts.get("kappa", 3.0))
        self._iterations_spin.setValue(        opts.get("iterations", 3))
        self._batch_spin.setValue(             opts.get("batch_size", 16))
        self._star_sigma_spin.setValue(        opts.get("star_sigma", 3.0))
        self._moffat_alpha_spin.setValue(      opts.get("moffat_alpha", 2.5))
        self._moffat_beta_spin.setValue(       opts.get("moffat_beta", 2.0))
        self._top_stars_spin.setValue(         opts.get("top_stars", 50))
        self._min_stars_spin.setValue(         opts.get("min_stars", 20))
        self._min_quality_spin.setValue(      opts.get("min_quality", 0.0))
        self._lm_centroid_check.setChecked(  opts.get("lm_centroid", False))
        self._lm_radius_spin.setValue(       opts.get("lm_radius", 8.0))
        self._lm_iterations_spin.setValue(   opts.get("lm_iterations", 15))
        self._min_inliers_spin.setValue(      opts.get("min_inliers", 10))
        self._triangle_iters_spin.setValue(    opts.get("triangle_iters", opts.get("ransac_iters", 1000)))
        self._triangle_thresh_spin.setValue(   opts.get("triangle_thresh", opts.get("ransac_thresh", 2.0)))
        self._match_radius_spin.setValue(      opts.get("match_radius", 30.0))
        self._set_combo(self._match_device_combo, opts.get("match_device", "auto"))
        self._set_combo(self._bayer_combo,     opts.get("bayer", "auto"))
        self._set_combo(self._bit_depth_combo, opts.get("bit_depth", "f32"))
        self._set_combo(self._tiff_compress_combo, opts.get("tiff_compression", "none"))
        smin = opts.get("stretch_min")
        self._stretch_min_edit.setText("" if smin is None else str(smin))
        smax = opts.get("stretch_max")
        self._stretch_max_edit.setText("" if smax is None else str(smax))
        self._set_text(self._save_master_edit, opts.get("save_master_dir", "./master"))
        self._set_combo(self._calib_method_combo, opts.get("calib_method", "kappa-sigma"))
        self._wsor_clip_spin.setValue(         opts.get("wsor_clip", 0.1))
        self._calib_kappa_spin.setValue(      opts.get("calib_kappa", 2.5))
        self._calib_iterations_spin.setValue( opts.get("calib_iterations", 5))
        self._set_combo(self._wb_mode_combo,   opts.get("white_balance", "none"))
        self._wb_red_spin.setValue(            opts.get("wb_red", 1.0))
        self._wb_green_spin.setValue(          opts.get("wb_green", 1.0))
        self._wb_blue_spin.setValue(           opts.get("wb_blue", 1.0))
        self._set_combo(self._bg_calib_combo, opts.get("bg_calibration", "none"))
        self._update_visibility()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        vbox = QVBoxLayout(container)
        vbox.setSpacing(12)
        vbox.setContentsMargins(12, 12, 12, 12)

        vbox.addWidget(self._build_io_group())
        vbox.addWidget(self._build_execution_group())
        vbox.addWidget(self._build_integration_group())
        vbox.addWidget(self._build_star_detect_group())
        vbox.addWidget(self._build_ransac_group())
        vbox.addWidget(self._build_sensor_group())
        vbox.addWidget(self._build_white_balance_group())
        vbox.addWidget(self._build_output_format_group())
        vbox.addWidget(self._build_background_group())
        vbox.addWidget(self._build_calibration_group())
        vbox.addStretch()

    # --- Section builders ---

    def _build_io_group(self) -> QGroupBox:
        box = QGroupBox("I / O")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        row_w = QWidget()
        row_h = QHBoxLayout(row_w)
        row_h.setContentsMargins(0, 0, 0, 0)
        row_h.setSpacing(4)
        self._output_edit = QLineEdit("output.fits")
        self._output_edit.setPlaceholderText("output.fits")
        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._browse_output)
        row_h.addWidget(self._output_edit)
        row_h.addWidget(btn_browse)
        form.addRow("Output file:", row_w)

        self._output_edit.textChanged.connect(self._on_output_changed)
        return box

    def _build_execution_group(self) -> QGroupBox:
        box = QGroupBox("Execution")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._backend_combo = QComboBox()
        self._backend_combo.addItems(query_backends())
        self._backend_combo.setToolTip(
            "auto: prefer GPU if available, fall back to CPU\n"
            "cpu: run all stages on CPU (OpenMP-accelerated)\n"
            "cuda: require NVIDIA CUDA GPU\n"
            "metal: require Apple Metal GPU"
        )
        form.addRow("Backend:", self._backend_combo)

        self._batch_spin = _int_spin(1, 64, 16)
        self._batch_lbl  = QLabel("GPU batch size:")
        form.addRow(self._batch_lbl, self._batch_spin)

        self._backend_combo.currentTextChanged.connect(self._update_visibility)
        return box

    def _build_integration_group(self) -> QGroupBox:
        box = QGroupBox("Integration")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._integration_combo = QComboBox()
        self._integration_combo.addItems(["kappa-sigma", "mean", "median", "auto-adaptive"])
        form.addRow("Method:", self._integration_combo)

        self._kappa_spin     = _dbl_spin(0.1, 20.0, 3.0, 1, 0.1)
        self._iterations_spin = _int_spin(1, 20, 3)
        self._kappa_lbl      = QLabel("Kappa (σ threshold):")
        self._iterations_lbl = QLabel("Max iterations:")
        form.addRow(self._kappa_lbl,      self._kappa_spin)
        form.addRow(self._iterations_lbl, self._iterations_spin)

        self._integration_combo.currentTextChanged.connect(self._update_visibility)
        return box

    def _build_star_detect_group(self) -> QGroupBox:
        box = QGroupBox("Star Detection")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._star_sigma_spin    = _dbl_spin(0.1, 20.0, 3.0, 1, 0.1)
        self._moffat_alpha_spin  = _dbl_spin(0.1, 15.0, 2.5, 1, 0.1)
        self._moffat_beta_spin   = _dbl_spin(0.1, 10.0, 2.0, 1, 0.1)
        self._top_stars_spin     = _int_spin(3, 500, 50)
        self._min_stars_spin     = _int_spin(3, 100, 20)
        self._min_quality_spin   = _dbl_spin(0.0, 1.0, 0.0, 2, 0.05)
        self._min_quality_spin.setToolTip(
            "Auto-reject frames scoring below this fraction of reference quality.\n"
            "0 = disabled (default). 0.5 = reject frames below 50% of ref quality."
        )

        form.addRow("Star sigma (σ):",   self._star_sigma_spin)
        form.addRow("Moffat alpha:",      self._moffat_alpha_spin)
        form.addRow("Moffat beta:",       self._moffat_beta_spin)
        form.addRow("Top-K stars:",       self._top_stars_spin)
        form.addRow("Min stars (detection gate):", self._min_stars_spin)
        form.addRow("Min quality (0=off):", self._min_quality_spin)

        self._lm_centroid_check = QCheckBox("LM Gaussian centroid fitting")
        self._lm_centroid_check.setToolTip(
            "Refine star centroids with Levenberg-Marquardt Gaussian fitting.\n"
            "Achieves ~5x better precision than center-of-mass (default: off)."
        )
        self._lm_centroid_check.stateChanged.connect(self._update_visibility)
        form.addRow("", self._lm_centroid_check)

        self._lm_radius_lbl = QLabel("LM radius (px):")
        self._lm_radius_spin = _dbl_spin(2.0, 20.0, 8.0, 1, 1.0)
        form.addRow(self._lm_radius_lbl, self._lm_radius_spin)

        self._lm_iterations_lbl = QLabel("LM iterations:")
        self._lm_iterations_spin = _int_spin(1, 50, 15)
        form.addRow(self._lm_iterations_lbl, self._lm_iterations_spin)

        return box

    def _build_ransac_group(self) -> QGroupBox:
        box = QGroupBox("Triangle Matching Alignment")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._triangle_iters_spin  = _int_spin(10, 10000, 1000)
        self._triangle_thresh_spin = _dbl_spin(0.1, 50.0, 2.0, 1, 0.1)
        self._match_radius_spin  = _dbl_spin(1.0, 500.0, 30.0, 1, 1.0)
        self._min_inliers_spin   = _int_spin(3, 100, 10)
        self._match_device_combo = QComboBox()
        self._match_device_combo.addItems(["auto", "cpu", "gpu"])
        self._match_device_lbl = QLabel("Match device:")

        form.addRow("Max iterations:",       self._triangle_iters_spin)
        form.addRow("Inlier threshold (px):", self._triangle_thresh_spin)
        form.addRow("Match radius (px):",     self._match_radius_spin)
        form.addRow("Min inliers (RANSAC):",  self._min_inliers_spin)
        form.addRow(self._match_device_lbl,   self._match_device_combo)
        return box

    def _build_sensor_group(self) -> QGroupBox:
        box = QGroupBox("Sensor")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._bayer_combo = QComboBox()
        self._bayer_combo.addItems(["auto", "none", "rggb", "bggr", "grbg", "gbrg"])
        self._bayer_combo.setToolTip(
            "auto: read BAYERPAT from the reference FITS header\n"
            "none: treat as monochrome (no debayering)\n"
            "rggb/bggr/grbg/gbrg: force a specific Bayer pattern"
        )
        form.addRow("Bayer pattern:", self._bayer_combo)
        return box

    def _build_white_balance_group(self) -> QGroupBox:
        box = QGroupBox("White Balance")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._wb_mode_combo = QComboBox()
        self._wb_mode_combo.addItems(["none", "camera", "auto", "manual"])
        self._wb_mode_combo.setToolTip(
            "none: no white balance correction\n"
            "camera: use WB from RAW file metadata\n"
            "auto: gray-world assumption (equalise channel means)\n"
            "manual: specify per-channel multipliers below"
        )
        self._wb_mode_combo.currentTextChanged.connect(self._update_visibility)
        form.addRow("Mode:", self._wb_mode_combo)

        self._wb_red_spin = QDoubleSpinBox()
        self._wb_red_spin.setRange(0.01, 10.0)
        self._wb_red_spin.setSingleStep(0.1)
        self._wb_red_spin.setDecimals(3)
        self._wb_red_spin.setValue(1.0)
        self._wb_red_lbl = QLabel("Red multiplier:")
        form.addRow(self._wb_red_lbl, self._wb_red_spin)

        self._wb_green_spin = QDoubleSpinBox()
        self._wb_green_spin.setRange(0.01, 10.0)
        self._wb_green_spin.setSingleStep(0.1)
        self._wb_green_spin.setDecimals(3)
        self._wb_green_spin.setValue(1.0)
        self._wb_green_lbl = QLabel("Green multiplier:")
        form.addRow(self._wb_green_lbl, self._wb_green_spin)

        self._wb_blue_spin = QDoubleSpinBox()
        self._wb_blue_spin.setRange(0.01, 10.0)
        self._wb_blue_spin.setSingleStep(0.1)
        self._wb_blue_spin.setDecimals(3)
        self._wb_blue_spin.setValue(1.0)
        self._wb_blue_lbl = QLabel("Blue multiplier:")
        form.addRow(self._wb_blue_lbl, self._wb_blue_spin)

        return box

    def _build_output_format_group(self) -> QGroupBox:
        box = QGroupBox("Output Format")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Bit depth — uses a standard-item model so items can be disabled.
        self._bit_depth_combo = QComboBox()
        bd_model = QStandardItemModel(self._bit_depth_combo)
        for txt in ["f32", "f16", "16", "8"]:
            item = QStandardItem(txt)
            bd_model.appendRow(item)
        self._bit_depth_combo.setModel(bd_model)
        self._bit_depth_combo.setToolTip(
            "f32 – 32-bit float (FITS or TIFF)\n"
            "f16 – 16-bit float (TIFF only)\n"
            "16  – 16-bit integer (TIFF or PNG)\n"
            "8   – 8-bit integer (TIFF or PNG)"
        )
        form.addRow("Bit depth:", self._bit_depth_combo)

        self._tiff_compress_combo = QComboBox()
        self._tiff_compress_combo.addItems(["none", "zip", "lzw", "rle"])
        self._tiff_compress_lbl = QLabel("TIFF compression:")
        form.addRow(self._tiff_compress_lbl, self._tiff_compress_combo)

        self._stretch_min_edit = QLineEdit()
        self._stretch_min_edit.setPlaceholderText("auto (image min)")
        self._stretch_max_edit = QLineEdit()
        self._stretch_max_edit.setPlaceholderText("auto (image max)")
        self._stretch_min_lbl = QLabel("Stretch min:")
        self._stretch_max_lbl = QLabel("Stretch max:")
        form.addRow(self._stretch_min_lbl, self._stretch_min_edit)
        form.addRow(self._stretch_max_lbl, self._stretch_max_edit)

        self._bit_depth_combo.currentTextChanged.connect(self._update_visibility)
        return box

    def _build_background_group(self) -> QGroupBox:
        box = QGroupBox("Background Normalization")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._bg_calib_combo = QComboBox()
        self._bg_calib_combo.addItems(["none", "per-channel", "rgb"])
        self._bg_calib_combo.setToolTip(
            "Per-frame background normalization before integration.\n"
            "  none: disabled (default)\n"
            "  per-channel: normalize R, G, B independently\n"
            "  rgb: normalize all channels by luminance stats\n\n"
            "Useful when sky brightness varies between frames\n"
            "(clouds, light pollution, moonrise)."
        )
        form.addRow("Mode:", self._bg_calib_combo)
        return box

    def _build_calibration_group(self) -> QGroupBox:
        box = QGroupBox("Calibration")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._calib_method_combo = QComboBox()
        self._calib_method_combo.addItems(["kappa-sigma", "winsorized-mean", "median"])
        self._calib_method_combo.setToolTip(
            "Stacking method for generating master calibration frames.\n"
            "Applied to all calibration tabs (Dark, Flat, Bias, Darkflat).\n"
            "Individual tabs can override this selection."
        )
        form.addRow("Method:", self._calib_method_combo)

        master_w = QWidget()
        master_h = QHBoxLayout(master_w)
        master_h.setContentsMargins(0, 0, 0, 0)
        master_h.setSpacing(4)
        self._save_master_edit = QLineEdit("./master")
        btn_master = QPushButton("Browse…")
        btn_master.setFixedWidth(80)
        btn_master.clicked.connect(self._browse_master_dir)
        master_h.addWidget(self._save_master_edit)
        master_h.addWidget(btn_master)
        form.addRow("Save masters to:", master_w)

        self._wsor_clip_spin = _dbl_spin(0.0, 0.49, 0.1, 2, 0.01)
        self._wsor_clip_spin.setToolTip(
            "Winsorized-mean clipping fraction per side.\n"
            "Valid range: [0.0, 0.49]. Default: 0.1."
        )
        self._wsor_clip_lbl = QLabel("Wsor clip fraction:")
        form.addRow(self._wsor_clip_lbl, self._wsor_clip_spin)

        self._calib_kappa_spin = _dbl_spin(0.1, 20.0, 2.5, 1, 0.1)
        self._calib_kappa_spin.setToolTip(
            "Kappa-sigma rejection threshold for calibration frame stacking.\n"
            "Values further than kappa*sigma from the mean are rejected.\n"
            "Default: 2.5"
        )
        self._calib_kappa_lbl = QLabel("Calib kappa:")
        form.addRow(self._calib_kappa_lbl, self._calib_kappa_spin)

        self._calib_iterations_spin = _int_spin(1, 20, 5)
        self._calib_iterations_spin.setToolTip(
            "Maximum clipping iterations for calibration kappa-sigma stacking.\n"
            "Default: 5"
        )
        self._calib_iterations_lbl = QLabel("Calib iterations:")
        form.addRow(self._calib_iterations_lbl, self._calib_iterations_spin)

        note = QLabel(
            "<i>Individual calibration tabs (Dark, Flat, Bias, Darkflat) can "
            "override this method independently.</i>"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow("", note)

        self._calib_method_combo.currentTextChanged.connect(self._update_visibility)
        self._calib_method_combo.currentTextChanged.connect(self.calib_method_changed.emit)
        return box

    # ------------------------------------------------------------------ #
    # Conditional visibility                                               #
    # ------------------------------------------------------------------ #

    def _update_visibility(self) -> None:
        """Show/hide widgets based on the current selection state."""
        integ    = self._integration_combo.currentText()
        backend  = self._backend_combo.currentText()
        out_path = self._output_edit.text()
        fmt      = detect_output_format(out_path)
        bd       = self._bit_depth_combo.currentText()
        cm       = self._calib_method_combo.currentText()

        # Kappa / iterations: only for kappa-sigma integration
        is_ks = (integ == "kappa-sigma")
        self._kappa_lbl.setVisible(is_ks)
        self._kappa_spin.setVisible(is_ks)
        self._iterations_lbl.setVisible(is_ks)
        self._iterations_spin.setVisible(is_ks)

        # Batch size / match device: only when backend is not CPU
        is_cpu = (backend == "cpu")
        self._batch_lbl.setVisible(not is_cpu)
        self._batch_spin.setVisible(not is_cpu)
        self._match_device_lbl.setVisible(not is_cpu)
        self._match_device_combo.setVisible(not is_cpu)

        # TIFF compression: only for TIFF output
        is_tiff = (fmt == "tiff")
        self._tiff_compress_lbl.setVisible(is_tiff)
        self._tiff_compress_combo.setVisible(is_tiff)

        # Stretch min/max: only for integer output
        is_int = (bd in ("8", "16"))
        self._stretch_min_lbl.setVisible(is_int)
        self._stretch_min_edit.setVisible(is_int)
        self._stretch_max_lbl.setVisible(is_int)
        self._stretch_max_edit.setVisible(is_int)

        # Bit depth combo: enable/disable items based on output format
        self._update_bit_depth_items(fmt)

        # Calibration parameters: show only what's relevant to the method
        self._wsor_clip_lbl.setVisible(cm == "winsorized-mean")
        self._wsor_clip_spin.setVisible(cm == "winsorized-mean")
        self._calib_kappa_lbl.setVisible(cm == "kappa-sigma")
        self._calib_kappa_spin.setVisible(cm == "kappa-sigma")
        self._calib_iterations_lbl.setVisible(cm == "kappa-sigma")
        self._calib_iterations_spin.setVisible(cm == "kappa-sigma")

        # White balance manual multipliers: only when mode is "manual"
        is_manual_wb = (self._wb_mode_combo.currentText() == "manual")
        self._wb_red_lbl.setVisible(is_manual_wb)
        self._wb_red_spin.setVisible(is_manual_wb)
        self._wb_green_lbl.setVisible(is_manual_wb)
        self._wb_green_spin.setVisible(is_manual_wb)
        self._wb_blue_lbl.setVisible(is_manual_wb)
        self._wb_blue_spin.setVisible(is_manual_wb)

        # LM centroid params: only when checkbox is checked
        is_lm = self._lm_centroid_check.isChecked()
        self._lm_radius_lbl.setVisible(is_lm)
        self._lm_radius_spin.setVisible(is_lm)
        self._lm_iterations_lbl.setVisible(is_lm)
        self._lm_iterations_spin.setVisible(is_lm)

    def _update_bit_depth_items(self, fmt: str) -> None:
        """Enable/disable bit-depth combo items according to output format.

        FITS  → only f32 valid
        TIFF  → all valid
        PNG   → only 8, 16 valid
        other → leave unchanged
        """
        model = self._bit_depth_combo.model()
        if not isinstance(model, QStandardItemModel):
            return

        valid: set[str]
        if fmt == "fits":
            valid = {"f32"}
        elif fmt == "tiff":
            valid = {"f32", "f16", "16", "8"}
        elif fmt == "png":
            valid = {"16", "8"}
        else:
            return  # unknown extension — leave items as-is

        current = self._bit_depth_combo.currentText()
        for row in range(model.rowCount()):
            item = model.item(row)
            enabled = item.text() in valid
            item.setEnabled(enabled)
            # Adjust flags for selectability
            if enabled:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)

        # Snap to a valid selection if current is now disabled
        if current not in valid:
            first_valid = next(iter(valid))
            self._set_combo(self._bit_depth_combo, first_valid)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _on_output_changed(self, text: str) -> None:
        self._update_visibility()
        self.output_path_changed.emit(text)

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose output file",
            self._output_edit.text(),
            "FITS (*.fits *.fit *.fts);;TIFF (*.tiff *.tif);;PNG (*.png);;All files (*)",
        )
        if path:
            self._output_edit.setText(path)

    def _browse_master_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Save master frames to", self._save_master_edit.text()
        )
        if path:
            self._save_master_edit.setText(path)

    @staticmethod
    def _parse_stretch(text: str) -> float | None:
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _set_text(widget: QLineEdit, value: str) -> None:
        widget.blockSignals(True)
        widget.setText(value)
        widget.blockSignals(False)

    @staticmethod
    def _set_combo(combo: QComboBox, value: str) -> None:
        idx = combo.findText(value)
        if idx >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)


# ------------------------------------------------------------------ #
# Spin box factory helpers                                            #
# ------------------------------------------------------------------ #

def _dbl_spin(lo: float, hi: float, default: float,
              decimals: int, step: float) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setValue(default)
    sb.setFixedWidth(110)
    return sb


def _int_spin(lo: int, hi: int, default: int) -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(default)
    sb.setFixedWidth(110)
    return sb
