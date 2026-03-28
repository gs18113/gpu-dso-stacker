"""
main.py — DSO Stacker GUI entry point

Launch with:
    python src/GUI/main.py
"""

from __future__ import annotations

import os
import sys

# Ensure src/GUI is on the path so sibling modules resolve correctly.
sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from main_window import MainWindow
from ui_density import get_ui_density


def _apply_dark_palette(app: QApplication) -> None:
    """Apply a clean dark Fusion palette."""
    density = get_ui_density()
    app.setStyle("Fusion")
    palette = QPalette()

    dark   = QColor(42,  42,  42)
    mid    = QColor(58,  58,  58)
    light  = QColor(75,  75,  75)
    text   = QColor(220, 220, 220)
    bright = QColor(255, 255, 255)
    accent = QColor(66,  135, 245)   # blue highlight
    base   = QColor(30,  30,  30)
    alt    = QColor(38,  38,  38)

    palette.setColor(QPalette.ColorRole.Window,          dark)
    palette.setColor(QPalette.ColorRole.WindowText,      text)
    palette.setColor(QPalette.ColorRole.Base,            base)
    palette.setColor(QPalette.ColorRole.AlternateBase,   alt)
    palette.setColor(QPalette.ColorRole.ToolTipBase,     mid)
    palette.setColor(QPalette.ColorRole.ToolTipText,     text)
    palette.setColor(QPalette.ColorRole.Text,            text)
    palette.setColor(QPalette.ColorRole.Button,          mid)
    palette.setColor(QPalette.ColorRole.ButtonText,      text)
    palette.setColor(QPalette.ColorRole.BrightText,      bright)
    palette.setColor(QPalette.ColorRole.Link,            accent)
    palette.setColor(QPalette.ColorRole.Highlight,       accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, bright)

    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(128, 128, 128))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       QColor(128, 128, 128))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(128, 128, 128))

    app.setPalette(palette)
    app.setStyleSheet(
        "QMainWindow, QWidget { background-color: #2a2a2a; color: #dcdcdc; }"
        "QMenuBar, QMenu { background-color: #303030; color: #dcdcdc; }"
        "QMenu::item:selected { background-color: #4287f5; color: #ffffff; }"
        "QTabWidget::pane { border: 1px solid #4a4a4a; }"
        "QTabBar::tab { background: #3a3a3a; color: #dcdcdc; border: 1px solid #4a4a4a; "
        f"padding: {density.tab_padding_v}px {density.tab_padding_h}px; "
        "margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }"
        "QTabBar::tab:selected { background: #474747; }"
        "QTabBar::tab:hover:!selected { background: #444444; }"
        "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #1f1f1f; "
        f"border: 1px solid #565656; border-radius: 4px; padding: {density.input_padding_v}px {density.input_padding_h}px; }}"
        f"QGroupBox {{ border: 1px solid #4a4a4a; border-radius: 6px; margin-top: {density.group_margin_top}px; padding: {density.group_padding}px; }}"
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        f"QHeaderView::section {{ background-color: #333333; color: #dcdcdc; border: 1px solid #4a4a4a; padding: {density.header_padding}px; }}"
        "QTableWidget { background-color: #1e1e1e; alternate-background-color: #252525; "
        "gridline-color: #3a3a3a; color: #dcdcdc; border: 1px solid #4a4a4a; }"
    )


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("DSO Stacker")
    app.setOrganizationName("dso-stacker")
    _apply_dark_palette(app)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
