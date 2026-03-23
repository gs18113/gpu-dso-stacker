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


def _apply_dark_palette(app: QApplication) -> None:
    """Apply a clean dark Fusion palette."""
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
