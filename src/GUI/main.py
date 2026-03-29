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


def _apply_app_stylesheet(app: QApplication) -> None:
    """Apply lightweight app-wide styling for a cleaner UI."""
    app.setStyleSheet(
        """
        QMainWindow {
            background-color: #2a2a2a;
        }
        QTabWidget::pane {
            border: 1px solid #4a4a4a;
            border-radius: 6px;
            background: #2f2f2f;
        }
        QTabBar::tab {
            background: #383838;
            border: 1px solid #4a4a4a;
            border-bottom: none;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            padding: 7px 14px;
            margin-right: 4px;
        }
        QTabBar::tab:selected {
            background: #454545;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #4a4a4a;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 8px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #d8d8d8;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
            background: #1f1f1f;
            border: 1px solid #4a4a4a;
            border-radius: 5px;
            padding: 5px 6px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 1px solid #4287f5;
        }
        QTableWidget {
            background: #1f1f1f;
            alternate-background-color: #252525;
            gridline-color: #3c3c3c;
            border: 1px solid #4a4a4a;
            border-radius: 6px;
        }
        QHeaderView::section {
            background: #333333;
            border: none;
            border-right: 1px solid #3f3f3f;
            border-bottom: 1px solid #3f3f3f;
            padding: 6px 8px;
            font-weight: 600;
        }
        QPushButton {
            background: #505050;
            border: 1px solid #5f5f5f;
            border-radius: 5px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background: #5b5b5b;
        }
        QPushButton:pressed {
            background: #4a4a4a;
        }
        QPushButton:disabled {
            color: #9a9a9a;
            background: #464646;
            border-color: #555555;
        }
        QPushButton#runButton {
            background: #2e7d32;
            border-color: #3f9a46;
            color: white;
            font-weight: 600;
        }
        QPushButton#runButton:hover {
            background: #388e3c;
        }
        QPushButton#abortButton {
            background: #c62828;
            border-color: #d84a4a;
            color: white;
            font-weight: 600;
        }
        QPushButton#abortButton:hover {
            background: #d32f2f;
        }
        QLabel#statusLabel {
            color: #dcdcdc;
            font-weight: 500;
        }
        """
    )


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("DSO Stacker")
    app.setOrganizationName("dso-stacker")
    _apply_dark_palette(app)
    _apply_app_stylesheet(app)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
