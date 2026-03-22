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

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QMessageBox

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


def _check_license() -> bool:
    """Show license dialog on first launch. Returns True if accepted."""
    settings = QSettings()
    if settings.value("license/accepted", False, type=bool):
        return True

    msg = QMessageBox()
    msg.setWindowTitle("License Agreement")
    msg.setText(
        "Please review and accept the license agreement to continue."
    )
    msg.setDetailedText(
        "DSO Stacker — License Agreement\n"
        "\n"
        "Copyright (c) 2026 DSO Stacker contributors. All rights reserved.\n"
        "\n"
        "This software is proprietary and confidential. Unauthorized\n"
        "copying, modification, distribution, or use is strictly\n"
        "prohibited. See the LICENSE file for full terms.\n"
        "\n"
        "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND.\n"
        "\n"
        "This software uses NVIDIA CUDA Toolkit and NVIDIA Performance\n"
        "Primitives (NPP), subject to the NVIDIA End User License Agreement:\n"
        "  https://docs.nvidia.com/cuda/eula/\n"
        "\n"
        "This software contains source code provided by NVIDIA Corporation.\n"
        "\n"
        "By using this software, you agree to:\n"
        "  1. The DSO Stacker license terms (see LICENSE)\n"
        "  2. The NVIDIA CUDA End User License Agreement\n"
        "  3. All third-party license terms (see THIRD_PARTY_LICENSES)"
    )
    msg.setIcon(QMessageBox.Icon.Information)
    accept_btn = msg.addButton("Accept", QMessageBox.ButtonRole.AcceptRole)
    msg.addButton("Decline", QMessageBox.ButtonRole.RejectRole)
    msg.exec()

    if msg.clickedButton() == accept_btn:
        settings.setValue("license/accepted", True)
        return True
    return False


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("DSO Stacker")
    app.setOrganizationName("dso-stacker")
    _apply_dark_palette(app)

    if not _check_license():
        sys.exit(0)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
