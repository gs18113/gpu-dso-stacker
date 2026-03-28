"""
ui_density.py — GUI spacing-density presets

Density can be selected with environment variable:
    DSO_GUI_DENSITY=compact|comfortable|airy

Default is "comfortable".
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class UiDensity:
    tab_padding_v: int
    tab_padding_h: int
    input_padding_v: int
    input_padding_h: int
    group_margin_top: int
    group_padding: int
    header_padding: int
    main_margin: int
    main_spacing: int
    toolbar_spacing: int
    action_button_h: int
    table_button_h: int
    log_font_size: int
    log_padding: int
    options_spacing: int
    options_margin: int


_DENSITY_PRESETS: dict[str, UiDensity] = {
    "compact": UiDensity(
        tab_padding_v=4, tab_padding_h=10,
        input_padding_v=2, input_padding_h=5,
        group_margin_top=8, group_padding=6, header_padding=3,
        main_margin=4, main_spacing=4, toolbar_spacing=6,
        action_button_h=30, table_button_h=30,
        log_font_size=12, log_padding=3,
        options_spacing=8, options_margin=8,
    ),
    "comfortable": UiDensity(
        tab_padding_v=6, tab_padding_h=12,
        input_padding_v=4, input_padding_h=6,
        group_margin_top=10, group_padding=8, header_padding=4,
        main_margin=6, main_spacing=6, toolbar_spacing=8,
        action_button_h=32, table_button_h=32,
        log_font_size=13, log_padding=4,
        options_spacing=12, options_margin=12,
    ),
    "airy": UiDensity(
        tab_padding_v=8, tab_padding_h=14,
        input_padding_v=6, input_padding_h=8,
        group_margin_top=12, group_padding=10, header_padding=5,
        main_margin=8, main_spacing=8, toolbar_spacing=10,
        action_button_h=34, table_button_h=34,
        log_font_size=13, log_padding=6,
        options_spacing=14, options_margin=14,
    ),
}


def get_ui_density_name() -> str:
    raw = os.getenv("DSO_GUI_DENSITY", "comfortable").strip().lower()
    return raw if raw in _DENSITY_PRESETS else "comfortable"


def get_ui_density() -> UiDensity:
    return _DENSITY_PRESETS[get_ui_density_name()]
