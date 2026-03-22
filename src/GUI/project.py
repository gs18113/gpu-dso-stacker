"""
project.py — ProjectState dataclass + YAML serialization.

Holds the complete GUI state: all frame lists with their per-tab
stacking methods, reference frame index for light frames, and all
stacking options that map 1-to-1 to CLI arguments.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import yaml


# Default option values — mirror CLI defaults exactly.
_DEFAULT_OPTIONS: dict = {
    "output_path": "output.fits",
    "use_cpu": False,
    "integration": "kappa-sigma",
    "kappa": 3.0,
    "iterations": 3,
    "batch_size": 16,
    "star_sigma": 3.0,
    "moffat_alpha": 2.5,
    "moffat_beta": 2.0,
    "top_stars": 50,
    "min_stars": 6,
    "ransac_iters": 1000,
    "ransac_thresh": 2.0,
    "match_radius": 30.0,
    "bayer": "auto",
    "bit_depth": "f32",
    "tiff_compression": "none",
    "stretch_min": None,
    "stretch_max": None,
    "save_master_dir": "./master",
    "wsor_clip": 0.1,
    "dark_method": "winsorized-mean",
    "bias_method": "winsorized-mean",
    "flat_method": "winsorized-mean",
    "darkflat_method": "winsorized-mean",
}


@dataclass
class ProjectState:
    """Complete serialisable GUI state."""

    light_files: list[str] = field(default_factory=list)
    light_reference_index: int = 0

    dark_files: list[str] = field(default_factory=list)
    flat_files: list[str] = field(default_factory=list)
    bias_files: list[str] = field(default_factory=list)
    darkflat_files: list[str] = field(default_factory=list)

    options: dict = field(default_factory=lambda: copy.deepcopy(_DEFAULT_OPTIONS))

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "light_frames": {
                "reference_index": self.light_reference_index,
                "files": list(self.light_files),
            },
            "dark_frames": {
                "method": self.options.get("dark_method", "winsorized-mean"),
                "files": list(self.dark_files),
            },
            "flat_frames": {
                "method": self.options.get("flat_method", "winsorized-mean"),
                "files": list(self.flat_files),
            },
            "bias_frames": {
                "method": self.options.get("bias_method", "winsorized-mean"),
                "files": list(self.bias_files),
            },
            "darkflat_frames": {
                "method": self.options.get("darkflat_method", "winsorized-mean"),
                "files": list(self.darkflat_files),
            },
            "options": copy.deepcopy(self.options),
        }

    def from_dict(self, d: dict) -> None:
        """Restore state from a dict (as produced by to_dict). Unknown keys
        are ignored for forward compatibility; missing keys fall back to
        defaults."""
        lf = d.get("light_frames", {})
        self.light_files = lf.get("files", [])
        self.light_reference_index = lf.get("reference_index", 0)

        self.dark_files = d.get("dark_frames", {}).get("files", [])
        self.flat_files = d.get("flat_frames", {}).get("files", [])
        self.bias_files = d.get("bias_frames", {}).get("files", [])
        self.darkflat_files = d.get("darkflat_frames", {}).get("files", [])

        opts = copy.deepcopy(_DEFAULT_OPTIONS)
        saved_opts = d.get("options", {})
        opts.update({k: v for k, v in saved_opts.items() if k in opts})

        # Per-tab methods may also live in the frame sub-dicts in older files.
        for tab, key in (
            ("dark", "dark_method"),
            ("flat", "flat_method"),
            ("bias", "bias_method"),
            ("darkflat", "darkflat_method"),
        ):
            tab_method = d.get(f"{tab}_frames", {}).get("method")
            if tab_method and key not in saved_opts:
                opts[key] = tab_method

        self.options = opts

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load(cls, path: str) -> "ProjectState":
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        if not isinstance(d, dict):
            raise ValueError(f"Invalid project file: {path}")
        state = cls()
        state.from_dict(d)
        return state

    def reset(self) -> None:
        """Restore to a clean empty-project state."""
        self.light_files = []
        self.light_reference_index = 0
        self.dark_files = []
        self.flat_files = []
        self.bias_files = []
        self.darkflat_files = []
        self.options = copy.deepcopy(_DEFAULT_OPTIONS)
