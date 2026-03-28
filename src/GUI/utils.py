"""
utils.py — Shared helpers for the dso_stacker GUI.

Provides:
  format_size()          — human-readable byte count string
  detect_output_format() — infer output format from file extension
  build_command()        — construct the dso_stacker argv list from ProjectState
"""

from __future__ import annotations

import os
import sys
import platform
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from project import ProjectState

_FITS_EXTS  = {".fits", ".fit", ".fts"}
_TIFF_EXTS  = {".tiff", ".tif"}
_PNG_EXTS   = {".png"}


def format_size(n_bytes: int) -> str:
    """Return a human-readable file size string (e.g. '23.4 MB')."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024.0:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} PB"


def detect_output_format(path: str) -> str:
    """Return 'fits', 'tiff', 'png', or 'unknown' based on file extension."""
    ext = Path(path).suffix.lower()
    if ext in _FITS_EXTS:
        return "fits"
    if ext in _TIFF_EXTS:
        return "tiff"
    if ext in _PNG_EXTS:
        return "png"
    return "unknown"


def _binary_path() -> Path:
    """Resolve the dso_stacker binary.

    Search order:
      1. PyInstaller bundle: <exe_dir>/bin/dso_stacker[-cpu|-gpu|-metal][.exe]
      2. Development layout:  <repo>/build/dso_stacker[-cpu|-gpu|-metal][.exe]

    Raises FileNotFoundError if the binary is absent.
    """
    suffix = ".exe" if platform.system() == "Windows" else ""
    exe_names = [
        f"dso_stacker{suffix}",
        f"dso_stacker-cpu{suffix}",
        f"dso_stacker-gpu{suffix}",
        f"dso_stacker-metal{suffix}",
    ]
    searched: list[Path] = []

    # 1. PyInstaller / frozen bundle (one-dir mode)
    #    PyInstaller 6+ places collected binaries under _internal/.
    if getattr(sys, "frozen", False):
        bundle_dir = Path(sys.executable).parent
        for subdir in ("bin", "_internal/bin"):
            for exe_name in exe_names:
                candidate = bundle_dir / subdir / exe_name
                searched.append(candidate)
                if candidate.is_file():
                    return candidate

    # 2. Standard development layout: <repo>/build/dso_stacker*
    build_dir = Path(__file__).parent.parent.parent / "build"
    for exe_name in exe_names:
        candidate = build_dir / exe_name
        searched.append(candidate)
        if candidate.is_file():
            return candidate

    locations = "\n".join(f"  {p}" for p in searched)
    raise FileNotFoundError(
        f"dso_stacker binary not found.\nSearched:\n{locations}\n"
        "Please build the project first:\n"
        "  cmake --build build --parallel $(nproc)"
    )


def build_command(
    project: "ProjectState",
    csv_path: str,
    calib_paths: dict[str, str],
) -> list[str]:
    """Build the dso_stacker argv list.

    Parameters
    ----------
    project:      ProjectState holding all options.
    csv_path:     Absolute path to the temporary 2-column light-frames CSV.
    calib_paths:  Dict mapping 'dark'/'flat'/'bias'/'darkflat' to the path
                  of the temp text list (or single FITS path) for that tab.
                  Omit a key to skip that calibration argument.

    Raises
    ------
    FileNotFoundError  if the binary is not built yet.
    """
    binary = str(_binary_path())
    opts = project.options

    argv: list[str] = [binary, "-f", csv_path, "-o", opts["output_path"]]

    # --- execution ---
    if opts.get("use_cpu"):
        argv.append("--cpu")

    # --- integration ---
    argv += ["--integration", opts["integration"]]
    if opts["integration"] == "kappa-sigma":
        argv += ["--kappa", str(opts["kappa"])]
        argv += ["--iterations", str(opts["iterations"])]
    if not opts.get("use_cpu"):
        argv += ["--batch-size", str(opts["batch_size"])]

    # --- star detection (always used with the GUI's 2-column CSV) ---
    argv += ["--star-sigma",    str(opts["star_sigma"])]
    argv += ["--moffat-alpha",  str(opts["moffat_alpha"])]
    argv += ["--moffat-beta",   str(opts["moffat_beta"])]
    argv += ["--top-stars",     str(opts["top_stars"])]
    argv += ["--min-stars",     str(opts["min_stars"])]

    # --- triangle matching ---
    argv += ["--triangle-iters",  str(opts["triangle_iters"])]
    argv += ["--triangle-thresh", str(opts["triangle_thresh"])]
    argv += ["--match-radius",  str(opts["match_radius"])]
    if not opts.get("use_cpu") and opts.get("match_device", "auto") != "auto":
        argv += ["--match-device", opts["match_device"]]

    # --- calibration ---
    for key, flag in (("dark", "--dark"), ("flat", "--flat"),
                      ("bias", "--bias"), ("darkflat", "--darkflat")):
        p = calib_paths.get(key)
        if p:
            argv += [flag, p]

    argv += ["--save-master-frames", opts["save_master_dir"]]
    argv += ["--wsor-clip", str(opts["wsor_clip"])]
    for key, flag in (
        ("dark_method",     "--dark-method"),
        ("bias_method",     "--bias-method"),
        ("flat_method",     "--flat-method"),
        ("darkflat_method", "--darkflat-method"),
    ):
        if opts.get(key, "winsorized-mean") != "winsorized-mean":
            argv += [flag, opts[key]]

    # --- sensor ---
    if opts.get("bayer", "auto") != "auto":
        argv += ["--bayer", opts["bayer"]]

    # --- output format ---
    argv += ["--bit-depth", opts["bit_depth"]]
    fmt = detect_output_format(opts["output_path"])
    if fmt == "tiff" and opts.get("tiff_compression", "none") != "none":
        argv += ["--tiff-compression", opts["tiff_compression"]]
    if opts["bit_depth"] in ("8", "16"):
        if opts.get("stretch_min") is not None:
            argv += ["--stretch-min", str(opts["stretch_min"])]
        if opts.get("stretch_max") is not None:
            argv += ["--stretch-max", str(opts["stretch_max"])]

    return argv


def write_csv(light_files: list[str], reference_index: int, csv_path: str) -> None:
    """Write a 2-column CSV for the dso_stacker -f argument."""
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("filepath, is_reference\n")
        for i, path in enumerate(light_files):
            ref = 1 if i == reference_index else 0
            fh.write(f"{path}, {ref}\n")


def write_calib_list(file_paths: list[str], list_path: str) -> None:
    """Write a text-file list of FITS paths (one per line)."""
    with open(list_path, "w", encoding="utf-8") as fh:
        for p in file_paths:
            fh.write(p + "\n")
