"""
fits_meta.py — Asynchronous image metadata reader (FITS and RAW).

Reads width, height, and the optional Bayer pattern from FITS or RAW camera
files.  FITS uses a minimal pure-Python header parser; RAW files use the
optional ``rawpy`` package (Python wrapper for LibRaw).

Runs in a QThreadPool worker so the UI thread is never blocked.

FITS header format (FITS Standard 4.0 §4):
  - Cards are 80 bytes each: "KEYWORD = VALUE / comment"
  - Packed 36 cards per 2880-byte block
  - Header ends at the first card whose keyword is "END"
"""

from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

_BLOCK = 2880
_CARD  = 80

_RAW_EXTS = {
    ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".dng",
    ".pef", ".srw", ".raw", ".3fr", ".iiq", ".rwl", ".nrw",
}

# Optional rawpy import for RAW metadata reading
try:
    import rawpy as _rawpy  # type: ignore[import-untyped]
    _HAS_RAWPY = True
except ImportError:
    _HAS_RAWPY = False


def _is_raw(path: str) -> bool:
    return Path(path).suffix.lower() in _RAW_EXTS


# Bayer pattern mapping from rawpy's 2x2 raw_pattern to string
_RAWPY_BAYER_MAP = {
    (0, 1, 1, 2): "RGGB",
    (2, 1, 1, 0): "BGGR",
    (1, 0, 2, 1): "GRBG",
    (1, 2, 0, 1): "GBRG",
}


def _read_raw_meta(path: str) -> dict[str, str]:
    """Read width, height, and Bayer pattern from a RAW camera file.

    Uses rawpy (LibRaw wrapper).  Returns empty dict on error or if
    rawpy is not installed.
    """
    if not _HAS_RAWPY:
        return {}
    try:
        with _rawpy.imread(path) as raw:
            found: dict[str, str] = {}
            found["NAXIS1"] = str(raw.sizes.width)
            found["NAXIS2"] = str(raw.sizes.height)
            # raw_pattern is a 2x2 numpy array for Bayer sensors
            try:
                pat = raw.raw_pattern[:2, :2]
                key = tuple(pat.flat)
                bayer = _RAWPY_BAYER_MAP.get(key)
                if bayer:
                    found["BAYERPAT"] = bayer
            except (AttributeError, IndexError):
                pass
            return found
    except Exception:
        return {}


def _read_fits_keywords(path: str, want: set[str]) -> dict[str, str]:
    """Return a dict of {keyword: raw_value_string} for the requested keys.

    Reads only the primary HDU header blocks; stops at END or EOF.
    Returns an empty dict on any I/O or format error.
    """
    found: dict[str, str] = {}
    try:
        with open(path, "rb") as fh:
            while True:
                block = fh.read(_BLOCK)
                if len(block) < _BLOCK:
                    break
                for i in range(36):
                    card = block[i * _CARD:(i + 1) * _CARD].decode("ascii", errors="replace")
                    key = card[:8].rstrip()
                    if key == "END":
                        return found
                    if key in want and "=" in card[8:10]:
                        # Value field is card[10:80]; strip whitespace and quotes
                        raw = card[10:80].split("/")[0].strip().strip("'").strip()
                        found[key] = raw
                        if len(found) == len(want):
                            return found
    except Exception:
        pass
    return found


class _MetaSignals(QObject):
    """Carrier object for signals, since QRunnable doesn't inherit QObject."""
    result = Signal(int, dict)   # (row, meta_dict)


class FitsMetaWorker(QRunnable):
    """Read FITS primary HDU header in a thread-pool worker.

    On completion emits ``signals.result(row, meta)`` where *meta* is::

        {"width": int|None, "height": int|None, "bayer": str|None}

    The signal is queued automatically because the receiver lives on
    the main thread (Qt's default cross-thread connection type).
    """

    def __init__(self, row: int, path: str) -> None:
        super().__init__()
        self.row = row
        self.path = path
        self.signals = _MetaSignals()

    @Slot()
    def run(self) -> None:
        meta: dict = {"width": None, "height": None, "bayer": None}
        if _is_raw(self.path):
            kw = _read_raw_meta(self.path)
        else:
            kw = _read_fits_keywords(self.path, {"NAXIS1", "NAXIS2", "BAYERPAT"})
        try:
            meta["width"]  = int(kw["NAXIS1"])
        except (KeyError, ValueError):
            pass
        try:
            meta["height"] = int(kw["NAXIS2"])
        except (KeyError, ValueError):
            pass
        if "BAYERPAT" in kw:
            meta["bayer"] = kw["BAYERPAT"].upper()
        self.signals.result.emit(self.row, meta)
