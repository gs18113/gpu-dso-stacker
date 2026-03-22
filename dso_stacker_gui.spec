# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the DSO Stacker GUI.

Produces a one-dir bundle at dist/DSOStacker/ containing:
  - DSOStacker[.exe]            GUI executable
  - bin/dso_stacker[.exe]       CLI binary (+ DLLs on Windows)

Usage (CI or local):
  1. Place the pre-built CLI binary in bundled_bin/ :
       bundled_bin/dso_stacker        (Linux)
       bundled_bin/dso_stacker.exe    (Windows, plus any .dll files)
  2. Run:  pyinstaller dso_stacker_gui.spec
"""

import os
import platform

# --- Collect CLI binary and any DLLs from bundled_bin/ ---
cli_binaries = []
bundled = 'bundled_bin'
if platform.system() == 'Windows':
    exe = os.path.join(bundled, 'dso_stacker.exe')
    if os.path.isfile(exe):
        cli_binaries.append((exe, 'bin'))
    for f in os.listdir(bundled):
        if f.lower().endswith('.dll'):
            cli_binaries.append((os.path.join(bundled, f), 'bin'))
else:
    exe = os.path.join(bundled, 'dso_stacker')
    if os.path.isfile(exe):
        cli_binaries.append((exe, 'bin'))

a = Analysis(
    ['src/GUI/main.py'],
    pathex=['src/GUI'],
    binaries=cli_binaries,
    datas=[],
    hiddenimports=[
        'PyQt6.sip',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DSOStacker',
    debug=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name='DSOStacker',
)
