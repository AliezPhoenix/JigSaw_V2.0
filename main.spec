# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files

# 收集qt_material的数据文件（fonts、themes、resources等）
qt_material_datas = collect_data_files('qt_material')
# 确保 PyInstaller 能找到项目根目录下的模块
spec_root = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    ['main.py'],
    pathex=[spec_root],
    binaries=[],
    datas=qt_material_datas,
    hiddenimports=['main_window', 'ImageViewerWidget', 'LogViewerWidget', 'DryPramasSetDialog', 'TransferPramasSetDialog','qt_material'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='JigSaw_v2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['gzp7z-4ntfo-001.ico'],
)
