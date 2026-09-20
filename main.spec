# -*- mode: python ; coding: utf-8 -*-
import os

# 确保 PyInstaller 能找到项目根目录下的模块
spec_root = os.path.dirname(os.path.abspath(SPEC))


def collect_mvs_binaries():
    """打包指定 MVS Runtime 目录下的全部 DLL。"""
    runtime_dir = r'C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64'
    if not os.path.isdir(runtime_dir):
        raise SystemExit(f'未找到 MVS Runtime 目录: {runtime_dir}')

    binaries = []
    for name in sorted(os.listdir(runtime_dir)):
        if name.lower().endswith('.dll'):
            binaries.append((os.path.join(runtime_dir, name), '.'))

    if not binaries:
        raise SystemExit(f'MVS Runtime 目录下没有 DLL: {runtime_dir}')

    print('MVS Runtime binaries from:', runtime_dir, 'count=', len(binaries))
    for src, _ in binaries:
        print('  +', os.path.basename(src))
    return binaries


mvs_binaries = collect_mvs_binaries()
mvs_dll_names = [os.path.basename(src) for src, _ in mvs_binaries]

a = Analysis(
    ['main.py'],
    pathex=[spec_root],
    binaries=mvs_binaries,
    datas=[],
    hiddenimports=['main_window', 'ImageViewerWidget', 'LogViewerWidget', 'DryPramasSetDialog', 'TransferPramasSetDialog', 'ui.theme'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.sip',
        'PySide2',
        'PySide6',
    ],
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
    name='JigSaw_v2.1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=mvs_dll_names,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['gzp7z-4ntfo-001.ico'],
)
