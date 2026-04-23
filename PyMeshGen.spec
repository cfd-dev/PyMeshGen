# -*- mode: python ; coding: utf-8 -*-
"""
PyMeshGen PyInstaller 打包配置文件
用于生成 PyMeshGen GUI 应用程序的独立可执行文件

使用方法:
    pyinstaller PyMeshGen.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_all

# 项目根目录
block_cipher = None
project_root = os.path.dirname(os.path.abspath(SPEC))

# 核心依赖 - 只收集必要的模块
hiddenimports = [
    # 核心依赖
    'numpy',
    'scipy',
    'matplotlib',
    'vtk',
    'PyQt5',
    'rtree',
    'PIL',

    # 项目模块
    'gui',
    'data_structure',
    'fileIO',
    'meshsize',
    'visualization',
    'adfront2',
    'optimize',
    'utils',
    'neural',
    'core',

    # PyQt5 相关
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.QtPrintSupport',
    'PyQt5.QtSvg',

    # 神经网络依赖
    'torch',
    'torch_geometric',
    'gym',
    'stable_baselines3',
    'trimesh',
    
    # 其他必要依赖
    'pandas',
    'requests',
    'urllib3',
    'charset_normalizer',
    'certifi',
    'idna',
    'six',
    'dateutil',
    'pytz',
    'cycler',
    'fonttools',
    'kiwisolver',
    'packaging',
    'pyparsing',
    'pillow',
    'contourpy',
]

# 数据文件收集
datas = [
    # 配置文件目录
    (os.path.join(project_root, 'config'), 'config'),
    (os.path.join(project_root, 'config', 'input'), 'config\\input'),

    # 文档和图标
    (os.path.join(project_root, 'docs'), 'docs'),

    # 第三方库
    (os.path.join(project_root, '3rd_party'), '3rd_party'),
]

# 收集项目中的数据文件
for subdir in ['fileIO', 'data_structure', 'meshsize', 'visualization',
               'adfront2', 'optimize', 'utils', 'neural', 'core', 'gui']:
    subdir_path = os.path.join(project_root, subdir)
    if os.path.exists(subdir_path):
        datas.append((subdir_path, subdir))

# 二进制文件
binaries = []

# 分析
a = Analysis(
    [os.path.join(project_root, 'start_gui.py')],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'jupyter_client',
        'ipykernel',
        'spyder',
        'pytest',
        'nose',
        'numpy.testing',
        'numpy.tests',
        'scipy.tests',
        'scipy.testing',
        'matplotlib.tests',
        'matplotlib.testing',
        'torch.testing',
        'torch.test',
        'pandas.tests',
        'pandas.testing',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 生成 PYZ 归档
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# 生成可执行文件
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PyMeshGen',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 设置为 False 以隐藏控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(project_root, 'docs', 'icon.ico') if os.path.exists(os.path.join(project_root, 'docs', 'icon.ico')) else None,
    version=None,
)

# 复制到 bin 目录
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PyMeshGen',
)
