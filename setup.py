#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMeshGen 安装配置脚本
用于通过 pip 安装 PyMeshGen 包

使用方法:
    # 开发模式安装
    pip install -e .
    
    # 正常安装
    pip install .
    
    # 构建分发包
    python setup.py sdist bdist_wheel
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# 获取项目根目录
project_root = Path(__file__).parent.absolute()

# 读取 README 文件作为长描述
readme_file = project_root / 'README_zh.md'
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')
else:
    long_description = "PyMeshGen - Python 非结构化网格生成器"

# 读取版本号
version_file = project_root / 'VERSION'
if version_file.exists():
    version = version_file.read_text(encoding='utf-8').strip()
else:
    version = "1.0.0"

# 读取依赖
requirements_file = project_root / 'requirements.txt'
if requirements_file.exists():
    requirements = []
    for line in requirements_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        # 跳过注释和空行
        if line and not line.startswith('#'):
            # 移除行内注释
            if '#' in line:
                line = line.split('#')[0].strip()
            if line:
                requirements.append(line)
else:
    requirements = []

# 项目元数据
setup(
    name='PyMeshGen',
    version=version,
    description='Python 非结构化网格生成器',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='PyMeshGen Team',
    author_email='pymeshgen@example.com',
    url='https://github.com/pymeshgen/PyMeshGen',
    license='MIT',
    
    # 包发现
    packages=find_packages(exclude=['tests', 'tests.*', 'build', 'dist']),
    
    # 包含包数据
    package_data={
        '': [
            '*.txt',
            '*.md',
            '*.json',
            '*.cas',
        ],
        'config': ['*'],
        'config/input': ['*'],
        'docs': ['*'],
        '3rd_party': ['**/*'],
    },
    
    # 包含额外文件
    data_files=[
        ('config', ['config/*.json']),
        ('config/input', ['config/input/*.cas']),
        ('docs', ['docs/*']),
    ],
    
    # 依赖
    install_requires=requirements,
    
    # 入口点
    entry_points={
        'console_scripts': [
            'pymeshgen=PyMeshGen:PyMeshGen',
            'pymeshgen-mixed=PyMeshGen:PyMeshGen_mixed',
        ],
        'gui_scripts': [
            'pymeshgen-gui=start_gui:main',
        ],
    },
    
    # Python 版本要求
    python_requires='>=3.8',
    
    # 分类
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    
    # 关键词
    keywords='mesh generation unstructured grid CFD numerical methods',
    
    # 项目链接
    project_urls={
        'Bug Reports': 'https://github.com/pymeshgen/PyMeshGen/issues',
        'Source': 'https://github.com/pymeshgen/PyMeshGen',
    },
)
