#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility shim for legacy setuptools workflows.

The canonical build configuration now lives in ``pyproject.toml``.
This file only keeps custom non-package data discovery so ``python -m build``
and older ``setuptools`` entrypoints produce distributions with the runtime
resources that PyMeshGen still expects.
"""

from __future__ import annotations

from pathlib import Path

from setuptools import setup


PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIRS = (
    "config",
    "config/input",
    "docs",
    "docs/design",
    "docs/images",
    "3rd_party/meshio/src",
    "3rd_party/triangle",
)


def _collect_data_files() -> list[tuple[str, list[str]]]:
    data_files: list[tuple[str, list[str]]] = []
    for relative_dir in DATA_DIRS:
        source_dir = PROJECT_ROOT / relative_dir
        if not source_dir.exists():
            continue

        files = [
            str(path.relative_to(PROJECT_ROOT))
            for path in source_dir.rglob("*")
            if path.is_file()
        ]
        if files:
            data_files.append((relative_dir, files))

    meshio_license = PROJECT_ROOT / "3rd_party/meshio/LICENSE.txt"
    if meshio_license.exists():
        data_files.append(("3rd_party/meshio", [str(meshio_license.relative_to(PROJECT_ROOT))]))

    top_level_files = []
    for relative_file in ("LICENSE.txt", "VERSION", "README.md", "README_zh.md", "README_PACKAGING.md"):
        file_path = PROJECT_ROOT / relative_file
        if file_path.exists():
            top_level_files.append(str(file_path.relative_to(PROJECT_ROOT)))
    if top_level_files:
        data_files.append((".", top_level_files))

    return data_files


if __name__ == "__main__":
    setup(data_files=_collect_data_files())
