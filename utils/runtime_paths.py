from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def find_resource_root(
    anchor_file: str,
    levels_up: int = 0,
    required_paths: Sequence[str] = (),
) -> Path:
    anchor_path = Path(anchor_file).resolve().parent
    module_root = anchor_path
    for _ in range(levels_up):
        module_root = module_root.parent

    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass))

    candidates.extend(
        [
            module_root,
            module_root.parent,
            Path(sys.prefix),
            Path(sys.base_prefix),
        ]
    )

    for candidate in _unique_paths(candidates):
        if not candidate.exists():
            continue
        if all((candidate / relative_path).exists() for relative_path in required_paths):
            return candidate

    return module_root


def add_existing_path(path: Path, prepend: bool = False) -> None:
    path_text = str(path)
    if not path.exists() or path_text in sys.path:
        return
    if prepend:
        sys.path.insert(0, path_text)
    else:
        sys.path.append(path_text)
