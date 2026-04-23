"""Triangle-backed Delaunay mesh generation via Triangle's CLI interface."""

from __future__ import annotations

import subprocess
import tempfile
from math import ceil, floor, sqrt
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils.runtime_paths import find_resource_root

try:
    from utils.geom_toolkit import point_in_polygon
except ModuleNotFoundError:
    from geom_toolkit import point_in_polygon


_RESOURCE_ROOT = find_resource_root(
    __file__,
    levels_up=1,
    required_paths=("3rd_party/triangle",),
)
_TRIANGLE_DIR = _RESOURCE_ROOT / "3rd_party" / "triangle"
_TRIANGLE_EXE = _TRIANGLE_DIR / "build" / "pymeshgen_triangle.exe"
_TRIANGLE_SOURCE = _TRIANGLE_DIR / "triangle.c"


def _candidate_vsdevcmd_paths() -> Iterable[Path]:
    yield Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat")
    yield Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
    yield Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat")
    yield Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
    yield Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat")
    yield Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat")


def _find_vsdevcmd() -> Optional[Path]:
    for path in _candidate_vsdevcmd_paths():
        if path.exists():
            return path
    return None


def _needs_rebuild(output_path: Path, sources: Sequence[Path]) -> bool:
    if not output_path.exists():
        return True
    output_mtime = output_path.stat().st_mtime
    return any(source.stat().st_mtime > output_mtime for source in sources)


def _build_triangle_exe() -> Path:
    build_dir = _TRIANGLE_EXE.parent
    build_dir.mkdir(parents=True, exist_ok=True)

    vsdevcmd = _find_vsdevcmd()
    if vsdevcmd is None:
        raise RuntimeError("未找到 Visual Studio C 编译环境，无法构建 Triangle 可执行文件")

    build_script = build_dir / "build_triangle.cmd"
    build_script.write_text(
        "\n".join(
            [
                "@echo off",
                f'call "{vsdevcmd}" -arch=x86 -host_arch=x64 >nul',
                (
                    "cl /nologo /O2 /DNO_TIMER /DCPU86 /D_CRT_SECURE_NO_WARNINGS "
                    f'/Fe:"{_TRIANGLE_EXE}" "{_TRIANGLE_SOURCE}"'
                ),
            ]
        ),
        encoding="ascii",
    )

    result = subprocess.run(
        ["cmd.exe", "/c", str(build_script)],
        cwd=str(build_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not _TRIANGLE_EXE.exists():
        raise RuntimeError(
            "Triangle 可执行文件构建失败:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return _TRIANGLE_EXE


def _ensure_triangle_exe() -> Path:
    sources = (_TRIANGLE_SOURCE, Path(__file__))
    if _needs_rebuild(_TRIANGLE_EXE, sources):
        return _build_triangle_exe()
    return _TRIANGLE_EXE


def _iter_leaf_nodes(nodes) -> Iterable[object]:
    for node in nodes:
        children = getattr(node, "children", None)
        if children:
            yield from _iter_leaf_nodes(children)
        else:
            yield node


def _point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-20:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    projection = a + t * ab
    return float(np.linalg.norm(point - projection))


def _point_in_domain(
    point: np.ndarray,
    outer_boundary: Optional[np.ndarray],
    holes: Sequence[np.ndarray],
) -> bool:
    if outer_boundary is not None and not point_in_polygon(point, outer_boundary):
        return False
    return not any(point_in_polygon(point, hole) for hole in holes)


def _compute_hole_seed(hole: np.ndarray) -> np.ndarray:
    centroid = np.mean(hole, axis=0)
    if point_in_polygon(centroid, hole):
        return centroid
    return np.mean(hole[: min(len(hole), 3)], axis=0)


def _normalize_point_strategy(point_strategy: str) -> str:
    normalized = str(point_strategy).strip().lower()
    if normalized in {"equilateral", "triangular", "hexagonal", "hex"}:
        return "equilateral"
    if normalized in {"cartesian", "grid", "rectilinear"}:
        return "cartesian"
    return "equilateral"


def _candidate_boundary_distance(
    candidate: np.ndarray,
    boundary_points: np.ndarray,
    boundary_edges: Sequence[Tuple[int, int]],
) -> float:
    min_boundary_distance = float("inf")
    for edge_start, edge_end in boundary_edges:
        dist = _point_to_segment_distance(
            candidate,
            boundary_points[edge_start],
            boundary_points[edge_end],
        )
        min_boundary_distance = min(min_boundary_distance, dist)
    return min_boundary_distance


def _accept_candidate(
    candidate: np.ndarray,
    sizing_system,
    outer_boundary: Optional[np.ndarray],
    holes: Sequence[np.ndarray],
    boundary_points: np.ndarray,
    boundary_edges: Sequence[Tuple[int, int]],
    accepted: List[np.ndarray],
    accepted_sizes: List[float],
    min_spacing_factor: float,
) -> Optional[float]:
    if not _point_in_domain(candidate, outer_boundary, holes):
        return None

    local_size = max(float(sizing_system.spacing_at(candidate)), 1e-8)
    min_boundary_distance = _candidate_boundary_distance(
        candidate,
        boundary_points,
        boundary_edges,
    )
    if min_boundary_distance < 0.3 * local_size:
        return None

    for existing, existing_size in zip(accepted, accepted_sizes):
        if np.linalg.norm(candidate - existing) < min_spacing_factor * min(
            local_size,
            existing_size,
        ):
            return None

    accepted.append(candidate)
    accepted_sizes.append(local_size)
    return local_size


def _iter_sorted_leaf_bounds(sizing_system) -> List[Tuple[Tuple[float, float, float, float], float]]:
    leaf_records = []
    for leaf in _iter_leaf_nodes(sizing_system.quad_tree):
        x_min, y_min, x_max, y_max = leaf.bounds
        center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=float)
        local_size = max(float(sizing_system.spacing_at(center)), 1e-8)
        leaf_records.append((leaf.bounds, local_size))
    leaf_records.sort(key=lambda item: item[1])
    return leaf_records


def _sample_interior_points_cartesian(
    sizing_system,
    outer_boundary: Optional[np.ndarray],
    holes: Sequence[np.ndarray],
    boundary_points: np.ndarray,
    boundary_edges: Sequence[Tuple[int, int]],
) -> np.ndarray:
    accepted: List[np.ndarray] = []
    accepted_sizes: List[float] = []

    for bounds, local_size in _iter_sorted_leaf_bounds(sizing_system):
        x_min, y_min, x_max, y_max = bounds
        dx = x_max - x_min
        dy = y_max - y_min
        candidate_points = [
            np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=float)
        ]
        if max(dx, dy) > 1.5 * local_size:
            candidate_points.extend(
                [
                    np.array([x_min + 0.25 * dx, y_min + 0.25 * dy], dtype=float),
                    np.array([x_min + 0.75 * dx, y_min + 0.75 * dy], dtype=float),
                ]
            )

        for candidate in candidate_points:
            _accept_candidate(
                candidate=candidate,
                sizing_system=sizing_system,
                outer_boundary=outer_boundary,
                holes=holes,
                boundary_points=boundary_points,
                boundary_edges=boundary_edges,
                accepted=accepted,
                accepted_sizes=accepted_sizes,
                min_spacing_factor=0.45,
            )

    if not accepted:
        return np.empty((0, 2), dtype=float)
    return np.asarray(accepted, dtype=float)


def _sample_interior_points_equilateral(
    sizing_system,
    outer_boundary: Optional[np.ndarray],
    holes: Sequence[np.ndarray],
    boundary_points: np.ndarray,
    boundary_edges: Sequence[Tuple[int, int]],
) -> np.ndarray:
    accepted: List[np.ndarray] = []
    accepted_sizes: List[float] = []
    if hasattr(sizing_system, "bg_bounds") and len(sizing_system.bg_bounds) == 4:
        origin_x = float(sizing_system.bg_bounds[0])
        origin_y = float(sizing_system.bg_bounds[1])
    elif len(boundary_points) > 0:
        origin_x = float(np.min(boundary_points[:, 0]))
        origin_y = float(np.min(boundary_points[:, 1]))
    else:
        origin_x = 0.0
        origin_y = 0.0

    eps = 1e-12
    for bounds, local_size in _iter_sorted_leaf_bounds(sizing_system):
        x_min, y_min, x_max, y_max = bounds
        row_step = 0.5 * sqrt(3.0) * local_size
        if row_step <= eps:
            continue

        accepted_in_leaf = 0
        row_min = floor((y_min - origin_y) / row_step) - 1
        row_max = ceil((y_max - origin_y) / row_step) + 1

        for row_idx in range(row_min, row_max + 1):
            y = origin_y + row_idx * row_step
            if y < y_min - eps or y > y_max + eps:
                continue

            row_offset = 0.5 * local_size if (row_idx & 1) else 0.0
            col_min = floor((x_min - origin_x - row_offset) / local_size) - 1
            col_max = ceil((x_max - origin_x - row_offset) / local_size) + 1

            for col_idx in range(col_min, col_max + 1):
                x = origin_x + row_offset + col_idx * local_size
                if x < x_min - eps or x > x_max + eps:
                    continue

                inserted = _accept_candidate(
                    candidate=np.array([x, y], dtype=float),
                    sizing_system=sizing_system,
                    outer_boundary=outer_boundary,
                    holes=holes,
                    boundary_points=boundary_points,
                    boundary_edges=boundary_edges,
                    accepted=accepted,
                    accepted_sizes=accepted_sizes,
                    min_spacing_factor=0.60,
                )
                if inserted is not None:
                    accepted_in_leaf += 1

        if accepted_in_leaf == 0:
            _accept_candidate(
                candidate=np.array(
                    [(x_min + x_max) * 0.5, (y_min + y_max) * 0.5],
                    dtype=float,
                ),
                sizing_system=sizing_system,
                outer_boundary=outer_boundary,
                holes=holes,
                boundary_points=boundary_points,
                boundary_edges=boundary_edges,
                accepted=accepted,
                accepted_sizes=accepted_sizes,
                min_spacing_factor=0.45,
            )

    if not accepted:
        return np.empty((0, 2), dtype=float)
    return np.asarray(accepted, dtype=float)


def _sample_interior_points(
    sizing_system,
    outer_boundary: Optional[np.ndarray],
    holes: Sequence[np.ndarray],
    boundary_points: np.ndarray,
    boundary_edges: Sequence[Tuple[int, int]],
    point_strategy: str,
) -> np.ndarray:
    strategy = _normalize_point_strategy(point_strategy)
    if strategy == "cartesian":
        return _sample_interior_points_cartesian(
            sizing_system=sizing_system,
            outer_boundary=outer_boundary,
            holes=holes,
            boundary_points=boundary_points,
            boundary_edges=boundary_edges,
        )
    return _sample_interior_points_equilateral(
        sizing_system=sizing_system,
        outer_boundary=outer_boundary,
        holes=holes,
        boundary_points=boundary_points,
        boundary_edges=boundary_edges,
    )


def _write_poly_file(
    poly_path: Path,
    all_points: np.ndarray,
    boundary_point_count: int,
    boundary_edges: Sequence[Tuple[int, int]],
    holes: Sequence[np.ndarray],
) -> None:
    lines = [f"{len(all_points)} 2 0 1"]
    for idx, point in enumerate(all_points):
        marker = 1 if idx < boundary_point_count else 0
        lines.append(f"{idx} {point[0]:.16g} {point[1]:.16g} {marker}")

    lines.append(f"{len(boundary_edges)} 1")
    for idx, (a, b) in enumerate(boundary_edges):
        lines.append(f"{idx} {a} {b} 1")

    lines.append(str(len(holes)))
    for idx, hole in enumerate(holes):
        seed = _compute_hole_seed(hole)
        lines.append(f"{idx} {seed[0]:.16g} {seed[1]:.16g}")

    lines.append("0")
    poly_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _parse_node_file(node_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    lines = [
        line.strip()
        for line in node_path.read_text(encoding="ascii").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    number_of_points, _, _, has_markers = map(int, lines[0].split()[:4])
    points = np.zeros((number_of_points, 2), dtype=float)
    boundary_mask = np.zeros(number_of_points, dtype=bool)

    for line in lines[1 : 1 + number_of_points]:
        parts = line.split()
        idx = int(parts[0])
        points[idx, 0] = float(parts[1])
        points[idx, 1] = float(parts[2])
        if has_markers:
            boundary_mask[idx] = int(parts[-1]) > 0

    return points, boundary_mask


def _parse_ele_file(ele_path: Path) -> np.ndarray:
    lines = [
        line.strip()
        for line in ele_path.read_text(encoding="ascii").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    number_of_triangles, corners_per_triangle, _ = map(int, lines[0].split()[:3])
    simplices = np.zeros((number_of_triangles, 3), dtype=np.int32)

    for line in lines[1 : 1 + number_of_triangles]:
        parts = line.split()
        idx = int(parts[0])
        simplices[idx, :] = [int(parts[i]) for i in range(1, 1 + min(corners_per_triangle, 3))]

    return simplices


def _triangle_switches() -> str:
    return "-pq20zQ"


def create_triangle_mesh(
    boundary_points: np.ndarray,
    boundary_edges: Sequence[Tuple[int, int]],
    sizing_system,
    holes: Optional[List[np.ndarray]] = None,
    outer_boundary: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    point_strategy: str = "equilateral",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a mesh using Jonathan Shewchuk's Triangle executable."""
    del seed

    holes = holes or []
    triangle_exe = _ensure_triangle_exe()

    interior_points = _sample_interior_points(
        sizing_system=sizing_system,
        outer_boundary=outer_boundary,
        holes=holes,
        boundary_points=boundary_points,
        boundary_edges=boundary_edges,
        point_strategy=point_strategy,
    )

    all_points = np.asarray(boundary_points, dtype=float)
    if interior_points.size > 0:
        all_points = np.vstack([all_points, interior_points])

    with tempfile.TemporaryDirectory(prefix="pymeshgen-triangle-") as temp_dir:
        temp_path = Path(temp_dir)
        poly_path = temp_path / "mesh.poly"
        _write_poly_file(
            poly_path=poly_path,
            all_points=all_points,
            boundary_point_count=len(boundary_points),
            boundary_edges=boundary_edges,
            holes=holes,
        )

        result = subprocess.run(
            [str(triangle_exe), _triangle_switches(), poly_path.name],
            cwd=str(temp_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Triangle 运行失败:\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        node_path = temp_path / "mesh.1.node"
        ele_path = temp_path / "mesh.1.ele"
        if not node_path.exists() or not ele_path.exists():
            raise RuntimeError(
                "Triangle 未生成预期输出文件:\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        points, boundary_mask = _parse_node_file(node_path)
        simplices = _parse_ele_file(ele_path)

    return points, simplices, boundary_mask
