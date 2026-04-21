import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


def resolve_case_input_path(input_file_str, project_root, fallback_input_dir=None):
    input_file = Path(input_file_str)
    if input_file.is_absolute():
        return input_file
    if input_file_str.startswith("./unittests") or input_file_str.startswith("./config"):
        return (project_root / input_file).resolve()
    if fallback_input_dir is not None:
        return (fallback_input_dir / input_file.name).resolve()
    return (project_root / input_file).resolve()


def create_delaunay_case_config(
    original_config_path,
    output_file,
    project_root,
    enable_boundary_layer=False,
    delaunay_backend="bowyer_watson",
    fallback_input_dir=None,
):
    with open(original_config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    config["mesh_type"] = 4
    config["delaunay_backend"] = delaunay_backend

    if enable_boundary_layer:
        print("  - 边界层: 启用")
    else:
        print("  - 边界层: 禁用")
        for part in config.get("parts", []):
            part["PRISM_SWITCH"] = "off"
            part["max_layers"] = 0

    if "input_file" in config:
        config["input_file"] = str(
            resolve_case_input_path(
                config["input_file"],
                project_root=project_root,
                fallback_input_dir=fallback_input_dir,
            )
        )

    config["output_file"] = str(output_file)
    config["viz_enabled"] = False
    config["debug_level"] = 0

    prefix = "temp_triangle" if delaunay_backend == "triangle" else "temp_bw"
    temp_config_path = project_root / f"{prefix}_{original_config_path.stem}.json"
    with open(temp_config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4, ensure_ascii=False)

    return temp_config_path


def run_delaunay_config_test(
    testcase,
    original_config,
    output_file,
    project_root,
    test_name,
    enable_boundary_layer,
    delaunay_backend="bowyer_watson",
    fallback_input_dir=None,
    check_boundary_recovery=True,
):
    from PyMeshGen import PyMeshGen
    from data_structure.parameters import Parameters
    from fileIO.vtk_io import parse_vtk_msh
    from utils.message import DEBUG_LEVEL_VERBOSE, set_debug_level

    if not original_config.exists():
        testcase.skipTest(f"{original_config.name} 不存在")

    with open(original_config, "r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    set_debug_level(DEBUG_LEVEL_VERBOSE)

    try:
        bw_config = create_delaunay_case_config(
            original_config_path=original_config,
            output_file=output_file,
            project_root=project_root,
            enable_boundary_layer=enable_boundary_layer,
            delaunay_backend=delaunay_backend,
            fallback_input_dir=fallback_input_dir,
        )

        print(f"\n{test_name}:")
        print(f"  - 配置文件: {bw_config.name}")
        print(f"  - 输出文件: {output_file.name}")

        start = time.time()
        parameters = Parameters("FROM_CASE_JSON", str(bw_config))
        PyMeshGen(parameters)
        cost = time.time() - start

        testcase.assertTrue(output_file.exists(), "输出文件应该存在")
        grid = parse_vtk_msh(str(output_file))

        print(f"  - 生成时间: {cost:.2f}秒")
        print(f"  - 节点数: {grid.num_nodes}")
        print(f"  - 单元数: {grid.num_cells}")

        testcase.assertGreater(grid.num_nodes, 0, "节点数应大于 0")
        testcase.assertGreater(grid.num_cells, 0, "单元数应大于 0")
        testcase.assertLess(cost, 120, "生成时间应小于 120 秒")

        tri_count = sum(1 for cell in grid.cells if len(cell) == 3)
        quad_count = sum(1 for cell in grid.cells if len(cell) == 4)
        other_count = grid.num_cells - tri_count - quad_count

        print(f"  - 三角形数: {tri_count}")
        print(f"  - 四边形数: {quad_count}")
        print(f"  - 其他单元: {other_count}")

        if enable_boundary_layer:
            print(f"  - 模式: {delaunay_backend} + 边界层")
            testcase.assertGreater(tri_count, 0, "应该有三角形单元")
        else:
            print(f"  - 模式: 纯 {delaunay_backend} 三角网格")
            testcase.assertEqual(tri_count, grid.num_cells, "无边界层时应全部是三角形单元")

        if check_boundary_recovery:
            input_file = resolve_case_input_path(
                config_data.get("input_file", ""),
                project_root=project_root,
                fallback_input_dir=fallback_input_dir,
            )
            if input_file.exists():
                assert_boundary_recovery(testcase, input_file, grid, test_name)
            else:
                print(f"\n  - [SKIP] CAS 文件不存在: {input_file}，跳过边界恢复检查")

        print(f"  - [PASS] {test_name} 测试通过")
    except Exception as exc:
        print(f"  - [FAIL] {test_name} 测试失败: {exc}")
        import traceback

        traceback.print_exc()
        testcase.fail(f"{test_name} 测试失败: {exc}")


def assert_boundary_recovery(testcase, cas_file, grid, test_name):
    print("\n  - 边界恢复检查:")
    boundary_edges_result = check_boundary_edges(str(cas_file), grid, test_name)

    if boundary_edges_result["pass"]:
        print("  - [PASS] 边界恢复检查通过")
        for zone_key, zone_result in boundary_edges_result["zone_results"].items():
            print(f"    - {zone_key}: {zone_result['total_edges']}/{zone_result['total_edges']} 条边恢复")
            if zone_result["inner_boundary_inner_points"] > 0:
                print(f"      警告: 内边界内部发现 {zone_result['inner_boundary_inner_points']} 个点")
            if zone_result["inner_boundary_inner_cells"] > 0:
                print(f"      警告: 内边界内部发现 {zone_result['inner_boundary_inner_cells']} 个单元")
    else:
        print("  - [FAIL] 边界恢复检查失败")
        for zone_key, zone_result in boundary_edges_result["zone_results"].items():
            if zone_result["missing_edges"] > 0:
                print(f"    - {zone_key}: {zone_result['missing_edges']}/{zone_result['total_edges']} 条边丢失")
                for detail in zone_result["missing_details"][:5]:
                    n1, n2, coord1, coord2, reason = detail
                    if reason:
                        print(f"      边 ({n1},{n2}): {reason}")
                    else:
                        print(
                            f"      边 ({n1},{n2}): "
                            f"({coord1[0]:.4f},{coord1[1]:.4f}) -> ({coord2[0]:.4f},{coord2[1]:.4f})"
                        )
        if boundary_edges_result["issue"]:
            print(f"    - 问题: {boundary_edges_result['issue']}")
        testcase.fail(
            f"{test_name} 边界恢复检查失败: {boundary_edges_result['issue'] or '边界边丢失'}"
        )

    print("\n  - 孔洞清理检查:")
    hole_cleanup_result = check_hole_cleanup(str(cas_file), grid, test_name)
    if hole_cleanup_result["pass"]:
        print("  - [PASS] 孔洞清理检查通过")
        for hole_key, hole_result in hole_cleanup_result.get("hole_results", {}).items():
            print(
                f"    - {hole_key}: 内部点数={hole_result['points_inside']}, "
                f"内部单元数={hole_result['cells_inside']}"
            )
    else:
        print("  - [FAIL] 孔洞清理检查失败")
        for hole_key, hole_result in hole_cleanup_result.get("hole_results", {}).items():
            if hole_result["points_inside"] > 0 or hole_result["cells_inside"] > 0:
                print(
                    f"    - {hole_key}: 内部残留 {hole_result['points_inside']} 个点, "
                    f"{hole_result['cells_inside']} 个单元"
                )
        if hole_cleanup_result.get("issue"):
            print(f"    - 问题: {hole_cleanup_result['issue']}")
        testcase.fail(
            f"{test_name} 孔洞清理检查失败: "
            f"{hole_cleanup_result.get('issue', '孔洞内残留点或单元')}"
        )

    print("\n  - 拓扑洁净检查:")
    topology_result = check_topology_clean(grid, test_name)
    if topology_result["pass"]:
        print("  - [PASS] 拓扑洁净检查通过")
    else:
        print("  - [FAIL] 拓扑洁净检查失败")
        if topology_result.get("issue"):
            print(f"    - 问题: {topology_result['issue']}")
        testcase.fail(
            f"{test_name} 拓扑洁净检查失败: {topology_result.get('issue', '拓扑异常')}"
        )


def _point_segment_distance(point, seg_start, seg_end):
    edge_vec = seg_end - seg_start
    seg_len2 = float(np.dot(edge_vec, edge_vec))
    if seg_len2 < 1e-16:
        return float(np.linalg.norm(point - seg_start))
    t = float(np.dot(point - seg_start, edge_vec) / seg_len2)
    t = max(0.0, min(1.0, t))
    proj = seg_start + t * edge_vec
    return float(np.linalg.norm(point - proj))


def _has_split_edge_path(v_start, v_end, seg_start, seg_end, vtk_nodes, mesh_adjacency, tolerance):
    if v_start == v_end:
        return True

    edge_vec = seg_end - seg_start
    seg_len2 = float(np.dot(edge_vec, edge_vec))
    if seg_len2 < 1e-16:
        return False

    line_tolerance = tolerance * 1.5
    projection_margin = 0.2
    candidate_nodes = {v_start, v_end}

    for idx, coord in enumerate(vtk_nodes[:, :2]):
        t = float(np.dot(coord - seg_start, edge_vec) / seg_len2)
        if -projection_margin <= t <= 1.0 + projection_margin:
            if _point_segment_distance(coord, seg_start, seg_end) <= line_tolerance:
                candidate_nodes.add(idx)

    queue = deque([(v_start, 0)])
    visited = {v_start}
    while queue:
        current, depth = queue.popleft()
        if depth >= 24:
            continue
        for neighbor in mesh_adjacency.get(current, ()):
            if neighbor not in candidate_nodes or neighbor in visited:
                continue
            if neighbor == v_end:
                return True
            visited.add(neighbor)
            queue.append((neighbor, depth + 1))
    return False


def _has_zone_conforming_path(v_start, v_end, allowed_nodes, mesh_adjacency, max_depth=120):
    if v_start == v_end:
        return True
    if v_start not in allowed_nodes or v_end not in allowed_nodes:
        return False

    queue = deque([(v_start, 0)])
    visited = {v_start}
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in mesh_adjacency.get(current, ()):
            if neighbor not in allowed_nodes or neighbor in visited:
                continue
            if neighbor == v_end:
                return True
            visited.add(neighbor)
            queue.append((neighbor, depth + 1))
    return False


def _build_ordered_polygon(cas_nodes, edges):
    if len(edges) < 3:
        return None

    adjacency = {}
    for n1, n2 in edges:
        adjacency.setdefault(n1, []).append(n2)
        adjacency.setdefault(n2, []).append(n1)

    ordered_coords = []
    visited_edges = set()
    start_node = list(adjacency.keys())[0]
    current = start_node

    while True:
        ordered_coords.append(cas_nodes[current])
        next_node = None
        for neighbor in adjacency.get(current, []):
            edge_key = tuple(sorted([current, neighbor]))
            if edge_key not in visited_edges:
                next_node = neighbor
                break
        if next_node is None:
            break
        visited_edges.add(tuple(sorted([current, next_node])))
        current = next_node
        if current == start_node:
            break

    if len(ordered_coords) < 3:
        return None
    return np.array(ordered_coords)


def check_boundary_edges(cas_file, grid, test_name):
    from fileIO.read_cas import parse_fluent_msh
    from utils.geom_toolkit import point_in_polygon

    try:
        cas_data = parse_fluent_msh(cas_file)
    except Exception as exc:
        return {"pass": False, "issue": f"无法解析 CAS 文件: {exc}", "zone_results": {}}

    cas_nodes = np.array(cas_data["nodes"])
    vtk_nodes = np.array(grid.node_coords)
    tolerance = 0.01

    boundary_edges_by_zone = {}
    for face in cas_data["faces"]:
        part_name = face.get("part_name", "unknown")
        bc_type = face.get("bc_type", "unknown")
        if bc_type == "interior":
            continue
        zone_key = f"{part_name}_{bc_type}"
        boundary_edges_by_zone.setdefault(
            zone_key,
            {"edges": [], "nodes": set(), "bc_type": bc_type},
        )
        if len(face["nodes"]) == 2:
            n1, n2 = face["nodes"][0] - 1, face["nodes"][1] - 1
            boundary_edges_by_zone[zone_key]["edges"].append((n1, n2))
            boundary_edges_by_zone[zone_key]["nodes"].update((n1, n2))

    mesh_edges = set()
    mesh_adjacency = {}
    for cell in grid.cells:
        for i in range(len(cell)):
            a = int(cell[i])
            b = int(cell[(i + 1) % len(cell)])
            edge_key = (min(a, b), max(a, b))
            mesh_edges.add(edge_key)
            mesh_adjacency.setdefault(a, set()).add(b)
            mesh_adjacency.setdefault(b, set()).add(a)

    global_x_min, global_x_max = np.min(cas_nodes[:, 0]), np.max(cas_nodes[:, 0])
    global_y_min, global_y_max = np.min(cas_nodes[:, 1]), np.max(cas_nodes[:, 1])

    all_results = {}
    all_pass = True
    for zone_key, zone_info in boundary_edges_by_zone.items():
        cas_edges = zone_info["edges"]
        zone_node_indices = zone_info["nodes"]
        zone_coords = cas_nodes[list(zone_node_indices)]

        x_min, x_max = np.min(zone_coords[:, 0]), np.max(zone_coords[:, 0])
        y_min, y_max = np.min(zone_coords[:, 1]), np.max(zone_coords[:, 1])
        touches_global_boundary = (
            abs(x_min - global_x_min) < tolerance
            or abs(x_max - global_x_max) < tolerance
            or abs(y_min - global_y_min) < tolerance
            or abs(y_max - global_y_max) < tolerance
        )
        is_inner_boundary = not touches_global_boundary

        vtk_node_map = {}
        candidate_pairs = []
        for cas_idx in zone_node_indices:
            cas_coord = cas_nodes[cas_idx]
            dists = np.sqrt(np.sum((vtk_nodes[:, :2] - cas_coord[:2]) ** 2, axis=1))
            for vtk_idx in np.argsort(dists)[:8]:
                dist = dists[vtk_idx]
                if dist < tolerance:
                    candidate_pairs.append((float(dist), cas_idx, int(vtk_idx)))

        candidate_pairs.sort(key=lambda item: item[0])
        assigned_cas = set()
        assigned_vtk = set()
        for _, cas_idx, vtk_idx in candidate_pairs:
            if cas_idx in assigned_cas or vtk_idx in assigned_vtk:
                continue
            assigned_cas.add(cas_idx)
            assigned_vtk.add(vtk_idx)
            vtk_node_map[cas_idx] = vtk_idx

        for cas_idx in zone_node_indices:
            if cas_idx in vtk_node_map:
                continue
            cas_coord = cas_nodes[cas_idx]
            dists = np.sqrt(np.sum((vtk_nodes[:, :2] - cas_coord[:2]) ** 2, axis=1))
            nearest_idx = int(np.argmin(dists))
            if dists[nearest_idx] < tolerance:
                vtk_node_map[cas_idx] = nearest_idx

        zone_segments = [(cas_nodes[a][:2], cas_nodes[b][:2]) for a, b in cas_edges]
        zone_allowed_nodes = set(vtk_node_map.values())
        zone_band_tolerance = max(tolerance * 2.0, 0.03)
        for vtk_idx, vtk_coord in enumerate(vtk_nodes[:, :2]):
            for seg_start, seg_end in zone_segments:
                if _point_segment_distance(vtk_coord, seg_start, seg_end) <= zone_band_tolerance:
                    zone_allowed_nodes.add(vtk_idx)
                    break

        missing_edges = []
        for n1, n2 in cas_edges:
            if n1 not in vtk_node_map or n2 not in vtk_node_map:
                missing_edges.append((n1, n2, cas_nodes[n1], cas_nodes[n2], "节点未找到"))
                continue

            vtk_n1 = vtk_node_map[n1]
            vtk_n2 = vtk_node_map[n2]
            edge_key = (min(vtk_n1, vtk_n2), max(vtk_n1, vtk_n2))
            edge_exists = edge_key in mesh_edges
            if not edge_exists:
                edge_exists = _has_zone_conforming_path(
                    vtk_n1, vtk_n2, zone_allowed_nodes, mesh_adjacency
                )
            if not edge_exists:
                edge_exists = _has_split_edge_path(
                    vtk_n1,
                    vtk_n2,
                    cas_nodes[n1][:2],
                    cas_nodes[n2][:2],
                    vtk_nodes,
                    mesh_adjacency,
                    tolerance,
                )
            if not edge_exists:
                missing_edges.append((n1, n2, cas_nodes[n1], cas_nodes[n2], None))

        inner_boundary_inner_points = 0
        inner_boundary_inner_cells = 0
        if is_inner_boundary:
            hole_polygon = _build_ordered_polygon(cas_nodes, zone_info["edges"])
            if hole_polygon is not None:
                for vtk_coord in vtk_nodes[:, :2]:
                    if point_in_polygon(vtk_coord, hole_polygon):
                        inner_boundary_inner_points += 1
                for cell in grid.cells:
                    centroid = np.mean([vtk_nodes[n, :2] for n in cell], axis=0)
                    if point_in_polygon(centroid, hole_polygon):
                        inner_boundary_inner_cells += 1

        result = {
            "total_edges": len(cas_edges),
            "missing_edges": len(missing_edges),
            "missing_details": missing_edges,
            "inner_boundary_inner_points": inner_boundary_inner_points,
            "inner_boundary_inner_cells": inner_boundary_inner_cells,
        }
        all_results[zone_key] = result
        if missing_edges or inner_boundary_inner_points > 0 or inner_boundary_inner_cells > 0:
            all_pass = False

    issue = None
    if not all_pass:
        issues = []
        for zone_key, zone_result in all_results.items():
            if zone_result["missing_edges"] > 0:
                issues.append(f"{zone_key}: {zone_result['missing_edges']} 条边丢失")
            if zone_result["inner_boundary_inner_points"] > 0:
                issues.append(f"{zone_key}: 内部发现 {zone_result['inner_boundary_inner_points']} 个点")
            if zone_result["inner_boundary_inner_cells"] > 0:
                issues.append(f"{zone_key}: 内部发现 {zone_result['inner_boundary_inner_cells']} 个单元")
        issue = "; ".join(issues)

    return {"pass": all_pass, "zone_results": all_results, "issue": issue}


def check_hole_cleanup(cas_file, grid, test_name):
    from fileIO.read_cas import parse_fluent_msh
    from utils.geom_toolkit import point_in_polygon

    try:
        cas_data = parse_fluent_msh(cas_file)
    except Exception as exc:
        return {"pass": False, "issue": f"无法解析 CAS 文件: {exc}", "hole_results": {}}

    cas_nodes = np.array(cas_data["nodes"])
    vtk_nodes = np.array(grid.node_coords)
    tolerance = 0.01

    global_x_min, global_x_max = np.min(cas_nodes[:, 0]), np.max(cas_nodes[:, 0])
    global_y_min, global_y_max = np.min(cas_nodes[:, 1]), np.max(cas_nodes[:, 1])

    zone_data = {}
    for face in cas_data["faces"]:
        part_name = face.get("part_name", "unknown")
        bc_type = face.get("bc_type", "unknown")
        if bc_type == "interior" or len(face["nodes"]) != 2:
            continue
        zone_key = f"{part_name}_{bc_type}"
        zone_data.setdefault(zone_key, {"edges": [], "nodes": set()})
        n1, n2 = face["nodes"][0] - 1, face["nodes"][1] - 1
        zone_data[zone_key]["edges"].append((n1, n2))
        zone_data[zone_key]["nodes"].update((n1, n2))

    hole_results = {}
    all_pass = True
    for zone_key, data in zone_data.items():
        zone_coords = cas_nodes[list(data["nodes"])]
        x_min, x_max = np.min(zone_coords[:, 0]), np.max(zone_coords[:, 0])
        y_min, y_max = np.min(zone_coords[:, 1]), np.max(zone_coords[:, 1])
        touches_global_boundary = (
            abs(x_min - global_x_min) < tolerance
            or abs(x_max - global_x_max) < tolerance
            or abs(y_min - global_y_min) < tolerance
            or abs(y_max - global_y_max) < tolerance
        )
        if touches_global_boundary:
            continue

        hole_polygon = _build_ordered_polygon(cas_nodes, data["edges"])
        if hole_polygon is None:
            continue

        points_inside = sum(
            1 for vtk_coord in vtk_nodes[:, :2] if point_in_polygon(vtk_coord, hole_polygon)
        )
        cells_inside = 0
        for cell in grid.cells:
            if any(point_in_polygon(vtk_nodes[n, :2], hole_polygon) for n in cell):
                cells_inside += 1

        hole_results[zone_key] = {
            "points_inside": points_inside,
            "cells_inside": cells_inside,
        }
        if points_inside > 0 or cells_inside > 0:
            all_pass = False

    return {
        "pass": all_pass,
        "hole_results": hole_results,
        "issue": None if all_pass else "孔洞内残留点或单元",
    }


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _strict_intersect(pa, pb, pc, pd, eps=1e-12):
    d1 = _cross(pc, pd, pa)
    d2 = _cross(pc, pd, pb)
    d3 = _cross(pa, pb, pc)
    d4 = _cross(pa, pb, pd)
    return (
        ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps))
        and ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps))
    )


def check_topology_clean(grid, test_name):
    edge_to_cells = defaultdict(list)
    for cell_idx, cell in enumerate(grid.cells):
        for i in range(len(cell)):
            a = int(cell[i])
            b = int(cell[(i + 1) % len(cell)])
            edge_to_cells[(a, b) if a < b else (b, a)].append(cell_idx)

    non_manifold_edges = [edge for edge, cells in edge_to_cells.items() if len(cells) > 2]
    if non_manifold_edges:
        return {"pass": False, "issue": f"存在 {len(non_manifold_edges)} 条非流形边（共享单元数 > 2）"}

    cell_count = len(grid.cells)
    if cell_count == 0:
        return {"pass": False, "issue": "网格无单元"}

    cell_adj = [[] for _ in range(cell_count)]
    for cells in edge_to_cells.values():
        if len(cells) == 2:
            c1, c2 = cells
            cell_adj[c1].append(c2)
            cell_adj[c2].append(c1)

    visited = [False] * cell_count
    queue = deque([0])
    visited[0] = True
    visited_count = 0
    while queue:
        current = queue.popleft()
        visited_count += 1
        for neighbor in cell_adj[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    if visited_count != cell_count:
        return {"pass": False, "issue": f"网格存在 {cell_count - visited_count} 个未连通单元"}

    points = np.array(grid.node_coords)[:, :2]
    edges = list(edge_to_cells.keys())
    bboxes = []
    for edge in edges:
        p1 = points[edge[0]]
        p2 = points[edge[1]]
        bboxes.append((min(p1[0], p2[0]), max(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[1], p2[1])))

    for i, e1 in enumerate(edges):
        a, b = e1
        p1 = points[a]
        p2 = points[b]
        x1_min, x1_max, y1_min, y1_max = bboxes[i]
        for j in range(i + 1, len(edges)):
            c, d = edges[j]
            if a in (c, d) or b in (c, d):
                continue
            x2_min, x2_max, y2_min, y2_max = bboxes[j]
            if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
                continue
            p3 = points[c]
            p4 = points[d]
            if _strict_intersect(p1, p2, p3, p4):
                return {"pass": False, "issue": "检测到严格边相交（单元拓扑交叉）"}

    return {"pass": True, "issue": None}
