from collections import defaultdict, deque

import numpy as np

from fileIO.read_cas import parse_fluent_msh
from utils.geom_toolkit import (
    point_in_polygon,
    point_to_segment_distance,
    segments_intersect_strict,
)


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
            if point_to_segment_distance(coord, seg_start, seg_end) <= line_tolerance:
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


def check_boundary_edges(cas_file, grid, test_name=None, tolerance=0.01, cas_data=None):
    if cas_data is None:
        try:
            cas_data = parse_fluent_msh(cas_file)
        except Exception as exc:
            return {"pass": False, "issue": f"无法解析 CAS 文件: {exc}", "zone_results": {}}

    cas_nodes = np.array(cas_data["nodes"])
    vtk_nodes = np.array(grid.node_coords)

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
                if point_to_segment_distance(vtk_coord, seg_start, seg_end) <= zone_band_tolerance:
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


def check_hole_cleanup(cas_file, grid, test_name=None, tolerance=0.01, cas_data=None):
    if cas_data is None:
        try:
            cas_data = parse_fluent_msh(cas_file)
        except Exception as exc:
            return {"pass": False, "issue": f"无法解析 CAS 文件: {exc}", "hole_results": {}}

    cas_nodes = np.array(cas_data["nodes"])
    vtk_nodes = np.array(grid.node_coords)

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


def check_topology_clean(grid, test_name=None):
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
            if segments_intersect_strict(p1, p2, p3, p4):
                return {"pass": False, "issue": "检测到严格边相交（单元拓扑交叉）"}

    return {"pass": True, "issue": None}
