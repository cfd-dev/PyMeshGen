from collections import defaultdict, deque

import numpy as np

from optimize.mesh_quality import triangle_shape_quality
from utils.geom_toolkit import segments_intersect_strict


def collect_boundary_edges_from_fronts(boundary_front):
    node_index_map = {}
    edges = []
    next_idx = 0
    for front in boundary_front:
        for node_elem in front.node_elems:
            node_hash = node_elem.hash
            if node_hash not in node_index_map:
                node_index_map[node_hash] = next_idx
                next_idx += 1
        if len(front.node_elems) >= 2:
            idx1 = node_index_map[front.node_elems[0].hash]
            idx2 = node_index_map[front.node_elems[1].hash]
            edges.append((idx1, idx2))
    return edges


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def _build_edge_to_tris(tris):
    edge_map = {}
    for tri_idx, tri in enumerate(tris):
        a, b, c = tri
        for u, v in ((a, b), (b, c), (c, a)):
            key = _edge_key(u, v)
            edge_map.setdefault(key, []).append(tri_idx)
    return edge_map


def _edge_exists(tris, a, b):
    target = _edge_key(a, b)
    for tri in tris:
        x, y, z = tri
        if (
            _edge_key(x, y) == target
            or _edge_key(y, z) == target
            or _edge_key(z, x) == target
        ):
            return True
    return False


def recover_boundary_edges_by_swaps(
    points_arr,
    simplices_arr,
    boundary_edges,
    max_iter_per_edge=400,
):
    triangles = [list(map(int, tri)) for tri in simplices_arr.tolist()]
    protected = {_edge_key(a, b) for a, b in boundary_edges}

    recovered = 0
    for v1, v2 in boundary_edges:
        if _edge_exists(triangles, v1, v2):
            continue

        for _ in range(max_iter_per_edge):
            if _edge_exists(triangles, v1, v2):
                recovered += 1
                break

            edge_map = _build_edge_to_tris(triangles)
            p1 = points_arr[v1, :2]
            p2 = points_arr[v2, :2]

            intersecting = []
            for (a, b), tri_ids in edge_map.items():
                if a in (v1, v2) or b in (v1, v2):
                    continue
                if (a, b) in protected:
                    continue
                p3 = points_arr[a, :2]
                p4 = points_arr[b, :2]
                if segments_intersect_strict(p1, p2, p3, p4):
                    mid = 0.5 * (p3 + p4)
                    dist = float(np.linalg.norm(mid - 0.5 * (p1 + p2)))
                    intersecting.append((dist, a, b, tri_ids))

            if not intersecting:
                break

            intersecting.sort(key=lambda item: item[0])
            flipped = False

            for _, a, b, tri_ids in intersecting:
                if len(tri_ids) != 2:
                    continue
                t1_idx, t2_idx = tri_ids
                t1 = triangles[t1_idx]
                t2 = triangles[t2_idx]

                c = next((x for x in t1 if x != a and x != b), None)
                d = next((x for x in t2 if x != a and x != b), None)
                if c is None or d is None or c == d:
                    continue

                new_edge = _edge_key(c, d)
                existing = edge_map.get(new_edge, [])
                if any(idx not in (t1_idx, t2_idx) for idx in existing):
                    continue

                pc = points_arr[c, :2]
                pd = points_arr[d, :2]
                pa = points_arr[a, :2]
                pb = points_arr[b, :2]

                if not segments_intersect_strict(pa, pb, pc, pd):
                    continue

                q1 = triangle_shape_quality(pc, pd, pa)
                q2 = triangle_shape_quality(pc, pd, pb)
                if q1 < 1e-6 or q2 < 1e-6:
                    continue

                triangles[t1_idx] = [c, d, a]
                triangles[t2_idx] = [c, d, b]
                flipped = True
                break

            if not flipped:
                break

    remaining = []
    for v1, v2 in boundary_edges:
        if not _edge_exists(triangles, v1, v2):
            remaining.append((v1, v2))

    return np.array(triangles, dtype=int), recovered, remaining


def is_topology_valid(points_arr, simplices_arr):
    """Quick topology validation for triangle connectivity and strict edge crossings."""
    edge_to_cells = defaultdict(list)
    for cell_idx, tri in enumerate(simplices_arr):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            edge_to_cells[_edge_key(u, v)].append(cell_idx)

    if any(len(cells) > 2 for cells in edge_to_cells.values()):
        return False

    tri_count = len(simplices_arr)
    if tri_count == 0:
        return False

    adjacency = [[] for _ in range(tri_count)]
    for cells in edge_to_cells.values():
        if len(cells) == 2:
            c1, c2 = cells
            adjacency[c1].append(c2)
            adjacency[c2].append(c1)

    visited = [False] * tri_count
    queue = deque([0])
    visited[0] = True
    visited_count = 0
    while queue:
        current = queue.popleft()
        visited_count += 1
        for neighbor in adjacency[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    if visited_count != tri_count:
        return False

    edges = list(edge_to_cells.keys())
    bboxes = []
    for edge in edges:
        p1 = points_arr[edge[0], :2]
        p2 = points_arr[edge[1], :2]
        bboxes.append(
            (
                min(p1[0], p2[0]),
                max(p1[0], p2[0]),
                min(p1[1], p2[1]),
                max(p1[1], p2[1]),
            )
        )

    for i, e1 in enumerate(edges):
        a, b = e1
        x1, x2, y1, y2 = bboxes[i]
        p1 = points_arr[a, :2]
        p2 = points_arr[b, :2]
        for j in range(i + 1, len(edges)):
            c, d = edges[j]
            if a in (c, d) or b in (c, d):
                continue
            u1, u2, v1, v2 = bboxes[j]
            if x2 < u1 or u2 < x1 or y2 < v1 or v2 < y1:
                continue
            p3 = points_arr[c, :2]
            p4 = points_arr[d, :2]
            if segments_intersect_strict(p1, p2, p3, p4):
                return False

    return True
