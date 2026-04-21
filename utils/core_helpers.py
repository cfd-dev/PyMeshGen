from adfront2.adfront2 import Adfront2
from adfront2.adfront2_hybrid import Adfront2Hybrid
from utils.message import debug, verbose


def is_mixed_mesh(mesh_type):
    return mesh_type == 3


def use_triangle_pipeline_for_qmorph(parameters):
    return (
        is_mixed_mesh(parameters.mesh_type)
        and parameters.triangle_to_quad_method == "q_morph"
    )


def select_delaunay_backend(parameters, has_boundary_layer=False):
    """选择 mesh_type=4 的 Delaunay 后端。

    当前策略：带边界层时优先使用 Triangle 生成内层三角形，
    以避免 Bowyer-Watson 在复杂近壁前沿上的拓扑不稳定问题。
    """
    configured_backend = str(
        getattr(parameters, "delaunay_backend", "bowyer_watson")
    ).strip().lower()
    if configured_backend not in {"bowyer_watson", "triangle"}:
        configured_backend = "bowyer_watson"
    if has_boundary_layer:
        return "triangle"
    return configured_backend


def create_interior_generator(
    parameters, front_heap, sizing_system, boundary_grid, visual_obj
):
    """创建内部区域网格生成器。
    
    根据网格类型选择适当的生成器：
    - q_morph 模式：使用 Adfront2（纯三角形推进）
    - 混合网格模式：使用 Adfront2Hybrid（混合推进）
    - 三角形网格模式：使用 Adfront2（纯三角形推进）
    """
    if use_triangle_pipeline_for_qmorph(parameters):
        return Adfront2(
            boundary_front=front_heap,
            sizing_system=sizing_system,
            node_coords=boundary_grid.node_coords,
            param_obj=parameters,
            visual_obj=visual_obj,
        )
    if is_mixed_mesh(parameters.mesh_type):
        # 使用混合网格生成器，推进过程中优先直接生成四边形
        return Adfront2Hybrid(
            boundary_front=front_heap,
            sizing_system=sizing_system,
            node_coords=boundary_grid.node_coords,
            param_obj=parameters,
            visual_obj=visual_obj,
        )
    return Adfront2(
        boundary_front=front_heap,
        sizing_system=sizing_system,
        node_coords=boundary_grid.node_coords,
        param_obj=parameters,
        visual_obj=visual_obj,
    )


def _cell_node_count(cell):
    if cell is None:
        return 0
    if hasattr(cell, "node_ids") and cell.node_ids is not None:
        return len(cell.node_ids)
    if hasattr(cell, "nodes") and cell.nodes is not None:
        return len(cell.nodes)
    if isinstance(cell, (list, tuple)):
        return len(cell)
    return 0


def _extract_node_ids(cell):
    if cell is None:
        return []
    if hasattr(cell, "node_ids") and cell.node_ids is not None:
        return list(cell.node_ids)
    if hasattr(cell, "nodes") and cell.nodes is not None:
        ids = []
        for node in cell.nodes:
            node_idx = getattr(node, "idx", None)
            if node_idx is not None:
                ids.append(node_idx)
        return ids
    if isinstance(cell, (list, tuple)):
        return list(cell)
    return []


def _iter_cells_for_analysis(mesh_data):
    if hasattr(mesh_data, "cell_container") and mesh_data.cell_container:
        return mesh_data.cell_container
    if hasattr(mesh_data, "cells") and mesh_data.cells:
        return mesh_data.cells
    return []


def log_parameters_debug_summary(parameters):
    debug(
        f"[参数调试] debug_level={parameters.debug_level}, mesh_type={parameters.mesh_type}, "
        f"input={parameters.input_file}, output={parameters.output_file}"
    )
    debug(f"[参数调试] 部件数量: {len(parameters.part_params)}")
    for i, part in enumerate(parameters.part_params):
        part_cfg = getattr(part, "part_params", None)
        if part_cfg is None:
            debug(f"[参数调试] 部件{i+1}: {part.part_name} (无part_params)")
            continue
        debug(
            f"[参数调试] 部件{i+1}: {part.part_name}, "
            f"PRISM_SWITCH={part_cfg.PRISM_SWITCH}, "
            f"multi_direction={part_cfg.multi_direction}, "
            f"max_layers={part_cfg.max_layers}, "
            f"full_layers={part_cfg.full_layers}, "
            f"first_height={part_cfg.first_height}, "
            f"max_size={part_cfg.max_size}"
        )


def log_mesh_debug_summary(mesh_data, summary_name):
    cells_for_analysis = _iter_cells_for_analysis(mesh_data)
    tri_count = 0
    quad_count = 0
    other_count = 0
    layer_stats = {}
    quad_samples = []

    for cell_idx, cell in enumerate(cells_for_analysis):
        n_nodes = _cell_node_count(cell)
        if n_nodes == 4:
            quad_count += 1
            if len(quad_samples) < 5:
                quad_samples.append((cell_idx, _extract_node_ids(cell)))
        elif n_nodes == 3:
            tri_count += 1
        else:
            other_count += 1

        layer = getattr(cell, "layer", None)
        if layer is None:
            continue
        if layer not in layer_stats:
            layer_stats[layer] = {"quads": 0, "tris": 0, "others": 0}
        if n_nodes == 4:
            layer_stats[layer]["quads"] += 1
        elif n_nodes == 3:
            layer_stats[layer]["tris"] += 1
        else:
            layer_stats[layer]["others"] += 1

    debug(
        f"[{summary_name}] 节点数量: {len(mesh_data.node_coords)}, "
        f"单元数量: {len(cells_for_analysis)}"
    )
    debug(
        f"[{summary_name}] 单元类型统计: "
        f"四边形={quad_count}, 三角形={tri_count}, 其他={other_count}"
    )

    for layer in sorted(layer_stats.keys()):
        stats = layer_stats[layer]
        debug(
            f"[{summary_name}] 第{layer}层: "
            f"{stats['quads']}个四边形 + {stats['tris']}个三角形 + {stats['others']}个其他"
        )

    if not quad_samples:
        return

    verbose(f"[{summary_name}] 四边形节点样本(最多5个):")
    for cell_idx, node_ids in quad_samples:
        verbose(f"[{summary_name}]   单元{cell_idx}: 节点{node_ids}")
        coord_text = []
        for node_id in node_ids:
            if not isinstance(node_id, int):
                continue
            if not (0 <= node_id < len(mesh_data.node_coords)):
                continue
            coord = mesh_data.node_coords[node_id]
            if len(coord) >= 2:
                coord_text.append(f"{node_id}:({coord[0]:.6f}, {coord[1]:.6f})")
            else:
                coord_text.append(f"{node_id}:{coord}")
        if coord_text:
            verbose(f"[{summary_name}]     坐标: {'; '.join(coord_text)}")
