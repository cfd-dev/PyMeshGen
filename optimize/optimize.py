import os
import numpy as np
import heapq
from itertools import combinations

from utils import geom_toolkit as geom_tool
from utils.timer import TimeSpan
from utils.message import info, debug, verbose, warning, error
from data_structure.basic_elements import Triangle, Quadrilateral
import sys
from pathlib import Path

# Ensure the optimize module directory is in the path
optimize_dir = str(Path(__file__).parent)
if optimize_dir not in sys.path:
    sys.path.insert(0, optimize_dir)

from mesh_quality import quadrilateral_quality2, triangle_shape_quality

# Import angle-based smoothing functions
from .angle_based_smoothing import (
    angle_based_smoothing,
    getme_method,
    smooth_mesh_angle_based,
    smooth_mesh_getme
)

# Import neural network smoothing functions (optional, requires torch)
try:
    from .nn_smoothing import (
        nn_smoothing_adam,
        smooth_mesh_drl,
        adam_optimization_smoothing,
        drl_smoothing
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn_smoothing_adam = None
    smooth_mesh_drl = None
    adam_optimization_smoothing = None
    drl_smoothing = None

def optimize_hybrid_grid(hybrid_grid):
    """调用外部混合网格优化软件进行优化"""  
    import subprocess
    tmp_file = "./out/tmp_mesh.vtk"
    hybrid_grid.save_to_vtkfile(tmp_file)

    max_iter = 10
    movement_factor = 0.3
    opt_exe = './optimize/laplacian_opt.exe'
    cmd = [opt_exe, tmp_file, str(max_iter), str(movement_factor)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    debug(f'Optimized VTK mesh written to: {tmp_file.replace(".vtk", "_opt.vtk")}')
    
    # 从优化后的VTK文件加载网格
    from fileIO.vtk_io import parse_vtk_msh
    optimized_grid = parse_vtk_msh(tmp_file.replace(".vtk", "_opt.vtk"))
    
    # 如果需要，删除临时文件
    if not __debug__:
        os.remove(tmp_file)
        os.remove(tmp_file.replace(".vtk", "_opt.vtk"))

    return optimized_grid



def hybrid_smooth(unstr_grid, max_iter=3):
    """混合平滑算法（结合角度优化和形态优化），优化同时具有三角形和四边形的混合网格"""
    timer = TimeSpan("开始混合平滑优化...")
    node_coords = np.array(unstr_grid.node_coords)
    original_coords = node_coords.copy()
    boundary_nodes = set(unstr_grid.boundary_nodes_list)

    # 新增凸性检查辅助函数
    def is_quad_convex(nodes):
        return geom_tool.is_convex(
            nodes[0], nodes[1], nodes[2], nodes[3], node_coords.tolist()
        )

    # 新增三角形质量检查辅助函数
    def is_triangle_valid(tri_nodes):
        """检查三角形质量是否可接受（放宽阈值）"""
        try:
            quality = triangle_shape_quality(
                node_coords[tri_nodes[0]],
                node_coords[tri_nodes[1]],
                node_coords[tri_nodes[2]]
            )
            return quality > 0.05
        except:
            return False

    # 改进的四边形理想位置计算函数
    def compute_quad_ideal_position(quad_nodes, target_node_idx):
        """计算四边形中目标节点的理想位置（基于Laplacian平滑）"""
        quad_coords = [node_coords[i] for i in quad_nodes]
        target_idx_in_quad = quad_nodes.index(target_node_idx)
        
        # 获取相邻节点索引
        prev_idx = (target_idx_in_quad - 1) % 4
        next_idx = (target_idx_in_quad + 1) % 4
        opposite_idx = (target_idx_in_quad + 2) % 4
        
        # 方法1：Laplacian平滑（向相邻节点平均位置移动）
        neighbor_avg = (quad_coords[prev_idx] + quad_coords[next_idx]) / 2
        vec_to_neighbor = neighbor_avg - quad_coords[target_idx_in_quad]
        ideal1 = quad_coords[target_idx_in_quad] + 0.6 * vec_to_neighbor
        
        # 方法2：向对角线中点移动
        opposite_midpoint = (quad_coords[opposite_idx] + quad_coords[target_idx_in_quad]) / 2
        vec_to_opposite = opposite_midpoint - quad_coords[target_idx_in_quad]
        ideal2 = quad_coords[target_idx_in_quad] + 0.4 * vec_to_opposite
        
        # 综合两种方法
        ideal_pos = 0.7 * ideal1 + 0.3 * ideal2
        return ideal_pos

    # 改进的三角形理想位置计算函数
    def compute_tri_ideal_position(tri_nodes, target_node_idx):
        """计算三角形中目标节点的理想位置（基于角度优化）"""
        tri_coords = [node_coords[i] for i in tri_nodes]
        target_idx_in_tri = tri_nodes.index(target_node_idx)
        
        other_indices = [i for i in range(3) if i != target_idx_in_tri]
        idx1, idx2 = other_indices[0], other_indices[1]
        
        # 计算当前角度
        angle_current = geom_tool.calculate_angle(
            tri_coords[idx1], tri_coords[target_idx_in_tri], tri_coords[idx2]
        )
        
        # 理想角度为60度（等边三角形）
        angle_ideal = 60.0
        
        # 计算角度差
        angle_diff = angle_ideal - angle_current
        
        # 计算当前边的单位向量
        vec1 = tri_coords[idx1] - tri_coords[target_idx_in_tri]
        vec2 = tri_coords[idx2] - tri_coords[target_idx_in_tri]
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 1e-10 and norm2 > 1e-10:
            unit_vec1 = vec1 / norm1
            unit_vec2 = vec2 / norm2
            
            # 计算角平分线方向
            bisector = unit_vec1 + unit_vec2
            bisector_norm = np.linalg.norm(bisector)
            
            if bisector_norm > 1e-10:
                bisector = bisector / bisector_norm
                
                # 根据角度差调整位置
                adjustment_factor = 0.3 * (angle_diff / 180.0)
                avg_edge_length = (norm1 + norm2) / 2
                ideal_pos = tri_coords[target_idx_in_tri] + adjustment_factor * avg_edge_length * bisector
            else:
                # 如果共线，向质心移动
                centroid = np.mean(tri_coords, axis=0)
                ideal_pos = centroid
        else:
            centroid = np.mean(tri_coords, axis=0)
            ideal_pos = centroid
        
        return ideal_pos

    for iter_idx in range(max_iter):
        # 存储每个节点的移动向量和权重
        displacements = np.zeros_like(node_coords)
        weights = np.zeros(len(node_coords))

        # 存储单元原始节点用于质量检查
        quad_originals = {}
        tri_originals = {}
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Quadrilateral):
                quad_originals[id(cell)] = np.array(
                    [node_coords[i] for i in cell.node_ids]
                )
            elif isinstance(cell, Triangle):
                tri_originals[id(cell)] = np.array(
                    [node_coords[i] for i in cell.node_ids]
                )

        # 计算当前网格质量，用于自适应调整
        quad_qualities = []
        tri_qualities = []
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Quadrilateral):
                quad_nodes = cell.node_ids
                coords = [node_coords[i] for i in quad_nodes]
                try:
                    q = quadrilateral_quality2(coords[0], coords[1], coords[2], coords[3])
                    quad_qualities.append(q)
                except:
                    quad_qualities.append(0.0)
            elif isinstance(cell, Triangle):
                tri_nodes = cell.node_ids
                coords = [node_coords[i] for i in tri_nodes]
                try:
                    q = triangle_shape_quality(coords[0], coords[1], coords[2])
                    tri_qualities.append(q)
                except:
                    tri_qualities.append(0.0)

        # 计算平均质量（仅考虑质量>0的单元）
        valid_quad_qualities = [q for q in quad_qualities if q > 0]
        valid_tri_qualities = [q for q in tri_qualities if q > 0]
        avg_quad_quality = np.mean(valid_quad_qualities) if valid_quad_qualities else 0.5
        avg_tri_quality = np.mean(valid_tri_qualities) if valid_tri_qualities else 0.5

        # 自适应调整位移系数（更保守的策略，避免产生非凸四边形）
        quad_displacement_factor = 0.2 * (1.0 - avg_quad_quality) + 0.05
        tri_displacement_factor = 0.15 * (1.0 - avg_tri_quality) + 0.05

        # 遍历所有单元进行贡献计算
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Quadrilateral):
                # 四边形优化目标：接近矩形（角度优化+边长比优化）
                quad_nodes = cell.node_ids

                for i in range(4):
                    curr_node = quad_nodes[i]
                    if curr_node in boundary_nodes:
                        continue

                    # 计算理想位置
                    ideal_point = compute_quad_ideal_position(quad_nodes, curr_node)

                    displacement = quad_displacement_factor * (ideal_point - node_coords[curr_node])
                    displacements[curr_node] += displacement
                    weights[curr_node] += 1.0

            elif isinstance(cell, Triangle):
                # 三角形优化目标：接近等边三角形（角度优化）
                tri_nodes = cell.node_ids

                for i in range(3):
                    curr_node = tri_nodes[i]
                    if curr_node in boundary_nodes:
                        continue

                    # 计算理想位置
                    ideal_point = compute_tri_ideal_position(tri_nodes, curr_node)

                    displacement = tri_displacement_factor * (ideal_point - node_coords[curr_node])
                    displacements[curr_node] += displacement
                    weights[curr_node] += 1.0

        # 应用平滑并更新坐标
        new_coords = node_coords.copy()
        for i in range(len(node_coords)):
            if weights[i] > 0 and i not in boundary_nodes:
                new_coords[i] += displacements[i] / weights[i]

        # 质量检查和修正（改进的接受策略，允许质量为0的单元）
        need_rollback = False
        
        # 计算新网格的质量
        new_quad_qualities = []
        new_tri_qualities = []
        
        # 检查四边形质量（包括非凸四边形）
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Quadrilateral) and len(cell.node_ids) == 4:
                new_nodes = [new_coords[i] for i in cell.node_ids]
                # 计算新质量（非凸四边形质量为0）
                try:
                    q = quadrilateral_quality2(new_nodes[0], new_nodes[1], new_nodes[2], new_nodes[3])
                    new_quad_qualities.append(q)
                except:
                    new_quad_qualities.append(0.0)

        # 检查三角形质量
        for cell in unstr_grid.cell_container:
            if isinstance(cell, Triangle) and len(cell.node_ids) == 3:
                tri_nodes = cell.node_ids
                # 计算新质量
                try:
                    q = triangle_shape_quality(
                        new_coords[tri_nodes[0]],
                        new_coords[tri_nodes[1]],
                        new_coords[tri_nodes[2]]
                    )
                    new_tri_qualities.append(q)
                except:
                    new_tri_qualities.append(0.0)

        # 改进的接受策略：检查有效单元（质量>0）的平均质量是否改善
        valid_new_quad_qualities = [q for q in new_quad_qualities if q > 0]
        valid_new_tri_qualities = [q for q in new_tri_qualities if q > 0]
        
        if valid_new_quad_qualities:
            new_avg_quad = np.mean(valid_new_quad_qualities)
            if new_avg_quad < avg_quad_quality * 0.98:  # 允许2%的下降
                need_rollback = True
        
        if valid_new_tri_qualities:
            new_avg_tri = np.mean(valid_new_tri_qualities)
            if new_avg_tri < avg_tri_quality * 0.98:  # 允许2%的下降
                need_rollback = True

        # 只有当所有单元保持质量要求时才接受更新
        if not need_rollback:
            node_coords = new_coords
            # 更新平均质量用于下一轮
            if new_quad_qualities:
                avg_quad_quality = np.mean(new_quad_qualities)
            if new_tri_qualities:
                avg_tri_quality = np.mean(new_tri_qualities)
        else:
            warning(f"检测到低质量单元，跳过第{iter_idx + 1}轮平滑")

        # 限制最大位移防止震荡
        max_disp = np.linalg.norm(node_coords - original_coords, axis=1).max()
        if max_disp < 1e-6:
            break

    # 更新回网格结构
    unstr_grid.node_coords = node_coords.tolist()
    timer.show_to_console("混合平滑完成.")
    return unstr_grid


def compute_rectangular_position(prev, curr, next):
    """计算理想矩形位置（基于相邻三点）"""
    vec1 = curr - prev
    vec2 = next - curr

    # 计算正交修正
    ideal_vec = vec1 - 2 * np.dot(vec1, vec2) / np.dot(vec2, vec2) * vec2
    return curr + ideal_vec


def edge_swap(unstr_grid):
    timer = TimeSpan("开始进行边交换优化...")
    node_coords = unstr_grid.node_coords
    edge_map = {}

    # 构建边到单元的映射
    for cell_idx, cell in enumerate(unstr_grid.cell_container):
        for i, j in combinations(sorted(cell.node_ids), 2):
            edge = (i, j)
            if edge not in edge_map:
                edge_map[edge] = []
            edge_map[edge].append(cell_idx)

    swapped = True
    num_swapped = 0
    while swapped:
        swapped = False
        # 使用当前edge_map的快照避免迭代修改问题
        for edge, cells in list(edge_map.items()):
            # 仅处理内部边（被两个单元共享）
            if len(cells) != 2:
                continue
            if edge == (194, 24) or edge == (24, 194):
                kkk = 0

            cell1_idx, cell2_idx = cells
            cell1 = unstr_grid.cell_container[cell1_idx]
            cell2 = unstr_grid.cell_container[cell2_idx]
            # 使用字符串类型比较代替isinstance，避免导入路径问题
            if not (type(cell1).__name__ == 'Triangle' and type(cell2).__name__ == 'Triangle'):
                continue  # 非三角形单元跳过

            # 确认公共边
            common_edge = set(cell1.node_ids) & set(cell2.node_ids)
            if len(common_edge) != 2:
                continue  # 数据异常

            a, b = sorted(common_edge)
            other_points = list(
                (set(cell1.node_ids) | set(cell2.node_ids)) - common_edge
            )
            if len(other_points) != 2:
                continue  # 无法构成四边形

            c, d = other_points

            # 凸性检查
            if not geom_tool.is_convex(a, c, b, d, node_coords):
                continue

            # 计算交换前的最小角
            # 修复：直接传递node_ids而不是Triangle对象
            current_min = min(
                geom_tool.calculate_min_angle(cell1.node_ids, node_coords),
                geom_tool.calculate_min_angle(cell2.node_ids, node_coords),
            )

            # 交换后的单元
            swapped_cell1 = [a, c, d]
            swapped_cell2 = [b, c, d]

            # 确保构成的单元节点逆时针
            if not geom_tool.is_left2d(node_coords[a], node_coords[c], node_coords[d]):
                swapped_cell1 = [d, c, a]
            if not geom_tool.is_left2d(node_coords[b], node_coords[c], node_coords[d]):
                swapped_cell2 = [d, c, b]

            # 有效性检查
            if not (
                geom_tool.is_valid_triangle(swapped_cell1, node_coords)
                and geom_tool.is_valid_triangle(swapped_cell2, node_coords)
            ):
                continue

            # 计算交换后的最小角
            swapped_min = min(
                geom_tool.calculate_min_angle(swapped_cell1, node_coords),
                geom_tool.calculate_min_angle(swapped_cell2, node_coords),
            )

            # 凸性检查
            # if not geom_tool.is_convex(a, c, b, d, node_coords):
            #     continue

            # 交换条件：最小角优化且不创建新边界边
            if swapped_min > current_min and not (
                c in unstr_grid.boundary_nodes_list
                and d in unstr_grid.boundary_nodes_list
            ):
                # 执行交换
                # 创建新的Triangle对象
                new_cell1 = Triangle(
                    node_coords[swapped_cell1[0]],
                    node_coords[swapped_cell1[1]],
                    node_coords[swapped_cell1[2]],
                    cell1.part_name,
                    cell1.idx,
                    node_ids=swapped_cell1,
                )
                new_cell2 = Triangle(
                    node_coords[swapped_cell2[0]],
                    node_coords[swapped_cell2[1]],
                    node_coords[swapped_cell2[2]],
                    cell2.part_name,
                    cell2.idx,
                    node_ids=swapped_cell2,
                )

                unstr_grid.cell_container[cell1_idx] = new_cell1  # 修改点
                unstr_grid.cell_container[cell2_idx] = new_cell2  # 修改点
                swapped = True
                num_swapped += 1

        # 重新构建边映射
        if swapped:
            edge_map = {}
            for cell_idx, cell in enumerate(unstr_grid.cell_container):
                for i, j in combinations(sorted(cell.node_ids), 2):
                    edge = (i, j)
                    if edge not in edge_map:
                        edge_map[edge] = []
                    edge_map[edge].append(cell_idx)

    info(f"共进行了{num_swapped}次边交换.")
    timer.show_to_console("边交换优化完成.")

    return unstr_grid


def laplacian_smooth(unstr_grid, num_iter=10):
    timer = TimeSpan("开始进行laplacian优化...")

    node_coords = unstr_grid.node_coords
    # 将键改为节点索引
    neighbors = {node_ids: set() for node_ids in range(len(node_coords))}

    for cell in unstr_grid.cell_container:
        for i, j in combinations(sorted(cell.node_ids), 2):
            neighbors[i].add(j)
            neighbors[j].add(i)

    # 迭代进行拉普拉斯平滑
    for i in range(num_iter):
        new_coords = []
        for node, coord in enumerate(node_coords):
            # 跳过边界节点（保持固定）
            if node in unstr_grid.boundary_nodes_list:
                new_coords.append(coord)
                continue

            if len(neighbors[node]) == 0:
                new_coords.append(coord)
                continue

            # 计算邻居节点的平均坐标（转换为numpy数组）
            neighbor_coords = np.array([node_coords[n] for n in neighbors[node]])
            avg_coord = np.mean(neighbor_coords, axis=0)

            # 使用numpy进行向量运算
            new_coord = 0.5 * np.array(coord) + 0.5 * avg_coord
            new_coords.append(new_coord.tolist())  # 转换回列表格式

        unstr_grid.node_coords = new_coords
        info(f"第{i+1}轮laplacian优化完成.")
    timer.show_to_console("laplacian优化完成.")

    return unstr_grid

def node_perturbation(unstr_grid, ratio=0.8):
    """对网格节点进行随机扰动（仅在XY平面内）
    Args:
        ratio: 扰动比例（相对于节点周边边长的平均值）
    """
    np.random.seed(42)  # 固定随机种子保证可重复性
    
    # 获取节点坐标和边界信息
    node_coords = np.array(unstr_grid.node_coords)
    boundary_nodes = set(unstr_grid.boundary_nodes_list)
    
    # 构建节点到边的映射
    node_edges = {}  # node_id -> list of edge lengths
    for cell in unstr_grid.cell_container:
        if hasattr(cell, 'node_ids'):
            nodes = cell.node_ids
            num_nodes = len(nodes)
            # 遍历单元的所有边
            for i in range(num_nodes):
                node1 = nodes[i]
                node2 = nodes[(i + 1) % num_nodes]
                
                # 计算边长（仅XY平面）
                coord1 = node_coords[node1]
                coord2 = node_coords[node2]
                edge_length = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                
                # 添加到两个节点的边长列表
                if node1 not in node_edges:
                    node_edges[node1] = []
                if node2 not in node_edges:
                    node_edges[node2] = []
                node_edges[node1].append(edge_length)
                node_edges[node2].append(edge_length)
    
    # 计算每个节点的最小边长
    node_scale = np.zeros(len(node_coords))
    for node_id, edges in node_edges.items():
        node_scale[node_id] = np.min(edges) if edges else 0.0
    
    # 仅扰动内部节点
    for i in range(len(node_coords)):
        if i in boundary_nodes:
            continue
        
        # 仅在XY平面内生成随机方向向量
        direction = np.random.normal(size=2)
        direction /= np.linalg.norm(direction)  # 单位向量
        
        # 计算扰动幅度（基于节点周边边长的最小值）
        max_shift = node_scale[i] * ratio
        shift = direction * np.random.uniform(0, max_shift)
        
        # 仅在XY平面应用扰动，保持Z坐标不变
        node_coords[i, 0] += shift[0]
        node_coords[i, 1] += shift[1]
        
    # 更新回网格结构
    unstr_grid.node_coords = node_coords.tolist()
    info(f"节点扰动完成，最大位移: {np.max(node_scale * ratio):.4f}")
    return unstr_grid