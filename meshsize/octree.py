"""八叉树背景网格

用于存储和查询三维尺寸场值。支持：
- 递归细分到可配置的最大深度
- 叶节点在8个角点存储尺寸值
- 三线性插值查询
- 尺寸值扩散
- 梯度限制

参考: 第6节
"""
import numpy as np
from collections import deque


class OctreeNode:
    """八叉树节点

    Attributes:
        bounds: (xmin, ymin, zmin, xmax, ymax, zmax) 包围盒
        level: 节点深度（0=根节点）
        children: 8个子节点（None=叶节点）
        size_values: (8,) 角点尺寸值（仅叶节点有效）
        is_leaf: 是否为叶节点
    """
    __slots__ = ['bounds', 'level', 'children', 'size_values', 'is_leaf']

    def __init__(self, bounds, level=0):
        self.bounds = bounds
        self.level = level
        self.children = None
        self.size_values = None
        self.is_leaf = True


# 八叉树子节点顺序：Section 6.3
# 0-3: zmin 平面 (SWU, SEU, NWU, NEU)
# 4-7: zmax 平面 (SWD, SED, NWD, NED)
CHILD_OFFSETS = [
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
]

# 8个角点的局部坐标 (xi, eta, zeta) ∈ {0,1}
CORNER_LOCAL = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]

# 6个邻居方向: -x, +x, -y, +y, -z, +z
NEIGHBOR_DIRS = [
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (0, 0, -1), (0, 0, 1),
]


def _divide_bounds(bounds):
    """将包围盒八等分

    Args:
        bounds: (xmin, ymin, zmin, xmax, ymax, zmax)

    Returns:
        list: 8个子包围盒
    """
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    xm = (xmin + xmax) / 2
    ym = (ymin + ymax) / 2
    zm = (zmin + zmax) / 2

    return [
        (xmin, ymin, zmin, xm, ym, zm),   # 0
        (xm, ymin, zmin, xmax, ym, zm),   # 1
        (xmin, ym, zmin, xm, ymax, zm),   # 2
        (xm, ym, zmin, xmax, ymax, zm),   # 3
        (xmin, ymin, zm, xm, ym, zmax),   # 4
        (xm, ymin, zm, xmax, ym, zmax),   # 5
        (xmin, ym, zm, xm, ymax, zmax),   # 6
        (xm, ym, zm, xmax, ymax, zmax),   # 7
    ]


def _point_in_bounds(point, bounds):
    """检查点是否在包围盒内"""
    x, y, z = point
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    return (xmin <= x <= xmax and
            ymin <= y <= ymax and
            zmin <= z <= zmax)


def _get_child_index(point, bounds):
    """计算点所在的子节点索引"""
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    xm = (xmin + xmax) / 2
    ym = (ymin + ymax) / 2
    zm = (zmin + zmax) / 2

    idx = 0
    if point[0] >= xm:
        idx += 1
    if point[1] >= ym:
        idx += 2
    if point[2] >= zm:
        idx += 4
    return idx


def _trilinear_interpolate(point, bounds, values):
    """三线性插值

    Args:
        point: (3,) 查询点
        bounds: (xmin, ymin, zmin, xmax, ymax, zmax) 包围盒
        values: (8,) 8个角点的尺寸值

    Returns:
        float: 插值结果
    """
    xmin, ymin, zmin, xmax, ymax, zmax = bounds

    # 局部坐标
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    xi = (point[0] - xmin) / dx if dx > 0 else 0.0
    eta = (point[1] - ymin) / dy if dy > 0 else 0.0
    zeta = (point[2] - zmin) / dz if dz > 0 else 0.0

    # 钳位到 [0, 1]
    xi = max(0.0, min(1.0, xi))
    eta = max(0.0, min(1.0, eta))
    zeta = max(0.0, min(1.0, zeta))

    # 8个形函数 (Section 3.1.2)
    N = [
        (1 - xi) * (1 - eta) * (1 - zeta),  # 0: (0,0,0)
        xi * (1 - eta) * (1 - zeta),          # 1: (1,0,0)
        xi * eta * (1 - zeta),                # 2: (1,1,0)
        (1 - xi) * eta * (1 - zeta),          # 3: (0,1,0)
        (1 - xi) * (1 - eta) * zeta,          # 4: (0,0,1)
        xi * (1 - eta) * zeta,                # 5: (1,0,1)
        xi * eta * zeta,                      # 6: (1,1,1)
        (1 - xi) * eta * zeta,                # 7: (0,1,1)
    ]

    return sum(N[i] * values[i] for i in range(8))


class Octree:
    """八叉树背景网格

    用于存储和查询三维尺寸场值。

    Args:
        bounds: (xmin, ymin, zmin, xmax, ymax, zmax) 全局包围盒
        max_depth: 最大细分深度
        default_size: 默认尺寸值
    """

    def __init__(self, bounds, max_depth=8, default_size=1.0):
        self.bounds = bounds
        self.max_depth = max_depth
        self.default_size = default_size
        self.root = OctreeNode(bounds, level=0)
        self.root.size_values = np.full(8, default_size, dtype=np.float64)
        self._leaf_count = 1

    @property
    def leaf_count(self):
        """叶节点数量"""
        return self._leaf_count

    def contains(self, point):
        """检查点是否在八叉树范围内"""
        return _point_in_bounds(point, self.bounds)

    def query(self, point):
        """查询点处的尺寸值（三线性插值）

        Args:
            point: (3,) 查询点坐标

        Returns:
            float: 插值后的尺寸值
        """
        if not _point_in_bounds(point, self.bounds):
            return self.default_size

        node = self._find_leaf(point)
        if node is None or node.size_values is None:
            return self.default_size

        return _trilinear_interpolate(point, node.bounds, node.size_values)

    def insert_size(self, point, target_size, growth_rate=1.3):
        """在指定点插入尺寸约束

        细分八叉树以解析目标尺寸，并设置尺寸值。

        Args:
            point: (3,) 目标点坐标
            target_size: 目标尺寸值
            growth_rate: 增长率（用于相邻节点尺寸过渡）
        """
        if not _point_in_bounds(point, self.bounds):
            return

        # 从根节点开始，递归细分到足够精细
        self._insert_recursive(self.root, point, target_size, growth_rate)

    def refine_for_source(self, source, growth_rate=1.3):
        """为尺寸源细化八叉树

        Args:
            source: 尺寸源对象（需有 center, size 属性）
            growth_rate: 增长率
        """
        self.insert_size(source.center, source.size, growth_rate)

    def diffuse(self, growth_rate=1.3, max_iterations=50):
        """扩散尺寸值

        从已设置的尺寸源向外扩散，确保尺寸值平滑过渡。

        Args:
            growth_rate: 增长率
            max_iterations: 最大迭代次数
        """
        # 收集所有叶节点
        leaves = []
        self._collect_leaves(self.root, leaves)

        if not leaves:
            return

        # 多次迭代扩散
        for _ in range(max_iterations):
            changed = False
            for leaf in leaves:
                # 获取相邻叶节点
                neighbors = self._find_neighbors(leaf)
                for neighbor in neighbors:
                    if neighbor is None or neighbor.size_values is None:
                        continue

                    # 检查角点尺寸值
                    for i in range(8):
                        corner = self._get_corner_coords(leaf, i)
                        # 在相邻节点中查找最近的角点值
                        for j in range(8):
                            nb_corner = self._get_corner_coords(neighbor, j)
                            dist = np.linalg.norm(np.array(corner) - np.array(nb_corner))
                            if dist < 1e-10:
                                # 同一位置的角点
                                max_allowed = neighbor.size_values[j] * growth_rate
                                if leaf.size_values[i] > max_allowed:
                                    leaf.size_values[i] = max_allowed
                                    changed = True

            if not changed:
                break

    def enforce_gradient(self, max_gradient=1.3, max_iterations=100):
        """强制执行梯度限制

        确保相邻节点的尺寸比不超过 max_gradient。

        Args:
            max_gradient: 最大尺寸比
            max_iterations: 最大迭代次数
        """
        for _ in range(max_iterations):
            changed = False
            self._enforce_gradient_recursive(self.root, max_gradient, changed_ref := [False])
            if not changed_ref[0]:
                break

    def _enforce_gradient_recursive(self, node, max_gradient, changed_ref):
        """递归执行梯度限制"""
        if node.is_leaf:
            return

        # 检查子节点之间的梯度
        for i in range(8):
            child_i = node.children[i]
            if child_i is None or not child_i.is_leaf:
                continue
            for j in range(i + 1, 8):
                child_j = node.children[j]
                if child_j is None or not child_j.is_leaf:
                    continue

                # 检查共享角点
                for ci in range(8):
                    corner_i = self._get_corner_coords(child_i, ci)
                    for cj in range(8):
                        corner_j = self._get_corner_coords(child_j, cj)
                        dist = np.linalg.norm(np.array(corner_i) - np.array(corner_j))
                        if dist < 1e-10:
                            # 同一位置，检查梯度
                            si = child_i.size_values[ci]
                            sj = child_j.size_values[cj]
                            if si > 0 and sj > 0:
                                ratio = si / sj
                                if ratio > max_gradient:
                                    child_i.size_values[ci] = sj * max_gradient
                                    changed_ref[0] = True
                                elif ratio < 1.0 / max_gradient:
                                    child_j.size_values[cj] = si * max_gradient
                                    changed_ref[0] = True

        # 递归处理子节点
        for child in node.children:
            if child is not None and not child.is_leaf:
                self._enforce_gradient_recursive(child, max_gradient, changed_ref)

    def _find_leaf(self, point):
        """查找包含点的叶节点"""
        node = self.root
        while not node.is_leaf:
            idx = _get_child_index(point, node.bounds)
            node = node.children[idx]
            if node is None:
                return None
        return node

    def _insert_recursive(self, node, point, target_size, growth_rate):
        """递归插入尺寸约束"""
        if node.is_leaf:
            # 更新当前叶节点的尺寸值
            if node.size_values is None:
                node.size_values = np.full(8, self.default_size, dtype=np.float64)

            # 计算点到各角点的距离，设置尺寸值
            for i in range(8):
                corner = self._get_corner_coords(node, i)
                dist = np.linalg.norm(np.array(point) - np.array(corner))

                # 几何增长公式 (Section 4.1)
                if dist < 1e-10:
                    new_size = target_size
                else:
                    k = max(1, int(np.ceil(np.log(dist * (growth_rate - 1) / target_size + 1) / np.log(growth_rate))))
                    new_size = target_size * (growth_rate ** k)

                # 取较小值（更精细的尺寸优先）
                if new_size < node.size_values[i]:
                    node.size_values[i] = new_size

            # 检查是否需要细分
            cell_size = max(
                node.bounds[3] - node.bounds[0],
                node.bounds[4] - node.bounds[1],
                node.bounds[5] - node.bounds[2],
            )

            if cell_size > target_size * 2 and node.level < self.max_depth:
                self._subdivide(node)
                # 重新插入到子节点
                idx = _get_child_index(point, node.bounds)
                self._insert_recursive(node.children[idx], point, target_size, growth_rate)
        else:
            # 非叶节点，继续递归
            idx = _get_child_index(point, node.bounds)
            self._insert_recursive(node.children[idx], point, target_size, growth_rate)

    def _subdivide(self, node):
        """细分节点为8个子节点"""
        if not node.is_leaf:
            return

        child_bounds = _divide_bounds(node.bounds)
        node.children = []
        node.is_leaf = False
        self._leaf_count += 7  # 8 children - 1 leaf = +7

        for i in range(8):
            child = OctreeNode(child_bounds[i], level=node.level + 1)
            child.size_values = np.full(8, self.default_size, dtype=np.float64)
            node.children.append(child)

        # 将父节点的尺寸值插值到子节点角点
        self._interpolate_to_children(node)

    def _interpolate_to_children(self, node):
        """将父节点的尺寸值插值到子节点角点"""
        if node.is_leaf or node.children is None:
            return

        for child in node.children:
            if child is None:
                continue
            for i in range(8):
                corner = self._get_corner_coords(child, i)
                child.size_values[i] = _trilinear_interpolate(
                    corner, node.bounds, node.size_values
                )

    def _get_corner_coords(self, node, corner_idx):
        """获取节点指定角点的坐标"""
        xmin, ymin, zmin, xmax, ymax, zmax = node.bounds
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        xi, eta, zeta = CORNER_LOCAL[corner_idx]
        return (
            xmin + xi * dx,
            ymin + eta * dy,
            zmin + zeta * dz,
        )

    def _collect_leaves(self, node, leaves):
        """收集所有叶节点"""
        if node.is_leaf:
            leaves.append(node)
        else:
            for child in node.children:
                if child is not None:
                    self._collect_leaves(child, leaves)

    def _find_neighbors(self, node):
        """查找相邻叶节点（简化实现：遍历所有叶节点）"""
        # 注意：这是简化实现，生产环境应使用邻居指针
        all_leaves = []
        self._collect_leaves(self.root, all_leaves)

        neighbors = []
        node_center = self._get_node_center(node)
        node_size = max(
            node.bounds[3] - node.bounds[0],
            node.bounds[4] - node.bounds[1],
            node.bounds[5] - node.bounds[2],
        )

        for leaf in all_leaves:
            if leaf is node:
                continue
            leaf_center = self._get_node_center(leaf)
            dist = np.linalg.norm(np.array(node_center) - np.array(leaf_center))
            # 相邻节点中心距离约为 node_size
            if dist < node_size * 1.5:
                neighbors.append(leaf)

        return neighbors

    def _get_node_center(self, node):
        """获取节点中心坐标"""
        xmin, ymin, zmin, xmax, ymax, zmax = node.bounds
        return (
            (xmin + xmax) / 2,
            (ymin + ymax) / 2,
            (zmin + zmax) / 2,
        )
