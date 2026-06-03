"""三维尺寸场控制器

管理尺寸源、八叉树背景网格，提供统一的尺寸查询接口。

遵循 duck-typed 协议:
- global_spacing: 全局参考尺寸
- spacing_at(point): 查询指定点处的局部尺寸

参考: 第3-4节
"""
import numpy as np
from meshsize.octree import Octree
from meshsize.size_sources import PointSource, BoxSource, SphereSource
from utils.message import info


class SizeField3D:
    """三维自适应尺寸场

    Args:
        surface_triangles: 表面三角形列表（可选，用于自动初始化）
        max_size: 全局最大尺寸
        min_size: 全局最小尺寸
        growth_rate: 尺寸增长率
        max_depth: 八叉树最大深度
    """

    def __init__(self, surface_triangles=None, max_size=1.0, min_size=0.0,
                 growth_rate=1.3, max_depth=8):
        self.global_spacing = max_size
        self.min_size = min_size
        self.growth_rate = growth_rate
        self.max_depth = max_depth
        self.sources = []
        self.octree = None
        self._built = False

        # 自动从表面网格初始化
        if surface_triangles is not None:
            self._init_from_surface(surface_triangles)

    def add_point_source(self, center, size, growth=None):
        """添加点源

        Args:
            center: (3,) 中心坐标
            size: 目标尺寸
            growth: 增长率（None 使用全局增长率）
        """
        if growth is None:
            growth = self.growth_rate
        source = PointSource(center, size, growth)
        self.sources.append(source)
        self._built = False

    def add_box_source(self, center, half_spans, size, growth=None, rotation=None):
        """添加盒源

        Args:
            center: (3,) 中心坐标
            half_spans: (3,) 三个方向的半跨度
            size: 盒内目标尺寸
            growth: 增长率（None 使用全局增长率）
            rotation: (3,) 旋转角度（度）
        """
        if growth is None:
            growth = self.growth_rate
        source = BoxSource(center, half_spans, size, growth, rotation)
        self.sources.append(source)
        self._built = False

    def add_sphere_source(self, center, radius, size, growth=None):
        """添加球源

        Args:
            center: (3,) 中心坐标
            radius: 球半径
            size: 球内目标尺寸
            growth: 增长率（None 使用全局增长率）
        """
        if growth is None:
            growth = self.growth_rate
        source = SphereSource(center, radius, size, growth)
        self.sources.append(source)
        self._built = False

    def spacing_at(self, point):
        """查询指定点处的局部尺寸

        Args:
            point: (3,) 查询点坐标

        Returns:
            float: 局部尺寸值
        """
        if self.octree is None:
            return self.global_spacing

        # 如果未构建，先构建八叉树
        if not self._built:
            self.build()

        return self.octree.query(point)

    def build(self):
        """构建尺寸场

        将所有尺寸源处理到八叉树中，并执行扩散和梯度限制。
        """
        if self.octree is None:
            # 从源的包围盒创建八叉树
            bounds = self._compute_bounds()
            self.octree = Octree(bounds, self.max_depth, self.global_spacing)

        # 处理所有尺寸源
        for source in self.sources:
            self.octree.refine_for_source(source, self.growth_rate)

        # 扩散
        self.octree.diffuse(self.growth_rate)

        # 梯度限制
        self.octree.enforce_gradient(self.growth_rate)

        self._built = True
        info(f"3D 尺寸场构建完成: 叶节点={self.octree.leaf_count}, 源数量={len(self.sources)}")

    def _init_from_surface(self, surface_triangles):
        """从表面网格初始化尺寸场

        Args:
            surface_triangles: 表面三角形列表
        """
        if not surface_triangles:
            return

        info(f"从表面网格初始化 3D 尺寸场...")

        # 1. 计算包围盒和全局尺寸
        all_coords = []
        min_edge = float('inf')
        max_edge = 0.0

        for tri in surface_triangles:
            for node in tri.nodes:
                all_coords.append(node.coords)

            # 计算边长
            for i in range(3):
                for j in range(i + 1, 3):
                    dx = tri.nodes[i].coords[0] - tri.nodes[j].coords[0]
                    dy = tri.nodes[i].coords[1] - tri.nodes[j].coords[1]
                    dz = tri.nodes[i].coords[2] - tri.nodes[j].coords[2]
                    edge_len = (dx * dx + dy * dy + dz * dz) ** 0.5
                    min_edge = min(min_edge, edge_len)
                    max_edge = max(max_edge, edge_len)

        # 2. 设置全局尺寸
        self.global_spacing = max_edge
        self.min_size = min_edge * 0.5

        # 3. 创建八叉树
        coords = np.array(all_coords)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        # 扩展包围盒
        extent = maxs - mins
        margin = max(extent) * 0.05
        bounds = (
            mins[0] - margin, mins[1] - margin, mins[2] - margin,
            maxs[0] + margin, maxs[1] + margin, maxs[2] + margin,
        )
        self.octree = Octree(bounds, self.max_depth, self.global_spacing)

        # 4. 插入表面边长约束
        for tri in surface_triangles:
            center = np.mean([n.coords for n in tri.nodes], axis=0)
            # 使用三角形的最小边长作为目标尺寸
            tri_min_edge = float('inf')
            for i in range(3):
                for j in range(i + 1, 3):
                    dx = tri.nodes[i].coords[0] - tri.nodes[j].coords[0]
                    dy = tri.nodes[i].coords[1] - tri.nodes[j].coords[1]
                    dz = tri.nodes[i].coords[2] - tri.nodes[j].coords[2]
                    edge_len = (dx * dx + dy * dy + dz * dz) ** 0.5
                    tri_min_edge = min(tri_min_edge, edge_len)

            self.octree.insert_size(center, tri_min_edge, self.growth_rate)

        # 5. 扩散和梯度限制
        self.octree.diffuse(self.growth_rate)
        self.octree.enforce_gradient(self.growth_rate)

        self._built = True
        info(f"  全局尺寸: {self.global_spacing:.4f}")
        info(f"  最小尺寸: {self.min_size:.4f}")
        info(f"  八叉树叶节点: {self.octree.leaf_count}")

    def _compute_bounds(self):
        """从尺寸源计算包围盒"""
        if not self.sources:
            # 默认包围盒
            return (-1, -1, -1, 1, 1, 1)

        all_points = []
        for source in self.sources:
            all_points.append(source.center)

        coords = np.array(all_points)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        # 扩展包围盒
        extent = maxs - mins
        margin = max(extent) * 0.5 if max(extent) > 0 else 1.0

        return (
            mins[0] - margin, mins[1] - margin, mins[2] - margin,
            maxs[0] + margin, maxs[1] + margin, maxs[2] + margin,
        )

    def get_stats(self):
        """获取尺寸场统计信息

        Returns:
            dict: 统计信息
        """
        if self.octree is None:
            return {
                'global_spacing': self.global_spacing,
                'min_size': self.min_size,
                'num_sources': len(self.sources),
                'octree_leaves': 0,
            }

        # 收集所有叶节点的尺寸值
        leaves = []
        self.octree._collect_leaves(self.octree.root, leaves)

        all_sizes = []
        for leaf in leaves:
            if leaf.size_values is not None:
                all_sizes.extend(leaf.size_values)

        if not all_sizes:
            return {
                'global_spacing': self.global_spacing,
                'min_size': self.min_size,
                'num_sources': len(self.sources),
                'octree_leaves': self.octree.leaf_count,
            }

        return {
            'global_spacing': self.global_spacing,
            'min_size': min(all_sizes),
            'max_size': max(all_sizes),
            'mean_size': sum(all_sizes) / len(all_sizes),
            'num_sources': len(self.sources),
            'octree_leaves': self.octree.leaf_count,
        }
