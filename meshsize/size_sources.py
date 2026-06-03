"""尺寸源类型

定义不同类型的尺寸源，用于指定局部区域的网格尺寸。

参考: 第3.5-3.9节
"""
import numpy as np
import math


class PointSource:
    """点源（带指数增长）

    在中心点指定尺寸，向外按几何级数增长。

    Args:
        center: (3,) 中心坐标
        size: 中心处的目标尺寸
        growth: 增长率（>1，越大增长越快）
    """

    def __init__(self, center, size, growth=1.3):
        self.center = list(center)
        self.size = max(size, 1e-30)
        self.growth = max(growth, 1.01)

    def size_at(self, point):
        """计算指定点处的尺寸

        使用几何增长公式 (Section 4.1):
        h(d) = h0 * r^k
        k = ceil(log(d*(r-1)/h0 + 1) / log(r))

        Args:
            point: (3,) 查询点坐标

        Returns:
            float: 该点处的尺寸值
        """
        dist = np.linalg.norm(np.array(point) - np.array(self.center))
        if dist < 1e-10:
            return self.size

        r = self.growth
        h0 = self.size

        # 求解层数 k
        arg = dist * (r - 1.0) / h0 + 1.0
        if arg <= 0:
            return self.size
        k = max(1, int(math.ceil(math.log(arg) / math.log(r))))

        return h0 * (r ** k)

    def distance_to(self, point):
        """到中心点的距离"""
        return np.linalg.norm(np.array(point) - np.array(self.center))


class BoxSource:
    """盒源（轴对齐盒，可选旋转）

    在盒内指定固定尺寸，盒外按距离增长。

    Args:
        center: (3,) 中心坐标
        half_spans: (3,) 三个方向的半跨度
        size: 盒内的目标尺寸
        growth: 盒外增长率
        rotation: (3,) 旋转角度（度），绕 x/y/z 轴
    """

    def __init__(self, center, half_spans, size, growth=1.3, rotation=None):
        self.center = np.array(center, dtype=np.float64)
        self.half_spans = np.array(half_spans, dtype=np.float64)
        self.size = max(size, 1e-30)
        self.growth = max(growth, 1.01)
        self.rotation = np.array(rotation if rotation is not None else [0, 0, 0],
                                  dtype=np.float64)

        # 构建旋转矩阵
        self._rot_matrix = self._build_rotation_matrix()
        self._inv_rot_matrix = self._rot_matrix.T

    def _build_rotation_matrix(self):
        """构建旋转矩阵（绕 x, y, z 轴依次旋转）"""
        rx, ry, rz = np.radians(self.rotation)

        # 绕 x 轴
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)],
        ])

        # 绕 y 轴
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)],
        ])

        # 绕 z 轴
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1],
        ])

        return Rz @ Ry @ Rx

    def _to_local(self, point):
        """将世界坐标转换为局部坐标（逆旋转）"""
        p = np.array(point, dtype=np.float64) - self.center
        return self._inv_rot_matrix @ p

    def contains(self, point):
        """检查点是否在盒内

        Args:
            point: (3,) 查询点坐标

        Returns:
            bool: True 表示在盒内
        """
        local = self._to_local(point)
        return all(abs(local[i]) <= self.half_spans[i] for i in range(3))

    def distance_to(self, point):
        """计算点到盒表面的距离

        Args:
            point: (3,) 查询点坐标

        Returns:
            float: 到盒表面的距离（盒内为0）
        """
        local = self._to_local(point)

        # 计算到各面的距离
        dist_sq = 0.0
        for i in range(3):
            excess = abs(local[i]) - self.half_spans[i]
            if excess > 0:
                dist_sq += excess ** 2

        return math.sqrt(dist_sq)

    def size_at(self, point):
        """计算指定点处的尺寸

        盒内返回指定尺寸，盒外按距离增长。

        Args:
            point: (3,) 查询点坐标

        Returns:
            float: 该点处的尺寸值
        """
        dist = self.distance_to(point)
        if dist < 1e-10:
            return self.size

        # 线性增长 (Section 3.6)
        return self.size + (self.growth - 1.0) * dist


class SphereSource:
    """球源

    在球内指定固定尺寸，球外按距离增长。

    Args:
        center: (3,) 中心坐标
        radius: 球半径
        size: 球内的目标尺寸
        growth: 球外增长率
    """

    def __init__(self, center, radius, size, growth=1.3):
        self.center = list(center)
        self.radius = max(radius, 1e-30)
        self.size = max(size, 1e-30)
        self.growth = max(growth, 1.01)

    def contains(self, point):
        """检查点是否在球内

        Args:
            point: (3,) 查询点坐标

        Returns:
            bool: True 表示在球内
        """
        dist = np.linalg.norm(np.array(point) - np.array(self.center))
        return dist <= self.radius

    def distance_to(self, point):
        """计算点到球表面的距离

        Args:
            point: (3,) 查询点坐标

        Returns:
            float: 到球表面的距离（球内为0）
        """
        dist = np.linalg.norm(np.array(point) - np.array(self.center))
        return max(0.0, dist - self.radius)

    def size_at(self, point):
        """计算指定点处的尺寸

        球内返回指定尺寸，球外按距离增长。

        Args:
            point: (3,) 查询点坐标

        Returns:
            float: 该点处的尺寸值
        """
        dist = self.distance_to(point)
        if dist < 1e-10:
            return self.size

        # 线性增长
        return self.size + (self.growth - 1.0) * dist
