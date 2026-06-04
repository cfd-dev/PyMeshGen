class UniformSizing3D:
    """均匀尺寸场，返回恒定间距"""

    def __init__(self, spacing=1.0):
        self.spacing_value = spacing

    @property
    def global_spacing(self):
        return self.spacing_value

    def spacing_at(self, point):
        return self.spacing_value


class SurfaceSizing3D:
    """基于表面网格的尺寸场，使用最近邻查询"""

    def __init__(self, surface_triangles):
        import numpy as np
        self._triangles = surface_triangles
        self._centers = []
        self._sizes = []
        self._global_spacing = 0.0

        # 计算每个三角形的中心和平均边长
        all_sizes = []
        for tri in surface_triangles:
            nodes = tri.nodes
            center = [0.0, 0.0, 0.0]
            for n in nodes:
                center[0] += n.coords[0]
                center[1] += n.coords[1]
                center[2] += n.coords[2]
            center[0] /= 3
            center[1] /= 3
            center[2] /= 3
            self._centers.append(center)

            # 最小边长（更保守的间距）
            min_edge = float('inf')
            for i in range(3):
                for j in range(i + 1, 3):
                    dx = nodes[i].coords[0] - nodes[j].coords[0]
                    dy = nodes[i].coords[1] - nodes[j].coords[1]
                    dz = nodes[i].coords[2] - nodes[j].coords[2]
                    edge_len = (dx * dx + dy * dy + dz * dz) ** 0.5
                    min_edge = min(min_edge, edge_len)
            self._sizes.append(min_edge)
            all_sizes.append(min_edge)

        self._centers = np.array(self._centers)
        self._sizes = np.array(self._sizes)
        self._global_spacing = float(np.mean(all_sizes)) if all_sizes else 1.0

    @property
    def global_spacing(self):
        return self._global_spacing

    def spacing_at(self, point):
        """查询最近表面三角形的间距"""
        import numpy as np
        pt = np.array(point)
        dists = np.linalg.norm(self._centers - pt, axis=1)
        idx = np.argmin(dists)
        return float(self._sizes[idx])
