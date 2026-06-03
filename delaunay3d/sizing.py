class UniformSizing3D:
    """均匀尺寸场，返回恒定间距"""

    def __init__(self, spacing=1.0):
        self.spacing_value = spacing

    @property
    def global_spacing(self):
        return self.spacing_value

    def spacing_at(self, point):
        return self.spacing_value
