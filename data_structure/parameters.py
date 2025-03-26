class Parameters:
    def __init__(self):
        self.debug_level = 2  # 调试级别，0-不输出，1-输出基本信息，2-输出详细信息


class PartMeshParameters:
    """网格生成部件参数"""

    def __init__(
        self,
        name,
        max_size=1.0,
        PRISM_SWITCH=False,
        first_height=0.1,
        max_layers=3,
        full_layers=0,
        growth_rate=1.2,
        growth_method="geometric",
        multi_direction=False,
    ):
        self.name = name  # 部件名称
        self.max_size = max_size  # 最大网格尺寸
        self.PRISM_SWITCH = PRISM_SWITCH  # 是否生成边界层网格
        self.first_height = first_height  # 第一层网格高度
        self.max_layers = max_layers  # 最大推进层数
        self.full_layers = full_layers  # 完整推进层数
        self.growth_rate = growth_rate  # 网格高度增长比例
        self.growth_method = growth_method  # 网格高度增长方法
        self.multi_direction = multi_direction  # 是否多方向推进
        self.front_list = []  # 阵面列表
