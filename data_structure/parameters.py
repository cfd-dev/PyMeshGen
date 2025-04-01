import json
from pathlib import Path
from message import set_debug_level


class Parameters:
    def __init__(self, get_param_from, json_file=None):
        # 配置文件来源，FROM_MAIN_JSON 或 FROM_CASE_JSON
        self.get_param_from = get_param_from
        self.config_file = []

        if json_file:
            json_file = Path(json_file).resolve()
        self.json_file = json_file

        # 算例参数
        self.debug_level = 0  # 调试级别，0-不输出，1-输出基本信息，2-输出详细信息
        self.part_params = []  # 网格生成部件参数
        self.input_file = []
        self.output_file = []
        self.viz_enabled = False

        self.load_cofig()
        set_debug_level(self.debug_level)

    # 配置文件读取函数
    def load_cofig(self):
        # 第一种方式，从main.json读取案例配置文件路径
        if self.get_param_from == "FROM_MAIN_JSON":
            self.json_file = Path(__file__).parent.parent / "config/main.json"
            with open(self.json_file, "r") as f1:
                main_config = json.load(f1)
                self.case_file = Path(main_config["case_file"]).resolve()
        elif self.get_param_from == "FROM_CASE_JSON":
            # 第二种方式，如果直接给定案例配置，则直接读取
            self.case_file = self.json_file

        with open(self.case_file, "r") as f2:
            config = json.load(f2)

        # 必要字段校验
        required_fields = [
            "debug_level",
            "input_file",
            "output_file",
            "parts",
            "viz_enabled",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"配置文件中缺少必要字段: {field}")

        self.debug_level = config["debug_level"]
        self.input_file = config["input_file"]
        self.output_file = config["output_file"]
        self.viz_enabled = config["viz_enabled"]

        self.part_params = []
        for part in config["parts"]:
            self.part_params.append(
                PartMeshParameters(
                    name=part["name"],
                    max_size=part["max_size"],
                    PRISM_SWITCH=part["PRISM_SWITCH"],
                    first_height=part["first_height"],
                    max_layers=part["max_layers"],
                    full_layers=part["full_layers"],
                    multi_direction=part["multi_direction"],
                )
            )


class PartMeshParameters:
    """网格生成部件参数"""

    def __init__(
        self,
        name,
        max_size=1.0,
        PRISM_SWITCH="off",
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
