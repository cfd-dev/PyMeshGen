import json
from pathlib import Path
import sys
import os

# 添加必要的路径 (保持兼容性)
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

# 使用直接导入，避免相对导入问题
from basic_elements import Connector, Part

# 导入message模块
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
        self.mesh_type = 1 # 1-三角形triangular，2-直角三角形right_trianglar，3-三角形/四边形混合2d_mixed
        self.viz_enabled = False

        self.load_cofig()
        set_debug_level(self.debug_level)

    def check_main_fields(self, config):
        required_fields = [
            "debug_level",
            "input_file",
            "output_file",
            #"mesh_type",
            "parts",
            "viz_enabled",
        ]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"配置文件中缺少必要字段: {field}")

    def check_part_fields(self, part):
        # TODO 部件必须参数视情况添加
        required_fields = [
            "part_name",
            # "max_size",
            # "PRISM_SWITCH",
            # "first_height",
            # "growth_rate",
            # "growth_method",
            # "max_layers",
            # "full_layers",
            # "multi_direction",
        ]

        for field in required_fields:
            if field not in part:
                raise ValueError(f"配置文件中缺少必要字段: {field}")

    def open_config_file(self):
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

        return config

    def load_cofig(self):
        """加载配置文件"""
        config = self.open_config_file()

        # 必要字段校验
        self.check_main_fields(config)

        self.debug_level = config["debug_level"]
        self.input_file = config["input_file"]
        self.output_file = config["output_file"]
        self.mesh_type = config.get("mesh_type", 1)
        self.viz_enabled = config["viz_enabled"]

        self.part_params = []
        for part in config["parts"]:
            # 处理包含多曲线定义的部件
            if "curves" in part:
                # 创建单个部件对象并添加所有curves
                merged_params = {
                    **{k: v for k, v in part.items() if k != "curves"},
                    "curves": part["curves"],  # 保留所有curve定义
                }
                self._create_part_params(merged_params)
            else:
                self._create_part_params(part)

    def update_part_params_from_mesh(self, mesh_data):
        """根据导入的网格数据更新部件参数，如果网格数据中的部件不在配置中，则添加默认参数"""
        if not hasattr(mesh_data, 'parts_info') or not mesh_data.parts_info:
            return  # 如果没有部件信息，直接返回

        # 获取当前配置中的部件名称
        existing_part_names = [part.part_name for part in self.part_params]

        # 遍历网格数据中的所有部件
        for part_name in mesh_data.parts_info.keys():
            if part_name not in existing_part_names:
                # 如果网格数据中的部件不在当前配置中，添加默认参数
                default_param = MeshParameters(
                    part_name=part_name,
                    max_size=1.0,
                    PRISM_SWITCH="off",
                    first_height=0.1,
                    growth_rate=1.2,
                    growth_method="geometric",
                    max_layers=3,
                    full_layers=0,
                    multi_direction=False,
                )

                # 创建默认的connector
                default_connector = Connector(
                    part_name=part_name,
                    curve_name="default",
                    param=default_param,
                )

                # 创建Part对象并添加到part_params列表
                new_part = Part(part_name, default_param, [default_connector])
                self.part_params.append(new_part)
                print(f"添加新部件到参数配置: {part_name}")

    def _create_part_params(self, params):
        """创建部件参数对象（支持单curve和多curve模式）"""

        self.check_part_fields(params)  # 校验必要字段

        # 处理部件参数，优先使用part参数，如果没有定义，则使用默认值
        part_param = MeshParameters(
            part_name=params["part_name"],
            max_size=params.get("max_size", 1.0),
            PRISM_SWITCH=params.get("PRISM_SWITCH", "off"),
            first_height=params.get("first_height", 0.1),
            growth_rate=params.get("growth_rate", 1.2),
            growth_method=params.get("growth_method", "geometric"),
            max_layers=params.get("max_layers", 3),
            full_layers=params.get("full_layers", 0),
            multi_direction=params.get("multi_direction", False),
        )

        # 处理curve定义
        connectors = []
        if "curves" in params:
            for curve in params["curves"]:
                # 对于每个curve，优先使用自己的参数，如果自己没有定义参数，则使用部件参数
                curve_param = MeshParameters(
                    part_name=params["part_name"],
                    max_size=curve.get("max_size", part_param.max_size),
                    PRISM_SWITCH=curve.get("PRISM_SWITCH", part_param.PRISM_SWITCH),
                    first_height=curve.get("first_height", part_param.first_height),
                    growth_rate=curve.get("growth_rate", part_param.growth_rate),
                    growth_method=curve.get("growth_method", part_param.growth_method),
                    max_layers=curve.get("max_layers", part_param.max_layers),
                    full_layers=curve.get("full_layers", part_param.full_layers),
                    multi_direction=curve.get(
                        "multi_direction", part_param.multi_direction
                    ),
                )

                connectors.append(
                    Connector(
                        part_name=params["part_name"],
                        curve_name=curve.get("curve_name"),
                        param=curve_param,
                    )
                )

        # 对于每个part自动创建一个other线对象（connector），
        # 用于收集未明确定义参数的曲线，或者不存在curve定义的情况
        connectors.append(
            Connector(
                part_name=params["part_name"],
                curve_name="default",
                param=part_param,
            )
        )

        current_part = Part(params["part_name"], part_param, connectors)
        self.part_params.append(current_part)


class MeshParameters:
    """网格生成参数"""

    def __init__(
        self,
        part_name,
        max_size=1.0,
        PRISM_SWITCH="off",
        first_height=0.1,
        growth_rate=1.2,
        growth_method="geometric",
        max_layers=3,
        full_layers=0,
        multi_direction=False,
    ):
        self.part_name = part_name  # 部件名称
        self.max_size = max_size  # 最大网格尺寸
        self.PRISM_SWITCH = PRISM_SWITCH  # 是否生成边界层网格
        self.first_height = first_height  # 第一层网格高度
        self.max_layers = max_layers  # 最大推进层数
        self.full_layers = full_layers  # 完整推进层数
        self.growth_rate = growth_rate  # 网格高度增长比例
        self.growth_method = growth_method  # 网格高度增长方法
        self.multi_direction = multi_direction  # 是否多方向推进
