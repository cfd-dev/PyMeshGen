import sys
import os
from pathlib import Path

# 获取项目根目录
project_root = str(Path(__file__).parent)

# 将项目根目录添加到sys.path，确保能导入core模块
sys.path.append(project_root)

# 导入核心网格生成模块
from core import generate_mesh
from data_structure.parameters import Parameters


# 全局GUI引用 - 保留用于向后兼容
_global_gui_instance = None


def set_gui_instance(gui_instance):
    """设置全局GUI实例（向后兼容）
    
    Args:
        gui_instance: GUI实例，用于在网格生成过程中输出信息和显示中间结果
    """
    global _global_gui_instance
    _global_gui_instance = gui_instance


def PyMeshGen(parameters=None, mesh_data=None):
    """PyMeshGen主函数（向后兼容）
    
    调用底层算法生成网格，支持两种调用方式：
    1. 通过parameters参数传递配置文件路径，从文件中读取网格数据
    2. 直接通过mesh_data参数传递网格数据
    
    Args:
        parameters: Parameters对象，包含网格生成的配置参数
    mesh_data: 可选，直接传入的网格数据对象，可以是Unstructured_Grid对象、字典或其他类型
    
    Returns:
        生成的网格数据
    """
    return generate_mesh(parameters, mesh_data, _global_gui_instance)


if __name__ == "__main__":
    """CLI入口
    
    命令行模式下运行网格生成器
    """
    import argparse

    parser = argparse.ArgumentParser(description="PyMeshGen 非结构网格生成器")
    parser.add_argument(
        "--case",
        type=str,
        default="",
        help="算例文件路径 (默认: 空)",
    )
    args = parser.parse_args()

    # 创建参数对象并应用命令行参数
    params = (
        Parameters("FROM_CASE_JSON", args.case)
        if args.case
        else Parameters("FROM_MAIN_JSON")
    )

    # 调用核心生成函数
    generate_mesh(params)

    input("Press Enter to continue...")
