import sys
import os
import json
import tempfile

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'data_structure'))
sys.path.insert(0, os.path.join(project_root, 'utils'))

from data_structure.parameters import Parameters
from gui.gui_new import SimplifiedPyMeshGenGUI
import tkinter as tk

# 创建一个简单的测试函数来验证GUI配置文件功能
def test_gui_config_functions():
    print("开始测试GUI配置文件功能...")
    
    # 创建一个临时的Tk根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 创建GUI实例
    gui = SimplifiedPyMeshGenGUI(root)
    
    # 测试创建配置
    print("1. 测试创建配置...")
    default_config = {
        "debug_level": 1,
        "input_file": "test.cas",
        "output_file": "test.vtk",
        "viz_enabled": True,
        "mesh_type": 3,
        "parts": [
            {
                "part_name": "airfoil",
                "part_params": {
                    "part_name": "airfoil",
                    "max_size": 0.5,
                    "PRISM_SWITCH": "on",
                    "first_height": 0.01,
                    "growth_rate": 1.2,
                    "growth_method": "geometric",
                    "max_layers": 5,
                    "full_layers": 2,
                    "multi_direction": False
                },
                "connectors": []
            }
        ]
    }
    
    # 测试从配置创建参数对象
    print("2. 测试从配置创建参数对象...")
    gui.create_params_from_config(default_config)
    
    if gui.params:
        print("   参数对象创建成功")
        print(f"   debug_level: {gui.params.debug_level}")
        print(f"   input_file: {gui.params.input_file}")
        print(f"   part_params数量: {len(gui.params.part_params)}")
    else:
        print("   参数对象创建失败")
    
    # 测试从参数对象创建配置
    print("3. 测试从参数对象创建配置...")
    config = gui.create_config_from_params()
    if config:
        print("   配置创建成功")
        print(f"   debug_level: {config['debug_level']}")
        print(f"   input_file: {config['input_file']}")
        print(f"   parts数量: {len(config['parts'])}")
    else:
        print("   配置创建失败")
    
    # 清理
    root.destroy()
    print("测试完成!")

if __name__ == "__main__":
    test_gui_config_functions()