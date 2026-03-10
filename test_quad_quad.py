#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试quad_quad配置的多方向推进功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 重定向输出到文件
output_file = open("test_quad_quad_output.txt", "w", encoding="utf-8")
original_stdout = sys.stdout
sys.stdout = output_file

try:
    from core import generate_mesh
    from data_structure.parameters import Parameters

    print("=" * 70)
    print("测试quad_quad配置 - 多方向推进")
    print("=" * 70)

    # 创建Parameters对象并加载配置
    parameters = Parameters("FROM_CASE_JSON", "config/quad_quad.json")

    # 打印配置信息
    print(f"输入文件: {parameters.input_file}")
    print(f"输出文件: {parameters.output_file}")
    print(f"调试级别: {parameters.debug_level}")
    print(f"部件数量: {len(parameters.part_params)}")

    for i, part in enumerate(parameters.part_params):
        print(f"\n部件 {i+1}: {part.part_name}")
        print(f"  - max_size: {part.part_params.max_size}")
        print(f"  - first_height: {part.part_params.first_height}")
        print(f"  - max_layers: {part.part_params.max_layers}")
        print(f"  - full_layers: {part.part_params.full_layers}")
        print(f"  - multi_direction: {part.part_params.multi_direction}")
        print(f"  - PRISM_SWITCH: {part.part_params.PRISM_SWITCH}")

    # 生成网格
    print("\n开始生成网格...")
    mesh_data = generate_mesh(parameters)
    print("网格生成完成!")

    # 恢复stdout并输出
    sys.stdout = original_stdout
    output_file.close()

    # 读取并打印结果
    with open("test_quad_quad_output.txt", "r", encoding="utf-8") as f:
        content = f.read()
        print(content)

    # 分析网格结果
    print("=" * 70)
    print("网格生成结果分析")
    print("=" * 70)
    print(f"节点数量: {len(mesh_data.node_coords)}")
    print(f"单元数量: {len(mesh_data.cells)}")

    # 优先使用cell_container（保留单元对象和layer信息）
    cells_for_analysis = (
        mesh_data.cell_container
        if hasattr(mesh_data, "cell_container") and mesh_data.cell_container
        else mesh_data.cells
    )

    # 统计四边形和三角形数量
    num_quads = 0
    num_tris = 0
    for cell in cells_for_analysis:
        if isinstance(cell, list):
            n_nodes = len(cell)
        else:
            if hasattr(cell, "node_ids") and cell.node_ids is not None:
                n_nodes = len(cell.node_ids)
            else:
                n_nodes = len(cell.nodes) if hasattr(cell, "nodes") else 0

        if n_nodes == 4:
            num_quads += 1
        elif n_nodes == 3:
            num_tris += 1

    print(f"四边形数量: {num_quads}")
    print(f"三角形数量: {num_tris}")
    print("=" * 70)

    # 分析每层的单元分布
    print("\n按层分析单元分布:")
    layers = {}
    for cell in cells_for_analysis:
        layer = getattr(cell, 'layer', None)
        if layer is not None:
            if layer not in layers:
                layers[layer] = {'quads': 0, 'tris': 0}

            if isinstance(cell, list):
                n_nodes = len(cell)
            else:
                if hasattr(cell, "node_ids") and cell.node_ids is not None:
                    n_nodes = len(cell.node_ids)
                else:
                    n_nodes = len(cell.nodes) if hasattr(cell, "nodes") else 0

            if n_nodes == 4:
                layers[layer]['quads'] += 1
            elif n_nodes == 3:
                layers[layer]['tris'] += 1

    for layer in sorted(layers.keys()):
        print(f"  第{layer}层: {layers[layer]['quads']}个四边形 + {layers[layer]['tris']}个三角形")

    # 验证预期结果
    print("\n" + "=" * 70)
    print("预期结果验证")
    print("=" * 70)
    expected_first_layer = "16四边形 + 4三角形"
    expected_other_layers = "20四边形"

    all_passed = True
    if 1 in layers:
        actual_1 = f"{layers[1]['quads']}四边形 + {layers[1]['tris']}三角形"
        print(f"第一层: {actual_1} (预期: {expected_first_layer})")
        if layers[1]['quads'] == 16 and layers[1]['tris'] == 4:
            print("  ✓ 第一层符合预期")
        else:
            print("  ✗ 第一层不符合预期!")
            all_passed = False
    else:
        print("第一层: 未找到layer=1单元信息")
        all_passed = False

    for layer in sorted(layers.keys()):
        if layer > 1:
            actual = f"{layers[layer]['quads']}四边形"
            print(f"第{layer}层: {actual} (预期: {expected_other_layers})")
            if layers[layer]['quads'] == 20:
                print("  ✓ 符合预期")
            else:
                print("  ✗ 不符合预期!")
                all_passed = False

    if not all_passed:
        raise AssertionError("quad_quad第一层网格结果不符合预期: 需要16个四边形+4个三角形")

except Exception as e:
    sys.stdout = original_stdout
    output_file.close()
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()
