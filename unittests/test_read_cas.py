import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "fileIO"))
sys.path.append(str(Path(__file__).parent / "visualization"))
import read_cas as rc
import mesh_visualization as viz

# 使用示例文件测试解析结果
file_path = "./test_files/naca0012-hybrid.cas"
grid = rc.parse_fluent_msh(file_path)
fig, ax = viz.visualize_mesh_2d(grid, BoundaryOnly=True)

# # 基础验证
# print(f"维度: {grid['dimensions']}")
# print(f"节点数量: {len(grid['nodes'])}")
# print(f"面数量: {sum(len(z['data']) for z in grid['zones'].values() if z['type']=='faces')}")
# print(f"边界条件数量: {sum(1 for z in grid['zones'].values() if z.get('bc_type'))}")

# # 验证wall面解析
# wall_zones = [z for z in grid['zones'].values()
#               if z['type'] == 'faces' and z.get('bc_type') == 'wall']
# print(f"Wall面数量: {sum(len(z['data']) for z in wall_zones)}")

# # 验证节点坐标
# print("前5个节点坐标示例:")
# for node in grid['nodes'][:5]:
#     print(node)
