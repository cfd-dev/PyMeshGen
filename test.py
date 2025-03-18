import sys
from pathlib import Path
import heapq

sys.path.append(str(Path(__file__).parent / "fileIO"))
sys.path.append(str(Path(__file__).parent / "data_structure"))
sys.path.append(str(Path(__file__).parent / "meshsize"))
sys.path.append(str(Path(__file__).parent / "visualization"))
import read_cas as rc
import front2d
import meshsize
import mesh_visualization as viz

# 使用示例文件测试解析结果
file_path = "./sample_grids/training/anw-hybrid.cas"
grid = rc.parse_fluent_msh(file_path)
fig, ax = viz.visualize_mesh_2d(grid, BoundaryOnly=True)

# 构造优先队列
front_heap = front2d.construct_initial_front(grid)

# 获取最小阵面
# while front_heap:
#     smallest = heapq.heappop(front_heap)
#     print(
#         f"边界类型: {smallest.bc_type}, 长度: {smallest.length:.4f}, 节点: {smallest.nodes}"
#     )
# 创建尺寸系统

sizing_system = meshsize.QuadtreeSizing(
    initial_front=front_heap, max_size=4, resolution=0.1, decay=1.2, fig=fig, ax=ax
)

# import torch
# print(f"PyTorch 版本: {torch.__version__}")          # 应显示 GPU 版本（如 2.0.0+cu117）
# print(f"CUDA 是否可用: {torch.cuda.is_available()}") # 应输出 True
# print(f"CUDA 版本: {torch.version.cuda}")            # 应输出 11.7

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

# import boundary_mesh_sample as bl_samp

# file_path = './sample/convex-60.cas'
# result = bl_samp.process_single_file(file_path)
