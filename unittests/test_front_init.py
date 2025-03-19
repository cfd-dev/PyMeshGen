sys.path.append(str(Path(__file__).parent / "fileIO"))
sys.path.append(str(Path(__file__).parent / "visualization"))
import read_cas as rc
import mesh_visualization as viz

# 使用示例文件测试解析结果
file_path = "./test_files/naca0012-hybrid.cas"
grid = rc.parse_fluent_msh(file_path)
fig, ax = viz.visualize_mesh_2d(grid, BoundaryOnly=True)

# 构造优先队列
front_heap = front2d.construct_initial_front(grid)

# 获取最小阵面
while front_heap:
    smallest = heapq.heappop(front_heap)
    print(
        f"边界类型: {smallest.bc_type}, 长度: {smallest.length:.4f}, 节点: {smallest.nodes}"
    )
