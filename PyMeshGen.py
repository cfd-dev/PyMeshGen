import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "fileIO"))
sys.path.append(str(Path(__file__).parent / "data_structure"))
sys.path.append(str(Path(__file__).parent / "meshsize"))
sys.path.append(str(Path(__file__).parent / "visualization"))
sys.path.append(str(Path(__file__).parent / "adfront2"))
sys.path.append(str(Path(__file__).parent / "optimize"))

from read_cas import parse_fluent_msh
from front2d import construct_initial_front
from meshsize import QuadtreeSizing
from adfront2 import Adfront2
from optimize import edge_swap, laplacian_smooth
from adlayers2 import Adlayers2, PartMeshParameters
from mesh_visualization import visualize_mesh_2d

# 读入边界网格
file_path = "./neural/sample_grids/concave.cas"
grid = parse_fluent_msh(file_path)
fig, ax = visualize_mesh_2d(grid, BoundaryOnly=True)

# 构造初始阵面
front_heap = construct_initial_front(grid)

# 计算网格尺寸场
sizing_system = QuadtreeSizing(
    initial_front=front_heap, max_size=4, resolution=0.1, decay=1.2, fig=fig, ax=ax
)
# sizing_system.draw_bgmesh()

unstr_grid_list = []
# 推进生成边界层网格
part_params = PartMeshParameters("wall", 2.0, True, 0.1, 5, 5)
adlayers = Adlayers2([part_params], front_heap, ax)
boundary_grid, front_heap = adlayers.generate_elements()
unstr_grid_list.append(boundary_grid)

# 推进生成网格
adfront2 = Adfront2(front_heap, sizing_system, boundary_grid.node_coords, ax=ax)
triangular_grid = adfront2.generate_elements()

# 网格质量优化
triangular_grid = edge_swap(triangular_grid)
triangular_grid = laplacian_smooth(triangular_grid, 3)
# triangular_grid.visualize_unstr_grid_2d()
unstr_grid_list.append(triangular_grid)

# 合并各向同性网格和边界层网格
global_unstr_grid = unstr_grid_list[0]
for unstr_grid in unstr_grid_list[1:]:
    global_unstr_grid.merge(unstr_grid)

global_unstr_grid.visualize_unstr_grid_2d()

# 输出网格文件
global_unstr_grid.save_to_vtkfile("./out/final_mesh.vtk")

input("Press Enter to continue...")
