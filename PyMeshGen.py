import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "fileIO"))
sys.path.append(str(Path(__file__).parent / "data_structure"))
sys.path.append(str(Path(__file__).parent / "meshsize"))
sys.path.append(str(Path(__file__).parent / "visualization"))
sys.path.append(str(Path(__file__).parent / "adfront2"))
sys.path.append(str(Path(__file__).parent / "optimize"))
import read_cas as rc
import front2d
import meshsize
import mesh_visualization as viz
import adfront2 as adfr
import optimize
from optimize import edge_swap, laplacian_smooth

# 读入边界网格
file_path = "./neural/sample_grids/inv_cylinder-8.cas"
grid = rc.parse_fluent_msh(file_path)
fig, ax = viz.visualize_mesh_2d(grid, BoundaryOnly=True)

# 构造初始阵面
front_heap = front2d.construct_initial_front(grid)

# 计算网格尺寸场
sizing_system = meshsize.QuadtreeSizing(
    initial_front=front_heap, max_size=4, resolution=0.1, decay=1.2, fig=fig, ax=ax
)
# sizing_system.draw_bgmesh()

# 推进生成网格
adfront2 = adfr.Adfront2(front_heap, sizing_system, ax=ax)
unstr_grid = adfront2.generate_elements()
unstr_grid.visualize_unstr_grid_2d()

# 网格质量优化
# unstr_grid = edge_swap(unstr_grid)
# unstr_grid.visualize_unstr_grid_2d()

unstr_grid = laplacian_smooth(unstr_grid)
unstr_grid.visualize_unstr_grid_2d()

# 输出网格文件
unstr_grid.save_to_vtkfile("./out/output_mesh.vtk")

input("Press Enter to continue...")
