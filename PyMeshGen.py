import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "fileIO"))
sys.path.append(str(Path(__file__).parent / "data_structure"))
sys.path.append(str(Path(__file__).parent / "meshsize"))
sys.path.append(str(Path(__file__).parent / "visualization"))
sys.path.append(str(Path(__file__).parent / "adfront2"))
import read_cas as rc
import front2d
import meshsize
import mesh_visualization as viz
import adfront2 as adfr

# 读入边界网格
file_path = "./neural/sample_grids/training/30p30n-hybrid-sample.cas"
grid = rc.parse_fluent_msh(file_path)
fig, ax = viz.visualize_mesh_2d(grid, BoundaryOnly=True)

# 构造初始阵面
front_heap = front2d.construct_initial_front(grid)

# 计算网格尺寸场
sizing_system = meshsize.QuadtreeSizing(
    initial_front=front_heap, max_size=4, resolution=0.1, decay=1.2, fig=fig, ax=ax
)

adfront2 = adfr.Adfront2(front_heap, sizing_system)
adfront2.generate_elements()
# 推进生成网格

# 网格质量优化

# 可视化网格
