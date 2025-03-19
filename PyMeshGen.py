import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "fileIO"))
sys.path.append(str(Path(__file__).parent / "data_structure"))
sys.path.append(str(Path(__file__).parent / "meshsize"))
sys.path.append(str(Path(__file__).parent / "visualization"))
import read_cas as rc
import front2d
import meshsize
import mesh_visualization as viz


file_path = "./neural/sample_grids/training/30p30n-hybrid-sample.cas"
grid = rc.parse_fluent_msh(file_path)
fig, ax = viz.visualize_mesh_2d(grid, BoundaryOnly=True)

# 构造优先队列
front_heap = front2d.construct_initial_front(grid)

sizing_system = meshsize.QuadtreeSizing(
    initial_front=front_heap, max_size=4, resolution=0.1, decay=1.2, fig=fig, ax=ax
)
