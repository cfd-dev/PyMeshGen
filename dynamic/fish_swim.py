import sys
from pathlib import Path
import numpy as np

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path / "fileIO"))
sys.path.append(str(root_path / "data_structure"))
sys.path.append(str(root_path / "meshsize"))
sys.path.append(str(root_path / "visualization"))
sys.path.append(str(root_path / "adfront2"))
sys.path.append(str(root_path / "optimize"))
sys.path.append(str(root_path / "utils"))

from read_cas import parse_fluent_msh
from mesh_visualization import Visualization
# from mesh_reconstruction import preprocess_grid

def RBF_func(dis, r0, basis="Wendland C2"):
    if basis == "Wendland C2":
        s = dis / r0
        return (1 - s) ** 4 * (4 * s + 1) * (s < 1)
    else:
        raise NotImplementedError("暂未实现其他基函数")
    
def extract_wall_nodes(input_grid):
    wall_nodes = set()
    for zone in input_grid["zones"].values():
        if zone["type"] == "faces" and zone.get("bc_type") == "wall":
            for face in zone["data"]:
                wall_nodes.update(face["nodes"])
    # 转为0-based索引
    return np.array(sorted([n - 1 for n in wall_nodes]), dtype=int)

def deform_wall(xCoord, yCoord, wallNodes, t, v, T, lamda, c):
    xCoord_new = xCoord.copy()
    yCoord_new = yCoord.copy()
    xCoord_new[wallNodes] += v * t
    nose_x = np.min(xCoord_new[wallNodes])
    x = xCoord_new[wallNodes] - nose_x
    y = yCoord[wallNodes]
    A = np.minimum(1, t / T) * (0.02 - 0.0825 * x + 0.1625 * x * x)
    dy = A * np.sin(2 * np.pi / lamda * (x - c * t))
    yCoord_new[wallNodes] = np.sign(y) * np.abs(y) + dy
    return xCoord_new, yCoord_new, dy

def compute_W(xCoord, yCoord, wallNodes, dy, r0, basis):
    nWall = len(wallNodes)
    X1 = xCoord[wallNodes][:, None]
    Y1 = yCoord[wallNodes][:, None]
    X2 = xCoord[wallNodes][None, :]
    Y2 = yCoord[wallNodes][None, :]
    dis = np.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2) + 1e-40
    fai = RBF_func(dis, r0, basis)
    return np.linalg.solve(fai, dy)

def deform_inner(xCoord, yCoord, wallNodes, W, t, v, r0, basis):
    nNodes = len(xCoord)
    nWall = len(wallNodes)
    xCoord_new = xCoord.copy()
    yCoord_new = yCoord.copy()
    mask = np.ones(nNodes, dtype=bool)
    mask[wallNodes] = False
    idx_inner = np.where(mask)[0]
    X = xCoord[idx_inner][:, None]
    Y = yCoord[idx_inner][:, None]
    Xw = xCoord[wallNodes][None, :]
    Yw = yCoord[wallNodes][None, :]
    dis = np.sqrt((X - Xw) ** 2 + (Y - Yw) ** 2) + 1e-40
    fai_inner = RBF_func(dis, r0, basis)
    dy_inner = np.dot(fai_inner, W)
    yCoord_new[idx_inner] += dy_inner
    xCoord_new[idx_inner] += v * t
    return xCoord_new, yCoord_new

if __name__ == "__main__":
    input_mesh = "unittests/test_files/2d_cases/naca0012.cas"
    input_grid = parse_fluent_msh(input_mesh)
    visual_obj = Visualization(True)
    visual_obj.plot_mesh(input_grid, boundary_only=False)

    xCoord = np.array([node[0] for node in input_grid['nodes']])
    yCoord = np.array([node[1] for node in input_grid['nodes']])
    wallNodes = extract_wall_nodes(input_grid)

    # 参数
    lamda = 1
    c = 0.1
    v = -0.2
    T = 2.0
    t = 0
    dt = 0.5
    r0 = 10.0
    basis = "Wendland C2"

    while t < 10:
        t += dt
        x_w, y_w, dy = deform_wall(xCoord, yCoord, wallNodes, t, v, T, lamda, c)

        W = compute_W(xCoord, yCoord, wallNodes, dy, r0, basis)

        x_new, y_new = deform_inner(x_w, y_w, wallNodes, W, t, v, r0, basis)

        visual_obj.plot_mesh_with_coords(input_grid, x_new, y_new, boundary_only=False)

        input("按回车继续到下一个时刻...")