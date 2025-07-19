import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 设置根目录路径，并将其加入sys.path，方便导入本地模块
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# 导入自定义的网格读取和可视化模块
from fileIO.read_cas import parse_fluent_msh
from visualization.mesh_visualization import Visualization
from utils.message import info, warning, error, debug

def RBF_func(rn, r0, basis="Wendland C2"):
    """
    通用RBF基函数，支持多种类型（字符串选择），支持向量化输入。
    rn: 距离（可为ndarray）
    r0: 紧支半径
    basis: 基函数类型（字符串）
        "Volume Spline"
        "Thin Plate Spline"
        "Multi-Quadric"
        "Inverse Multi-Quadric"
        "Inverse Quadric"
        "Wendland C0"
        "Wendland C2"
        "Wendland C4"
        "Wendland C6"
        "compact TPS C0"
        "compact TPS C1"
        "compact TPS C2a"
        "compact TPS C2b"
    返回：RBF值（与rn同shape）
    """
    rn = np.asarray(rn)
    if basis == "Volume Spline":
        fai = rn
    elif basis == "Thin Plate Spline":
        with np.errstate(divide='ignore', invalid='ignore'):
            fai = rn**2 * np.log10(np.where(rn > 0, rn, 1))
            fai[rn == 0] = 0.0
    elif basis == "Multi-Quadric":
        c = 1e-4
        fai = np.sqrt(c**2 + rn**2)
    elif basis == "Inverse Multi-Quadric":
        c = 1e-4
        fai = 1.0 / np.sqrt(c**2 + rn**2)
    elif basis == "Inverse Quadric":
        fai = 1.0 / (1 + rn**2)
    else:
        # 紧支基函数
        ksi = rn / r0
        fai = np.zeros_like(ksi)
        mask = ksi < 1
        if basis == "Wendland C0":
            fai[mask] = (1 - ksi[mask])**2
        elif basis == "Wendland C2":
            fai[mask] = (1 - ksi[mask])**4 * (4.0 * ksi[mask] + 1)
        elif basis == "Wendland C4":
            fai[mask] = (1 - ksi[mask])**6 * (35.0 * ksi[mask]**2 + 18.0 * ksi[mask] + 3)
        elif basis == "Wendland C6":
            fai[mask] = (1 - ksi[mask])**8 * (32.0 * ksi[mask]**3 + 25.0 * ksi[mask]**2 + 8.0 * ksi[mask] + 1)
        elif basis == "compact TPS C0":
            fai[mask] = (1 - ksi[mask])**5
        elif basis == "compact TPS C1":
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ksi = np.zeros_like(ksi)
                log_ksi[mask] = np.log(ksi[mask])
                fai[mask] = 3 + 80 * ksi[mask]**2 - 120 * ksi[mask]**3 + 45 * ksi[mask]**4 - 8 * ksi[mask]**5 + 60 * ksi[mask]**2 * log_ksi[mask]
        elif basis == "compact TPS C2a":
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ksi = np.zeros_like(ksi)
                log_ksi[mask] = np.log(ksi[mask])
                fai[mask] = 1 - 30 * ksi[mask]**2 - 10 * ksi[mask]**3 + 45 * ksi[mask]**4 - 6 * ksi[mask]**5 - 60 * ksi[mask]**3 * log_ksi[mask]
        elif basis == "compact TPS C2b":
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ksi = np.zeros_like(ksi)
                log_ksi[mask] = np.log(ksi[mask])
                fai[mask] = 1 - 20 * ksi[mask]**2 + 80 * ksi[mask]**3 - 45 * ksi[mask]**4 - 16 * ksi[mask]**5 + 60 * ksi[mask]**4 * log_ksi[mask]
        else:
            raise ValueError(f"未知的RBF基函数类型: {basis}")
        # 超出支撑半径的部分加极小值，防止数值问题
        fai[~mask] = 1e-40
        fai = 1 - fai
    return fai
    
def extract_wall_nodes(input_grid):
    """
    提取网格中物面（wall）节点的索引，返回0-based索引数组。
    参数:
        input_grid: 解析后的网格数据结构
    返回:
        wall_nodes: 物面节点索引数组
    """
    wall_nodes = set()
    for zone in input_grid["zones"].values():
        if zone["type"] == "faces" and zone.get("bc_type") == "wall":
            for face in zone["data"]:
                wall_nodes.update(face["nodes"])
    # 转为0-based索引
    return np.array(sorted([n - 1 for n in wall_nodes]), dtype=int)

def deform_wall(xCoord, yCoord, wallNodes, t, v, T, lamda, c):
    """
    计算物面节点在当前时刻的变形（鱼体游动），返回新的x、y坐标和y方向位移dy
    参数:
        xCoord, yCoord: 所有节点的原始坐标
        wallNodes: 物面节点索引
        t: 当前时刻
        v: 游动速度
        T: 周期
        lamda: 波长
        c: 波速
    返回:
        xCoord_new, yCoord_new: 变形后的节点坐标
        dy: 物面节点y方向位移
    """
    xCoord_new = xCoord.copy()
    yCoord_new = yCoord.copy()
    # 物面节点x方向整体平移
    xCoord_new[wallNodes] += v * t
    # 计算鱼头位置
    nose_x = np.min(xCoord_new[wallNodes])
    x = xCoord_new[wallNodes] - nose_x
    y = yCoord[wallNodes]
    # 计算振幅
    A = np.minimum(1, t / T) * (0.02 - 0.0825 * x + 0.1625 * x * x)
    # 计算y方向变形
    dy = A * np.sin(2 * np.pi / lamda * (x - c * t))
    # 更新物面节点y坐标
    yCoord_new[wallNodes] = np.sign(y) * np.abs(y) + dy
    return xCoord_new, yCoord_new, dy

def compute_W(xCoord, yCoord, wallNodes, dy, r0, basis):
    """
    计算RBF插值的权重系数W，使物面节点的插值正好等于dy
    参数:
        xCoord, yCoord: 所有节点的原始坐标
        wallNodes: 物面节点索引
        dy: 物面节点y方向位移
        r0: RBF紧支半径
        basis: RBF基函数类型
    返回:
        W: 权重系数数组
    """
    # 构造距离矩阵
    # 注意插值系数计算都用原始坐标进行，而非变形后的坐标
    X1 = xCoord[wallNodes][:, None]  # 提取物面节点x坐标，形状为 (nWallNodes, 1)
    Y1 = yCoord[wallNodes][:, None]  # 提取物面节点y坐标，形状为 (nWallNodes, 1)
    X2 = xCoord[wallNodes][None, :]  # 提取物面节点x坐标，形状为 (1, nWallNodes)
    Y2 = yCoord[wallNodes][None, :]  # 提取物面节点y坐标，形状为 (1, nWallNodes)
    dis = np.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2) + 1e-40
    # 计算RBF矩阵
    fai = RBF_func(dis, r0, basis)
    # # 求解线性方程组
    return np.linalg.solve(fai, dy)

def deform_interior_nodes(x_w, y_w, xCoord, yCoord, wallNodes, W, t, v, r0, basis):
    """
    利用RBF插值和权重W，计算内场节点的变形。
    参数:
        x_w, y_w: 当前物面已经变形后的节点坐标
        xCoord, yCoord: 所有节点的原始坐标
        wallNodes: 物面节点索引
        W: RBF权重
        t, v, r0, basis: 相关参数
    返回:
        xCoord_new, yCoord_new: 变形后的所有节点坐标
    """
    nNodes = len(xCoord)
    xCoord_new = x_w.copy()
    yCoord_new = y_w.copy()
    # 标记内场节点
    mask = np.ones(nNodes, dtype=bool)
    mask[wallNodes] = False
    idx_inner = np.where(mask)[0]
    # 计算内场节点到所有物面节点的距离,
    # 注意插值系数计算都用原始坐标进行，而非变形后的坐标
    X = xCoord[idx_inner][:, None]
    Y = yCoord[idx_inner][:, None]
    Xw = xCoord[wallNodes][None, :]
    Yw = yCoord[wallNodes][None, :]
    dis = np.sqrt((X - Xw) ** 2 + (Y - Yw) ** 2) + 1e-40
    # 计算RBF插值
    fai_inner = RBF_func(dis, r0, basis)
    dy_inner = np.dot(fai_inner, W)
    # 更新内场节点坐标
    yCoord_new[idx_inner] = y_w[idx_inner] + dy_inner
    xCoord_new[idx_inner] = x_w[idx_inner] + v * t
    return xCoord_new, yCoord_new

def update_grid_coords(dynamic_grid, x_new, y_new):
    """
    用新的x、y坐标更新网格节点信息
    参数:
        dynamic_grid: 网格数据结构
        x_new, y_new: 新坐标
    """
    for i, node in enumerate(dynamic_grid["nodes"]):
        dynamic_grid["nodes"][i] = (x_new[i], y_new[i]) if len(node) == 2 else (x_new[i], y_new[i], node[2])
    return dynamic_grid

if __name__ == "__main__":
    # 读取初始网格文件
    initial_mesh_file = "unittests/test_files/2d_cases/naca0012-tri_coarse.cas"
    initial_grid = parse_fluent_msh(initial_mesh_file)
    # 创建可视化对象并显示初始网格
    visual_obj = Visualization(True)
    visual_obj.plot_mesh(initial_grid, boundary_only=False)
    visual_obj.set_range(-3, 1, -0.5, 0.5)
    # input("按回车继续到下一个时刻...")
    visual_obj.save_png(f"./out/t=0.jpg")

    # 提取节点坐标和物面节点索引
    xCoord = np.array([node[0] for node in initial_grid['nodes']])
    yCoord = np.array([node[1] for node in initial_grid['nodes']])
    wallNodes = extract_wall_nodes(initial_grid)

    # 物理和运动参数
    lamda = 1      # 波长
    c = 0.1        # 波速
    v = -0.2       # 游动速度
    T = 2.0        # 周期
    t = 0          # 起始时间
    dt = 0.5       # 时间步长
    r0 = 2.0      # RBF紧支半径
    basis = "Wendland C2"  # RBF基函数类型Wendland C2，Volume Spline
    
    # 拷贝初始网格用于动态变形
    dynamic_grid = initial_grid.copy()
    
    # 时间推进主循环
    while t < 10:
        t += dt
        info(f"t = {t:.2f} ")

        # 1. 物面节点变形
        info("1. 正在计算物面节点变形...")
        x_w, y_w, dy = deform_wall(xCoord, yCoord, wallNodes, t, v, T, lamda, c)

        # 2. 计算RBF权重
        info("2. 正在计算RBF权重系数...")
        W = compute_W(xCoord, yCoord, wallNodes, dy, r0, basis)

        # 3. 内场节点变形
        info("3. 正在插值计算内场节点变形...")
        x_new, y_new = deform_interior_nodes(x_w, y_w, xCoord, yCoord, wallNodes, W, t, v, r0, basis)

        # 4. 更新网格节点坐标
        info("4. 正在更新网格节点坐标...")
        update_grid_coords(dynamic_grid, x_new, y_new)

        # 5. 新建figure并绘制当前网格
        info("5. 正在刷新可视化窗口...")
        visual_obj.ax.clear()
        visual_obj.plot_mesh(dynamic_grid, boundary_only=False)
        visual_obj.set_range(-3, 1, -0.5, 0.5)
        visual_obj.save_png(f"./out/t={t}.jpg")
        plt.pause(0.1)

        # 6. 等待用户输入，进入下一步
        # input(f"t={t}, 按回车继续到下一个时刻...")
    input("时间推进结束！")