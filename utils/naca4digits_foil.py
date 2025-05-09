import numpy as np
import matplotlib.pyplot as plt


def naca4digit(number: str, num_points: int = 100):
    m = int(number[0]) / 100  # 最大弯度（%）
    p = int(number[1]) / 10  # 最大弯度位置（×10%）
    t = int(number[2:]) / 100  # 最大厚度（%）
    c = 1  # 翼弦长
    return naca4digit(m, p, t, c, num_points)


def naca4digit(m, p, t, c, num_points):
    """
    生成 NACA 四位数翼型的坐标点

    参数：
    m: 最大弯度（Camber），占弦长的百分比，取值范围为 0 到 9
    p: 最大弯度位置距离前缘的距离与弦长之比，取值范围为 0 到 1
    t: 翼型的最大厚度与弦长之比，取值范围为 0 到 1
    c: 翼弦长
    num_points: 生成坐标点的数量

    返回：
    coords: 形状为 (num_points * 2, 2) 的数组，表示翼型的坐标点
    example:
        NACA4412 翼型, 最大相对弯度4%弦长=0.04c，最大弯度位置4*10%c=0.4c，最大相对厚度12%=0.12c
        m = 0.04  # 最大相对弯度，最大弯度占弦长的百分比
        p = 0.4  # 最大弯度位置：最大弯曲处距离翼根的距离与弦长之比
        t = 0.12  # 最大相对厚度：翼型的最大厚度与弦长之比
    """

    # 指数增长因子
    r = 1.1

    # 初始化 x 数组
    x = np.zeros(num_points)
    x[0] = 0.0

    # 生成非均匀分布的 x 坐标
    for i in range(1, num_points):
        exponent = 100 - i + 1
        x[i] = x[i - 1] + (1 - x[i - 1]) / (r**exponent)

    # 归一化处理
    x = x / np.max(x)

    # 计算厚度分布 yt
    term = x / c
    yt = (
        5
        * t
        * c
        * (
            0.2969 * np.sqrt(term)
            - 0.1260 * term
            - 0.3516 * term**2
            + 0.2843 * term**3
            - 0.1036 * term**4
        )
    )

    # 计算中弧线 yc
    yc = np.zeros_like(x)

    # 分段计算 yc
    idx1 = np.where(x < p * c)
    if len(idx1[0]) > 0:
        term1 = x[idx1] / c
        yc[idx1] = (m / (p**2)) * (2 * p * term1 - term1**2)

    idx2 = np.where(x >= p * c)
    if len(idx2[0]) > 0:
        term2 = x[idx2] / c
        yc[idx2] = (m / ((1 - p) ** 2)) * ((1 - 2 * p) + 2 * p * term2 - term2**2)

    # 计算斜率角 theta
    theta = np.arctan2(yc, x)

    # 计算上下表面坐标
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # 合并坐标点：上表面反转后与下表面拼接
    upper = np.column_stack((xu[::-1], yu[::-1]))
    lower = np.column_stack((xl, yl))
    coords = np.vstack((upper, lower))

    return coords


if __name__ == "__main__":

    m = 0.06  # 最大弯度，占弦长的百分比
    p = 0.4  # 最大弯度位置：最大弯曲处距离翼根的距离与翼弦长之比
    t = 0.12  # 最大相对厚度：翼型的最大厚度与翼弦长之比
    coords = naca4digit(m, p, t, c=1.0, num_points=100)

    # 绘制翼型
    plt.plot(coords[:, 0], coords[:, 1], "b-")
    # 按照coords最大最小值限制坐标轴
    plt.xlim(np.min(coords[:, 0]), np.max(coords[:, 0]))
    plt.ylim(np.min(coords[:, 1]), np.max(coords[:, 1]))
    title = f"NACA {int(100*m)}{int(p*10)}{int(t*100)} Airfoil"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
