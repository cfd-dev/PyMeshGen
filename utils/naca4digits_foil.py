from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class NACA4DigitFoil:
    def __init__(self, *args, **kwargs):
        """
        初始化 NACA 四位数翼型

        支持两种调用方式：
        1. NACA4DigitFoil("4412", num_points=100)
        2. NACA4DigitFoil(m=0.04, p=0.4, t=0.12, c=1.0, num_points=100)

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

        if len(args) == 1 and isinstance(args[0], str):
            # 处理字符串参数调用方式
            self.number = args[0]
            self.num_points = kwargs.get("num_points", 100)
            self.m = int(self.number[0]) / 100
            self.p = int(self.number[1]) / 10
            self.t = int(self.number[2:]) / 100
            self.c = 1.0
        else:
            # 处理关键字参数调用方式
            self.m = kwargs.get("m", 0.0)
            self.p = kwargs.get("p", 0.0)
            self.t = kwargs.get("t", 0.12)
            self.c = kwargs.get("c", 1.0)
            self.num_points = kwargs.get("num_points", 100)
            self.number = f"{int(self.m*100)}{int(self.p*10):01d}{int(self.t*100):02d}"
        self.coords = self.generate_coords()  # 生成坐标点

    def generate_coords(self):
        """
        生成 NACA 四位数翼型的坐标点
        """

        # 指数增长因子
        r = 1.1

        # 初始化 x 数组
        x = np.zeros(self.num_points)
        x[0] = 0.0

        # 生成非均匀分布的 x 坐标
        for i in range(1, self.num_points):
            exponent = 100 - i
            x[i] = x[i - 1] + (1 - x[i - 1]) / (r**exponent)

        # 归一化处理
        x = x / np.max(x)

        # 计算厚度分布 yt
        term = x / self.c
        yt = (
            5
            * self.t
            * self.c
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
        idx1 = np.where(x < self.p * self.c)
        if len(idx1[0]) > 0:
            term1 = x[idx1] / self.c
            yc[idx1] = (self.m / (self.p**2)) * (2 * self.p * term1 - term1**2)

        idx2 = np.where(x >= self.p * self.c)
        if len(idx2[0]) > 0:
            term2 = x[idx2] / self.c
            yc[idx2] = (self.m / ((1 - self.p) ** 2)) * (
                (1 - 2 * self.p) + 2 * self.p * term2 - term2**2
            )

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
        self.coords = np.vstack((upper, lower))

        return self.coords

    def plot_foil(self):
        """绘制 NACA 四位数翼型"""
        plt.plot(self.coords[:, 0], self.coords[:, 1], "b-")
        plt.xlim(np.min(self.coords[:, 0]), np.max(self.coords[:, 0]))
        plt.ylim(np.min(self.coords[:, 1]), np.max(self.coords[:, 1]))
        title = f"NACA {self.number} Airfoil"
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.grid(True)
        plt.show(block=False)

    def write_foil(self, filename):
        """将翼型坐标写入文件"""
        total_points = len(self.coords)
        with open(filename, "w") as f:
            f.write(f"# NACA {self.number} Airfoil\n")
            f.write(f"# {total_points} points\n")
            for x, y in self.coords:
                f.write(f"{x} {y} 0\n")
        print(f"翼型坐标已写入文件 {filename}")


def generate_naca_serials(save_folder: str = "naca_serials"):
    """
    生成 NACA 翼型的所有可能序列
    """
    save_folder = Path(save_folder)  # 转为Path对象

    if not save_folder.exists():
        save_folder.mkdir(parents=True)
        print(f"文件夹 {save_folder} 已创建")

    # m取值范围0.00-0.09，p取值范围0.3-0.6，t取值范围0.10-0.40
    m_values = np.arange(0, 10, 1) / 100
    p_values = np.arange(3, 7, 1) / 10
    t_values = np.arange(10, 41, 10) / 100

    count = 0
    for m in m_values:
        for p in p_values:
            for t in t_values:
                foil = NACA4DigitFoil(m=m, p=p, t=t, c=1.0, num_points=100)
                file_name = save_folder / f"NACA{foil.number}.dat"
                foil.write_foil(file_name)
                count += 1

    print(f"共生成 {count} 个翼型文件")


if __name__ == "__main__":
    # 示例调用1
    # naca_foil = NACA4DigitFoil(m=0.06, p=0.4, t=0.12, c=1.0, num_points=100)

    # 示例调用2
    # naca_foil = NACA4DigitFoil("6412", num_points=100)

    # naca_foil.plot_foil()

    generate_naca_serials()
