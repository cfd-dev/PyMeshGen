# 对生成的unstr_grid进行网格质量检查，计算三角形单元的扭曲度，并绘制直方图


def calculate_triangle_twist(triangle):
    """计算三角形的扭曲度"""
    p1, p2, p3 = triangle.p1, triangle.p2, triangle.p3
    a = geom_tool.calculate_distance(p1, p2)
    b = geom_tool.calculate_distance(p2, p3)
    c = geom_tool.calculate_distance(p3, p1)
    s = (a + b + c) / 2
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    return 4 * area / (a * b * c)


def check_mesh_quality(unstr_grid):
    """检查网格质量"""
    twist_values = []
    for triangle in unstr_grid.cell_container:
        twist = calculate_triangle_twist(triangle)
        twist_values.append(twist)

    plt.hist(twist_values, bins=50, alpha=0.7, color="blue")

    plt.xlabel("Twist Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Triangle Twists")
    plt.grid(True)
    plt.show()


# 调用网格质量检查函数
check_mesh_quality(unstr_grid)
