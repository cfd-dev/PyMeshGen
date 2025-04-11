from vtk_io import VTK_ELEMENT_TYPE
from basic_elements import Unstructured_Grid, NodeElement, Triangle


def read_stl(filename):
    """读取ASCII格式STL文件，返回节点坐标和三角形单元"""
    node_coords = []
    coord_map = {}  # 坐标到索引的映射，用于去重
    cell_idx_container = []

    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith("facet normal"):
            # 解析三角形面片
            vertices = []
            i += 2  # 跳过 facet normal 和 outer loop
            while not lines[i].startswith("endloop"):
                if lines[i].startswith("vertex"):
                    parts = lines[i].split()
                    coord = tuple(map(float, parts[1:4]))  # (x,y,z)
                    if coord not in coord_map:
                        coord_map[coord] = len(node_coords)
                        node_coords.append(list(coord[:2]))  # TODO  只取XY坐标
                    vertices.append(coord_map[coord])
                i += 1  # 跳过 endloop
            i += 1  # 跳过 endfacet

            # STL面片为三角形，确保三个顶点
            if len(vertices) == 3:
                cell_idx_container.append(vertices)
            else:
                raise ValueError(f"STL文件中的三角形面片顶点数量不正确: {vertices}")

    # 自动检测边界边（出现次数为1的边）
    edge_count = {}
    for cell in cell_idx_container:
        for i in range(3):
            edge = tuple(sorted([cell[i], cell[(i + 1) % 3]]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # 收集边界节点
    boundary_nodes = set()
    for edge, count in edge_count.items():
        if count == 1:
            boundary_nodes.update(edge)

    boundary_nodes_idx = list(boundary_nodes)
    cell_type_container = [VTK_ELEMENT_TYPE.TRI.value] * len(cell_idx_container)

    return node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container


def write_stl(
    filename, node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container
):
    """将网格写入ASCII STL文件（仅支持三角形单元）"""
    with open(filename, "w") as f:
        f.write("solid PyMeshGen\n")

        for cell_idx, cell_type in zip(cell_idx_container, cell_type_container):
            if VTK_ELEMENT_TYPE(cell_type) != VTK_ELEMENT_TYPE.TRI:
                continue  # 跳过非三角形单元

            # 获取三个顶点坐标
            p1 = node_coords[cell_idx[0]] + [0.0]
            p2 = node_coords[cell_idx[1]] + [0.0]
            p3 = node_coords[cell_idx[2]] + [0.0]

            # 计算法向量
            v1 = np.subtract(p2, p1)
            v2 = np.subtract(p3, p1)
            normal = np.cross(v1, v2)
            normal /= np.linalg.norm(normal)

            # 写入面片
            f.write(f"facet normal {' '.join(map(str, normal))}\n")
            f.write(" outer loop\n")
            f.write(f"  vertex {' '.join(map(str, p1))}\n")
            f.write(f"  vertex {' '.join(map(str, p2))}\n")
            f.write(f"  vertex {' '.join(map(str, p3))}\n")
            f.write(" endloop\n")
            f.write("endfacet\n")

        f.write("endsolid PyMeshGen\n")


def reconstruct_mesh_from_stl(
    node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container
):
    """将STL数据重建为Unstructured_Grid对象"""

    # 创建带边界标记的节点对象
    node_elements = []
    for idx, coord in enumerate(node_coords):
        bc_type = "boundary" if idx in boundary_nodes_idx else "interior"
        node_elements.append(NodeElement(coord, idx, bc_type=bc_type))

    # 创建单元容器
    cell_container = []
    for idx, (cell_nodes, cell_type) in enumerate(
        zip(cell_idx_container, cell_type_container)
    ):
        nodes = [node_elements[i] for i in cell_nodes]
        cell = Triangle(nodes[0], nodes[1], nodes[2], idx)
        cell_container.append(cell)

    # 创建边界节点列表
    boundary_nodes = []
    for node in node_elements:
        if node.idx in boundary_nodes_idx:
            boundary_nodes.append(node)

    return Unstructured_Grid(cell_container, node_coords, boundary_nodes)


def parse_stl_msh(filename):
    """解析STL文件生成网格对象"""
    node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container = read_stl(
        filename
    )
    return reconstruct_mesh_from_stl(
        node_coords, cell_idx_container, boundary_nodes_idx, cell_type_container
    )
