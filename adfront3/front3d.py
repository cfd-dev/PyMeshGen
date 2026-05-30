import numpy as np

from data_structure.basic_elements import is_node_element


class Front3D:
    """三维阵面（三角形面），用于三维阵面推进法"""

    __slots__ = [
        'node_elems', 'idx', 'bc_type', 'part_name',
        'priority', 'al', 'center', 'area', 'normal',
        'bbox', 'hash', 'node_ids', 'face_key',
    ]

    def __init__(self, node_elem1, node_elem2, node_elem3,
                 idx=None, bc_type=None, part_name=None, al=3.0):
        if not (is_node_element(node_elem1)
                and is_node_element(node_elem2)
                and is_node_element(node_elem3)):
            raise TypeError("Front3D 需要 NodeElement 类型的节点")

        self.node_elems = [node_elem1, node_elem2, node_elem3]
        self.node_ids = [n.idx for n in self.node_elems]
        self.idx = idx
        self.bc_type = bc_type
        self.part_name = part_name
        self.priority = False
        self.al = al

        p0 = np.array(node_elem1.coords, dtype=float)
        p1 = np.array(node_elem2.coords, dtype=float)
        p2 = np.array(node_elem3.coords, dtype=float)

        self.center = [(p0[0] + p1[0] + p2[0]) / 3.0,
                       (p0[1] + p1[1] + p2[1]) / 3.0,
                       (p0[2] + p1[2] + p2[2]) / 3.0]

        v1 = p1 - p0
        v2 = p2 - p0
        cross = np.cross(v1, v2)
        cross_norm = np.linalg.norm(cross)

        self.area = 0.5 * cross_norm
        if cross_norm > 1e-30:
            self.normal = (cross / cross_norm).tolist()
        else:
            self.normal = [0.0, 0.0, 0.0]

        coords = [node_elem1.coords, node_elem2.coords, node_elem3.coords]
        self.bbox = (
            min(c[0] for c in coords),
            min(c[1] for c in coords),
            min(c[2] for c in coords),
            max(c[0] for c in coords),
            max(c[1] for c in coords),
            max(c[2] for c in coords),
        )

        # 使用排序元组确保面哈希与节点顺序无关且确定性
        self.face_key = tuple(sorted([node_elem1.hash, node_elem2.hash, node_elem3.hash]))
        self.hash = hash(self.face_key)

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority
        return self.area < other.area

    def __eq__(self, other):
        if not isinstance(other, Front3D):
            return False
        return self.hash == other.hash

    def __hash__(self):
        return self.hash
