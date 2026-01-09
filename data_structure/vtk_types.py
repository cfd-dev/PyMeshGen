from enum import IntEnum


class VTKCellType(IntEnum):
    """
    VTK单元类型枚举
    
    参考VTK文档中的单元类型定义
    https://vtk.org/doc/nightly/html/vtkCellType_8h.html
    """
    EMPTY = 0
    VERTEX = 1
    POLY_VERTEX = 2
    LINE = 3
    POLY_LINE = 4
    TRIANGLE = 5
    TRIANGLE_STRIP = 6
    POLYGON = 7
    PIXEL = 8
    QUAD = 9
    TETRA = 10
    VOXEL = 11
    HEXAHEDRON = 12
    WEDGE = 13
    PYRAMID = 14
    PENTAGONAL_PRISM = 15
    HEXAGONAL_PRISM = 16


class VTKCellTypeName:
    """
    VTK单元类型名称映射
    
    提供单元类型到名称的映射，便于调试和日志输出
    """
    TYPE_NAMES = {
        VTKCellType.EMPTY: "Empty",
        VTKCellType.VERTEX: "Vertex",
        VTKCellType.POLY_VERTEX: "PolyVertex",
        VTKCellType.LINE: "Line",
        VTKCellType.POLY_LINE: "PolyLine",
        VTKCellType.TRIANGLE: "Triangle",
        VTKCellType.TRIANGLE_STRIP: "TriangleStrip",
        VTKCellType.POLYGON: "Polygon",
        VTKCellType.PIXEL: "Pixel",
        VTKCellType.QUAD: "Quadrilateral",
        VTKCellType.TETRA: "Tetrahedron",
        VTKCellType.VOXEL: "Voxel",
        VTKCellType.HEXAHEDRON: "Hexahedron",
        VTKCellType.WEDGE: "Wedge",
        VTKCellType.PYRAMID: "Pyramid",
        VTKCellType.PENTAGONAL_PRISM: "PentagonalPrism",
        VTKCellType.HEXAGONAL_PRISM: "HexagonalPrism",
    }

    @classmethod
    def get_name(cls, cell_type):
        """
        获取单元类型的名称
        
        Args:
            cell_type: VTKCellType枚举值
            
        Returns:
            str: 单元类型的名称
        """
        return cls.TYPE_NAMES.get(cell_type, "Unknown")


def get_vtk_cell_type_name(cell_type):
    """
    获取VTK单元类型的名称（便捷函数）
    
    Args:
        cell_type: VTKCellType枚举值
        
    Returns:
        str: 单元类型的名称
    """
    return VTKCellTypeName.get_name(cell_type)
