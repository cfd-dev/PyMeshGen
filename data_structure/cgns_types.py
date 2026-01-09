from enum import IntEnum


class CGNSElementType(IntEnum):
    """
    CGNS单元类型枚举
    
    参考CGNS标准文档中的单元类型定义
    https://cgns.github.io/CGNS_docs_current/sids/elemtypes.html
    """
    ELEMENT_TYPE_NULL = 0
    ELEMENT_TYPE_USER_DEFINED = 1
    NODE = 2
    BAR_2 = 3
    BAR_3 = 4
    TRI_3 = 5
    TRI_6 = 6
    QUAD_4 = 7
    QUAD_8 = 8
    QUAD_9 = 9
    TETRA_4 = 10
    TETRA_10 = 11
    PYRA_5 = 12
    PYRA_14 = 13
    PENTA_6 = 14
    PENTA_15 = 15
    PENTA_18 = 16
    HEXA_8 = 17
    HEXA_20 = 18
    HEXA_27 = 19
    MIXED = 20
    PYRA_13 = 21


class CGNSElementTypeName:
    """
    CGNS单元类型名称映射
    
    提供单元类型到名称的映射，便于调试和日志输出
    """
    TYPE_NAMES = {
        CGNSElementType.ELEMENT_TYPE_NULL: "ElementTypeNull",
        CGNSElementType.ELEMENT_TYPE_USER_DEFINED: "ElementTypeUserDefined",
        CGNSElementType.NODE: "NODE",
        CGNSElementType.BAR_2: "BAR_2",
        CGNSElementType.BAR_3: "BAR_3",
        CGNSElementType.TRI_3: "TRI_3",
        CGNSElementType.TRI_6: "TRI_6",
        CGNSElementType.QUAD_4: "QUAD_4",
        CGNSElementType.QUAD_8: "QUAD_8",
        CGNSElementType.QUAD_9: "QUAD_9",
        CGNSElementType.TETRA_4: "TETRA_4",
        CGNSElementType.TETRA_10: "TETRA_10",
        CGNSElementType.PYRA_5: "PYRA_5",
        CGNSElementType.PYRA_14: "PYRA_14",
        CGNSElementType.PENTA_6: "PENTA_6",
        CGNSElementType.PENTA_15: "PENTA_15",
        CGNSElementType.PENTA_18: "PENTA_18",
        CGNSElementType.HEXA_8: "HEXA_8",
        CGNSElementType.HEXA_20: "HEXA_20",
        CGNSElementType.HEXA_27: "HEXA_27",
        CGNSElementType.MIXED: "MIXED",
        CGNSElementType.PYRA_13: "PYRA_13",
    }

    @classmethod
    def get_name(cls, element_type):
        """
        获取单元类型的名称
        
        Args:
            element_type: CGNSElementType枚举值
            
        Returns:
            str: 单元类型的名称
        """
        return cls.TYPE_NAMES.get(element_type, "Unknown")


def get_cgns_element_type_name(element_type):
    """
    获取CGNS单元类型的名称（便捷函数）
    
    Args:
        element_type: CGNSElementType枚举值
        
    Returns:
        str: 单元类型的名称
    """
    return CGNSElementTypeName.get_name(element_type)


class CGNSBCType(IntEnum):
    """
    CGNS边界条件类型枚举
    
    参考CGNS标准文档中的边界条件类型定义
    https://cgns.github.io/CGNS_docs_current/sids/bc.html
    """
    BC_TYPE_NULL = 0
    BC_TYPE_USER_DEFINED = 1
    BC_AXISYMMETRIC_WEDGE = 2
    BC_DEGENERATE_LINE = 3
    BC_DEGENERATE_POINT = 4
    BC_DIRICHLET = 5
    BC_EXTRAPOLATE = 6
    BC_FARFIELD = 7
    BC_GENERAL = 8
    BC_INFLOW = 9
    BC_INFLOW_SUBSONIC = 10
    BC_INFLOW_SUPERSONIC = 11
    BC_NEUMANN = 12
    BC_OUTFLOW = 13
    BC_OUTFLOW_SUBSONIC = 14
    BC_OUTFLOW_SUPERSONIC = 15
    BC_SYMMETRY_PLANE = 16
    BC_SYMMETRY_POLAR = 17
    BC_TUNNEL_INFLOW = 18
    BC_TUNNEL_OUTFLOW = 19
    BC_WALL = 20
    BC_WALL_INVISCID = 21
    BC_WALL_VISCOUS = 22
    BC_WALL_VISCOUS_HEAT_FLUX = 23
    BC_WALL_VISCOUS_ISOTHERMAL = 24
    FAMILY_SPECIFIED = 25


class CGNSBCTypeName:
    """
    CGNS边界条件类型名称映射
    
    提供边界条件类型到名称的映射，便于调试和日志输出
    """
    TYPE_NAMES = {
        CGNSBCType.BC_TYPE_NULL: "BCTypeNull",
        CGNSBCType.BC_TYPE_USER_DEFINED: "BCTypeUserDefined",
        CGNSBCType.BC_AXISYMMETRIC_WEDGE: "BCAxisymmetricWedge",
        CGNSBCType.BC_DEGENERATE_LINE: "BCDegenerateLine",
        CGNSBCType.BC_DEGENERATE_POINT: "BCDegeneratePoint",
        CGNSBCType.BC_DIRICHLET: "BCDirichlet",
        CGNSBCType.BC_EXTRAPOLATE: "BCExtrapolate",
        CGNSBCType.BC_FARFIELD: "BCFarfield",
        CGNSBCType.BC_GENERAL: "BCGeneral",
        CGNSBCType.BC_INFLOW: "BCInflow",
        CGNSBCType.BC_INFLOW_SUBSONIC: "BCInflowSubsonic",
        CGNSBCType.BC_INFLOW_SUPERSONIC: "BCInflowSupersonic",
        CGNSBCType.BC_NEUMANN: "BCNeumann",
        CGNSBCType.BC_OUTFLOW: "BCOutflow",
        CGNSBCType.BC_OUTFLOW_SUBSONIC: "BCOutflowSubsonic",
        CGNSBCType.BC_OUTFLOW_SUPERSONIC: "BCOutflowSupersonic",
        CGNSBCType.BC_SYMMETRY_PLANE: "BCSymmetryPlane",
        CGNSBCType.BC_SYMMETRY_POLAR: "BCSymmetryPolar",
        CGNSBCType.BC_TUNNEL_INFLOW: "BCTunnelInflow",
        CGNSBCType.BC_TUNNEL_OUTFLOW: "BCTunnelOutflow",
        CGNSBCType.BC_WALL: "BCWall",
        CGNSBCType.BC_WALL_INVISCID: "BCWallInviscid",
        CGNSBCType.BC_WALL_VISCOUS: "BCWallViscous",
        CGNSBCType.BC_WALL_VISCOUS_HEAT_FLUX: "BCWallViscousHeatFlux",
        CGNSBCType.BC_WALL_VISCOUS_ISOTHERMAL: "BCWallViscousIsothermal",
        CGNSBCType.FAMILY_SPECIFIED: "FamilySpecified",
    }

    @classmethod
    def get_name(cls, bc_type):
        """
        获取边界条件类型的名称
        
        Args:
            bc_type: CGNSBCType枚举值
            
        Returns:
            str: 边界条件类型的名称
        """
        return cls.TYPE_NAMES.get(bc_type, "Unknown")


def get_cgns_bc_type_name(bc_type):
    """
    获取CGNS边界条件类型的名称（便捷函数）
    
    Args:
        bc_type: CGNSBCType枚举值
        
    Returns:
        str: 边界条件类型的名称
    """
    return CGNSBCTypeName.get_name(bc_type)
