"""
Fluent网格文件类型定义
包含Fluent .cas/.msh文件中使用的各种类型常量
"""

# Fluent网格类型定义
FLUENT_FACE_TYPES = {"MIXED": 0, "LINEAR": 2, "TRI": 3, "QUAD": 4}

# Fluent单元类型定义
FLUENT_CELL_TYPES = {
    "MIXED": 0,
    "TRI": 1,
    "TET": 2,
    "QUAD": 3,
    "HEX": 4,
    "PYRAMID": 5,
    "WEDGE": 6,
}

# Fluent边界条件类型定义
FLUENT_BOUNDARY_TYPES = {
    "INTERIOR": 2,
    "WALL": 3,
    "PRESSURE_INLET": 4,
    "PRESSURE_OUTLET": 5,
    "SYMMETRY": 7,
    "PRESSURE_FAR": 9,
    "VELOCITY_INLET": 10,
    "PERIODIC": 12,
    "MASS_FLOW_INLET": 20,
    "INTERFACE": 24,
    "OUTFLOW": 36,
    "AXIS": 37,
}

# CELL区域类型
CELL_ZONE_TYPE = {"DEAD": 0, "FLUID": 1}
