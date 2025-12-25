#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据转换模块
用于将不同类型的网格数据转换为统一格式，供底层算法使用
"""

def convert_to_internal_mesh_format(mesh_data):
    """
    将不同类型的网格数据转换为底层算法所需的内部格式

    Args:
        mesh_data: 网格数据对象，可以是MeshData对象、字典或其他类型

    Returns:
        转换后的内部网格格式，格式如下：
        {
            'nodes': 节点坐标列表，每个元素为[x, y, z]
            'faces': 面列表，每个元素为包含nodes, right_cell, bc_type, part_name等信息的字典
            'zones': 部件信息，格式：{part_name: part_data}
            'dimensions': 网格维度
        }
    """
    internal_mesh = {
        'nodes': [],
        'faces': [],
        'zones': {},
        'dimensions': 2,  # 默认2D网格
        'node_count': 0,
        'face_count': 0,
        'cell_count': 0,
    }

    # 处理不同类型的mesh_data
    if mesh_data is None:
        return internal_mesh

    # MeshData对象类型
    elif hasattr(mesh_data, 'node_coords') and hasattr(mesh_data, 'cells'):
        # 先获取节点坐标
        node_coords = mesh_data.node_coords
        
        # 先只考虑2维，后续再拓展到3维
        # 先获取网格维度，用于后续的节点坐标处理
        # if hasattr(mesh_data, 'dimensions'):
        #     mesh_dimensions = mesh_data.dimensions
        # else:
        #     # 根据节点坐标推断维度
        #     if node_coords and len(node_coords[0]) > 2:
        #         mesh_dimensions = 3
        #     else:
        #         mesh_dimensions = 2
        
        # 设置网格维度
        mesh_dimensions = 2
        internal_mesh['dimensions'] = mesh_dimensions
        
        # 对于2D网格，去掉z坐标
        processed_nodes = []
        for node in node_coords:
            # 确保节点坐标是列表或元组
            if isinstance(node, (list, tuple)):
                # 如果是3D坐标且网格是2D，只保留x和y
                if mesh_dimensions == 2 and len(node) >= 2:
                    processed_nodes.append(node[:2])
                else:
                    processed_nodes.append(node)
            else:
                processed_nodes.append(node)
        
        # 设置处理后的节点坐标
        internal_mesh['nodes'] = processed_nodes
        internal_mesh['node_count'] = len(processed_nodes)

        # 优先使用parts_info来构建faces（如果存在）
        if hasattr(mesh_data, 'parts_info') and mesh_data.parts_info:
            # 使用parts_info来构建边界faces
            faces = []
            for part_name, part_data in mesh_data.parts_info.items():
                # 如果part_data有faces信息，使用它
                if isinstance(part_data, dict) and 'faces' in part_data and part_data['faces']:
                    for face_data in part_data['faces']:
                        if isinstance(face_data, dict) and 'nodes' in face_data:
                            # 确保节点索引是1基的（fluent格式）
                            nodes = [n + 1 if isinstance(n, int) else n for n in face_data['nodes']]  # 转换为1基索引
                            face = {
                                "nodes": nodes,
                                "left_cell": face_data.get('left_cell', 0),
                                "right_cell": face_data.get('right_cell', 0),
                                "bc_type": part_data.get('type', 'wall'),
                                "part_name": part_name
                            }
                            faces.append(face)
                else:
                    # 如果parts_info没有具体faces，但有face_count，可能需要其他处理方式
                    pass

            # 如果通过parts_info创建的faces有效，使用它们；否则回退到基于cells的处理
            if faces:
                internal_mesh['faces'] = faces
            else:
                # 回退到基于cells的处理
                faces = []
                for i, cell in enumerate(mesh_data.cells):
                    if len(cell) == 3:  # 三角形
                        # 创建三角形的三条边作为边界faces
                        for j in range(3):
                            node1_idx = cell[j]
                            node2_idx = cell[(j + 1) % 3]
                            face = {
                                "nodes": [node1_idx + 1, node2_idx + 1],  # fluent网格从1开始计数
                                "left_cell": i + 1,  # 当前单元
                                "right_cell": 0,  # 边界face，右侧无单元
                                "bc_type": "wall",  # 默认边界类型
                                "part_name": part_name if 'part_name' in locals() else "default"  # 使用part名称或默认
                            }
                            faces.append(face)
                internal_mesh['faces'] = faces
        else:
            # 如果没有parts_info，基于cells创建边界faces
            faces = []
            for i, cell in enumerate(mesh_data.cells):
                if len(cell) == 3:  # 三角形
                    # 创建三角形的三条边作为边界faces
                    for j in range(3):
                        node1_idx = cell[j]
                        node2_idx = cell[(j + 1) % 3]
                        face = {
                            "nodes": [node1_idx + 1, node2_idx + 1],  # fluent网格从1开始计数
                            "left_cell": i + 1,  # 当前单元
                            "right_cell": 0,  # 边界face，右侧无单元
                            "bc_type": "wall",  # 默认边界类型
                            "part_name": "default"  # 默认部件名称
                        }
                        faces.append(face)
                elif len(cell) == 4:  # 四边形
                    # 创建四边形的四条边作为边界faces
                    for j in range(4):
                        node1_idx = cell[j]
                        node2_idx = cell[(j + 1) % 4]
                        face = {
                            "nodes": [node1_idx + 1, node2_idx + 1],  # fluent网格从1开始计数
                            "left_cell": i + 1,  # 当前单元
                            "right_cell": 0,  # 边界face，右侧无单元
                            "bc_type": "wall",  # 默认边界类型
                            "part_name": "default"  # 默认部件名称
                        }
                        faces.append(face)
            internal_mesh['faces'] = faces

        internal_mesh['face_count'] = len(internal_mesh['faces'])

        # 添加部件信息
        if hasattr(mesh_data, 'parts_info') and mesh_data.parts_info:
            internal_mesh['zones'] = mesh_data.parts_info

    # 字典类型
    elif isinstance(mesh_data, dict):
        # 先获取节点坐标
        node_coords = mesh_data.get('node_coords', [])
        
        # 先获取网格维度，用于后续的节点坐标处理
        if 'dimensions' in mesh_data:
            mesh_dimensions = mesh_data['dimensions']
        else:
            # 根据节点坐标推断维度
            if node_coords and len(node_coords[0]) > 2:
                mesh_dimensions = 3
            else:
                mesh_dimensions = 2
        
        # 设置网格维度
        internal_mesh['dimensions'] = mesh_dimensions
        
        # 对于2D网格，去掉z坐标
        processed_nodes = []
        for node in node_coords:
            # 确保节点坐标是列表或元组
            if isinstance(node, (list, tuple)):
                # 如果是3D坐标且网格是2D，只保留x和y
                if mesh_dimensions == 2 and len(node) >= 2:
                    processed_nodes.append(node[:2])
                else:
                    processed_nodes.append(node)
            else:
                processed_nodes.append(node)
        
        # 设置处理后的节点坐标
        internal_mesh['nodes'] = processed_nodes
        internal_mesh['node_count'] = len(processed_nodes)

        # 优先使用字典中的faces信息（如果存在）
        if 'faces' in mesh_data and mesh_data['faces']:
            # 直接使用faces信息
            faces = []
            for face_data in mesh_data['faces']:
                if isinstance(face_data, dict):
                    # 确保格式正确
                    face = {
                        "nodes": face_data.get("nodes", []),
                        "left_cell": face_data.get("left_cell", 0),
                        "right_cell": face_data.get("right_cell", 0),
                        "bc_type": face_data.get("bc_type", "wall"),
                        "part_name": face_data.get("part_name", "default")
                    }
                    faces.append(face)
            internal_mesh['faces'] = faces
        elif 'parts_info' in mesh_data and mesh_data['parts_info']:
            # 使用parts_info来构建faces
            faces = []
            for part_name, part_data in mesh_data['parts_info'].items():
                if isinstance(part_data, dict) and 'faces' in part_data and part_data['faces']:
                    for face_data in part_data['faces']:
                        if isinstance(face_data, dict) and 'nodes' in face_data:
                            # 确保节点索引是1基的（fluent格式）
                            nodes = [n + 1 if isinstance(n, int) else n for n in face_data['nodes']]  # 转换为1基索引
                            face = {
                                "nodes": nodes,
                                "left_cell": face_data.get('left_cell', 0),
                                "right_cell": face_data.get('right_cell', 0),
                                "bc_type": part_data.get('type', 'wall'),
                                "part_name": part_name
                            }
                            faces.append(face)

            # 如果通过parts_info创建的faces有效，使用它们；否则回退到基于cells的处理
            if faces:
                internal_mesh['faces'] = faces
            else:
                # 回退到基于cells的处理
                faces = []
                cells = mesh_data.get('cells', [])
                for i, cell in enumerate(cells):
                    if isinstance(cell, (list, tuple)) and len(cell) >= 2:
                        # 为每个cell创建边界faces
                        if len(cell) == 3:  # 三角形
                            for j in range(3):
                                node1_idx = cell[j]
                                node2_idx = cell[(j + 1) % 3]
                                face = {
                                    "nodes": [node1_idx + 1, node2_idx + 1],  # fluent网格从1开始计数
                                    "left_cell": i + 1,  # 当前单元
                                    "right_cell": 0,  # 边界face，右侧无单元
                                    "bc_type": "wall",  # 默认边界类型
                                    "part_name": part_name if 'part_name' in locals() else "default"  # 使用part名称或默认
                                }
                                faces.append(face)
                        elif len(cell) == 4:  # 四边形
                            for j in range(4):
                                node1_idx = cell[j]
                                node2_idx = cell[(j + 1) % 4]
                                face = {
                                    "nodes": [node1_idx + 1, node2_idx + 1],  # fluent网格从1开始计数
                                    "left_cell": i + 1,  # 当前单元
                                    "right_cell": 0,  # 边界face，右侧无单元
                                    "bc_type": "wall",  # 默认边界类型
                                    "part_name": part_name if 'part_name' in locals() else "default"  # 使用part名称或默认
                                }
                                faces.append(face)
                internal_mesh['faces'] = faces
        else:
            # 如果没有faces或parts_info信息，基于cells创建
            faces = []
            cells = mesh_data.get('cells', [])
            for i, cell in enumerate(cells):
                if isinstance(cell, (list, tuple)) and len(cell) >= 2:
                    # 为每个cell创建边界faces
                    if len(cell) == 3:  # 三角形
                        for j in range(3):
                            node1_idx = cell[j]
                            node2_idx = cell[(j + 1) % 3]
                            face = {
                                "nodes": [node1_idx + 1, node2_idx + 1],  # fluent网格从1开始计数
                                "left_cell": i + 1,  # 当前单元
                                "right_cell": 0,  # 边界face，右侧无单元
                                "bc_type": "wall",  # 默认边界类型
                                "part_name": mesh_data.get('part_name', 'default')  # 默认部件名称
                            }
                            faces.append(face)
                    elif len(cell) == 4:  # 四边形
                        for j in range(4):
                            node1_idx = cell[j]
                            node2_idx = cell[(j + 1) % 4]
                            face = {
                                "nodes": [node1_idx + 1, node2_idx + 1],  # fluent网格从1开始计数
                                "left_cell": i + 1,  # 当前单元
                                "right_cell": 0,  # 边界face，右侧无单元
                                "bc_type": "wall",  # 默认边界类型
                                "part_name": mesh_data.get('part_name', 'default')  # 默认部件名称
                            }
                            faces.append(face)
            internal_mesh['faces'] = faces

        internal_mesh['face_count'] = len(internal_mesh['faces'])
        internal_mesh['zones'] = mesh_data.get('parts_info', {})

    # 其他类型
    else:
        raise TypeError(f"不支持的网格数据类型: {type(mesh_data)}")

    return internal_mesh

def convert_to_mesh_data(mesh_dict, file_path=None, mesh_type=None):
    """
    将字典格式的网格数据转换为MeshData对象
    
    Args:
        mesh_dict: 字典格式的网格数据
        file_path: 网格文件路径
        mesh_type: 网格类型
        
    Returns:
        MeshData对象
    """
    from data_structure.mesh_data import MeshData
    
    mesh_data = MeshData(file_path=file_path, mesh_type=mesh_type)
    mesh_data.from_dict(mesh_dict)
    return mesh_data