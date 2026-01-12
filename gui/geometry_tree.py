#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一模型树组件
用于显示和管理几何、网格和部件信息
三层结构：几何、网格、部件
"""

from PyQt5.QtWidgets import (QTreeWidget, QTreeWidgetItem, QWidget, QVBoxLayout,
                             QHeaderView, QMenu, QAction)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon


class GeometryTreeWidget:
    """统一模型树组件 - 三层结构：几何、网格、部件"""

    def __init__(self, parent=None):
        """
        初始化统一模型树组件

        Args:
            parent: 父窗口
        """
        self.parent = parent
        self.widget = QWidget()
        
        self.geometry_data = None
        self.geometry_name = "几何"
        self.mesh_data = None
        self.mesh_name = "网格"
        self.parts_data = None
        self.parts_name = "部件"

        self._create_tree_widget()
        self._setup_ui()
        self._init_tree_structure()

    def _create_tree_widget(self):
        """创建树形控件"""
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["名称", "数量"])
        self.tree.setColumnWidth(0, 200)
        self.tree.setColumnWidth(1, 60)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)

        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)

        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)

    def _setup_ui(self):
        """设置UI布局"""
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        layout.addWidget(self.tree)
        self.widget.setLayout(layout)

    def _init_tree_structure(self):
        """初始化三层模型树结构"""
        self.tree.clear()

        geometry_item = QTreeWidgetItem(self.tree)
        geometry_item.setText(0, self.geometry_name)
        geometry_item.setText(1, "")
        geometry_item.setExpanded(True)
        geometry_item.setCheckState(0, Qt.Checked)
        geometry_item.setData(0, Qt.UserRole, "geometry")

        geometry_vertices_item = QTreeWidgetItem(geometry_item)
        geometry_vertices_item.setText(0, "点")
        geometry_vertices_item.setText(1, "0")
        geometry_vertices_item.setCheckState(0, Qt.Checked)
        geometry_vertices_item.setData(0, Qt.UserRole, ("geometry", "vertices"))

        geometry_edges_item = QTreeWidgetItem(geometry_item)
        geometry_edges_item.setText(0, "线")
        geometry_edges_item.setText(1, "0")
        geometry_edges_item.setCheckState(0, Qt.Checked)
        geometry_edges_item.setData(0, Qt.UserRole, ("geometry", "edges"))

        geometry_faces_item = QTreeWidgetItem(geometry_item)
        geometry_faces_item.setText(0, "面")
        geometry_faces_item.setText(1, "0")
        geometry_faces_item.setCheckState(0, Qt.Checked)
        geometry_faces_item.setData(0, Qt.UserRole, ("geometry", "faces"))

        geometry_bodies_item = QTreeWidgetItem(geometry_item)
        geometry_bodies_item.setText(0, "体")
        geometry_bodies_item.setText(1, "0")
        geometry_bodies_item.setCheckState(0, Qt.Checked)
        geometry_bodies_item.setData(0, Qt.UserRole, ("geometry", "bodies"))

        mesh_item = QTreeWidgetItem(self.tree)
        mesh_item.setText(0, self.mesh_name)
        mesh_item.setText(1, "")
        mesh_item.setExpanded(True)
        mesh_item.setCheckState(0, Qt.Checked)
        mesh_item.setData(0, Qt.UserRole, "mesh")

        mesh_vertices_item = QTreeWidgetItem(mesh_item)
        mesh_vertices_item.setText(0, "点")
        mesh_vertices_item.setText(1, "0")
        mesh_vertices_item.setCheckState(0, Qt.Checked)
        mesh_vertices_item.setData(0, Qt.UserRole, ("mesh", "vertices"))

        mesh_edges_item = QTreeWidgetItem(mesh_item)
        mesh_edges_item.setText(0, "线")
        mesh_edges_item.setText(1, "0")
        mesh_edges_item.setCheckState(0, Qt.Checked)
        mesh_edges_item.setData(0, Qt.UserRole, ("mesh", "edges"))

        mesh_faces_item = QTreeWidgetItem(mesh_item)
        mesh_faces_item.setText(0, "面")
        mesh_faces_item.setText(1, "0")
        mesh_faces_item.setCheckState(0, Qt.Checked)
        mesh_faces_item.setData(0, Qt.UserRole, ("mesh", "faces"))

        mesh_bodies_item = QTreeWidgetItem(mesh_item)
        mesh_bodies_item.setText(0, "体")
        mesh_bodies_item.setText(1, "0")
        mesh_bodies_item.setCheckState(0, Qt.Checked)
        mesh_bodies_item.setData(0, Qt.UserRole, ("mesh", "bodies"))

        parts_item = QTreeWidgetItem(self.tree)
        parts_item.setText(0, self.parts_name)
        parts_item.setText(1, "0")
        parts_item.setExpanded(True)
        parts_item.setCheckState(0, Qt.Checked)
        parts_item.setData(0, Qt.UserRole, "parts")

    def load_geometry(self, shape, geometry_name="几何"):
        """
        加载几何模型到树中

        Args:
            shape: OpenCASCADE TopoDS_Shape对象
            geometry_name: 几何模型名称
        """
        self.geometry_data = shape
        self.geometry_name = geometry_name

        geometry_item = self.tree.topLevelItem(0)
        if geometry_item:
            geometry_item.setText(0, geometry_name)

        self._clear_geometry_elements()
        self._extract_geometry_elements(shape)

    def load_mesh(self, mesh_data, mesh_name="网格"):
        """
        加载网格模型到树中

        Args:
            mesh_data: 网格数据对象
            mesh_name: 网格模型名称
        """
        self.mesh_data = mesh_data
        self.mesh_name = mesh_name

        mesh_item = self.tree.topLevelItem(1)
        if mesh_item:
            mesh_item.setText(0, mesh_name)

        self._clear_mesh_elements()
        self._extract_mesh_elements(mesh_data)

    def load_parts(self, parts_data=None):
        """
        加载部件信息到树中

        Args:
            parts_data: 部件数据对象（可选，如果不提供则从mesh_data中提取）
        """
        if parts_data is not None:
            self.parts_data = parts_data
        elif self.mesh_data is not None:
            self.parts_data = self.mesh_data

        self._clear_parts_elements()
        self._extract_parts_elements(self.parts_data)

    def _clear_geometry_elements(self):
        """清除几何元素"""
        geometry_item = self.tree.topLevelItem(0)
        if geometry_item:
            for i in range(geometry_item.childCount()):
                category_item = geometry_item.child(i)
                for j in range(category_item.childCount()):
                    category_item.removeChild(category_item.child(0))
                category_item.setText(1, "0")

    def _clear_mesh_elements(self):
        """清除网格元素"""
        mesh_item = self.tree.topLevelItem(1)
        if mesh_item:
            for i in range(mesh_item.childCount()):
                category_item = mesh_item.child(i)
                for j in range(category_item.childCount()):
                    category_item.removeChild(category_item.child(0))
                category_item.setText(1, "0")

    def _clear_parts_elements(self):
        """清除部件元素"""
        parts_item = self.tree.topLevelItem(2)
        if parts_item:
            for i in range(parts_item.childCount()):
                parts_item.removeChild(parts_item.child(0))
            parts_item.setText(1, "0")

    def _extract_geometry_elements(self, shape):
        """
        从形状中提取几何元素并添加到树中

        Args:
            shape: OpenCASCADE TopoDS_Shape对象
        """
        if shape is None:
            return

        geometry_item = self.tree.topLevelItem(0)
        if geometry_item is None:
            return

        vertices_item = geometry_item.child(0)
        edges_item = geometry_item.child(1)
        faces_item = geometry_item.child(2)
        bodies_item = geometry_item.child(3)

        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID

        vertex_count = 0
        edge_count = 0
        face_count = 0
        body_count = 0

        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        while explorer.More():
            vertex = explorer.Current()
            vertex_item = QTreeWidgetItem(vertices_item)
            vertex_item.setText(0, f"点_{vertex_count}")
            vertex_item.setText(1, "")
            vertex_item.setCheckState(0, Qt.Checked)
            vertex_item.setData(0, Qt.UserRole, ("geometry", "vertex", vertex, vertex_count))

            pnt = self._get_vertex_point(vertex)
            if pnt:
                vertex_item.setToolTip(0, f"坐标: ({pnt.X():.3f}, {pnt.Y():.3f}, {pnt.Z():.3f})")

            vertex_count += 1
            explorer.Next()

        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while explorer.More():
            edge = explorer.Current()
            edge_item = QTreeWidgetItem(edges_item)
            edge_item.setText(0, f"线_{edge_count}")
            edge_item.setText(1, "")
            edge_item.setCheckState(0, Qt.Checked)
            edge_item.setData(0, Qt.UserRole, ("geometry", "edge", edge, edge_count))

            edge_length = self._get_edge_length(edge)
            if edge_length is not None:
                edge_item.setToolTip(0, f"长度: {edge_length:.3f}")

            edge_count += 1
            explorer.Next()

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            face_item = QTreeWidgetItem(faces_item)
            face_item.setText(0, f"面_{face_count}")
            face_item.setText(1, "")
            face_item.setCheckState(0, Qt.Checked)
            face_item.setData(0, Qt.UserRole, ("geometry", "face", face, face_count))

            face_area = self._get_face_area(face)
            if face_area is not None:
                face_item.setToolTip(0, f"面积: {face_area:.3f}")

            face_count += 1
            explorer.Next()

        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            solid = explorer.Current()
            body_item = QTreeWidgetItem(bodies_item)
            body_item.setText(0, f"体_{body_count}")
            body_item.setText(1, "")
            body_item.setCheckState(0, Qt.Checked)
            body_item.setData(0, Qt.UserRole, ("geometry", "body", solid, body_count))

            body_volume = self._get_solid_volume(solid)
            if body_volume is not None:
                body_item.setToolTip(0, f"体积: {body_volume:.3f}")

            body_count += 1
            explorer.Next()

        vertices_item.setText(1, str(vertex_count))
        edges_item.setText(1, str(edge_count))
        faces_item.setText(1, str(face_count))
        bodies_item.setText(1, str(body_count))

    def _extract_mesh_elements(self, mesh_data):
        """
        从网格数据中提取网格元素并添加到树中

        Args:
            mesh_data: 网格数据对象
        """
        if mesh_data is None:
            return

        mesh_item = self.tree.topLevelItem(1)
        if mesh_item is None:
            return

        vertices_item = mesh_item.child(0)
        edges_item = mesh_item.child(1)
        faces_item = mesh_item.child(2)
        bodies_item = mesh_item.child(3)

        vertex_count = 0
        edge_count = 0
        face_count = 0
        body_count = 0

        if hasattr(mesh_data, 'node_coords'):
            node_coords = mesh_data.node_coords
            vertex_count = len(node_coords)

            for i, coord in enumerate(node_coords):
                vertex_item = QTreeWidgetItem(vertices_item)
                vertex_item.setText(0, f"点_{i}")
                vertex_item.setText(1, "")
                vertex_item.setCheckState(0, Qt.Checked)
                vertex_item.setData(0, Qt.UserRole, ("mesh", "vertex", i, coord))

                coord_str = f"({coord[0]:.3f}, {coord[1]:.3f}"
                if len(coord) > 2:
                    coord_str += f", {coord[2]:.3f})"
                else:
                    coord_str += ", 0.000)"
                vertex_item.setToolTip(0, f"坐标: {coord_str}")

        if hasattr(mesh_data, 'cells'):
            cells = mesh_data.cells
            face_count = len(cells)

            for i, cell in enumerate(cells):
                face_item = QTreeWidgetItem(faces_item)
                face_item.setText(0, f"面_{i}")
                face_item.setText(1, "")
                face_item.setCheckState(0, Qt.Checked)
                face_item.setData(0, Qt.UserRole, ("mesh", "face", i, cell))

                num_nodes = len(cell)
                face_item.setToolTip(0, f"节点数: {num_nodes}")

        vertices_item.setText(1, str(vertex_count))
        edges_item.setText(1, str(edge_count))
        faces_item.setText(1, str(face_count))
        bodies_item.setText(1, str(body_count))

    def _extract_parts_elements(self, parts_data):
        """
        从数据中提取部件并添加到树中

        Args:
            parts_data: 部件数据对象
        """
        parts_item = self.tree.topLevelItem(2)
        if parts_item is None:
            return

        part_count = 0

        if parts_data is None:
            parts_item.setText(1, "0")
            return

        if hasattr(parts_data, 'parts_info') and parts_data.parts_info:
            if isinstance(parts_data.parts_info, list):
                for part_info in parts_data.parts_info:
                    part_name = part_info.get('part_name', f'部件_{part_count}')
                    bc_type = part_info.get('bc_type', '')
                    checked = bc_type != 'interior'

                    part_item = QTreeWidgetItem(parts_item)
                    part_item.setText(0, part_name)
                    part_item.setText(1, "")
                    part_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                    part_item.setData(0, Qt.UserRole, ("parts", part_info, part_count))

                    if bc_type:
                        part_item.setToolTip(0, f"边界条件: {bc_type}")

                    part_count += 1
            elif isinstance(parts_data.parts_info, dict):
                for part_name, part_data in parts_data.parts_info.items():
                    if part_name not in ['type', 'node_coords', 'cells', 'num_points', 'num_cells', 'unstr_grid']:
                        bc_type = part_data.get('bc_type', '') if isinstance(part_data, dict) else ''
                        checked = bc_type != 'interior'

                        part_item = QTreeWidgetItem(parts_item)
                        part_item.setText(0, part_name)
                        part_item.setText(1, "")
                        part_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                        part_item.setData(0, Qt.UserRole, ("parts", part_data, part_count))

                        if bc_type:
                            part_item.setToolTip(0, f"边界条件: {bc_type}")

                        part_count += 1
        elif hasattr(parts_data, 'boundary_info') and parts_data.boundary_info:
            for part_name, part_data in parts_data.boundary_info.items():
                bc_type = part_data.get('bc_type', '') if isinstance(part_data, dict) else ''
                checked = bc_type != 'interior'

                part_item = QTreeWidgetItem(parts_item)
                part_item.setText(0, part_name)
                part_item.setText(1, "")
                part_item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                part_item.setData(0, Qt.UserRole, ("parts", part_data, part_count))

                if bc_type:
                    part_item.setToolTip(0, f"边界条件: {bc_type}")

                part_count += 1
        else:
            default_part_item = QTreeWidgetItem(parts_item)
            default_part_item.setText(0, "默认部件")
            default_part_item.setText(1, "")
            default_part_item.setCheckState(0, Qt.Checked)
            default_part_item.setData(0, Qt.UserRole, ("parts", None, 0))
            part_count = 1

        parts_item.setText(1, str(part_count))

    def _get_vertex_point(self, vertex):
        """
        获取顶点的坐标

        Args:
            vertex: OpenCASCADE TopoDS_Vertex对象

        Returns:
            gp_Pnt对象
        """
        from OCC.Core.BRep import BRep_Tool
        try:
            return BRep_Tool.Pnt(vertex)
        except:
            return None

    def _get_edge_length(self, edge):
        """
        获取边的长度

        Args:
            edge: OpenCASCADE TopoDS_Edge对象

        Returns:
            边的长度
        """
        from OCC.Core.GCPnts import GCPnts_AbscissaPoint
        from OCC.Core.BRep import BRep_Tool

        try:
            curve = BRep_Tool.Curve(edge)
            if curve:
                geom_curve, first, last = curve
                if geom_curve:
                    length = GCPnts_AbscissaPoint.Length(geom_curve, first, last)
                    return length
        except:
            pass
        return None

    def _get_face_area(self, face):
        """
        获取面的面积

        Args:
            face: OpenCASCADE TopoDS_Face对象

        Returns:
            面的面积
        """
        from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
        from OCC.Core.GProp import GProp_GProps

        try:
            props = GProp_GProps()
            brepgprop_SurfaceProperties(face, props)
            return props.Mass()
        except:
            return None

    def _get_solid_volume(self, solid):
        """
        获取体的体积

        Args:
            solid: OpenCASCADE TopoDS_Solid对象

        Returns:
            体的体积
        """
        from OCC.Core.BRepGProp import brepgprop_VolumeProperties
        from OCC.Core.GProp import GProp_GProps

        try:
            props = GProp_GProps()
            brepgprop_VolumeProperties(solid, props)
            return props.Mass()
        except:
            return None

    def _on_item_changed(self, item, column):
        """
        树项改变时的回调

        Args:
            item: 改变的树项
            column: 改变的列
        """
        if column == 0:
            self._update_child_items(item, item.checkState(0))
            self._update_parent_item(item)

            element_data = item.data(0, Qt.UserRole)
            self._handle_visibility_change(item, element_data)

    def _on_item_clicked(self, item, column):
        """
        树项点击时的回调

        Args:
            item: 点击的树项
            column: 点击的列
        """
        element_data = item.data(0, Qt.UserRole)
        self._handle_selection_change(item, element_data)

    def _handle_visibility_change(self, item, element_data):
        """
        处理可见性改变

        Args:
            item: 树项
            element_data: 元素数据
        """
        if not hasattr(self.parent, 'on_model_tree_visibility_changed'):
            return

        if isinstance(element_data, str):
            self.parent.on_model_tree_visibility_changed(element_data, item.checkState(0) == Qt.Checked)
        elif isinstance(element_data, tuple) and len(element_data) >= 2:
            category = element_data[0]
            
            # 对于部件，element_data 格式为 ("parts", part_data, part_count)
            # 我们只需要传递 category 和 element_index
            if category == 'parts' and len(element_data) >= 3:
                element_index = element_data[2]
                self.parent.on_model_tree_visibility_changed(category, element_index, item.checkState(0) == Qt.Checked)
            else:
                element_type = element_data[1]
                if len(element_data) >= 4:
                    element_index = element_data[3]
                    self.parent.on_model_tree_visibility_changed(category, element_type, element_index, item.checkState(0) == Qt.Checked)
                else:
                    self.parent.on_model_tree_visibility_changed(category, element_type, item.checkState(0) == Qt.Checked)

    def _handle_selection_change(self, item, element_data):
        """
        处理选择改变

        Args:
            item: 树项
            element_data: 元素数据
        """
        if not hasattr(self.parent, 'on_model_tree_selected'):
            return

        if isinstance(element_data, tuple) and len(element_data) >= 3:
            category = element_data[0]
            
            # 对于部件，element_data 格式为 ("parts", part_data, part_count)
            if category == 'parts' and len(element_data) >= 3:
                part_data = element_data[1]
                element_index = element_data[2]
                # 对于部件，element_type 可以是部件名称
                element_type = item.text(0)
                self.parent.on_model_tree_selected(category, element_type, element_index, part_data)
            elif len(element_data) >= 4:
                element_type = element_data[1]
                element_index = element_data[3]
                element_obj = element_data[2] if len(element_data) >= 3 else None
                self.parent.on_model_tree_selected(category, element_type, element_index, element_obj)

    def _update_child_items(self, item, check_state):
        """
        更新子项的选中状态

        Args:
            item: 父项
            check_state: 选中状态
        """
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, check_state)

    def _update_parent_item(self, item):
        """
        更新父项的选中状态

        Args:
            item: 子项
        """
        parent = item.parent()
        if parent is None:
            return

        all_checked = True
        all_unchecked = True

        for i in range(parent.childCount()):
            child = parent.child(i)
            if child.checkState(0) == Qt.Checked:
                all_unchecked = False
            elif child.checkState(0) == Qt.Unchecked:
                all_checked = False
            else:
                all_checked = False
                all_unchecked = False

        if all_checked:
            parent.setCheckState(0, Qt.Checked)
        elif all_unchecked:
            parent.setCheckState(0, Qt.Unchecked)
        else:
            parent.setCheckState(0, Qt.PartiallyChecked)

    def _show_context_menu(self, position):
        """
        显示右键菜单

        Args:
            position: 鼠标位置
        """
        item = self.tree.itemAt(position)
        if item is None:
            return

        element_data = item.data(0, Qt.UserRole)

        menu = QMenu()

        show_action = QAction("显示", self.tree)
        show_action.triggered.connect(lambda: self._set_item_visibility(item, True))
        menu.addAction(show_action)

        hide_action = QAction("隐藏", self.tree)
        hide_action.triggered.connect(lambda: self._set_item_visibility(item, False))
        menu.addAction(hide_action)

        expand_action = QAction("展开", self.tree)
        expand_action.triggered.connect(lambda: item.setExpanded(True))
        menu.addAction(expand_action)

        collapse_action = QAction("折叠", self.tree)
        collapse_action.triggered.connect(lambda: item.setExpanded(False))
        menu.addAction(collapse_action)

        if not menu.isEmpty():
            menu.exec_(self.tree.mapToGlobal(position))

    def _set_item_visibility(self, item, visible):
        """
        设置项的可见性

        Args:
            item: 树项
            visible: 是否可见
        """
        item.setCheckState(0, Qt.Checked if visible else Qt.Unchecked)

    def get_visible_elements(self, category=None, element_type=None):
        """
        获取可见的元素

        Args:
            category: 类别（"geometry", "mesh", "parts"），None表示所有类别
            element_type: 元素类型（"vertices", "edges", "faces", "bodies"），None表示所有类型

        Returns:
            可见元素字典
        """
        visible_elements = {}

        for i in range(self.tree.topLevelItemCount()):
            category_item = self.tree.topLevelItem(i)
            category_data = category_item.data(0, Qt.UserRole)

            if category is not None and category_data != category:
                continue

            if category_data is None:
                continue

            if category_data not in visible_elements:
                visible_elements[category_data] = {}

            for j in range(category_item.childCount()):
                element_type_item = category_item.child(j)
                element_type_data = element_type_item.data(0, Qt.UserRole)

                if isinstance(element_type_data, tuple) and len(element_type_data) >= 2:
                    elem_category, elem_type = element_type_data[0], element_type_data[1]

                    if element_type is not None and elem_type != element_type:
                        continue

                    if elem_type is None:
                        continue

                    if elem_type not in visible_elements[category_data]:
                        visible_elements[category_data][elem_type] = []

                    for k in range(element_type_item.childCount()):
                        element_item = element_type_item.child(k)

                        if element_item.checkState(0) == Qt.Checked:
                            element_data = element_item.data(0, Qt.UserRole)
                            if isinstance(element_data, tuple) and len(element_data) >= 4:
                                elem_category_data, elem_type_data, elem_obj, elem_index = element_data
                                if elem_type_data == elem_type:
                                    visible_elements[category_data][elem_type].append((elem_index, elem_obj))

        if category is not None and category in visible_elements:
            if element_type is not None and element_type in visible_elements[category]:
                return visible_elements[category][element_type]
            return visible_elements[category]

        return visible_elements

    def get_visible_parts(self):
        """
        获取可见的部件

        Returns:
            可见部件名称列表
        """
        visible_parts = []

        parts_item = self.tree.topLevelItem(2)
        if parts_item is None:
            return visible_parts

        for i in range(parts_item.childCount()):
            part_item = parts_item.child(i)
            if part_item.checkState(0) == Qt.Checked:
                part_name = part_item.text(0)
                visible_parts.append(part_name)

        return visible_parts

    def set_element_visibility(self, category, element_type, element_index, visible):
        """
        设置特定元素的可见性

        Args:
            category: 类别（"geometry", "mesh", "parts"）
            element_type: 元素类型（"vertices", "edges", "faces", "bodies"）
            element_index: 元素索引
            visible: 是否可见
        """
        for i in range(self.tree.topLevelItemCount()):
            category_item = self.tree.topLevelItem(i)
            category_data = category_item.data(0, Qt.UserRole)

            if category_data != category:
                continue

            for j in range(category_item.childCount()):
                element_type_item = category_item.child(j)
                element_type_data = element_type_item.data(0, Qt.UserRole)

                if isinstance(element_type_data, tuple) and len(element_type_data) >= 2:
                    elem_category, elem_type = element_type_data[0], element_type_data[1]

                    if elem_type != element_type:
                        continue

                    for k in range(element_type_item.childCount()):
                        element_item = element_type_item.child(k)
                        element_data = element_item.data(0, Qt.UserRole)

                        if isinstance(element_data, tuple) and len(element_data) >= 4:
                            elem_category, elem_type, elem_obj, elem_index_data = element_data
                            if elem_index_data == element_index:
                                element_item.setCheckState(0, Qt.Checked if visible else Qt.Unchecked)
                                return

    def set_category_visibility(self, category, element_type, visible):
        """
        设置类别的可见性

        Args:
            category: 类别（"geometry", "mesh", "parts"）
            element_type: 元素类型（"vertices", "edges", "faces", "bodies"）
            visible: 是否可见
        """
        for i in range(self.tree.topLevelItemCount()):
            category_item = self.tree.topLevelItem(i)
            category_data = category_item.data(0, Qt.UserRole)

            if category_data != category:
                continue

            for j in range(category_item.childCount()):
                element_type_item = category_item.child(j)
                element_type_data = element_type_item.data(0, Qt.UserRole)

                if isinstance(element_type_data, tuple) and len(element_type_data) >= 2:
                    elem_category, elem_type = element_type_data[0], element_type_data[1]

                    if elem_type != element_type:
                        continue

                    element_type_item.setCheckState(0, Qt.Checked if visible else Qt.Unchecked)
                    return

    def clear(self):
        """清空树"""
        self._init_tree_structure()
        self.geometry_data = None
        self.mesh_data = None
        self.parts_data = None
