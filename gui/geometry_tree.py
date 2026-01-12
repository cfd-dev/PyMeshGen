#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
几何模型树组件
用于显示和管理几何模型的顶点、边、面、体等元素
"""

from PyQt5.QtWidgets import (QTreeWidget, QTreeWidgetItem, QWidget, QVBoxLayout,
                             QHeaderView, QMenu, QAction)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon


class GeometryTreeWidget:
    """几何模型树组件"""

    def __init__(self, parent=None):
        """
        初始化几何模型树组件

        Args:
            parent: 父窗口
        """
        self.parent = parent
        self.widget = QWidget()
        self.geometry_data = None
        self.geometry_name = "几何模型"

        self._create_tree_widget()
        self._setup_ui()

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

    def load_geometry(self, shape, geometry_name="几何模型"):
        """
        加载几何模型到树中

        Args:
            shape: OpenCASCADE TopoDS_Shape对象
            geometry_name: 几何模型名称
        """
        self.geometry_data = shape
        self.geometry_name = geometry_name

        self.tree.clear()

        root_item = QTreeWidgetItem(self.tree)
        root_item.setText(0, geometry_name)
        root_item.setText(1, "")
        root_item.setExpanded(True)
        root_item.setCheckState(0, Qt.Checked)

        vertices_item = QTreeWidgetItem(root_item)
        vertices_item.setText(0, "Vertices")
        vertices_item.setText(1, "0")
        vertices_item.setCheckState(0, Qt.Checked)
        vertices_item.setData(0, Qt.UserRole, "vertices")

        edges_item = QTreeWidgetItem(root_item)
        edges_item.setText(0, "Edges")
        edges_item.setText(1, "0")
        edges_item.setCheckState(0, Qt.Checked)
        edges_item.setData(0, Qt.UserRole, "edges")

        faces_item = QTreeWidgetItem(root_item)
        faces_item.setText(0, "Faces")
        faces_item.setText(1, "0")
        faces_item.setCheckState(0, Qt.Checked)
        faces_item.setData(0, Qt.UserRole, "faces")

        bodies_item = QTreeWidgetItem(root_item)
        bodies_item.setText(0, "Bodies")
        bodies_item.setText(1, "0")
        bodies_item.setCheckState(0, Qt.Checked)
        bodies_item.setData(0, Qt.UserRole, "bodies")

        if shape is not None:
            self._extract_geometry_elements(shape, vertices_item, edges_item, faces_item, bodies_item)

    def _extract_geometry_elements(self, shape, vertices_item, edges_item, faces_item, bodies_item):
        """
        从形状中提取几何元素并添加到树中

        Args:
            shape: OpenCASCADE TopoDS_Shape对象
            vertices_item: 顶点树项
            edges_item: 边树项
            faces_item: 面树项
            bodies_item: 体树项
        """
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        from OCC.Core.gp import gp_Pnt

        vertex_count = 0
        edge_count = 0
        face_count = 0
        body_count = 0

        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        while explorer.More():
            vertex = explorer.Current()
            vertex_item = QTreeWidgetItem(vertices_item)
            vertex_item.setText(0, f"Vertex_{vertex_count}")
            vertex_item.setText(1, "")
            vertex_item.setCheckState(0, Qt.Checked)
            vertex_item.setData(0, Qt.UserRole, ("vertex", vertex, vertex_count))

            pnt = self._get_vertex_point(vertex)
            if pnt:
                vertex_item.setToolTip(0, f"坐标: ({pnt.X():.3f}, {pnt.Y():.3f}, {pnt.Z():.3f})")

            vertex_count += 1
            explorer.Next()

        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while explorer.More():
            edge = explorer.Current()
            edge_item = QTreeWidgetItem(edges_item)
            edge_item.setText(0, f"Edge_{edge_count}")
            edge_item.setText(1, "")
            edge_item.setCheckState(0, Qt.Checked)
            edge_item.setData(0, Qt.UserRole, ("edge", edge, edge_count))

            edge_length = self._get_edge_length(edge)
            if edge_length is not None:
                edge_item.setToolTip(0, f"长度: {edge_length:.3f}")

            edge_count += 1
            explorer.Next()

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            face_item = QTreeWidgetItem(faces_item)
            face_item.setText(0, f"Face_{face_count}")
            face_item.setText(1, "")
            face_item.setCheckState(0, Qt.Checked)
            face_item.setData(0, Qt.UserRole, ("face", face, face_count))

            face_area = self._get_face_area(face)
            if face_area is not None:
                face_item.setToolTip(0, f"面积: {face_area:.3f}")

            face_count += 1
            explorer.Next()

        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            solid = explorer.Current()
            body_item = QTreeWidgetItem(bodies_item)
            body_item.setText(0, f"Body_{body_count}")
            body_item.setText(1, "")
            body_item.setCheckState(0, Qt.Checked)
            body_item.setData(0, Qt.UserRole, ("body", solid, body_count))

            body_volume = self._get_solid_volume(solid)
            if body_volume is not None:
                body_item.setToolTip(0, f"体积: {body_volume:.3f}")

            body_count += 1
            explorer.Next()

        vertices_item.setText(1, str(vertex_count))
        edges_item.setText(1, str(edge_count))
        faces_item.setText(1, str(face_count))
        bodies_item.setText(1, str(body_count))

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
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
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

            if hasattr(self.parent, 'on_geometry_visibility_changed'):
                element_type = item.data(0, Qt.UserRole)
                if isinstance(element_type, str):
                    self.parent.on_geometry_visibility_changed(element_type, item.checkState(0) == Qt.Checked)
                elif isinstance(element_type, tuple):
                    elem_type, elem_data, elem_index = element_type
                    self.parent.on_geometry_element_visibility_changed(elem_type, elem_index, item.checkState(0) == Qt.Checked)

    def _on_item_clicked(self, item, column):
        """
        树项点击时的回调

        Args:
            item: 点击的树项
            column: 点击的列
        """
        element_data = item.data(0, Qt.UserRole)
        if isinstance(element_data, tuple) and len(element_data) == 3:
            elem_type, elem_data, elem_index = element_data
            if hasattr(self.parent, 'on_geometry_element_selected'):
                self.parent.on_geometry_element_selected(elem_type, elem_data, elem_index)

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

        if isinstance(element_data, str):
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

        elif isinstance(element_data, tuple) and len(element_data) == 3:
            elem_type, elem_data, elem_index = element_data

            show_action = QAction("显示", self.tree)
            show_action.triggered.connect(lambda: self._set_item_visibility(item, True))
            menu.addAction(show_action)

            hide_action = QAction("隐藏", self.tree)
            hide_action.triggered.connect(lambda: self._set_item_visibility(item, False))
            menu.addAction(hide_action)

            if hasattr(self.parent, 'on_geometry_element_selected'):
                select_action = QAction("选中", self.tree)
                select_action.triggered.connect(lambda: self.parent.on_geometry_element_selected(elem_type, elem_data, elem_index))
                menu.addAction(select_action)

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

    def get_visible_elements(self, element_type=None):
        """
        获取可见的几何元素

        Args:
            element_type: 元素类型（"vertices", "edges", "faces", "bodies"），None表示所有类型

        Returns:
            可见元素列表
        """
        visible_elements = {}

        root = self.tree.topLevelItem(0)
        if root is None:
            return visible_elements

        for i in range(root.childCount()):
            category_item = root.child(i)
            category = category_item.data(0, Qt.UserRole)

            if element_type is not None and category != element_type:
                continue

            if category_item.checkState(0) != Qt.Checked:
                continue

            visible_elements[category] = []

            for j in range(category_item.childCount()):
                element_item = category_item.child(j)

                if element_item.checkState(0) == Qt.Checked:
                    element_data = element_item.data(0, Qt.UserRole)
                    if isinstance(element_data, tuple) and len(element_data) == 3:
                        elem_type, elem_data, elem_index = element_data
                        visible_elements[category].append((elem_index, elem_data))

        if element_type is not None and element_type in visible_elements:
            return visible_elements[element_type]

        return visible_elements

    def set_element_visibility(self, element_type, element_index, visible):
        """
        设置特定元素的可见性

        Args:
            element_type: 元素类型（"vertices", "edges", "faces", "bodies"）
            element_index: 元素索引
            visible: 是否可见
        """
        root = self.tree.topLevelItem(0)
        if root is None:
            return

        for i in range(root.childCount()):
            category_item = root.child(i)
            category = category_item.data(0, Qt.UserRole)

            if category == element_type:
                for j in range(category_item.childCount()):
                    element_item = category_item.child(j)
                    element_data = element_item.data(0, Qt.UserRole)

                    if isinstance(element_data, tuple) and len(element_data) == 3:
                        elem_type, elem_data, elem_index = element_data
                        if elem_index == element_index:
                            element_item.setCheckState(0, Qt.Checked if visible else Qt.Unchecked)
                            break

    def set_category_visibility(self, category, visible):
        """
        设置类别的可见性

        Args:
            category: 类别名称（"vertices", "edges", "faces", "bodies"）
            visible: 是否可见
        """
        root = self.tree.topLevelItem(0)
        if root is None:
            return

        for i in range(root.childCount()):
            category_item = root.child(i)
            if category_item.data(0, Qt.UserRole) == category:
                category_item.setCheckState(0, Qt.Checked if visible else Qt.Unchecked)
                break

    def clear(self):
        """清空树"""
        self.tree.clear()
        self.geometry_data = None
