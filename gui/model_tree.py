#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一模型树组件
用于显示和管理几何、网格和部件信息
三层结构：几何、网格、部件
"""

from PyQt5.QtWidgets import (QTreeWidget, QTreeWidgetItem, QWidget, QVBoxLayout,
                             QHeaderView, QMenu, QAction, QActionGroup, QDialog, QShortcut)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QKeySequence
from gui.ui_utils import PARTS_INFO_RESERVED_KEYS


class ModelTreeWidget:
    """统一模型树组件 - 三层结构：几何、网格、部件"""

    MAX_TREE_ITEMS = 100000  # 最大树项数量，超过则使用虚拟化
    LAZY_LOAD_THRESHOLD = 10000  # 延迟加载阈值

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
        self._pending_label_restore = None

        self._updating_items = False  # 标志：是否正在更新项（防止递归调用）

        self._create_tree_widget()
        self._setup_ui()
        self._init_tree_structure()

    def _get_parent_handler(self, handler_name):
        """获取父级处理函数（优先使用part_manager）"""
        manager = getattr(self.parent, 'part_manager', None)
        if manager and hasattr(manager, handler_name):
            return getattr(manager, handler_name)
        if hasattr(self.parent, handler_name):
            return getattr(self.parent, handler_name)
        return None

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
        self.tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self._delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self.tree)
        self._delete_shortcut.activated.connect(self._delete_selected_geometry_elements)

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

        self.tree.blockSignals(True)

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
        geometry_vertices_item.setExpanded(False)
        geometry_vertices_item.setData(0, Qt.UserRole, ("geometry", "vertices"))

        geometry_edges_item = QTreeWidgetItem(geometry_item)
        geometry_edges_item.setText(0, "线")
        geometry_edges_item.setText(1, "0")
        geometry_edges_item.setCheckState(0, Qt.Checked)
        geometry_edges_item.setExpanded(False)
        geometry_edges_item.setData(0, Qt.UserRole, ("geometry", "edges"))

        geometry_faces_item = QTreeWidgetItem(geometry_item)
        geometry_faces_item.setText(0, "面")
        geometry_faces_item.setText(1, "0")
        geometry_faces_item.setCheckState(0, Qt.Checked)
        geometry_faces_item.setExpanded(False)
        geometry_faces_item.setData(0, Qt.UserRole, ("geometry", "faces"))

        geometry_bodies_item = QTreeWidgetItem(geometry_item)
        geometry_bodies_item.setText(0, "体")
        geometry_bodies_item.setText(1, "0")
        geometry_bodies_item.setCheckState(0, Qt.Checked)
        geometry_bodies_item.setExpanded(False)
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

        self.tree.blockSignals(False)

    def load_geometry(self, shape, geometry_name="几何"):
        """
        加载几何模型到树中（使用分批加载避免阻塞）

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

        self.tree.blockSignals(True)

        vertex_count = 0
        edge_count = 0
        face_count = 0
        body_count = 0

        self.tree.blockSignals(False)

        vertices_item.setText(1, "加载中...")
        edges_item.setText(1, "加载中...")
        faces_item.setText(1, "加载中...")
        bodies_item.setText(1, "加载中...")

        self._batch_load_geometry_elements(shape, vertices_item, edges_item, faces_item, bodies_item)

    def set_pending_label_restore(self, old_shape, new_shape):
        self._pending_label_restore = (old_shape, new_shape)

    def restore_geometry_labels(self, old_shape, new_shape):
        """根据旧几何索引保持元素名称不变"""
        if old_shape is None or new_shape is None:
            return
        try:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
        except Exception:
            return

        geometry_item = self.tree.topLevelItem(0)
        if geometry_item is None:
            return
        cleanup_handler = self._get_parent_handler('cleanup_geometry_actors')
        if cleanup_handler:
            cleanup_handler()
        self.tree.blockSignals(True)

        type_pairs = [
            ("vertices", TopAbs_VERTEX, "点_"),
            ("edges", TopAbs_EDGE, "线_"),
            ("faces", TopAbs_FACE, "面_"),
            ("bodies", TopAbs_SOLID, "体_"),
        ]

        old_maps = {}
        for key, occ_type, _ in type_pairs:
            idx_map = {}
            explorer = TopExp_Explorer(old_shape, occ_type)
            idx = 0
            while explorer.More():
                idx_map[idx] = explorer.Current()
                idx += 1
                explorer.Next()
            old_maps[key] = idx_map

        def _match_old_index(new_shape_obj, key):
            for old_idx, old_shape_obj in old_maps.get(key, {}).items():
                try:
                    if hasattr(old_shape_obj, "IsEqual") and old_shape_obj.IsEqual(new_shape_obj):
                        return old_idx
                    if hasattr(old_shape_obj, "IsSame") and old_shape_obj.IsSame(new_shape_obj):
                        return old_idx
                except Exception:
                    continue
            return None

        for i in range(geometry_item.childCount()):
            element_type_item = geometry_item.child(i)
            element_type_data = element_type_item.data(0, Qt.UserRole)
            if not (isinstance(element_type_data, tuple) and len(element_type_data) >= 2):
                continue
            _, elem_type = element_type_data[0], element_type_data[1]
            label_prefix = None
            for key, _, prefix in type_pairs:
                if key == elem_type:
                    label_prefix = prefix
                    break
            if label_prefix is None:
                continue
            children = []
            for j in range(element_type_item.childCount()):
                element_item = element_type_item.child(j)
                element_data = element_item.data(0, Qt.UserRole)
                if not (isinstance(element_data, tuple) and len(element_data) >= 4):
                    children.append((j, element_item))
                    continue
                category, element_type, element_obj, element_index = element_data
                if element_type != elem_type:
                    children.append((element_index, element_item))
                    continue
                old_index = _match_old_index(element_obj, elem_type)
                if old_index is None:
                    children.append((element_index, element_item))
                    continue
                element_item.setText(0, f"{label_prefix}{old_index}")
                element_item.setData(0, Qt.UserRole, (category, element_type, element_obj, old_index))
                children.append((old_index, element_item))

            if children:
                for _ in range(element_type_item.childCount()):
                    element_type_item.takeChild(0)
                for _, child in sorted(children, key=lambda x: x[0]):
                    element_type_item.addChild(child)
        self.tree.blockSignals(False)
        handler = self._get_parent_handler('_update_geometry_element_display')
        if handler:
            handler()

    def _batch_load_geometry_elements(self, shape, vertices_item, edges_item, faces_item, bodies_item):
        """
        分批加载几何元素到树中

        Args:
            shape: OpenCASCADE TopoDS_Shape对象
            vertices_item: 顶点树项
            edges_item: 边树项
            faces_item: 面树项
            bodies_item: 体树项
        """
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID

        BATCH_SIZE = 1000  # 每批处理的元素数量（增加以提高批量处理效率）

        vertex_explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        body_explorer = TopExp_Explorer(shape, TopAbs_SOLID)

        vertex_count = 0
        edge_count = 0
        face_count = 0
        body_count = 0

        def process_batch():
            nonlocal vertex_count, edge_count, face_count, body_count

            self.tree.blockSignals(True)

            processed = 0

            for _ in range(BATCH_SIZE):
                if vertex_explorer.More():
                    vertex = vertex_explorer.Current()
                    
                    if vertex_count < self.LAZY_LOAD_THRESHOLD:
                        vertex_item = QTreeWidgetItem(vertices_item)
                        vertex_item.setText(0, f"点_{vertex_count}")
                        vertex_item.setText(1, "")
                        vertex_item.setCheckState(0, Qt.Checked)
                        vertex_item.setData(0, Qt.UserRole, ("geometry", "vertices", vertex, vertex_count))
                    
                    vertex_count += 1
                    vertex_explorer.Next()
                    processed += 1

            for _ in range(BATCH_SIZE):
                if edge_explorer.More():
                    edge = edge_explorer.Current()
                    
                    if edge_count < self.LAZY_LOAD_THRESHOLD:
                        edge_item = QTreeWidgetItem(edges_item)
                        edge_item.setText(0, f"线_{edge_count}")
                        edge_item.setText(1, "")
                        edge_item.setCheckState(0, Qt.Checked)
                        edge_item.setData(0, Qt.UserRole, ("geometry", "edges", edge, edge_count))
                    
                    edge_count += 1
                    edge_explorer.Next()
                    processed += 1

            for _ in range(BATCH_SIZE):
                if face_explorer.More():
                    face = face_explorer.Current()
                    
                    if face_count < self.LAZY_LOAD_THRESHOLD:
                        face_item = QTreeWidgetItem(faces_item)
                        face_item.setText(0, f"面_{face_count}")
                        face_item.setText(1, "")
                        face_item.setCheckState(0, Qt.Checked)
                        face_item.setData(0, Qt.UserRole, ("geometry", "faces", face, face_count))
                    
                    face_count += 1
                    face_explorer.Next()
                    processed += 1

            for _ in range(BATCH_SIZE):
                if body_explorer.More():
                    solid = body_explorer.Current()
                    
                    if body_count < self.LAZY_LOAD_THRESHOLD:
                        body_item = QTreeWidgetItem(bodies_item)
                        body_item.setText(0, f"体_{body_count}")
                        body_item.setText(1, "")
                        body_item.setCheckState(0, Qt.Checked)
                        body_item.setData(0, Qt.UserRole, ("geometry", "bodies", solid, body_count))
                    
                    body_count += 1
                    body_explorer.Next()
                    processed += 1

            vertices_item.setText(1, str(vertex_count))
            edges_item.setText(1, str(edge_count))
            faces_item.setText(1, str(face_count))
            bodies_item.setText(1, str(body_count))

            self.tree.blockSignals(False)

            if vertex_explorer.More() or edge_explorer.More() or face_explorer.More() or body_explorer.More():
                QTimer.singleShot(0, process_batch)
            else:
                handler = self._get_parent_handler('_update_geometry_element_display')
                if handler:
                    handler()
                if self._pending_label_restore:
                    old_shape, new_shape = self._pending_label_restore
                    self._pending_label_restore = None
                    self.restore_geometry_labels(old_shape, new_shape)

        QTimer.singleShot(0, process_batch)

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

        self.tree.blockSignals(True)

        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        while explorer.More():
            vertex = explorer.Current()
            
            if vertex_count < self.LAZY_LOAD_THRESHOLD:
                vertex_item = QTreeWidgetItem(vertices_item)
                vertex_item.setText(0, f"点_{vertex_count}")
                vertex_item.setText(1, "")
                vertex_item.setCheckState(0, Qt.Checked)
                vertex_item.setData(0, Qt.UserRole, ("geometry", "vertices", vertex, vertex_count))
            
            vertex_count += 1
            explorer.Next()

        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while explorer.More():
            edge = explorer.Current()
            
            if edge_count < self.LAZY_LOAD_THRESHOLD:
                edge_item = QTreeWidgetItem(edges_item)
                edge_item.setText(0, f"线_{edge_count}")
                edge_item.setText(1, "")
                edge_item.setCheckState(0, Qt.Checked)
                edge_item.setData(0, Qt.UserRole, ("geometry", "edges", edge, edge_count))
            
            edge_count += 1
            explorer.Next()

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            
            if face_count < self.LAZY_LOAD_THRESHOLD:
                face_item = QTreeWidgetItem(faces_item)
                face_item.setText(0, f"面_{face_count}")
                face_item.setText(1, "")
                face_item.setCheckState(0, Qt.Checked)
                face_item.setData(0, Qt.UserRole, ("geometry", "faces", face, face_count))
            
            face_count += 1
            explorer.Next()

        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            solid = explorer.Current()
            
            if body_count < self.LAZY_LOAD_THRESHOLD:
                body_item = QTreeWidgetItem(bodies_item)
                body_item.setText(0, f"体_{body_count}")
                body_item.setText(1, "")
                body_item.setCheckState(0, Qt.Checked)
                body_item.setData(0, Qt.UserRole, ("geometry", "bodies", solid, body_count))
            
            body_count += 1
            explorer.Next()

        vertices_item.setText(1, str(vertex_count))
        edges_item.setText(1, str(edge_count))
        faces_item.setText(1, str(face_count))
        bodies_item.setText(1, str(body_count))

        self.tree.blockSignals(False)

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

        self.tree.blockSignals(True)

        if hasattr(mesh_data, 'node_coords'):
            node_coords = mesh_data.node_coords
            vertex_count = len(node_coords)

            if vertex_count <= self.LAZY_LOAD_THRESHOLD:
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
            else:
                for i in range(0, min(vertex_count, self.MAX_TREE_ITEMS)):
                    coord = node_coords[i]
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

                if vertex_count > self.MAX_TREE_ITEMS:
                    summary_item = QTreeWidgetItem(vertices_item)
                    summary_item.setText(0, f"... (还有 {vertex_count - self.MAX_TREE_ITEMS} 个节点)")
                    summary_item.setText(1, "")
                    summary_item.setCheckState(0, Qt.Checked)
                    summary_item.setData(0, Qt.UserRole, ("mesh", "vertex_summary", self.MAX_TREE_ITEMS, vertex_count))

        if hasattr(mesh_data, 'cell_container'):
            cells = mesh_data.cell_container
            face_count = 0
            line_count = 0
            volume_count = 0
            for cell in cells:
                if cell is None:
                    continue
                cell_name = cell.__class__.__name__
                
                # 处理具体单元类型
                if cell_name in ('Triangle', 'Quadrilateral'):
                    face_count += 1
                elif cell_name in ('Tetrahedron', 'Pyramid', 'Prism', 'Hexahedron'):
                    volume_count += 1
                # 处理GenericCell（包括线段单元）
                elif hasattr(cell, 'node_ids'):
                    node_count = len(cell.node_ids)
                    if node_count == 2:
                        line_count += 1
                    elif node_count in (3, 4) and getattr(mesh_data, 'dimension', 2) == 2:
                        face_count += 1
                    elif node_count in (4, 5, 6, 8) and getattr(mesh_data, 'dimension', 3) == 3:
                        volume_count += 1
                # 处理list/tuple类型的单元（兼容性）
                elif isinstance(cell, (list, tuple)):
                    node_count = len(cell)
                    if node_count == 2:
                        line_count += 1
                    elif node_count in (3, 4) and getattr(mesh_data, 'dimension', 2) == 2:
                        face_count += 1
                    elif node_count in (4, 5, 6, 8) and getattr(mesh_data, 'dimension', 3) == 3:
                        volume_count += 1
            if hasattr(mesh_data, 'volume_cells') and mesh_data.volume_cells:
                body_count = len(mesh_data.volume_cells)
            elif volume_count:
                body_count = volume_count

            if hasattr(mesh_data, 'boundary_info') and mesh_data.boundary_info:
                boundary_lines = 0
                for part_data in mesh_data.boundary_info.values():
                    for face in part_data.get('faces', []):
                        nodes = face.get('nodes', [])
                        if len(nodes) == 2:
                            boundary_lines += 1
                if boundary_lines:
                    line_count = boundary_lines
            if hasattr(mesh_data, 'line_cells') and mesh_data.line_cells and line_count == 0:
                line_count = len(mesh_data.line_cells)
            edge_count = line_count

            if hasattr(mesh_data, 'volume_cells') and mesh_data.volume_cells:
                if body_count <= self.LAZY_LOAD_THRESHOLD:
                    for i, cell in enumerate(mesh_data.volume_cells):
                        body_item = QTreeWidgetItem(bodies_item)
                        body_item.setText(0, f"体_{i}")
                        body_item.setText(1, "")
                        body_item.setCheckState(0, Qt.Checked)
                        body_item.setData(0, Qt.UserRole, ("mesh", "body", i, cell))
                        body_item.setToolTip(0, f"节点数: {len(cell)}")
                else:
                    for i in range(0, min(body_count, self.MAX_TREE_ITEMS)):
                        cell = mesh_data.volume_cells[i]
                        body_item = QTreeWidgetItem(bodies_item)
                        body_item.setText(0, f"体_{i}")
                        body_item.setText(1, "")
                        body_item.setCheckState(0, Qt.Checked)
                        body_item.setData(0, Qt.UserRole, ("mesh", "body", i, cell))
                        body_item.setToolTip(0, f"节点数: {len(cell)}")
                    if body_count > self.MAX_TREE_ITEMS:
                        summary_item = QTreeWidgetItem(bodies_item)
                        summary_item.setText(0, f"... (还有 {body_count - self.MAX_TREE_ITEMS} 个单元)")
                        summary_item.setText(1, "")
                        summary_item.setCheckState(0, Qt.Checked)
                        summary_item.setData(0, Qt.UserRole, ("mesh", "body_summary", self.MAX_TREE_ITEMS, body_count))

            if face_count <= self.LAZY_LOAD_THRESHOLD:
                for i, cell in enumerate(cells):
                    face_item = QTreeWidgetItem(faces_item)
                    face_item.setText(0, f"面_{i}")
                    face_item.setText(1, "")
                    face_item.setCheckState(0, Qt.Checked)
                    face_item.setData(0, Qt.UserRole, ("mesh", "face", i, cell))

                    num_nodes = len(cell.node_ids) if hasattr(cell, 'node_ids') else len(cell)
                    face_item.setToolTip(0, f"节点数: {num_nodes}")
            else:
                for i in range(0, min(face_count, self.MAX_TREE_ITEMS)):
                    cell = cells[i]
                    face_item = QTreeWidgetItem(faces_item)
                    face_item.setText(0, f"面_{i}")
                    face_item.setText(1, "")
                    face_item.setCheckState(0, Qt.Checked)
                    face_item.setData(0, Qt.UserRole, ("mesh", "face", i, cell))

                    num_nodes = len(cell.node_ids) if hasattr(cell, 'node_ids') else len(cell)
                    face_item.setToolTip(0, f"节点数: {num_nodes}")

                if face_count > self.MAX_TREE_ITEMS:
                    summary_item = QTreeWidgetItem(faces_item)
                    summary_item.setText(0, f"... (还有 {face_count - self.MAX_TREE_ITEMS} 个单元)")
                    summary_item.setText(1, "")
                    summary_item.setCheckState(0, Qt.Checked)
                    summary_item.setData(0, Qt.UserRole, ("mesh", "face_summary", self.MAX_TREE_ITEMS, face_count))
        elif hasattr(mesh_data, 'cells'):
            cells = mesh_data.cells
            face_count = 0
            line_count = 0
            volume_count = 0
            for cell in cells:
                if cell is None:
                    continue
                if hasattr(cell, 'node_ids'):
                    node_count = len(cell.node_ids)
                elif isinstance(cell, (list, tuple)):
                    node_count = len(cell)
                else:
                    continue
                if node_count == 2:
                    line_count += 1
                elif node_count in (3, 4) and getattr(mesh_data, 'dimension', 2) == 2:
                    face_count += 1
                elif node_count in (4, 5, 6, 8) and getattr(mesh_data, 'dimension', 3) == 3:
                    volume_count += 1
            if face_count == 0 and volume_count == 0 and line_count == 0 and cells:
                face_count = len(cells)
            if volume_count:
                body_count = volume_count
            if hasattr(mesh_data, 'boundary_info') and mesh_data.boundary_info:
                boundary_lines = 0
                for part_data in mesh_data.boundary_info.values():
                    for face in part_data.get('faces', []):
                        nodes = face.get('nodes', [])
                        if len(nodes) == 2:
                            boundary_lines += 1
                if boundary_lines:
                    line_count = boundary_lines
            if hasattr(mesh_data, 'line_cells') and mesh_data.line_cells and line_count == 0:
                line_count = len(mesh_data.line_cells)
            edge_count = line_count

            if face_count <= self.LAZY_LOAD_THRESHOLD:
                for i, cell in enumerate(cells):
                    face_item = QTreeWidgetItem(faces_item)
                    face_item.setText(0, f"面_{i}")
                    face_item.setText(1, "")
                    face_item.setCheckState(0, Qt.Checked)
                    face_item.setData(0, Qt.UserRole, ("mesh", "face", i, cell))

                    num_nodes = len(cell.node_ids) if hasattr(cell, 'node_ids') else len(cell)
                    face_item.setToolTip(0, f"节点数: {num_nodes}")
            else:
                for i in range(0, min(face_count, self.MAX_TREE_ITEMS)):
                    cell = cells[i]
                    face_item = QTreeWidgetItem(faces_item)
                    face_item.setText(0, f"面_{i}")
                    face_item.setText(1, "")
                    face_item.setCheckState(0, Qt.Checked)
                    face_item.setData(0, Qt.UserRole, ("mesh", "face", i, cell))

                    num_nodes = len(cell.node_ids) if hasattr(cell, 'node_ids') else len(cell)
                    face_item.setToolTip(0, f"节点数: {num_nodes}")

                if face_count > self.MAX_TREE_ITEMS:
                    summary_item = QTreeWidgetItem(faces_item)
                    summary_item.setText(0, f"... (还有 {face_count - self.MAX_TREE_ITEMS} 个单元)")
                    summary_item.setText(1, "")
                    summary_item.setCheckState(0, Qt.Checked)
                    summary_item.setData(0, Qt.UserRole, ("mesh", "face_summary", self.MAX_TREE_ITEMS, face_count))

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

        self.tree.blockSignals(True)

        # 处理字典类型的数据
        if isinstance(parts_data, dict) and 'parts_info' in parts_data:
            parts_info = parts_data['parts_info']
            if isinstance(parts_info, list):
                for part_info in parts_info:
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
            elif isinstance(parts_info, dict):
                for part_name, part_data in parts_info.items():
                    if part_name not in PARTS_INFO_RESERVED_KEYS:
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
        elif isinstance(parts_data, dict):
            for part_name, part_data in parts_data.items():
                if part_name not in PARTS_INFO_RESERVED_KEYS:
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
        elif hasattr(parts_data, 'parts_info') and parts_data.parts_info:
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
                    if part_name not in PARTS_INFO_RESERVED_KEYS:
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
            default_part_item.setText(0, "DefaultPart")
            default_part_item.setText(1, "")
            default_part_item.setCheckState(0, Qt.Checked)
            default_part_item.setData(0, Qt.UserRole, ("parts", None, 0))
            part_count = 1

        parts_item.setText(1, str(part_count))

        self.tree.blockSignals(False)

    def _on_item_changed(self, item, column):
        """
        树项改变时的回调

        Args:
            item: 改变的树项
            column: 改变的列
        """
        if column == 0:
            if self._updating_items:
                return

            self._updating_items = True
            self._update_child_items(item, item.checkState(0))
            self._update_parent_item(item)
            self._updating_items = False

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
        handler = self._get_parent_handler('on_model_tree_visibility_changed')
        if not handler:
            return

        if isinstance(element_data, str):
            handler(element_data, item.checkState(0) == Qt.Checked)
        elif isinstance(element_data, tuple) and len(element_data) >= 2:
            category = element_data[0]

            # 对于部件，element_data 格式为 ("parts", part_data, part_count)
            # 我们只需要传递 category 和 element_index
            if category == 'parts' and len(element_data) >= 3:
                element_index = element_data[2]
                handler(category, element_index, item.checkState(0) == Qt.Checked)
            else:
                element_type = element_data[1]
                if len(element_data) >= 4:
                    element_index = element_data[3]
                    handler(category, element_type, element_index, item.checkState(0) == Qt.Checked)
                else:
                    handler(category, element_type, item.checkState(0) == Qt.Checked)

    def _handle_selection_change(self, item, element_data):
        """
        处理选择改变

        Args:
            item: 树项
            element_data: 元素数据
        """
        handler = self._get_parent_handler('on_model_tree_selected')
        if not handler:
            return

        if isinstance(element_data, tuple) and len(element_data) >= 3:
            category = element_data[0]
            
            # 对于部件，element_data 格式为 ("parts", part_data, part_count)
            if category == 'parts' and len(element_data) >= 3:
                part_data = element_data[1]
                element_index = element_data[2]
                # 对于部件，element_type 可以是部件名称
                element_type = item.text(0)
                handler(category, element_type, element_index, part_data)
            elif len(element_data) >= 4:
                element_type = element_data[1]
                element_index = element_data[3]
                element_obj = element_data[2] if len(element_data) >= 3 else None
                handler(category, element_type, element_index, element_obj)

    def _update_child_items(self, item, check_state):
        """
        更新子项的选中状态

        Args:
            item: 父项
            check_state: 选中状态
        """
        self.tree.blockSignals(True)
        def update_descendants(parent_item):
            for i in range(parent_item.childCount()):
                child = parent_item.child(i)
                child.setCheckState(0, check_state)
                update_descendants(child)
        update_descendants(item)
        self.tree.blockSignals(False)

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

        self.tree.blockSignals(True)
        if all_checked:
            parent.setCheckState(0, Qt.Checked)
        elif all_unchecked:
            parent.setCheckState(0, Qt.Unchecked)
        else:
            parent.setCheckState(0, Qt.PartiallyChecked)
        self.tree.blockSignals(False)
        self._update_parent_item(parent)

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

        if element_data in ("geometry", "mesh", "parts"):
            menu.addSeparator()
            display_group = QActionGroup(self.tree)
            display_group.setExclusive(True)
            full_action = QAction("整体显示", self.tree)
            full_action.setCheckable(True)
            full_action.setChecked(getattr(self.parent, 'display_mode', 'full') == "full")
            full_action.triggered.connect(lambda: self._set_display_mode("full"))
            display_group.addAction(full_action)
            menu.addAction(full_action)

            element_action = QAction("元素显示", self.tree)
            element_action.setCheckable(True)
            element_action.setChecked(getattr(self.parent, 'display_mode', 'full') == "elements")
            element_action.triggered.connect(lambda: self._set_display_mode("elements"))
            display_group.addAction(element_action)
            menu.addAction(element_action)

        if isinstance(element_data, tuple) and len(element_data) >= 3:
            category = element_data[0]
            element_type = element_data[1]
            element_obj = element_data[2]

            if category == 'geometry':
                menu.addSeparator()

                if element_type == 'vertices':
                    view_coords_action = QAction("查看坐标", self.tree)
                    view_coords_action.triggered.connect(lambda: self._show_vertex_properties(element_obj, item))
                    menu.addAction(view_coords_action)

                elif element_type == 'edges':
                    view_length_action = QAction("查看长度", self.tree)
                    view_length_action.triggered.connect(lambda: self._show_edge_properties(element_obj, item))
                    menu.addAction(view_length_action)

                elif element_type == 'faces':
                    view_area_action = QAction("查看面积", self.tree)
                    view_area_action.triggered.connect(lambda: self._show_face_properties(element_obj, item))
                    menu.addAction(view_area_action)

                elif element_type == 'bodies':
                    view_volume_action = QAction("查看体积", self.tree)
                    view_volume_action.triggered.connect(lambda: self._show_solid_properties(element_obj, item))
                    menu.addAction(view_volume_action)

                delete_action = QAction("删除元素", self.tree)
                delete_action.triggered.connect(self._delete_selected_geometry_elements)
                menu.addAction(delete_action)

        if element_data == "parts":
            menu.addSeparator()
            create_part_action = QAction("创建部件", self.tree)
            create_part_action.triggered.connect(lambda: self._create_part_dialog())
            menu.addAction(create_part_action)
        if element_data == "geometry":
            menu.addSeparator()
            delete_action = QAction("删除几何", self.tree)
            delete_action.triggered.connect(self._open_geometry_delete_dialog)
            menu.addAction(delete_action)
        elif isinstance(element_data, tuple) and len(element_data) >= 2:
            category = element_data[0]
            if category == "parts":
                menu.addSeparator()
                params_action = QAction("设置部件参数", self.tree)
                params_action.triggered.connect(lambda: self._open_part_params_dialog(item))
                menu.addAction(params_action)
                add_elements_action = QAction("添加元素到部件", self.tree)
                add_elements_action.triggered.connect(lambda: self._open_add_elements_dialog(item))
                menu.addAction(add_elements_action)

        if not menu.isEmpty():
            menu.exec_(self.tree.mapToGlobal(position))

    def _show_vertex_properties(self, vertex, item):
        """
        显示顶点属性（按需计算）

        Args:
            vertex: OpenCASCADE TopoDS_Vertex对象
            item: 树项
        """
        from OCC.Core.BRep import BRep_Tool

        try:
            pnt = BRep_Tool.Pnt(vertex)
            coords = f"({pnt.X():.6f}, {pnt.Y():.6f}, {pnt.Z():.6f})"

            if hasattr(self.parent, 'log_info'):
                self.parent.log_info(f"顶点坐标: {coords}")

            self.tree.blockSignals(True)
            item.setToolTip(0, f"坐标: ({pnt.X():.3f}, {pnt.Y():.3f}, {pnt.Z():.3f})")
            self.tree.blockSignals(False)
        except Exception as e:
            if hasattr(self.parent, 'log_info'):
                self.parent.log_info(f"获取顶点坐标失败: {str(e)}")

    def _show_edge_properties(self, edge, item):
        """
        显示边属性（按需计算）

        Args:
            edge: OpenCASCADE TopoDS_Edge对象
            item: 树项
        """
        from OCC.Core.GCPnts import GCPnts_AbscissaPoint
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

        try:
            adaptor = BRepAdaptor_Curve(edge)
            length = GCPnts_AbscissaPoint.Length(adaptor)

            if hasattr(self.parent, 'log_info'):
                self.parent.log_info(f"边长度: {length:.6f}")

            self.tree.blockSignals(True)
            item.setToolTip(0, f"长度: {length:.3f}")
            self.tree.blockSignals(False)
        except Exception as e:
            if hasattr(self.parent, 'log_info'):
                self.parent.log_info(f"获取边长度失败: {str(e)}")

    def _show_face_properties(self, face, item):
        """
        显示面属性（按需计算）

        Args:
            face: OpenCASCADE TopoDS_Face对象
            item: 树项
        """
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps

        try:
            props = GProp_GProps()
            brepgprop.SurfaceProperties(face, props)
            area = props.Mass()

            if hasattr(self.parent, 'log_info'):
                self.parent.log_info(f"面面积: {area:.6f}")

            self.tree.blockSignals(True)
            item.setToolTip(0, f"面积: {area:.3f}")
            self.tree.blockSignals(False)
        except Exception as e:
            if hasattr(self.parent, 'log_info'):
                self.parent.log_info(f"获取面面积失败: {str(e)}")

    def _show_solid_properties(self, solid, item):
        """
        显示体属性（按需计算）

        Args:
            solid: OpenCASCADE TopoDS_Solid对象
            item: 树项
        """
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps

        try:
            props = GProp_GProps()
            brepgprop.VolumeProperties(solid, props)
            volume = props.Mass()

            if hasattr(self.parent, 'log_info'):
                self.parent.log_info(f"体体积: {volume:.6f}")

            self.tree.blockSignals(True)
            item.setToolTip(0, f"体积: {volume:.3f}")
            self.tree.blockSignals(False)
        except Exception as e:
            if hasattr(self.parent, 'log_info'):
                self.parent.log_info(f"获取体体积失败: {str(e)}")

    def _set_item_visibility(self, item, visible):
        """
        设置项的可见性

        Args:
            item: 树项
            visible: 是否可见
        """
        target_state = Qt.Checked if visible else Qt.Unchecked
        if item.checkState(0) == target_state:
            return
        item.setCheckState(0, target_state)

    def _set_display_mode(self, mode):
        """设置全局显示模式并刷新"""
        if hasattr(self.parent, 'view_controller') and hasattr(self.parent.view_controller, 'set_display_mode'):
            self.parent.view_controller.set_display_mode(mode)
            return
        if hasattr(self.parent, 'display_mode'):
            self.parent.display_mode = mode
        handler = self._get_parent_handler('on_display_mode_changed')
        if handler:
            handler(mode)

    def _create_part_dialog(self):
        """显示创建部件对话框"""
        from .create_part_dialog import CreatePartDialog
        existing_dialog = getattr(self, "_create_part_dialog_ref", None)
        if existing_dialog and existing_dialog.isVisible():
            existing_dialog.raise_()
            existing_dialog.activateWindow()
            return

        dialog = CreatePartDialog(self.parent, self.geometry_data, self.mesh_data)
        dialog.setModal(False)
        dialog.setWindowModality(Qt.NonModal)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.accepted.connect(lambda: self._on_part_dialog_accepted(dialog))
        dialog.finished.connect(self._on_part_dialog_finished)
        self._create_part_dialog_ref = dialog
        dialog.show()

    def _on_part_dialog_accepted(self, dialog):
        part_info = dialog.get_part_info()
        self._add_part_to_tree(part_info)

    def _on_part_dialog_finished(self, result):
        self._create_part_dialog_ref = None
    
    def _add_part_to_tree(self, part_info):
        """
        添加部件到树中
        
        Args:
            part_info: 部件信息字典
        """
        parts_item = self.tree.topLevelItem(2)
        if parts_item is None:
            return
        
        self.tree.blockSignals(True)
        
        part_name = part_info.get("part_name", "新部件")
        geometry_elements = part_info.get("geometry_elements", {})
        mesh_elements = part_info.get("mesh_elements", {})
        
        part_item = QTreeWidgetItem(parts_item)
        part_item.setText(0, part_name)
        part_item.setText(1, "")
        part_item.setCheckState(0, Qt.Checked)
        part_item.setData(0, Qt.UserRole, ("parts", part_info, parts_item.childCount()))
        
        geo_count = sum(len(v) for v in geometry_elements.values())
        mesh_count = sum(len(v) for v in mesh_elements.values())
        total_count = geo_count + mesh_count
        
        if total_count > 0:
            part_item.setToolTip(0, f"几何元素: {geo_count}, 网格元素: {mesh_count}")
        
        parts_item.setText(1, str(parts_item.childCount()))
        
        self.tree.blockSignals(False)
        
        handler = self._get_parent_handler('on_part_created')
        if handler:
            handler(part_info)

    def _open_add_elements_dialog(self, part_item):
        """打开向部件添加元素的对话框"""
        element_data = part_item.data(0, Qt.UserRole)
        if not (isinstance(element_data, tuple) and len(element_data) >= 2):
            return
        part_data = element_data[1]
        part_name = part_item.text(0)

        geometry_elements = {}
        mesh_elements = {}
        if isinstance(part_data, dict):
            geometry_elements = part_data.get("geometry_elements", {}) or {}
            mesh_elements = part_data.get("mesh_elements", {}) or {}

        from .create_part_dialog import AddElementsToPartDialog
        dialog = AddElementsToPartDialog(
            self.parent,
            self.geometry_data,
            self.mesh_data,
            part_name=part_name,
            geometry_elements=geometry_elements,
            mesh_elements=mesh_elements,
        )
        dialog.setModal(False)
        dialog.setWindowModality(Qt.NonModal)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.accepted.connect(lambda: self._on_add_elements_dialog_accepted(dialog, part_item))
        dialog.show()

    def _open_geometry_delete_dialog(self):
        handler = self._get_parent_handler('open_geometry_delete_dialog')
        if handler:
            handler()

    def _delete_selected_geometry_elements(self):
        handler = self._get_parent_handler('delete_geometry_selected_elements')
        if handler:
            handler()

    def _on_add_elements_dialog_accepted(self, dialog, part_item):
        part_info = dialog.get_part_info()
        handler = self._get_parent_handler("on_part_elements_added")
        if handler:
            handler(part_info)
        part_item.setToolTip(0, f"几何元素: {sum(len(v) for v in part_info.get('geometry_elements', {}).values())}, 网格元素: {sum(len(v) for v in part_info.get('mesh_elements', {}).values())}")

    def _open_part_params_dialog(self, part_item):
        element_data = part_item.data(0, Qt.UserRole)
        if not (isinstance(element_data, tuple) and len(element_data) >= 2):
            return
        part_name = part_item.text(0)
        handler = self._get_parent_handler("edit_mesh_params_for_part")
        if handler:
            handler(part_name)

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

        return visible_elements

    def get_geometry_type_states(self):
        """获取几何类型勾选状态"""
        geometry_item = self.tree.topLevelItem(0)
        if geometry_item is None:
            return {}

        type_order = [
            ('vertices', 0),
            ('edges', 1),
            ('faces', 2),
            ('bodies', 3)
        ]

        states = {}
        for type_name, index in type_order:
            child = geometry_item.child(index)
            if child is not None:
                states[type_name] = child.checkState(0) != Qt.Unchecked

        return states

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

    def is_category_visible(self, category):
        """检查顶层类别是否可见"""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == category:
                return item.checkState(0) != Qt.Unchecked
        return False

    def set_all_parts_visible(self, visible=True):
        """
        设置所有部件的可见性

        Args:
            visible: 是否可见
        """
        parts_item = self.tree.topLevelItem(2)
        if parts_item is None:
            return

        self.tree.blockSignals(True)

        for i in range(parts_item.childCount()):
            part_item = parts_item.child(i)
            part_item.setCheckState(0, Qt.Checked if visible else Qt.Unchecked)

        self.tree.blockSignals(False)

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
        self.geometry_name = "几何"
        self.mesh_name = "网格"
        self.parts_name = "部件"
        self._init_tree_structure()
        self.geometry_data = None
        self.mesh_data = None
        self.parts_data = None
