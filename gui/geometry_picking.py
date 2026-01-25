# -*- coding: utf-8 -*-
"""
几何拾取助手
基于VTK拾取器实现点/线/面/体的鼠标拾取
"""

from typing import Callable, Dict, Optional

import vtk
from PyQt5.QtCore import QObject, QEvent, Qt, QRect, QSize
from PyQt5.QtWidgets import QRubberBand


class GeometryPickingHelper:
    """几何元素拾取辅助类"""

    def __init__(
        self,
        mesh_display,
        gui=None,
        on_pick: Optional[Callable[[str, object, int], None]] = None,
        on_unpick: Optional[Callable[[str, object, int], None]] = None,
        on_confirm: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        on_delete: Optional[Callable[[], None]] = None,
    ):
        self.mesh_display = mesh_display
        self.gui = gui
        self._on_pick = on_pick
        self._on_unpick = on_unpick
        self._on_confirm = on_confirm
        self._on_cancel = on_cancel
        self._on_delete = on_delete
        self._enabled = False
        self._observer_id = None
        self._observer_right_id = None
        self._observer_key_id = None
        self._saved_display_mode = None
        self._highlighted_actors: Dict[vtk.vtkActor, Dict[str, object]] = {}
        self._area_selecting = False
        self._area_picker = vtk.vtkRenderedAreaPicker()
        self._area_filter = _AreaSelectionFilter(self)

        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.01)
        
        self._point_picker = vtk.vtkCellPicker()
        self._point_picker.SetTolerance(0.02)
        self._point_pick_enabled = False
        self._point_pick_observer_id = None
        self._point_pick_right_id = None
        self._point_pick_key_id = None
        self._on_point_pick = None
        self._on_point_pick_confirm = None
        self._on_point_pick_cancel = None
        self._on_point_pick_exit = None
        self._point_highlight_actor = None
        
        self._snap_enabled = False
        self._snap_tolerance = 0.1
        self._geometry_points_cache = []
        self._snap_line_actor = None
        self._snap_point_actor = None
        self._last_pick_pos = None
        self._last_snap_pos = None

    def set_callbacks(
        self,
        on_pick: Optional[Callable[[str, object, int], None]] = None,
        on_unpick: Optional[Callable[[str, object, int], None]] = None,
        on_confirm: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        on_delete: Optional[Callable[[], None]] = None,
    ):
        self._on_pick = on_pick
        self._on_unpick = on_unpick
        self._on_confirm = on_confirm
        self._on_cancel = on_cancel
        self._on_delete = on_delete

    def enable(self):
        """启用拾取"""
        if self._enabled:
            return
        if not self.mesh_display or not getattr(self.mesh_display, "frame", None):
            return

        self._ensure_geometry_display_mode()
        self.mesh_display.frame.installEventFilter(self._area_filter)
        interactor = self._get_interactor()
        if interactor is None:
            return

        self._observer_id = interactor.AddObserver("LeftButtonPressEvent", self._on_left_button_press)
        self._observer_right_id = interactor.AddObserver("RightButtonPressEvent", self._on_right_button_press)
        self._observer_key_id = interactor.AddObserver("KeyPressEvent", self._on_key_press)
        self._enabled = True

    def disable(self, restore_display_mode=True):
        """禁用拾取"""
        if not self._enabled:
            return

        interactor = self._get_interactor()
        if interactor is not None:
            if self._observer_id is not None:
                interactor.RemoveObserver(self._observer_id)
            if self._observer_right_id is not None:
                interactor.RemoveObserver(self._observer_right_id)
            if self._observer_key_id is not None:
                interactor.RemoveObserver(self._observer_key_id)
        self._observer_id = None
        self._observer_right_id = None
        self._observer_key_id = None
        self._clear_highlights()
        self._enabled = False
        if restore_display_mode:
            self._restore_display_mode()
        else:
            self._saved_display_mode = None
        if getattr(self.mesh_display, "frame", None) is not None:
            self.mesh_display.frame.removeEventFilter(self._area_filter)

    def cleanup(self, restore_display_mode=True):
        """清理资源"""
        self.disable(restore_display_mode=restore_display_mode)
        self.stop_point_pick()

    def set_point_pick_callbacks(self, on_pick=None, on_confirm=None, on_cancel=None, on_exit=None):
        """设置点拾取回调"""
        self._on_point_pick = on_pick
        self._on_point_pick_confirm = on_confirm
        self._on_point_pick_cancel = on_cancel
        self._on_point_pick_exit = on_exit

    def start_point_pick(self, on_pick=None, on_confirm=None, on_cancel=None, on_exit=None):
        """启动点拾取"""
        if self._point_pick_enabled:
            return
        
        if on_pick is not None:
            self._on_point_pick = on_pick
        if on_confirm is not None:
            self._on_point_pick_confirm = on_confirm
        if on_cancel is not None:
            self._on_point_pick_cancel = on_cancel
        if on_exit is not None:
            self._on_point_pick_exit = on_exit
        
        interactor = self._get_interactor()
        if interactor is None:
            return
        
        self._point_pick_observer_id = interactor.AddObserver("LeftButtonPressEvent", self._on_point_pick_press)
        self._point_pick_right_id = interactor.AddObserver("RightButtonPressEvent", self._on_point_pick_right_press)
        self._point_pick_key_id = interactor.AddObserver("KeyPressEvent", self._on_point_pick_key_press)
        self._point_pick_move_id = interactor.AddObserver("MouseMoveEvent", self._on_point_pick_move)
        self._point_pick_enabled = True

    def stop_point_pick(self):
        """停止点拾取"""
        if not self._point_pick_enabled:
            return
        
        interactor = self._get_interactor()
        if interactor is not None:
            if self._point_pick_observer_id is not None:
                interactor.RemoveObserver(self._point_pick_observer_id)
            if self._point_pick_right_id is not None:
                interactor.RemoveObserver(self._point_pick_right_id)
            if self._point_pick_key_id is not None:
                interactor.RemoveObserver(self._point_pick_key_id)
            if self._point_pick_move_id is not None:
                interactor.RemoveObserver(self._point_pick_move_id)
        
        self._point_pick_observer_id = None
        self._point_pick_right_id = None
        self._point_pick_key_id = None
        self._point_pick_move_id = None
        self._point_pick_enabled = False
        self._remove_point_highlight()
        self._remove_snap_visualization()

    def is_point_pick_active(self):
        """检查点拾取是否激活"""
        return self._point_pick_enabled

    def set_snap_enabled(self, enabled):
        """设置是否启用磁吸"""
        self._snap_enabled = enabled
        if enabled:
            self._update_geometry_points_cache()

    def set_snap_tolerance(self, tolerance):
        """设置磁吸容差"""
        self._snap_tolerance = tolerance

    def _update_geometry_points_cache(self):
        """更新几何点缓存"""
        self._geometry_points_cache = []
        if not self.gui:
            return
        
        elements = self._get_all_geometry_elements()
        vertices = elements.get("vertices", [])
        
        from OCC.Core.BRep import BRep_Tool
        
        for vertex_obj, vertex_index in vertices:
            try:
                pnt = BRep_Tool.Pnt(vertex_obj)
                self._geometry_points_cache.append((pnt.X(), pnt.Y(), pnt.Z()))
            except Exception:
                continue

    def _find_nearest_point(self, point):
        """找到最近的几何点"""
        if not self._geometry_points_cache:
            return None
        
        min_dist = float('inf')
        nearest_point = None
        
        for cached_point in self._geometry_points_cache:
            dist = ((point[0] - cached_point[0]) ** 2 + 
                   (point[1] - cached_point[1]) ** 2 + 
                   (point[2] - cached_point[2]) ** 2) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                nearest_point = cached_point
        
        if min_dist <= self._snap_tolerance:
            return nearest_point
        return None

    def _on_point_pick_press(self, obj, event):
        """点拾取左键按下事件"""
        if not self._point_pick_enabled:
            return
        if not self._on_point_pick:
            return
        
        interactor = self._get_interactor()
        if interactor is None:
            return
        
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return
        
        click_pos = interactor.GetEventPosition()
        picked = self._point_picker.Pick(click_pos[0], click_pos[1], 0, renderer)
        
        if picked:
            pos = self._point_picker.GetPickPosition()
            
            if self._snap_enabled:
                nearest_point = self._find_nearest_point(pos)
                if nearest_point:
                    pos = nearest_point
            
            self._remove_snap_visualization()
            self._show_point_highlight(pos[0], pos[1], pos[2])
            self._on_point_pick((pos[0], pos[1], pos[2]))
        
        style = interactor.GetInteractorStyle()
        if style:
            style.OnLeftButtonDown()

    def _on_point_pick_right_press(self, obj, event):
        """点拾取右键按下事件"""
        if not self._point_pick_enabled:
            return
        
        if self._on_point_pick_cancel:
            self._on_point_pick_cancel()
        
        interactor = self._get_interactor()
        if interactor is None:
            return
        
        style = interactor.GetInteractorStyle()
        if style:
            style.OnRightButtonDown()

    def _on_point_pick_key_press(self, obj, event):
        """点拾取键盘事件"""
        if not self._point_pick_enabled:
            return
        
        interactor = self._get_interactor()
        if interactor is None:
            return
        
        key = interactor.GetKeySym()
        
        if key in ("Escape", "Esc"):
            if self._on_point_pick_exit:
                self._on_point_pick_exit()
            return
        
        if key in ("Return", "Enter", "KP_Enter"):
            if self._on_point_pick_confirm:
                self._on_point_pick_confirm()
            return
        
        style = interactor.GetInteractorStyle()
        if style:
            style.OnKeyPress()

    def _on_point_pick_move(self, obj, event):
        """点拾取鼠标移动事件"""
        if not self._point_pick_enabled:
            return
        if not self._snap_enabled:
            return
        
        interactor = self._get_interactor()
        if interactor is None:
            return
        
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return
        
        click_pos = interactor.GetEventPosition()
        picked = self._point_picker.Pick(click_pos[0], click_pos[1], 0, renderer)
        
        if picked:
            pos = self._point_picker.GetPickPosition()
            self._last_pick_pos = pos
            
            nearest_point = self._find_nearest_point(pos)
            if nearest_point:
                self._last_snap_pos = nearest_point
                self._show_snap_visualization(pos, nearest_point)
            else:
                self._last_snap_pos = None
                self._remove_snap_visualization()
        else:
            self._last_pick_pos = None
            self._last_snap_pos = None
            self._remove_snap_visualization()

    def _show_snap_visualization(self, pick_pos, snap_pos):
        """显示磁吸可视化效果"""
        if not hasattr(self.mesh_display, "renderer"):
            return
        
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return
        
        self._remove_snap_visualization()
        
        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(pick_pos[0], pick_pos[1], pick_pos[2])
        line_source.SetPoint2(snap_pos[0], snap_pos[1], snap_pos[2])
        line_source.Update()
        
        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputConnection(line_source.GetOutputPort())
        
        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
        line_actor.GetProperty().SetLineWidth(2)
        line_actor.GetProperty().SetLineStipplePattern(0xAAAA)
        line_actor.GetProperty().SetLineStippleRepeatFactor(1)
        
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(snap_pos[0], snap_pos[1], snap_pos[2])
        sphere.SetRadius(0.03)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        sphere.Update()
        
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())
        
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
        sphere_actor.GetProperty().SetOpacity(0.8)
        
        renderer.AddActor(line_actor)
        renderer.AddActor(sphere_actor)
        
        self._snap_line_actor = line_actor
        self._snap_point_actor = sphere_actor
        
        if hasattr(self.mesh_display, "render_window"):
            self.mesh_display.render_window.Render()

    def _remove_snap_visualization(self):
        """移除磁吸可视化效果"""
        if self._snap_line_actor is not None:
            renderer = getattr(self.mesh_display, "renderer", None)
            if renderer is not None:
                renderer.RemoveActor(self._snap_line_actor)
            self._snap_line_actor = None
        
        if self._snap_point_actor is not None:
            renderer = getattr(self.mesh_display, "renderer", None)
            if renderer is not None:
                renderer.RemoveActor(self._snap_point_actor)
            self._snap_point_actor = None
        
        if hasattr(self.mesh_display, "render_window"):
            self.mesh_display.render_window.Render()

    def _show_point_highlight(self, x, y, z):
        """显示点高亮"""
        if not hasattr(self.mesh_display, "renderer"):
            return
        
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return
        
        self._remove_point_highlight()
        
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(x, y, z)
        sphere.SetRadius(0.05)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        sphere.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        actor.GetProperty().SetOpacity(0.8)
        
        renderer.AddActor(actor)
        if hasattr(self.mesh_display, "render_window"):
            self.mesh_display.render_window.Render()
        
        self._point_highlight_actor = actor

    def _remove_point_highlight(self):
        """移除点高亮"""
        if self._point_highlight_actor is None:
            return
        
        if not hasattr(self.mesh_display, "renderer"):
            return
        
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return
        
        renderer.RemoveActor(self._point_highlight_actor)
        if hasattr(self.mesh_display, "render_window"):
            self.mesh_display.render_window.Render()
        
        self._point_highlight_actor = None

    def _get_interactor(self):
        if getattr(self.mesh_display, "frame", None) is None:
            return None
        return self.mesh_display.frame.GetRenderWindow().GetInteractor()

    def _ensure_geometry_display_mode(self):
        if not self.gui:
            return
        if not hasattr(self.gui, "display_mode"):
            return
        if self.gui.display_mode != "elements":
            self._saved_display_mode = self.gui.display_mode
        self.gui.display_mode = "elements"
        handler = None
        if hasattr(self.gui, "part_manager") and hasattr(self.gui.part_manager, "on_display_mode_changed"):
            handler = self.gui.part_manager.on_display_mode_changed
        elif hasattr(self.gui, "on_display_mode_changed"):
            handler = self.gui.on_display_mode_changed
        if handler:
            handler("elements")

    def _restore_display_mode(self):
        if self._saved_display_mode is None or not self.gui:
            return
        self.gui.display_mode = "elements"
        self._saved_display_mode = None

    def _on_left_button_press(self, obj, event):
        if not self._enabled:
            return
        if self._area_selecting:
            return
        interactor = self._get_interactor()
        if interactor is None:
            return
        style = interactor.GetInteractorStyle()

        click_pos = interactor.GetEventPosition()
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return

        actor_map = self._build_actor_map()
        if not actor_map:
            if style:
                style.OnLeftButtonDown()
            return

        self._picker.Pick(click_pos[0], click_pos[1], 0, renderer)
        actor = self._picker.GetActor()

        if actor is not None and actor in actor_map:
            element_info = actor_map[actor]
            if actor not in self._highlighted_actors:
                self._highlight_actor(actor, element_info["element_type"])
                if self._on_pick:
                    self._on_pick(
                        element_info["element_type"],
                        element_info["element_obj"],
                        element_info["element_index"],
                    )
                if self.gui and hasattr(self.gui, "part_manager"):
                    self.gui.part_manager.on_geometry_element_selected(
                        element_info["element_type"],
                        element_info["element_obj"],
                        element_info["element_index"],
                    )
            elif self.gui and hasattr(self.gui, "part_manager"):
                self.gui.part_manager.on_geometry_element_selected(
                    element_info["element_type"],
                    element_info["element_obj"],
                    element_info["element_index"],
                )
            if hasattr(self.mesh_display, "render_window"):
                self.mesh_display.render_window.Render()
        else:
            if style:
                style.OnLeftButtonDown()

    def _on_right_button_press(self, obj, event):
        if not self._enabled:
            return
        if self._area_selecting:
            return
        interactor = self._get_interactor()
        if interactor is None:
            return
        style = interactor.GetInteractorStyle()

        click_pos = interactor.GetEventPosition()
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return

        actor_map = self._build_actor_map()
        if not actor_map:
            if style:
                style.OnRightButtonDown()
            return

        self._picker.Pick(click_pos[0], click_pos[1], 0, renderer)
        actor = self._picker.GetActor()

        if actor is not None and actor in actor_map and actor in self._highlighted_actors:
            element_info = actor_map[actor]
            self._unhighlight_actor(actor)
            if self._on_unpick:
                self._on_unpick(
                    element_info["element_type"],
                    element_info["element_obj"],
                    element_info["element_index"],
                )
            if hasattr(self.mesh_display, "render_window"):
                self.mesh_display.render_window.Render()
        elif style:
            style.OnRightButtonDown()

    def _on_key_press(self, obj, event):
        interactor = self._get_interactor()
        if interactor is None:
            return
        key = interactor.GetKeySym()
        if key in ("Escape", "Esc"):
            if self._on_cancel:
                self._on_cancel()
            return
        if key in ("Delete", "BackSpace"):
            if self._on_delete:
                self._on_delete()
            return
        if key in ("Return", "Enter", "KP_Enter"):
            if self._on_confirm:
                self._on_confirm()
            return
        style = interactor.GetInteractorStyle()
        if style:
            style.OnKeyPress()

    def _build_actor_map(self) -> Dict[vtk.vtkActor, Dict[str, object]]:
        if not self.gui or not hasattr(self.gui, "geometry_actors_cache"):
            return {}

        actor_map: Dict[vtk.vtkActor, Dict[str, object]] = {}
        elements = self._get_all_geometry_elements()
        type_map = {
            "vertices": "vertex",
            "edges": "edge",
            "faces": "face",
            "bodies": "body",
        }

        for elem_type, elements_list in elements.items():
            actors_dict = self.gui.geometry_actors_cache.get(elem_type, {})
            for elem_index, elem_obj in elements_list:
                actor = actors_dict.get(elem_index)
                if actor is None:
                    continue
                actor_map[actor] = {
                    "element_type": type_map.get(elem_type, elem_type),
                    "element_index": elem_index,
                    "element_obj": elem_obj,
                }

        return actor_map

    def _select_by_rect(self, rect: QRect, mode: str):
        renderer = getattr(self.mesh_display, "renderer", None)
        interactor = self._get_interactor()
        if renderer is None or interactor is None:
            return
        size = interactor.GetSize()
        if not size or size[1] <= 0:
            return
        x0 = rect.left()
        x1 = rect.right()
        y0 = size[1] - rect.top()
        y1 = size[1] - rect.bottom()
        if self._area_picker.AreaPick(x0, y0, x1, y1, renderer):
            frustum = self._area_picker.GetFrustum()
            if frustum is not None:
                self._select_by_frustum(frustum, mode=mode)

    def _select_by_frustum(self, frustum, mode="intersect"):
        actor_map = self._build_actor_map()
        if not actor_map:
            return
        for actor, element_info in actor_map.items():
            mapper = actor.GetMapper()
            if mapper is None:
                continue
            polydata = mapper.GetInput()
            if polydata is None:
                continue
            if mode == "contain":
                selected = self._polydata_fully_inside(polydata, frustum)
            else:
                selected = self._polydata_intersects(polydata, frustum)
            if not selected:
                continue
            if actor in self._highlighted_actors:
                self._unhighlight_actor(actor)
                if self._on_unpick:
                    self._on_unpick(
                        element_info["element_type"],
                        element_info["element_obj"],
                        element_info["element_index"],
                    )
            else:
                self._highlight_actor(actor, element_info["element_type"])
                if self._on_pick:
                    self._on_pick(
                        element_info["element_type"],
                        element_info["element_obj"],
                        element_info["element_index"],
                    )
                if self.gui and hasattr(self.gui, "part_manager"):
                    self.gui.part_manager.on_geometry_element_selected(
                        element_info["element_type"],
                        element_info["element_obj"],
                        element_info["element_index"],
                    )

    def _polydata_intersects(self, polydata, frustum):
        extractor = vtk.vtkExtractSelectedFrustum()
        extractor.SetFrustum(frustum)
        extractor.SetInputData(polydata)
        extractor.Update()
        output = extractor.GetOutput()
        return output is not None and output.GetNumberOfCells() > 0

    def _polydata_fully_inside(self, polydata, frustum):
        points = polydata.GetPoints()
        if points is None or points.GetNumberOfPoints() == 0:
            return False
        for i in range(points.GetNumberOfPoints()):
            p = points.GetPoint(i)
            if frustum.EvaluateFunction(p) > 1e-6:
                return False
        return True

    def _highlight_actor(self, actor, element_type):
        if actor in self._highlighted_actors:
            return
        prop = actor.GetProperty()
        self._highlighted_actors[actor] = {
            "color": prop.GetColor(),
            "opacity": prop.GetOpacity(),
            "line_width": prop.GetLineWidth(),
            "point_size": prop.GetPointSize(),
            "edge_visibility": prop.GetEdgeVisibility(),
            "edge_color": prop.GetEdgeColor(),
        }
        prop.SetColor(1.0, 1.0, 0.0)
        if element_type in ("edge", "edges"):
            prop.SetLineWidth(max(prop.GetLineWidth(), 3.0))
        elif element_type in ("vertex", "vertices"):
            prop.SetPointSize(max(prop.GetPointSize(), 10.0))
        else:
            prop.SetOpacity(max(prop.GetOpacity(), 0.9))
            prop.EdgeVisibilityOn()
            prop.SetEdgeColor(0.0, 0.0, 0.0)

    def _unhighlight_actor(self, actor):
        if actor not in self._highlighted_actors:
            return
        props = self._highlighted_actors.pop(actor)
        prop = actor.GetProperty()
        prop.SetColor(*props["color"])
        prop.SetOpacity(props["opacity"])
        prop.SetLineWidth(props["line_width"])
        prop.SetPointSize(props["point_size"])
        if props["edge_visibility"]:
            prop.EdgeVisibilityOn()
        else:
            prop.EdgeVisibilityOff()
        prop.SetEdgeColor(*props["edge_color"])

    def _clear_highlights(self):
        for actor in list(self._highlighted_actors.keys()):
            self._unhighlight_actor(actor)

    def _get_all_geometry_elements(self) -> Dict[str, list]:
        if self.gui and hasattr(self.gui, "part_manager") and hasattr(self.gui.part_manager, "_get_all_geometry_elements_from_tree"):
            return self.gui.part_manager._get_all_geometry_elements_from_tree()
        return {"vertices": [], "edges": [], "faces": [], "bodies": []}

    def get_selected_elements(self) -> Dict[str, list]:
        selected = {"vertices": set(), "edges": set(), "faces": set(), "bodies": set()}
        actor_map = self._build_actor_map()
        type_map = {
            "vertex": "vertices",
            "edge": "edges",
            "face": "faces",
            "body": "bodies",
            "vertices": "vertices",
            "edges": "edges",
            "faces": "faces",
            "bodies": "bodies",
        }
        for actor in self._highlighted_actors.keys():
            element_info = actor_map.get(actor)
            if not element_info:
                continue
            key = type_map.get(element_info.get("element_type"))
            if key:
                selected[key].add(element_info.get("element_obj"))
        return {key: list(values) for key, values in selected.items()}

    def clear_selection(self):
        self._clear_highlights()
        if hasattr(self.mesh_display, "render_window"):
            self.mesh_display.render_window.Render()


class _AreaSelectionFilter(QObject):
    def __init__(self, helper: GeometryPickingHelper):
        super().__init__()
        self.helper = helper
        self._origin = None
        self._mode = None
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, helper.mesh_display.frame)
        self._rubber_band.setAttribute(Qt.WA_TranslucentBackground, True)
        self._rubber_band.setStyleSheet(
            "QRubberBand { background-color: transparent; border: 2px dashed rgba(255, 210, 0, 255); }"
        )

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress and event.modifiers() & Qt.AltModifier:
            if event.button() == Qt.LeftButton:
                self._mode = "intersect"
            elif event.button() == Qt.RightButton:
                self._mode = "contain"
            else:
                return False
            self._origin = event.pos()
            self.helper._area_selecting = True
            self._rubber_band.setGeometry(QRect(self._origin, QSize()))
            self._rubber_band.show()
            return True

        if event.type() == QEvent.MouseMove and self._origin is not None:
            rect = QRect(self._origin, event.pos()).normalized()
            self._rubber_band.setGeometry(rect)
            return True

        if event.type() == QEvent.MouseButtonRelease and self._origin is not None:
            rect = QRect(self._origin, event.pos()).normalized()
            self._rubber_band.hide()
            self._origin = None
            self.helper._area_selecting = False
            if rect.width() > 2 and rect.height() > 2 and self._mode:
                self.helper._select_by_rect(rect, self._mode)
                if hasattr(self.helper.mesh_display, "render_window"):
                    self.helper.mesh_display.render_window.Render()
            self._mode = None
            return True

        if event.type() == QEvent.FocusOut:
            self._origin = None
            self._mode = None
            self._rubber_band.hide()
            self.helper._area_selecting = False

        return False


