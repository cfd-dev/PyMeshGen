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
    ):
        self.mesh_display = mesh_display
        self.gui = gui
        self._on_pick = on_pick
        self._on_unpick = on_unpick
        self._enabled = False
        self._observer_id = None
        self._observer_right_id = None
        self._saved_display_mode = None
        self._highlighted_actors: Dict[vtk.vtkActor, Dict[str, object]] = {}
        self._area_selecting = False
        self._area_picker = vtk.vtkRenderedAreaPicker()
        self._area_filter = _AreaSelectionFilter(self)

        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.0005)

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
        self._enabled = True

    def disable(self):
        """禁用拾取"""
        if not self._enabled:
            return

        interactor = self._get_interactor()
        if interactor is not None:
            if self._observer_id is not None:
                interactor.RemoveObserver(self._observer_id)
            if self._observer_right_id is not None:
                interactor.RemoveObserver(self._observer_right_id)
        self._observer_id = None
        self._observer_right_id = None
        self._clear_highlights()
        self._enabled = False
        self._restore_display_mode()
        if getattr(self.mesh_display, "frame", None) is not None:
            self.mesh_display.frame.removeEventFilter(self._area_filter)

    def cleanup(self):
        """清理资源"""
        self.disable()

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
        self.gui.display_mode = self._saved_display_mode
        handler = None
        if hasattr(self.gui, "part_manager") and hasattr(self.gui.part_manager, "on_display_mode_changed"):
            handler = self.gui.part_manager.on_display_mode_changed
        elif hasattr(self.gui, "on_display_mode_changed"):
            handler = self.gui.on_display_mode_changed
        if handler:
            handler(self._saved_display_mode)
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

        if actor is not None and actor in actor_map:
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
        else:
            if style:
                style.OnRightButtonDown()

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


class _AreaSelectionFilter(QObject):
    def __init__(self, helper: GeometryPickingHelper):
        super().__init__()
        self.helper = helper
        self._origin = None
        self._mode = None
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, helper.mesh_display.frame)

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
