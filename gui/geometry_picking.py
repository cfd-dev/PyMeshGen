# -*- coding: utf-8 -*-
"""
几何拾取助手
基于VTK拾取器实现点/线/面/体的鼠标拾取
"""

from typing import Callable, Dict, Optional

import vtk


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

        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.0005)

    def enable(self):
        """启用拾取"""
        if self._enabled:
            return
        if not self.mesh_display or not getattr(self.mesh_display, "frame", None):
            return

        self._ensure_geometry_display_mode()
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
        self._enabled = False
        self._restore_display_mode()

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

    def _get_all_geometry_elements(self) -> Dict[str, list]:
        if self.gui and hasattr(self.gui, "part_manager") and hasattr(self.gui.part_manager, "_get_all_geometry_elements_from_tree"):
            return self.gui.part_manager._get_all_geometry_elements_from_tree()
        return {"vertices": [], "edges": [], "faces": [], "bodies": []}
