# -*- coding: utf-8 -*-
"""
几何拾取助手
基于VTK拾取器实现点/线/面/体的鼠标拾取
"""

import math
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
        self._world_picker = vtk.vtkWorldPointPicker()
        self._point_pick_enabled = False
        self._point_pick_observer_id = None
        self._point_pick_right_id = None
        self._point_pick_key_id = None
        self._on_point_pick = None
        self._on_point_pick_confirm = None
        self._on_point_pick_cancel = None
        self._on_point_pick_exit = None
        self._point_highlight_actor = None
        
        # FIXME 磁吸功能无法正确实现，待修复
        self._snap_enabled = False
        self._snap_pixel_tolerance = 12.0  # 默认像素容差（12像素，提高磁吸灵敏度）
        self._geometry_points_cache = []
        self._snap_line_actor = None
        self._snap_point_actor = None
        self._last_pick_pos = None
        self._last_snap_pos = None
        self._is_snapped = False  # 标记当前是否磁吸到现有点
        self._geometry_vertex_highlight_actors = {}  # 存储高亮的几何点actor {vertex_obj: actor}

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
        self._is_snapped = False
        self._remove_point_highlight()
        self._remove_snap_visualization()
        self._clear_geometry_vertex_highlights()

    def is_point_pick_active(self):
        """检查点拾取是否激活"""
        return self._point_pick_enabled

    def set_snap_enabled(self, enabled):
        """设置是否启用磁吸"""
        self._snap_enabled = enabled
        if enabled:
            if not self._geometry_points_cache:
                self._update_geometry_points_cache()

    def set_snap_pixel_tolerance(self, pixel_tolerance):
        """设置磁吸像素容差"""
        self._snap_pixel_tolerance = pixel_tolerance

    def refresh_geometry_cache(self):
        """手动刷新几何点缓存"""
        self._geometry_points_cache = []
        self._update_geometry_points_cache()

    def _update_geometry_points_cache(self):
        """更新几何点缓存"""
        if not self.gui:
            return
        
        try:
            elements = self._get_all_geometry_elements()
            vertices = elements.get("vertices", [])
            
            if not vertices:
                self._geometry_points_cache = []
                return
            
            from OCC.Core.BRep import BRep_Tool
            
            new_cache = []
            for vertex_obj, vertex_index in vertices:
                try:
                    pnt = BRep_Tool.Pnt(vertex_obj)
                    new_cache.append((pnt.X(), pnt.Y(), pnt.Z()))
                except Exception:
                    continue
            
            self._geometry_points_cache = new_cache
            
        except Exception as e:
            print(f"更新几何点缓存失败: {e}")
            self._geometry_points_cache = []

    def _world_to_display_coords(self, world_point, renderer):
        """将世界坐标转换为显示坐标（屏幕像素坐标）"""
        if renderer is None:
            return None

        try:
            # 使用 vtkCoordinate 进行坐标转换（更可靠的方法）
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToWorld()
            coord.SetValue(world_point[0], world_point[1], world_point[2])
            display_point = coord.GetComputedDisplayValue(renderer)

            # 返回2D屏幕坐标 (x, y)
            return (display_point[0], display_point[1])
        except Exception:
            # 回退到渲染器方法
            try:
                renderer.SetWorldPoint(world_point[0], world_point[1], world_point[2], 1.0)
                renderer.WorldToDisplay()
                display_point = renderer.GetDisplayPoint()
                return (display_point[0], display_point[1])
            except Exception:
                return None

    def _find_nearest_point_from_screen_pos(self, screen_pos, renderer, pixel_tolerance=None):
        """基于屏幕位置找到最近的几何点"""
        if not self._geometry_points_cache or renderer is None:
            return None

        # 使用实例变量的容差值，如果未提供参数
        if pixel_tolerance is None:
            pixel_tolerance = getattr(self, '_snap_pixel_tolerance', 12.0)

        # 获取渲染器尺寸，用于Y坐标转换
        renderer_size = renderer.GetSize()
        screen_height = renderer_size[1] if renderer_size else 0

        min_pixel_dist_sq = float('inf')
        nearest_point = None
        tolerance_sq = pixel_tolerance ** 2

        for cached_point in self._geometry_points_cache:
            # 将缓存的几何点世界坐标转换为屏幕坐标
            cached_display = self._world_to_display_coords(cached_point, renderer)
            if cached_display is None:
                continue

            # VTK的显示坐标原点在左下角，需要转换为左上角坐标系
            # 鼠标事件位置使用左上角坐标系
            vtk_y = cached_display[1]
            mouse_y = screen_pos[1]

            # 转换Y坐标：VTK Y = 屏幕高度 - 鼠标 Y
            # 或者：鼠标 Y = 屏幕高度 - VTK Y
            converted_vtk_y = screen_height - vtk_y

            # 计算屏幕像素距离的平方
            pixel_dist_sq = (screen_pos[0] - cached_display[0]) ** 2 + \
                           (mouse_y - converted_vtk_y) ** 2

            if pixel_dist_sq < min_pixel_dist_sq:
                min_pixel_dist_sq = pixel_dist_sq
                nearest_point = cached_point

        if min_pixel_dist_sq <= tolerance_sq:
            return nearest_point
        return None


    def _pick_on_plane(self, click_pos, renderer):
        """在空白区域拾取坐标（使用参考平面）"""
        if renderer is None:
            return None
        
        camera = renderer.GetActiveCamera()
        if camera is None:
            return None
        
        self._world_picker.Pick(click_pos[0], click_pos[1], 0, renderer)
        pick_pos = self._world_picker.GetPickPosition()
        
        if pick_pos is not None and not (pick_pos[0] == 0 and pick_pos[1] == 0 and pick_pos[2] == 0):
            return pick_pos
        
        focal_point = camera.GetFocalPoint()
        position = camera.GetPosition()
        
        click_vec = [click_pos[0] - renderer.GetSize()[0] / 2,
                    click_pos[1] - renderer.GetSize()[1] / 2, 0]
        
        view_width = camera.GetViewAngle()
        aspect = renderer.GetAspect()
        
        if aspect > 0:
            view_height = 2 * abs(position[2] - focal_point[2]) * abs(math.tan(math.radians(view_width / 2)))
            view_width = view_height * aspect
        else:
            view_width = 2 * abs(position[2] - focal_point[2]) * abs(math.tan(math.radians(view_width / 2)))
        
        if click_pos[0] != 0 or click_pos[1] != 0:
            scale_factor = abs(position[2] - focal_point[2]) / (renderer.GetSize()[1] / 2) if renderer.GetSize()[1] > 0 else 1
            x_offset = (click_pos[0] - renderer.GetSize()[0] / 2) * scale_factor
            y_offset = (click_pos[1] - renderer.GetSize()[1] / 2) * scale_factor
            
            return [focal_point[0] + x_offset, focal_point[1] + y_offset, focal_point[2]]
        
        return focal_point

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

        # 首先获取鼠标点击位置对应的世界坐标（使用参考平面）
        world_pos = self._pick_on_plane(click_pos, renderer)
        if world_pos is None:
            # 如果无法获取世界坐标，尝试从几何体拾取
            picked = self._point_picker.Pick(click_pos[0], click_pos[1], 0, renderer)
            if picked:
                world_pos = self._point_picker.GetPickPosition()
            else:
                world_pos = [0.0, 0.0, 0.0]

        self._is_snapped = False
        final_pos = world_pos

        # 只有在启用磁吸时才进行磁吸判断
        if self._snap_enabled:
            # 基于鼠标屏幕位置进行磁吸判断（正确的方法）
            nearest_point = self._find_nearest_point_from_screen_pos(click_pos, renderer)
            if nearest_point:
                final_pos = nearest_point
                self._is_snapped = True

        if final_pos is not None:
            if self._is_snapped:
                # 拾取现有点：直接高亮现有点（黄色），不显示临时预览点
                vertex_obj = self._find_vertex_by_point(final_pos)
                if vertex_obj:
                    self._highlight_geometry_vertex(vertex_obj)
                # 不调用 _show_point_highlight 和 _show_snap_visualization
            else:
                # 拾取新点：显示临时预览点（黄色）
                self._show_point_highlight(final_pos[0], final_pos[1], final_pos[2])
                self._remove_snap_visualization()
                self._clear_geometry_vertex_highlights()

            self._on_point_pick((final_pos[0], final_pos[1], final_pos[2]))

            # 只在拾取新点时移除预览点
            if not self._is_snapped:
                self._remove_point_highlight()

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
        """点拾取鼠标移动事件 - 实时显示磁吸效果"""
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

        # 基于鼠标屏幕位置进行磁吸判断（实时预览）
        nearest_point = self._find_nearest_point_from_screen_pos(click_pos, renderer)
        if nearest_point:
            # 获取鼠标点击的世界坐标用于可视化
            pos = self._pick_on_plane(click_pos, renderer)
            if pos is None:
                pos = [0.0, 0.0, 0.0]

            self._last_pick_pos = pos
            self._last_snap_pos = nearest_point
            self._is_snapped = True
            self._show_snap_visualization(pos, nearest_point)
            # 不在这里高亮几何点，只在实际点击时高亮
            # vertex_obj = self._find_vertex_by_point(nearest_point)
            # if vertex_obj:
            #     self._highlight_geometry_vertex(vertex_obj)
        else:
            pos = self._pick_on_plane(click_pos, renderer)
            if pos is not None:
                self._last_pick_pos = pos
                self._last_snap_pos = None
                self._is_snapped = False
                self._remove_snap_visualization()
                # 不在这里清除几何点高亮，只在拾取到非几何点时清除
                # self._clear_geometry_vertex_highlights()
            else:
                self._last_pick_pos = None
                self._last_snap_pos = None
                self._is_snapped = False
                # self._clear_geometry_vertex_highlights()

    def _show_snap_visualization(self, pick_pos, snap_pos):
        """显示磁吸可视化效果"""
        if not hasattr(self.mesh_display, "renderer"):
            return
        
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return
        
        self._remove_snap_visualization()
        
        is_aligned = (abs(pick_pos[0] - snap_pos[0]) < 1e-6 and 
                     abs(pick_pos[1] - snap_pos[1]) < 1e-6 and 
                     abs(pick_pos[2] - snap_pos[2]) < 1e-6)
        
        if not is_aligned:
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
            
            renderer.AddActor(line_actor)
            self._snap_line_actor = line_actor
        
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(snap_pos[0], snap_pos[1], snap_pos[2])
        sphere.SetRadius(0.05 if is_aligned else 0.03)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        sphere.Update()
        
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())
        
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        
        if is_aligned:
            sphere_actor.GetProperty().SetColor(0.2, 0.8, 0.2)
        else:
            sphere_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
        
        sphere_actor.GetProperty().SetOpacity(0.9)
        
        renderer.AddActor(sphere_actor)
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

    def _highlight_geometry_vertex(self, vertex_obj):
        """高亮几何点（黄色）"""
        if not hasattr(self.mesh_display, "renderer"):
            return
        
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return
        
        if vertex_obj in self._geometry_vertex_highlight_actors:
            return
        
        try:
            from OCC.Core.BRep import BRep_Tool
            pnt = BRep_Tool.Pnt(vertex_obj)
            x, y, z = pnt.X(), pnt.Y(), pnt.Z()
        except Exception:
            return
        
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(x, y, z)
        sphere.SetRadius(0.06)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        sphere.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        actor.GetProperty().SetOpacity(0.9)
        
        renderer.AddActor(actor)
        if hasattr(self.mesh_display, "render_window"):
            self.mesh_display.render_window.Render()
        
        self._geometry_vertex_highlight_actors[vertex_obj] = actor

    def _unhighlight_geometry_vertex(self, vertex_obj):
        """取消高亮几何点"""
        if vertex_obj not in self._geometry_vertex_highlight_actors:
            return
        
        if not hasattr(self.mesh_display, "renderer"):
            return
        
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return
        
        actor = self._geometry_vertex_highlight_actors.pop(vertex_obj)
        renderer.RemoveActor(actor)
        if hasattr(self.mesh_display, "render_window"):
            self.mesh_display.render_window.Render()

    def _clear_geometry_vertex_highlights(self):
        """清除所有几何点高亮"""
        if not self._geometry_vertex_highlight_actors:
            return
        
        if not hasattr(self.mesh_display, "renderer"):
            return
        
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return
        
        for vertex_obj in list(self._geometry_vertex_highlight_actors.keys()):
            actor = self._geometry_vertex_highlight_actors.pop(vertex_obj)
            renderer.RemoveActor(actor)
        
        if hasattr(self.mesh_display, "render_window"):
            self.mesh_display.render_window.Render()

    def _find_vertex_by_point(self, point):
        """根据坐标找到对应的几何点对象（使用像素距离转换）"""
        if not self.gui:
            return None

        try:
            elements = self._get_all_geometry_elements()
            vertices = elements.get("vertices", [])

            if not vertices:
                return None

            from OCC.Core.BRep import BRep_Tool

            renderer = getattr(self.mesh_display, "renderer", None)
            if renderer is None:
                # 回退到世界坐标距离比较
                snap_tolerance = getattr(self, '_snap_pixel_tolerance', 12.0) * 0.01  # 粗略转换
                for vertex_obj, vertex_index in vertices:
                    try:
                        pnt = BRep_Tool.Pnt(vertex_obj)
                        vertex_point = (pnt.X(), pnt.Y(), pnt.Z())

                        dist_sq = (vertex_point[0] - point[0]) ** 2 + \
                                 (vertex_point[1] - point[1]) ** 2 + \
                                 (vertex_point[2] - point[2]) ** 2

                        if dist_sq <= snap_tolerance ** 2:
                            return vertex_obj
                    except Exception:
                        continue
                return None

            # 使用像素距离进行比较
            pick_display = self._world_to_display_coords(point, renderer)
            if pick_display is None:
                return None

            pixel_tolerance = getattr(self, '_snap_pixel_tolerance', 12.0)
            tolerance_sq = pixel_tolerance ** 2

            for vertex_obj, vertex_index in vertices:
                try:
                    pnt = BRep_Tool.Pnt(vertex_obj)
                    vertex_point = (pnt.X(), pnt.Y(), pnt.Z())

                    vertex_display = self._world_to_display_coords(vertex_point, renderer)
                    if vertex_display is None:
                        continue

                    pixel_dist_sq = (pick_display[0] - vertex_display[0]) ** 2 + \
                                   (pick_display[1] - vertex_display[1]) ** 2

                    if pixel_dist_sq <= tolerance_sq:
                        return vertex_obj
                except Exception:
                    continue

            return None
        except Exception:
            return None

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


