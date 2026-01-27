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
        
        # FIXME 磁吸现有点的功能未能正确运行，待检查修复
        self._snap_enabled = False
        self._snap_pixel_tolerance = 12.0  # 默认像素容差（12像素）
        self._geometry_points_cache = []
        self._snap_line_actor = None
        self._snap_point_actor = None
        self._last_pick_pos = None
        self._last_snap_pos = None
        self._is_snapped = False  # 标记当前是否磁吸到现有点
        self._geometry_vertex_highlight_actors = {}  # 存储高亮的几何点actor {vertex_obj: actor}
        self._picked_points = []  # 存储已拾取的点列表 [(x, y, z, vertex_obj), ...]
        self._picked_point_actors = []  # 存储已拾取点的actor列表
        self._last_picked_vertex = None  # 最近一次拾取到的几何点对象

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

        # 启动点拾取时更新几何点缓存
        if self._snap_enabled:
            self._update_geometry_points_cache()

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
        self._clear_picked_points()
        self._geometry_points_cache = []

    def is_point_pick_active(self):
        """检查点拾取是否激活"""
        return self._point_pick_enabled

    def clear_pick_highlights(self):
        """清除所有拾取高亮点（供创建几何时调用）"""
        self._clear_picked_points()
        self._remove_point_highlight()
        self._remove_snap_visualization()
        self._clear_geometry_vertex_highlights()

    def get_last_picked_vertex(self):
        """获取最近一次拾取到的几何点对象（如果有的话）"""
        return self._last_picked_vertex

    def set_snap_enabled(self, enabled):
        """设置是否启用磁吸"""
        self._snap_enabled = enabled
        if enabled:
            # 启用磁吸时强制更新缓存
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
            for vertex_index, vertex_obj in vertices:
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

    def _get_pixel_to_world_scale(self, world_point, renderer):
        """计算世界坐标中1像素对应的世界距离"""
        if renderer is None or world_point is None:
            return 0.01  # 默认值

        camera = renderer.GetActiveCamera()
        if camera is None:
            return 0.01

        try:
            focal_point = camera.GetFocalPoint()
            camera_pos = camera.GetPosition()
            distance = math.sqrt(
                (focal_point[0] - camera_pos[0]) ** 2 +
                (focal_point[1] - camera_pos[1]) ** 2 +
                (focal_point[2] - camera_pos[2]) ** 2
            )

            view_angle = camera.GetViewAngle()
            viewport_height = renderer.GetSize()[1]
            if viewport_height > 0 and view_angle > 0:
                world_height = 2 * distance * math.tan(math.radians(view_angle / 2))
                pixels_per_unit = viewport_height / world_height if world_height > 0 else 100
                return 1.0 / pixels_per_unit if pixels_per_unit > 0 else 0.01

            return 0.01
        except Exception:
            return 0.01

    def _find_nearest_point_from_screen_pos(self, screen_pos, renderer, pixel_tolerance=None):
        """基于屏幕位置找到最近的几何点"""
        if not self._geometry_points_cache or renderer is None:
            return None

        # 使用实例变量的容差值，如果未提供参数
        if pixel_tolerance is None:
            pixel_tolerance = getattr(self, '_snap_pixel_tolerance', 12.0)

        min_pixel_dist_sq = float('inf')
        nearest_point = None
        tolerance_sq = pixel_tolerance ** 2

        for cached_point in self._geometry_points_cache:
            # 将缓存的几何点世界坐标转换为屏幕坐标
            cached_display = self._world_to_display_coords(cached_point, renderer)
            if cached_display is None:
                continue

            # VTK的显示坐标原点在左下角，鼠标事件位置也是左下角原点
            # 直接比较屏幕坐标即可
            pixel_dist_sq = (screen_pos[0] - cached_display[0]) ** 2 + \
                           (screen_pos[1] - cached_display[1]) ** 2

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

        world_pos = self._pick_on_plane(click_pos, renderer)
        if world_pos is None:
            picked = self._point_picker.Pick(click_pos[0], click_pos[1], 0, renderer)
            if picked:
                world_pos = self._point_picker.GetPickPosition()
            else:
                world_pos = [0.0, 0.0, 0.0]

        self._is_snapped = False
        final_pos = world_pos
        vertex_obj = None
        vertex_index = None

        if self._snap_enabled:
            nearest_point = self._find_nearest_point_from_screen_pos(click_pos, renderer)
            if nearest_point:
                final_pos = nearest_point
                self._is_snapped = True
                result = self._find_vertex_by_point(final_pos)
                if result:
                    vertex_obj, vertex_index = result

        if final_pos is not None:
            if self._is_snapped and vertex_obj:
                self._highlight_geometry_vertex(vertex_obj)
                self._remove_snap_visualization()
            else:
                self._remove_snap_visualization()
                self._clear_geometry_vertex_highlights()

            actor = self._show_picked_point_highlight(final_pos[0], final_pos[1], final_pos[2])
            if actor:
                self._picked_points.append((final_pos[0], final_pos[1], final_pos[2], vertex_obj))
                self._picked_point_actors.append(actor)

            self._last_picked_vertex = vertex_obj
            self._on_point_pick((final_pos[0], final_pos[1], final_pos[2]), vertex_obj)

            if self.gui and hasattr(self.gui, 'status_bar'):
                if self._is_snapped and vertex_obj is not None:
                    vertex_name = f"点_{vertex_index}" if vertex_index is not None else "未知点"
                    self.gui.status_bar.update_status(f"磁吸拾取现有几何点: {vertex_name}({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
                else:
                    self.gui.status_bar.update_status(f"已拾取新点: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")

        style = interactor.GetInteractorStyle()
        if style:
            style.OnLeftButtonDown()

    def _on_point_pick_right_press(self, obj, event):
        """点拾取右键按下事件 - 取消上一次拾取的点"""
        if not self._point_pick_enabled:
            return
        
        if self._picked_points:
            self._remove_last_picked_point()
            
            if self.gui and hasattr(self.gui, 'status_bar'):
                self.gui.status_bar.update_status(f"已取消上一次拾取，剩余 {len(self._picked_points)} 个点")
        else:
            if self._on_point_pick_cancel:
                self._on_point_pick_cancel()
        
        interactor = self._get_interactor()
        if interactor is None:
            return
        
        style = interactor.GetInteractorStyle()
        if style:
            style.OnRightButtonDown()

    def _remove_last_picked_point(self):
        """移除最后一次拾取的点和对应的高亮actor"""
        if not self._picked_points or not self._picked_point_actors:
            return
        
        self._picked_points.pop()
        last_actor = self._picked_point_actors.pop()
        
        if not hasattr(self.mesh_display, "renderer"):
            return
        
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return
        
        renderer.RemoveActor(last_actor)
        if hasattr(self.mesh_display, "render_window"):
            self.mesh_display.render_window.Render()

    def _clear_picked_points(self):
        """清除所有已拾取的点和对应的高亮actor"""
        if not self._picked_points:
            return
        
        if not hasattr(self.mesh_display, "renderer"):
            self._picked_points = []
            self._picked_point_actors = []
            return
        
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            self._picked_points = []
            self._picked_point_actors = []
            return
        
        for actor in self._picked_point_actors:
            renderer.RemoveActor(actor)
        
        self._picked_points = []
        self._picked_point_actors = []
        
        if hasattr(self.mesh_display, "render_window"):
            self.mesh_display.render_window.Render()

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

            # 在状态栏显示磁吸提示
            if self.gui and hasattr(self.gui, 'status_bar'):
                distance = math.sqrt(
                    (pos[0] - nearest_point[0]) ** 2 +
                    (pos[1] - nearest_point[1]) ** 2 +
                    (pos[2] - nearest_point[2]) ** 2
                )
                # 计算屏幕像素距离
                nearest_display = self._world_to_display_coords(nearest_point, renderer)
                if nearest_display:
                    pixel_distance = math.sqrt(
                        (click_pos[0] - nearest_display[0]) ** 2 +
                        (click_pos[1] - nearest_display[1]) ** 2
                    )
                    # 获取 vertex 的索引和名称
                    result = self._find_vertex_by_point(nearest_point)
                    if result:
                        _, vertex_index = result
                        vertex_name = f"点_{vertex_index}" if vertex_index is not None else "未知点"
                        if distance < 1e-6:
                            self.gui.status_bar.update_status(f"磁吸现有几何点: {vertex_name}({nearest_point[0]:.3f}, {nearest_point[1]:.3f}, {nearest_point[2]:.3f}) [像素距离: {pixel_distance:.1f}]")
                        else:
                            self.gui.status_bar.update_status(f"磁吸现有几何点: {vertex_name}({nearest_point[0]:.3f}, {nearest_point[1]:.3f}, {nearest_point[2]:.3f}) 世界距离: {distance:.3f} [像素距离: {pixel_distance:.1f}]")

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

                # 在状态栏显示自由拾取提示
                if self.gui and hasattr(self.gui, 'status_bar'):
                    self.gui.status_bar.update_status(f"自由拾取: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

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
            # 创建虚线连接线
            line_source = vtk.vtkLineSource()
            line_source.SetPoint1(pick_pos[0], pick_pos[1], pick_pos[2])
            line_source.SetPoint2(snap_pos[0], snap_pos[1], snap_pos[2])
            line_source.Update()

            line_mapper = vtk.vtkPolyDataMapper()
            line_mapper.SetInputConnection(line_source.GetOutputPort())

            line_actor = vtk.vtkActor()
            line_actor.SetMapper(line_mapper)
            line_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # 绿色
            line_actor.GetProperty().SetLineWidth(3)  # 加粗
            line_actor.GetProperty().SetLineStipplePattern(0xF0F0)  # 更明显的虚线
            line_actor.GetProperty().SetLineStippleRepeatFactor(2)
            line_actor.GetProperty().SetOpacity(0.8)

            renderer.AddActor(line_actor)
            self._snap_line_actor = line_actor

        # 创建磁吸点指示器（绿色圆环）
        # 计算与像素容差一致的世界坐标半径
        pixel_tolerance = getattr(self, '_snap_pixel_tolerance', 12.0)
        pixel_to_world = self._get_pixel_to_world_scale(snap_pos, renderer)
        ring_radius = pixel_tolerance * pixel_to_world

        num_points = 64
        points_list = []
        for i in range(num_points + 1):
            angle = 2 * math.pi * i / num_points
            px = snap_pos[0] + ring_radius * math.cos(angle)
            py = snap_pos[1] + ring_radius * math.sin(angle)
            pz = snap_pos[2]
            points_list.append((px, py, pz))
        
        points = vtk.vtkPoints()
        for i, (x, y, z) in enumerate(points_list):
            points.InsertNextPoint(x, y, z)
        
        line_source = vtk.vtkLineSource()
        line_source.SetPoints(points)
        
        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputConnection(line_source.GetOutputPort())
        
        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        
        if is_aligned:
            line_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
        else:
            line_actor.GetProperty().SetColor(0.0, 0.8, 0.0)
        
        line_actor.GetProperty().SetLineWidth(2.5)
        
        renderer.AddActor(line_actor)
        self._snap_point_actor = line_actor

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
        """显示点高亮（蓝色，使用与几何点一致的显示方式）"""
        if not hasattr(self.mesh_display, "renderer"):
            return
        
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return
        
        self._remove_point_highlight()
        
        points = vtk.vtkPoints()
        points.InsertNextPoint(x, y, z)
        
        vertices = vtk.vtkCellArray()
        vertex_id = vtk.vtkVertex()
        vertex_id.GetPointIds().SetId(0, 0)
        vertices.InsertNextCell(vertex_id)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 0.0, 1.0)
        actor.GetProperty().SetPointSize(10)
        
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

    def _show_picked_point_highlight(self, x, y, z):
        """显示已拾取点的高亮（蓝色，与几何点显示方式一致），返回actor以便后续管理"""
        if not hasattr(self.mesh_display, "renderer"):
            return None
        
        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return None
        
        points = vtk.vtkPoints()
        points.InsertNextPoint(x, y, z)
        
        vertices = vtk.vtkCellArray()
        vertex_id = vtk.vtkVertex()
        vertex_id.GetPointIds().SetId(0, 0)
        vertices.InsertNextCell(vertex_id)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 0.0, 1.0)
        actor.GetProperty().SetPointSize(10)
        
        renderer.AddActor(actor)
        if hasattr(self.mesh_display, "render_window"):
            self.mesh_display.render_window.Render()
        
        return actor

    def _show_snap_point_highlight(self, x, y, z):
        """显示磁吸点高亮（蓝色，与几何点显示方式一致）"""
        if not hasattr(self.mesh_display, "renderer"):
            return

        renderer = getattr(self.mesh_display, "renderer", None)
        if renderer is None:
            return

        self._remove_point_highlight()

        points = vtk.vtkPoints()
        points.InsertNextPoint(x, y, z)
        
        vertices = vtk.vtkCellArray()
        vertex_id = vtk.vtkVertex()
        vertex_id.GetPointIds().SetId(0, 0)
        vertices.InsertNextCell(vertex_id)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 0.0, 1.0)
        actor.GetProperty().SetPointSize(10)

        renderer.AddActor(actor)
        if hasattr(self.mesh_display, "render_window"):
            self.mesh_display.render_window.Render()

        self._point_highlight_actor = actor

    def _highlight_geometry_vertex(self, vertex_obj):
        """高亮几何点（蓝色，与几何点显示方式一致）"""
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
        
        points = vtk.vtkPoints()
        points.InsertNextPoint(x, y, z)
        
        vertices = vtk.vtkCellArray()
        vertex_id = vtk.vtkVertex()
        vertex_id.GetPointIds().SetId(0, 0)
        vertices.InsertNextCell(vertex_id)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 0.0, 1.0)
        actor.GetProperty().SetPointSize(10)
        
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

            for vertex_index, vertex_obj in vertices:
                try:
                    pnt = BRep_Tool.Pnt(vertex_obj)
                    vertex_point = (pnt.X(), pnt.Y(), pnt.Z())

                    vertex_display = self._world_to_display_coords(vertex_point, renderer)
                    if vertex_display is None:
                        continue

                    pixel_dist_sq = (pick_display[0] - vertex_display[0]) ** 2 + \
                                   (pick_display[1] - vertex_display[1]) ** 2

                    if pixel_dist_sq <= tolerance_sq:
                        return vertex_obj, vertex_index
                except Exception:
                    continue

            return None, None
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


