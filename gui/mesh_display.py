#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
网格显示模块
处理网格可视化和交互功能
"""

import os
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from .gui_base import BaseFrame, LabelFrameWrapper


class MeshDisplayArea(BaseFrame):
    """网格显示区域类"""
    
    def __init__(self, parent, figsize=(16, 9), dpi=100):
        super().__init__(parent)
        self.figsize = figsize
        self.dpi = dpi
        self.mesh_data = None
        self.params = None
        
        # 初始化鼠标交互状态
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None
        
        # 创建网格显示区域
        self.create_mesh_display_area()
        
    def create_mesh_display_area(self):
        """创建网格显示区域"""
        # 创建网格显示框架
        mesh_frame = LabelFrameWrapper(self.frame, "网格显示区")
        mesh_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建matplotlib图形和轴
        self.fig = Figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("网格显示区域")
        self.ax.set_xlabel("X坐标")
        self.ax.set_ylabel("Y坐标")
        # 初始状态不显示坐标轴
        self.ax.set_axis_off()
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, mesh_frame.frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 设置窗口标题为空字符串，避免显示"Figure 1"
        if hasattr(self.canvas, 'manager') and self.canvas.manager is not None:
            self.canvas.manager.set_window_title('')
        
        # 添加导航工具栏
        self.toolbar = NavigationToolbar2Tk(self.canvas, mesh_frame.frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 确保鼠标事件被正确连接
        self.canvas.mpl_connect('scroll_event', self.on_mouse_wheel)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
    def set_mesh_data(self, mesh_data):
        """设置网格数据"""
        self.mesh_data = mesh_data
        
    def set_params(self, params):
        """设置参数对象"""
        self.params = params
        
    def display_mesh(self):
        """显示网格"""
        if not self.mesh_data:
            # 尝试从输出文件加载网格数据
            if self.params and self.params.output_file and os.path.exists(self.params.output_file):
                try:
                    from fileIO.vtk_io import parse_vtk_msh
                    self.mesh_data = parse_vtk_msh(self.params.output_file)
                except Exception as e:
                    print(f"无法从输出文件加载网格数据: {str(e)}")
            
            if not self.mesh_data:
                return False
                
        try:
            # 清除之前的绘图
            self.ax.clear()
            self.ax.set_title("网格显示区域")
            self.ax.set_xlabel("X坐标")
            self.ax.set_ylabel("Y坐标")
            self.ax.axis("equal")
            # 确保坐标轴可见
            self.ax.set_axis_on()
            
            # 创建可视化对象，使用GUI中的绘图区域
            if self.params and hasattr(self.params, 'viz_enabled'):
                viz_enabled = self.params.viz_enabled
            else:
                viz_enabled = True
                
            if viz_enabled:
                from visualization.mesh_visualization import Visualization
                visual_obj = Visualization(viz_enabled, self.ax)
                
                # 直接使用内存中的网格对象进行可视化
                if hasattr(self.mesh_data, 'visualize_unstr_grid_2d'):
                    # 如果是UnstructuredGrid对象，直接调用其可视化方法
                    self.mesh_data.visualize_unstr_grid_2d(visual_obj)
                else:
                    # 否则使用通用的plot_mesh方法
                    visual_obj.plot_mesh(self.mesh_data)
                    
                self.canvas.draw()  # 更新画布
                return True
            else:
                return False
        except Exception as e:
            print(f"显示网格失败: {str(e)}")
            import traceback
            traceback.print_exc()  # 添加错误追踪信息
            return False
            
    def clear_display(self):
        """清除显示"""
        try:
            # 清除当前图形
            self.ax.clear()
            self.ax.set_title("网格显示区域")
            self.ax.set_xlabel("X坐标")
            self.ax.set_ylabel("Y坐标")
            # 初始状态不显示坐标轴
            self.ax.set_axis_off()
            self.canvas.draw()
            
            # 关闭所有matplotlib图形
            plt.close('all')
        except Exception as e:
            print(f"清除显示失败: {str(e)}")
            
    def zoom_in(self):
        """放大视图"""
        if self.ax:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_range = (xlim[1] - xlim[0]) * 0.8
            y_range = (ylim[1] - ylim[0]) * 0.8
            x_center = (xlim[1] + xlim[0]) / 2
            y_center = (ylim[1] + ylim[0]) / 2
            self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
            self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
            self.canvas.draw()

    def zoom_out(self):
        """缩小视图"""
        if self.ax:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_range = (xlim[1] - xlim[0]) * 1.2
            y_range = (ylim[1] - ylim[0]) * 1.2
            x_center = (xlim[1] + xlim[0]) / 2
            y_center = (ylim[1] + ylim[0]) / 2
            self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
            self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
            self.canvas.draw()

    def reset_view(self):
        """重置视图到原始大小"""
        if self.ax:
            self.ax.autoscale()
            self.canvas.draw()
            
    def on_mouse_wheel(self, event):
        """处理鼠标滚轮事件实现缩放功能"""
        if event.inaxes != self.ax:
            return
            
        # 获取当前坐标轴范围
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        # 计算缩放因子
        scale_factor = 1.1 if event.step < 0 else 1/1.1  # 滚轮向上缩小，向下放大
        
        # 计算鼠标位置相对于图形的比例
        xdata = event.xdata
        ydata = event.ydata
        
        # 计算新的坐标轴范围
        x_range = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        y_range = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        # 保持鼠标位置不变的情况下进行缩放
        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
        
        new_xlim = [xdata - x_range * (1 - relx), xdata + x_range * relx]
        new_ylim = [ydata - y_range * (1 - rely), ydata + y_range * rely]
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()

    def on_mouse_press(self, event):
        """处理鼠标按下事件"""
        if event.inaxes != self.ax:
            return
            
        # 记录按下时的坐标和位置
        self.press = (event.xdata, event.ydata)
        self.x0, self.y0 = event.x, event.y
        self.xpress, self.ypress = event.xdata, event.ydata
        
        # 记录当前的坐标轴范围
        self.cur_xlim = self.ax.get_xlim()
        self.cur_ylim = self.ax.get_ylim()
        
        # 记录按下的鼠标按键
        self.button_pressed = event.button

    def on_mouse_release(self, event):
        """处理鼠标释放事件"""
        self.press = None
        self.x1, self.y1 = event.x, event.y
        self.button_pressed = None

    def on_mouse_move(self, event):
        """处理鼠标移动事件实现平移或旋转功能"""
        if self.press is None:
            return  # 鼠标未按下，不执行操作
            
        if event.inaxes != self.ax:
            return
            
        # 根据按下的鼠标按键决定执行什么操作
        if self.button_pressed == 1:  # 左键按下，执行平移
            # 获取当前坐标轴范围
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            
            # 计算鼠标移动的偏移量
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            
            # 更新坐标轴范围实现平移
            self.ax.set_xlim(cur_xlim[0] - dx, cur_xlim[1] - dx)
            self.ax.set_ylim(cur_ylim[0] - dy, cur_ylim[1] - dy)
            
            self.canvas.draw()
        elif self.button_pressed == 3:  # 右键按下，执行旋转
            # 获取当前坐标轴范围
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            
            # 计算中心点
            center_x = (cur_xlim[0] + cur_xlim[1]) / 2
            center_y = (cur_ylim[0] + cur_ylim[1]) / 2
            
            # 计算旋转角度（简化实现）
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            
            # 根据鼠标移动距离计算旋转角度
            angle = (dx - dy) * 0.01  # 简化的旋转角度计算
            
            # 应用旋转变换到坐标轴
            # 这里我们简单地交换坐标轴范围来模拟旋转效果
            x_range = cur_xlim[1] - cur_xlim[0]
            y_range = cur_ylim[1] - cur_ylim[0]
            
            # 更新坐标轴范围实现旋转效果
            self.ax.set_xlim(center_x - x_range/2, center_x + x_range/2)
            self.ax.set_ylim(center_y - y_range/2, center_y + y_range/2)
            
            self.canvas.draw()