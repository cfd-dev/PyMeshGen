#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
详细的VTK OpenGL测试脚本，模拟项目中的实际使用情况
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import vtk
import tkinter as tk
from tkinter import ttk

class VTKMeshDisplay:
    """模拟项目中的网格显示类"""
    
    def __init__(self, parent):
        self.parent = parent
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.embedded = False
        self.create_mesh_display_area()
        
    def create_mesh_display_area(self):
        """创建VTK网格显示区域"""
        # 创建主框架
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建VTK渲染器
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)  # 设置为白色背景
        
        # 创建VTK渲染窗口
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(800, 600)
        
        # 创建一个Tkinter框架来容纳VTK窗口
        self.vtk_frame = ttk.Frame(main_frame)
        self.vtk_frame.pack(fill=tk.BOTH, expand=True)
        
        # 初始化交互器
        self.interactor = None
        self.embedded = False
        
        # 延迟嵌入VTK窗口，确保框架已经完全创建
        self.vtk_frame.after(100, self.embed_vtk_window)
        
        # 重置相机
        self.renderer.ResetCamera()
    
    def embed_vtk_window(self):
        """将VTK窗口嵌入到Tkinter框架中"""
        try:
            print("开始嵌入VTK窗口...")
            
            # 创建VTK交互器
            self.interactor = vtk.vtkRenderWindowInteractor()
            self.interactor.SetRenderWindow(self.render_window)
            
            # 检测操作系统
            import platform
            system = platform.system()
            print(f"操作系统: {system}")
            
            # Windows系统使用特殊处理
            if system == "Windows":
                try:
                    print("使用Windows特殊处理...")
                    # 获取窗口句柄
                    window_id = self.vtk_frame.winfo_id()
                    print(f"窗口句柄: {window_id}")
                    
                    # 设置窗口信息
                    self.render_window.SetWindowInfo(str(window_id))
                    
                    # 初始化交互器
                    print("初始化交互器...")
                    self.interactor.Initialize()
                    
                    # 设置交互器样式，支持鼠标操作
                    style = vtk.vtkInteractorStyleTrackballCamera()
                    self.interactor.SetInteractorStyle(style)
                    
                    # 设置窗口大小
                    self.vtk_frame.update_idletasks()
                    width = self.vtk_frame.winfo_width()
                    height = self.vtk_frame.winfo_height()
                    if width <= 1 or height <= 1:
                        width, height = 800, 600
                    self.render_window.SetSize(width, height)
                    
                    print("VTK窗口嵌入成功")
                    self.embedded = True
                    return True
                    
                except Exception as e:
                    print(f"Windows窗口ID方法失败: {str(e)}")
                    return False
            
            print("VTK窗口嵌入完成")
            self.embedded = True
            return True
            
        except Exception as e:
            print(f"嵌入VTK窗口时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            self.embedded = False
            return False
    
    def display_simple_mesh(self):
        """显示一个简单的网格"""
        try:
            if not self.embedded:
                print("VTK窗口未嵌入，尝试嵌入...")
                self.embed_vtk_window()
                
            if not self.embedded:
                print("VTK窗口嵌入失败")
                return False
                
            print("创建测试网格...")
            
            # 创建一个简单的几何体（球体）
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(1.0)
            sphere.SetPhiResolution(20)
            sphere.SetThetaResolution(20)
            
            # 创建映射器
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            # 创建演员
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # 清除之前的演员
            self.renderer.RemoveAllViewProps()
            
            # 添加到渲染器
            self.renderer.AddActor(actor)
            self.renderer.ResetCamera()
            
            print("渲染网格...")
            # 强制渲染
            self.render_window.Render()
            
            return True
            
        except Exception as e:
            print(f"显示网格时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def test_vtk_detailed():
    """详细测试VTK功能"""
    print("开始详细测试VTK功能...")
    
    try:
        # 创建Tkinter窗口
        root = tk.Tk()
        root.title("VTK详细测试")
        root.geometry("1000x800")
        
        # 创建网格显示区域
        print("创建网格显示区域...")
        mesh_display = VTKMeshDisplay(root)
        
        # 创建控制按钮
        control_frame = ttk.Frame(root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        def test_display():
            print("测试显示网格...")
            success = mesh_display.display_simple_mesh()
            if success:
                print("网格显示成功!")
            else:
                print("网格显示失败!")
        
        def close_window():
            print("关闭窗口...")
            root.destroy()
        
        test_btn = ttk.Button(control_frame, text="测试显示", command=test_display)
        test_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = ttk.Button(control_frame, text="关闭", command=close_window)
        close_btn.pack(side=tk.LEFT, padx=5)
        
        # 设置定时器进行测试
        root.after(1000, test_display)  # 1秒后测试显示
        root.after(5000, close_window)  # 5秒后关闭
        
        print("启动GUI...")
        root.mainloop()
        
        return True
        
    except Exception as e:
        print(f"详细测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("VTK详细测试脚本")
    print("=" * 50)
    success = test_vtk_detailed()
    if success:
        print("\nVTK详细测试完成!")
    else:
        print("\nVTK详细测试失败!")