#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单的VTK OpenGL测试脚本，用于诊断wglMakeCurrent错误
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import vtk
import tkinter as tk
from tkinter import ttk

def test_vtk_opengl():
    """测试VTK OpenGL功能"""
    print("开始测试VTK OpenGL功能...")
    
    try:
        # 创建Tkinter窗口
        root = tk.Tk()
        root.title("VTK OpenGL测试")
        root.geometry("800x600")
        
        # 创建主框架
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建VTK相关组件
        print("创建VTK渲染器...")
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.1, 0.2, 0.4)  # 设置背景色
        
        print("创建VTK渲染窗口...")
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(600, 400)
        
        # 创建一个简单的几何体（球体）用于测试
        print("创建测试几何体...")
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
        
        # 添加到渲染器
        renderer.AddActor(actor)
        renderer.ResetCamera()
        
        # 创建Tkinter框架用于容纳VTK窗口
        vtk_frame = ttk.Frame(main_frame)
        vtk_frame.pack(fill=tk.BOTH, expand=True)
        
        print("尝试嵌入VTK窗口...")
        # 获取窗口句柄
        window_id = vtk_frame.winfo_id()
        print(f"窗口句柄: {window_id}")
        
        # 设置窗口信息
        render_window.SetWindowInfo(str(window_id))
        
        # 创建交互器
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        
        # 初始化交互器
        print("初始化交互器...")
        interactor.Initialize()
        
        # 设置交互器样式
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)
        
        print("设置窗口大小...")
        render_window.SetSize(600, 400)
        
        print("开始渲染...")
        render_window.Render()
        
        # 添加按钮用于测试交互
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        def close_window():
            print("关闭窗口...")
            root.destroy()
        
        close_btn = ttk.Button(button_frame, text="关闭", command=close_window)
        close_btn.pack()
        
        print("VTK OpenGL测试成功!")
        print("如果出现wglMakeCurrent错误，将在几秒钟内显示...")
        
        # 设置定时器关闭窗口
        root.after(5000, close_window)
        
        # 运行GUI
        root.mainloop()
        
        return True
        
    except Exception as e:
        print(f"VTK OpenGL测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vtk_opengl()
    if success:
        print("\nVTK OpenGL测试完成!")
    else:
        print("\nVTK OpenGL测试失败!")