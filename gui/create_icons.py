#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图标资源生成脚本
为GUI界面创建简单的图标
"""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_icon_folder():
    """创建图标文件夹"""
    icon_dir = os.path.join(os.path.dirname(__file__), "icons")
    if not os.path.exists(icon_dir):
        os.makedirs(icon_dir)
    return icon_dir

def create_file_icon(size=(24, 24), color="#3498db"):
    """创建文件图标"""
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 绘制文件形状
    w, h = size
    margin = 4
    draw.rectangle([margin, margin, w-margin, h-margin], fill=color, outline="black")
    
    # 绘制文件折角
    corner_size = 6
    points = [
        (w - margin, margin + corner_size),
        (w - margin - corner_size, margin),
        (w - margin, margin)
    ]
    draw.polygon(points, fill="white", outline="black")
    
    return img

def create_folder_icon(size=(24, 24), color="#f39c12"):
    """创建文件夹图标"""
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 绘制文件夹形状
    w, h = size
    margin = 4
    tab_width = 8
    tab_height = 4
    
    # 绘制文件夹标签
    draw.rectangle([margin, margin, margin + tab_width, margin + tab_height], fill=color, outline="black")
    
    # 绘制文件夹主体
    draw.rectangle([margin, margin + tab_height - 1, w - margin, h - margin], fill=color, outline="black")
    
    return img

def create_config_icon(size=(24, 24), color="#9b59b6"):
    """创建配置图标（齿轮）"""
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 绘制齿轮形状
    w, h = size
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 3
    
    # 绘制齿轮外圈
    draw.ellipse([center_x - radius, center_y - radius, 
                  center_x + radius, center_y + radius], 
                 fill=color, outline="black")
    
    # 绘制齿轮内圈
    inner_radius = radius // 2
    draw.ellipse([center_x - inner_radius, center_y - inner_radius, 
                  center_x + inner_radius, center_y + inner_radius], 
                 fill="white", outline="black")
    
    # 绘制齿轮齿
    teeth = 8
    for i in range(teeth):
        angle = i * (360 / teeth)
        # 简化绘制，只画几个点表示齿轮齿
        if i % 2 == 0:
            x = center_x + radius * np.cos(np.radians(angle))
            y = center_y + radius * np.sin(np.radians(angle))
            draw.ellipse([x-2, y-2, x+2, y+2], fill=color, outline="black")
    
    return img

def create_mesh_icon(size=(24, 24), color="#2ecc71"):
    """创建网格图标"""
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 绘制网格形状
    w, h = size
    margin = 4
    
    # 绘制三角形网格
    points = [
        (w // 2, margin),  # 顶点
        (w - margin, h - margin),  # 右下
        (margin, h - margin)  # 左下
    ]
    draw.polygon(points, fill=color, outline="black")
    
    # 绘制内部网格线
    center_x, center_y = w // 2, h // 2
    draw.line([(w // 2, margin), (center_x, center_y)], fill="black", width=1)
    draw.line([(center_x, center_y), (w - margin, h - margin)], fill="black", width=1)
    draw.line([(center_x, center_y), (margin, h - margin)], fill="black", width=1)
    
    return img

def create_part_icon(size=(24, 24), color="#e74c3c"):
    """创建部件图标"""
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 绘制部件形状（立方体）
    w, h = size
    margin = 4
    
    # 绘制立方体正面
    draw.rectangle([margin, margin, w - margin - 4, h - margin], fill=color, outline="black")
    
    # 绘制立方体顶面
    points = [
        (margin, margin),
        (w - margin - 4, margin),
        (w - margin, margin - 4),
        (margin + 4, margin - 4)
    ]
    draw.polygon(points, fill=color, outline="black")
    
    # 绘制立方体侧面
    points = [
        (w - margin - 4, margin),
        (w - margin - 4, h - margin),
        (w - margin, h - margin - 4),
        (w - margin, margin - 4)
    ]
    draw.polygon(points, fill=color, outline="black")
    
    return img

def create_all_icons():
    """创建所有图标"""
    icon_dir = create_icon_folder()
    
    # 创建各种图标
    icons = [
        ("file.png", create_file_icon),
        ("folder.png", create_folder_icon),
        ("config.png", create_config_icon),
        ("mesh.png", create_mesh_icon),
        ("part.png", create_part_icon)
    ]
    
    for filename, create_func in icons:
        icon = create_func()
        icon_path = os.path.join(icon_dir, filename)
        icon.save(icon_path)
        print(f"已创建图标: {icon_path}")

if __name__ == "__main__":
    create_all_icons()