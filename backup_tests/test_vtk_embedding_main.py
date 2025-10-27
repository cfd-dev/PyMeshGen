"""
测试VTK窗口嵌入到主界面的网格视图区
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.mesh_display import MeshDisplayArea
from fileIO.read_cas import parse_fluent_msh

class TestMainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VTK窗口嵌入测试")
        self.root.geometry("1200x800")
        
        # 状态栏变量
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建左侧面板（控制面板）
        self.left_panel = ttk.Frame(self.main_frame, width=300)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        self.left_panel.pack_propagate(False)
        
        # 创建右侧面板（网格显示区域）
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 在右侧面板中创建网格显示区域
        self.create_mesh_display()
        
        # 在左侧面板中添加控制按钮
        self.create_control_panel()
        
        # 状态栏
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 初始化网格数据
        self.mesh_data = None
        
    def create_mesh_display(self):
        """创建网格显示区域"""
        # 创建标题
        title_label = ttk.Label(self.right_panel, text="网格显示区域", font=("Arial", 12, "bold"))
        title_label.pack(pady=5)
        
        # 创建网格显示区域
        self.mesh_display = MeshDisplayArea(self.right_panel)
        self.mesh_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 检查VTK窗口是否正确嵌入
        if hasattr(self.mesh_display, 'embedded') and self.mesh_display.embedded:
            self.status_var.set("VTK窗口已成功嵌入到网格视图区")
        else:
            self.status_var.set("警告：VTK窗口可能未正确嵌入")
    
    def create_control_panel(self):
        """创建控制面板"""
        # 标题
        title_label = ttk.Label(self.left_panel, text="控制面板", font=("Arial", 12, "bold"))
        title_label.pack(pady=10)
        
        # 分隔线
        ttk.Separator(self.left_panel, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # 文件操作区域
        file_frame = ttk.LabelFrame(self.left_panel, text="文件操作")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 导入CAS文件按钮
        import_btn = ttk.Button(file_frame, text="导入CAS文件", command=self.import_cas_file)
        import_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建测试网格按钮
        test_btn = ttk.Button(file_frame, text="创建测试网格", command=self.create_test_mesh)
        test_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 清除显示按钮
        clear_btn = ttk.Button(file_frame, text="清除显示", command=self.clear_display)
        clear_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 视图控制区域
        view_frame = ttk.LabelFrame(self.left_panel, text="视图控制")
        view_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 重置视图按钮
        reset_btn = ttk.Button(view_frame, text="重置视图", command=self.reset_view)
        reset_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # 信息显示区域
        info_frame = ttk.LabelFrame(self.left_panel, text="网格信息")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 信息文本框
        self.info_text = tk.Text(info_frame, height=10, width=30)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.configure(yscrollcommand=scrollbar.set)
    
    def import_cas_file(self):
        """导入CAS文件"""
        cas_file = "config/input/convex.cas"
        
        if not os.path.exists(cas_file):
            messagebox.showerror("错误", f"CAS文件不存在: {cas_file}")
            return
        
        try:
            # 解析CAS文件
            self.status_var.set("正在解析CAS文件...")
            self.root.update()
            
            mesh_data = parse_fluent_msh(cas_file)
            
            # 设置网格数据
            self.mesh_display.set_mesh_data(mesh_data)
            
            # 显示网格
            self.mesh_display.display_mesh()
            
            # 更新网格信息
            self.update_mesh_info(mesh_data)
            
            self.status_var.set(f"成功导入CAS文件: {os.path.basename(cas_file)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"导入CAS文件失败: {str(e)}")
            self.status_var.set("导入CAS文件失败")
    
    def create_test_mesh(self):
        """创建测试网格"""
        try:
            # 创建简单的测试网格数据
            mesh_data = {
                'nodes': [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.5, 0.5, 1.0]
                ],
                'cells': [
                    [0, 1, 2, 3],  # 底面四边形
                    [0, 1, 4],     # 三角形1
                    [1, 2, 4],     # 三角形2
                    [2, 3, 4],     # 三角形3
                    [3, 0, 4]      # 三角形4
                ],
                'dimensions': 3
            }
            
            # 设置网格数据
            self.mesh_display.set_mesh_data(mesh_data)
            
            # 显示网格
            self.mesh_display.display_mesh()
            
            # 更新网格信息
            self.update_mesh_info(mesh_data)
            
            self.status_var.set("成功创建测试网格")
            
        except Exception as e:
            messagebox.showerror("错误", f"创建测试网格失败: {str(e)}")
            self.status_var.set("创建测试网格失败")
    
    def clear_display(self):
        """清除显示"""
        try:
            self.mesh_display.clear_display()
            self.info_text.delete(1.0, tk.END)
            self.status_var.set("已清除显示")
        except Exception as e:
            messagebox.showerror("错误", f"清除显示失败: {str(e)}")
    
    def reset_view(self):
        """重置视图"""
        try:
            self.mesh_display.reset_view()
            self.status_var.set("视图已重置")
        except Exception as e:
            messagebox.showerror("错误", f"重置视图失败: {str(e)}")
    
    def update_mesh_info(self, mesh_data):
        """更新网格信息"""
        self.info_text.delete(1.0, tk.END)
        
        info = []
        info.append(f"节点数量: {len(mesh_data.get('nodes', []))}")
        info.append(f"单元数量: {len(mesh_data.get('cells', []))}")
        info.append(f"维度: {mesh_data.get('dimensions', '未知')}")
        
        if 'zones' in mesh_data:
            info.append(f"区域数量: {len(mesh_data['zones'])}")
            for zone_name, zone_data in mesh_data['zones'].items():
                zone_type = zone_data.get('type', '未知')
                info.append(f"  - {zone_name}: {zone_type}")
        
        self.info_text.insert(tk.END, "\n".join(info))
    
    def run(self):
        """运行主窗口"""
        self.root.mainloop()

def main():
    """主函数"""
    print("开始测试VTK窗口嵌入到主界面的网格视图区...")
    
    # 创建并运行测试窗口
    app = TestMainWindow()
    app.run()
    
    print("测试完成！")

if __name__ == "__main__":
    main()