#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
网格生成工作线程模块
提供异步网格生成功能，避免UI冻结
"""

from PyQt5.QtCore import QThread, pyqtSignal, QObject
import sys
import os

# 设置项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 添加子目录到路径
for subdir in ["fileIO", "data_structure", "meshsize", "visualization", "adfront2", "optimize", "utils"]:
    subdir_path = os.path.join(PROJECT_ROOT, subdir)
    if subdir_path not in sys.path:
        sys.path.insert(0, subdir_path)

PROGRESS_STEPS = [
    (10, "正在初始化网格生成参数..."),
    (20, "正在读取输入网格数据..."),
    (30, "正在构造初始阵面..."),
    (40, "正在计算网格尺寸场..."),
    (50, "正在生成边界层网格..."),
    (60, "正在推进生成网格..."),
    (70, "正在优化网格质量..."),
    (80, "正在合并网格..."),
    (90, "正在保存网格文件..."),
    (100, "网格生成完成")
]


class MeshGenerationSignals(QObject):
    """网格生成信号类"""
    progress = pyqtSignal(int, str)  # 进度值, 进度描述
    finished = pyqtSignal(object)     # 完成信号, 传递生成的网格数据
    error = pyqtSignal(str)           # 错误信号, 传递错误信息
    log = pyqtSignal(str)             # 日志信号, 传递日志信息


class MeshGenerationThread(QThread):
    """网格生成工作线程类"""

    def __init__(self, params, mesh_data, gui_instance=None):
        super().__init__()
        self.params = params
        self.mesh_data = mesh_data
        self.gui_instance = gui_instance
        self.signals = MeshGenerationSignals()
        self._is_running = True
        self._progress_steps = PROGRESS_STEPS

    def run(self):
        """执行网格生成任务"""
        try:
            from core import generate_mesh
            from utils.message import set_gui_instance, info

            # 创建GUI适配器，用于将线程信号转发到GUI
            gui_adapter = GUIAdapter(self.signals, self._is_running)
            
            # 设置GUI实例到消息系统
            set_gui_instance(gui_adapter)

            # 发送初始进度
            self.signals.progress.emit(5, "准备开始网格生成...")

            # 执行网格生成
            result_mesh = generate_mesh(self.params, self.mesh_data, gui_adapter)

            # 发送完成信号
            if self._is_running:
                self.signals.progress.emit(100, "网格生成完成")
                self.signals.finished.emit(result_mesh)
            else:
                self.signals.log.emit("网格生成已被用户取消")

        except Exception as e:
            import traceback
            error_msg = f"网格生成失败: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)

        finally:
            # 重置消息系统
            from utils.message import set_gui_instance
            set_gui_instance(None)

    def stop(self):
        """停止网格生成任务"""
        self._is_running = False
        self.signals.log.emit("正在停止网格生成...")
        self.wait(1000)  # 等待线程结束，最多等待1秒
        if self.isRunning():
            self.terminate()


class GUIAdapter:
    """GUI适配器类，用于将线程信号转发到GUI"""

    def __init__(self, signals, is_running_flag):
        self.signals = signals
        self._is_running = is_running_flag
        self._current_step = 0
        self.ax = None  # 添加ax属性，设置为None以避免core.py中的可视化调用
        self.canvas = None  # 添加canvas属性
        self._progress_steps = PROGRESS_STEPS

    def _update_progress(self, step_index):
        """更新进度"""
        if step_index < len(self._progress_steps):
            progress, description = self._progress_steps[step_index]
            self.signals.progress.emit(progress, description)
            self._current_step = step_index

    def append_info_output(self, message):
        """添加信息输出"""
        self.signals.log.emit(message)

    def log_info(self, message):
        """记录信息"""
        self.signals.log.emit(message)

    def log_error(self, message):
        """记录错误"""
        self.signals.log.emit(f"[ERROR] {message}")

    def log_warning(self, message):
        """记录警告"""
        self.signals.log.emit(f"[WARNING] {message}")

    def update_status(self, message):
        """更新状态"""
        self.signals.log.emit(message)

    # 添加其他可能的GUI方法
    def __getattr__(self, name):
        """处理未定义的属性访问"""
        # 返回一个空函数，避免AttributeError
        return lambda *args, **kwargs: None
