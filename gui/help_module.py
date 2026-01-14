import os
import subprocess
import sys
from PyQt5.QtWidgets import QMessageBox


class HelpModule:
    """帮助模块 - 负责显示关于信息和用户手册"""

    def __init__(self, gui):
        self.gui = gui

    def show_about(self):
        """显示关于对话框"""
        about_text = """PyMeshGen v1.0\n\n基于Python的网格生成工具\n\n© 2025 CFD Dev"""
        QMessageBox.about(self.gui, "关于", about_text)

    def show_user_manual(self):
        """显示用户手册 - 打开UserGuide.pdf或UserGuide.md文件"""
        try:
            pdf_path = os.path.join(self.gui.project_root, "docs", "UserGuide.pdf")

            if not os.path.exists(pdf_path):
                pdf_path = os.path.join(self.gui.project_root, "..", "docs", "UserGuide.pdf")
                if not os.path.exists(pdf_path):
                    md_path = os.path.join(self.gui.project_root, "docs", "UserGuide.md")
                    if os.path.exists(md_path):
                        pdf_path = md_path
                    else:
                        md_path = os.path.join(self.gui.project_root, "..", "docs", "UserGuide.md")
                        if os.path.exists(md_path):
                            pdf_path = md_path
                        else:
                            QMessageBox.warning(self.gui, "警告", f"用户手册文件不存在:\n{os.path.join('docs', 'UserGuide.pdf')}")
                            self.gui.log_info("用户手册文件不存在")
                            self.gui.update_status("手册文件不存在")
                            return

            if sys.platform.startswith('darwin'):
                subprocess.call(['open', pdf_path])
            elif sys.platform.startswith('win'):
                os.startfile(pdf_path)
            elif sys.platform.startswith('linux'):
                subprocess.call(['xdg-open', pdf_path])
            else:
                subprocess.call(['xdg-open', pdf_path])

            self.gui.log_info(f"已打开用户手册: {pdf_path}")
            self.gui.update_status("用户手册已打开")

        except Exception as e:
            QMessageBox.critical(self.gui, "错误", f"打开用户手册时出错:\n{str(e)}")
            self.gui.log_info(f"打开用户手册出错: {str(e)}")
            self.gui.update_status("打开手册失败")
            pdf_path = os.path.join(self.gui.project_root, "docs", "UserGuide.pdf")

            if not os.path.exists(pdf_path):
                pdf_path = os.path.join(self.gui.project_root, "..", "docs", "UserGuide.pdf")
                if not os.path.exists(pdf_path):
                    md_path = os.path.join(self.gui.project_root, "docs", "UserGuide.md")
                    if os.path.exists(md_path):
                        pdf_path = md_path
                    else:
                        md_path = os.path.join(self.gui.project_root, "..", "docs", "UserGuide.md")
                        if os.path.exists(md_path):
                            pdf_path = md_path
                        else:
                            QMessageBox.warning(self.gui, "警告", f"用户手册文件不存在:\n{os.path.join('docs', 'UserGuide.pdf')}")
                            self.gui.log_info("用户手册文件不存在")
                            self.gui.update_status("手册文件不存在")
                            return

            if sys.platform.startswith('darwin'):
                subprocess.call(['open', pdf_path])
            elif sys.platform.startswith('win'):
                os.startfile(pdf_path)
            elif sys.platform.startswith('linux'):
                subprocess.call(['xdg-open', pdf_path])
            else:
                subprocess.call(['xdg-open', pdf_path])

            self.gui.log_info(f"已打开用户手册: {pdf_path}")
            self.gui.update_status("用户手册已打开")

    def show_quick_start(self):
        """显示快速入门"""
        quick_start_text = """PyMeshGen 快速入门\n\n1. 启动程序后，点击"新建配置"创建新项目\n2. 在左侧部件列表中添加几何部件\n3. 配置网格生成参数\n4. 点击"生成网格"按钮生成网格\n5. 使用视图工具查看和操作网格"""
        QMessageBox.about(self.gui, "快速入门", quick_start_text)

    def check_for_updates(self):
        """检查更新"""
        self.gui.log_info("检查更新功能暂未实现")

    def show_shortcuts(self):
        """显示快捷键"""
        shortcuts_text = """常用快捷键：\n\nCtrl+N: 新建工程\nCtrl+O: 打开工程\nCtrl+S: 保存工程\nCtrl+I: 导入网格\nCtrl+E: 导出网格\nCtrl+G: 导入几何\nCtrl+Shift+E: 导出几何\nF5: 生成网格\nF6: 显示网格\nF11: 全屏显示\nEsc: 退出全屏"""
        QMessageBox.about(self.gui, "快捷键", shortcuts_text)
