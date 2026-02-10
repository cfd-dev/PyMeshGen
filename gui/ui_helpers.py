class UIHelpers:
    """UI工具类 - 提供日志记录、状态更新和UI元素切换等辅助功能"""

    def __init__(self, gui):
        self.gui = gui

    def log_info(self, message):
        """记录信息日志"""
        if hasattr(self.gui, 'info_output'):
            self.gui.info_output.append_info_output(f"[INFO] {message}")

    def log_error(self, message):
        """记录错误日志"""
        if hasattr(self.gui, 'info_output'):
            self.gui.info_output.append_info_output(f"[ERROR] {message}")

    def log_warning(self, message):
        """记录警告日志"""
        if hasattr(self.gui, 'info_output'):
            self.gui.info_output.append_info_output(f"[WARNING] {message}")

    def log_debug(self, message):
        """记录调试日志"""
        if hasattr(self.gui, 'info_output'):
            self.gui.info_output.append_info_output(f"[DEBUG] {message}")

    def log_verbose(self, message):
        """记录详细日志"""
        if hasattr(self.gui, 'info_output'):
            self.gui.info_output.append_info_output(f"[VERBOSE] {message}")

    def update_status(self, message):
        """更新状态栏信息"""
        if hasattr(self.gui, 'status_bar'):
            self.gui.status_bar.update_status(message)
