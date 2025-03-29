import time
from typing import Optional


class TimeSpan:
    def __init__(self, title: str = ""):
        self.last_time = time.perf_counter()
        self.now_time = self.last_time
        self.title = title
        if title:
            print(f"{title}")

    def reset(self):
        """重置计时器"""
        self.last_time = time.perf_counter()
        self.now_time = self.last_time

    def show_span(self, title: str = ""):
        """显示时间间隔（基础版）"""
        self._calculate_span()
        print(f"{title} Time elapsed: {time_span:.3f} seconds")

    def show_to_log(self, title: str = "", log_func: Optional[callable] = None):
        """记录时间到日志（需提供日志函数）"""
        self._calculate_span()
        log_content = f"t: {self.now_time:.3f}  dt: {self.time_span:.3f}  {title}"

        if log_func:
            log_func(log_content)
        else:
            print(log_content)

    def show_to_console(self, title: str = ""):
        """带格式的屏幕输出"""
        self._calculate_span()
        print(
            f"{title}"
            f"  timeSpan: {self.time_span:.3f}  presentTime: {self.now_time:.3f}\n"
        )

    def _calculate_span(self):
        """内部计算时间间隔"""
        self.now_time = time.perf_counter()
        self.time_span = self.now_time - self.last_time
        self.last_time = self.now_time
