"""
消息输出模块
提供统一的日志输出功能，支持控制台和 GUI 输出
"""

DEBUG_LEVEL_INFO = 0
DEBUG_LEVEL_DEBUG = 1
DEBUG_LEVEL_VERBOSE = 2

current_debug_level = DEBUG_LEVEL_INFO

_LEVEL_THRESHOLDS = {
    "INFO": DEBUG_LEVEL_INFO,
    "WARNING": DEBUG_LEVEL_INFO,
    "ERROR": DEBUG_LEVEL_INFO,
    "DEBUG": DEBUG_LEVEL_DEBUG,
    "VERBOSE": DEBUG_LEVEL_VERBOSE,
}

_gui_instance = None


def set_gui_instance(gui_instance):
    """设置GUI实例引用"""
    global _gui_instance
    _gui_instance = gui_instance


def format_message(level, message):
    """格式化消息"""
    return f"[{level}] {message}"


def _output_to_gui(gui_instance, message):
    """输出消息到GUI"""
    if not gui_instance:
        return
    
    if hasattr(gui_instance, "info_output") and hasattr(gui_instance.info_output, "append_info_output"):
        gui_instance.info_output.append_info_output(message)
    elif hasattr(gui_instance, "append_info_output"):
        gui_instance.append_info_output(message)


def _output_to_console(message):
    """输出消息到控制台"""
    print(message)


def gui_log(gui_instance, message):
    """输出日志到GUI"""
    _output_to_gui(gui_instance, message)


def gui_progress(gui_instance, step):
    """更新GUI进度"""
    if gui_instance and hasattr(gui_instance, "_update_progress"):
        gui_instance._update_progress(step)


def gui_info(gui_instance, message):
    """输出INFO级别消息到GUI"""
    gui_log(gui_instance, format_message("INFO", message))


def gui_warning(gui_instance, message):
    """输出WARNING级别消息到GUI"""
    gui_log(gui_instance, format_message("WARNING", message))


def gui_error(gui_instance, message):
    """输出ERROR级别消息到GUI"""
    gui_log(gui_instance, format_message("ERROR", message))


def gui_debug(gui_instance, message):
    """输出DEBUG级别消息到GUI"""
    gui_log(gui_instance, format_message("DEBUG", message))


def gui_verbose(gui_instance, message):
    """输出VERBOSE级别消息到GUI"""
    gui_log(gui_instance, format_message("VERBOSE", message))


def set_debug_level(level):
    """设置调试级别"""
    global current_debug_level
    current_debug_level = level


def _log(level, message):
    """内部日志函数"""
    if current_debug_level >= _LEVEL_THRESHOLDS[level]:
        formatted = format_message(level, message)
        _output_to_gui(_gui_instance, formatted)
        _output_to_console(formatted)


def info(message):
    """输出INFO级别消息"""
    _log("INFO", message)


def error(message, raise_exception=True):
    """输出ERROR级别消息，并可选抛出异常中断流程"""
    _log("ERROR", message)
    if raise_exception:
        raise RuntimeError(message)


def warning(message):
    """输出WARNING级别消息"""
    _log("WARNING", message)


def debug(message):
    """输出DEBUG级别消息"""
    _log("DEBUG", message)


def verbose(message):
    """输出VERBOSE级别消息"""
    _log("VERBOSE", message)
