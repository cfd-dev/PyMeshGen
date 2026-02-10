# 定义不同的调试级别
DEBUG_LEVEL_INFO = 0
DEBUG_LEVEL_DEBUG = 1
DEBUG_LEVEL_VERBOSE = 2

# 默认调试级别
current_debug_level = DEBUG_LEVEL_INFO

_LEVEL_THRESHOLDS = {
    "INFO": DEBUG_LEVEL_INFO,
    "WARNING": DEBUG_LEVEL_INFO,
    "ERROR": DEBUG_LEVEL_INFO,
    "DEBUG": DEBUG_LEVEL_DEBUG,
    "VERBOSE": DEBUG_LEVEL_VERBOSE,
}

# GUI实例引用
_gui_instance = None


def set_gui_instance(gui_instance):
    """
    设置GUI实例引用，用于将消息输出到GUI信息窗口。

    :param gui_instance: GUI实例
    """
    global _gui_instance
    _gui_instance = gui_instance


def set_debug_level(level):
    """
    设置当前的调试级别。

    :param level: 要设置的调试级别
    """
    global current_debug_level
    current_debug_level = level


def _output_message(prefix, message):
    """内部函数，用于输出带前缀的消息。"""
    formatted_message = f"[{prefix}] {message}"
    
    # 如果有GUI实例，同时输出到GUI信息窗口
    if _gui_instance is not None:
        _gui_instance.append_info_output(formatted_message)
    
    # 同时也输出到控制台
    print(formatted_message)


def _log(level, message):
    if current_debug_level >= _LEVEL_THRESHOLDS[level]:
        _output_message(level, message)


def info(message):
    """输出信息级别的消息。"""
    _log("INFO", message)


def error(message):
    """输出错误级别的消息。"""
    _log("ERROR", message)


def warning(message):
    """输出警告级别的消息。"""
    _log("WARNING", message)


def debug(message):
    """输出调试级别的消息。"""
    _log("DEBUG", message)


def verbose(message):
    """输出详细级别的消息。"""
    _log("VERBOSE", message)
