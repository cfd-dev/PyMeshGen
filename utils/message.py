# 定义不同的调试级别
DEBUG_LEVEL_INFO = 0
DEBUG_LEVEL_DEBUG = 1
DEBUG_LEVEL_VERBOSE = 2

# 当前的调试级别
current_debug_level = DEBUG_LEVEL_INFO


def set_debug_level(level):
    """
    设置当前的调试级别。

    :param level: 要设置的调试级别
    """
    global current_debug_level
    current_debug_level = level


def info(message):
    """
    输出信息级别的消息。

    :param message: 要输出的消息
    """
    if current_debug_level >= DEBUG_LEVEL_INFO:
        print(f"[INFO] {message}")


def error(message):
    """
    输出错误级别的消息。

    :param message: 要输出的消息
    """
    if current_debug_level >= DEBUG_LEVEL_INFO:
        print(f"[ERROR] {message}")


def warning(message):
    """
    输出警告级别的消息。

    :param message: 要输出的消息
    """
    if current_debug_level >= DEBUG_LEVEL_INFO:
        print(f"[WARNING] {message}")


def debug(message):
    """
    输出调试级别的消息。

    :param message: 要输出的消息
    """
    if current_debug_level >= DEBUG_LEVEL_DEBUG:
        print(f"[DEBUG] {message}")


def verbose(message):
    """
    输出详细级别的消息。

    :param message: 要输出的消息
    """
    if current_debug_level >= DEBUG_LEVEL_VERBOSE:
        print(f"[VERBOSE] {message}")
