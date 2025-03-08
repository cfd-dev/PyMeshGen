import re

# 定义正则表达式（注意使用原始字符串避免转义问题）
pattern = r'^\(\s*45\s+\(\s*\d+\s+[\w-]+\s+[\w-]+\s*\)\s*\(\s*\)\s*\)$'
# pattern = r'^\(45 \(\d+ [a-zA-Z0-9-]+ [a-zA-Z0-9-]+\)\)\(\)$'
# pattern = r'^$45 \(\d+(?: [a-zA-Z0-9-]+)+$$$\)$'

# 编译正则表达式
regex = re.compile(pattern)

# 测试用例
test_cases = [
    "(45 (2 fluid unspecified)())",
    "(45 (3 interior interior-unspecified)())",
    "(45 (4 wall bc-2)())",
    "(45 (5 pressure-far-field bc-3)())",
    "(45 (6 invalid))",          # 无效格式（缺少部分）
    "45 (7 test test)()",        # 无效格式（缺少外层括号）
    "(45 (8 too many parts here)())"  # 无效格式（多余的内容）
]

# 验证测试用例
for test in test_cases:
    if regex.match(test):
        print(f"匹配成功: {test}")
    else:
        print(f"匹配失败: {test}")