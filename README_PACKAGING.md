# PyMeshGen 打包工具使用说明

## 🚀 快速开始

### 一键打包（推荐）

**Windows 用户**：双击运行 `build.bat`

**命令行方式**：
```bash
# 基本打包（仅生成可执行文件）
python build.py

# 清理后打包并创建 ZIP 便携版
python build.py --clean --zip

# 创建所有格式（可执行文件 + 安装包 + ZIP）
python build.py --all
```

## 📦 输出文件

| 文件 | 位置 | 说明 |
|------|------|------|
| 可执行文件 | `dist/PyMeshGen.exe` | 独立运行的程序 |
| ZIP 便携版 | `releases/PyMeshGen-v*.zip` | 压缩的便携版本 |
| Windows 安装包 | `installer/PyMeshGen-Setup-*.exe` | 标准安装程序（需 Inno Setup） |

## 🔧 命令行参数

| 参数 | 说明 |
|------|------|
| `--clean` | 清理之前的构建文件 |
| `--debug` | 启用调试模式（显示控制台窗口） |
| `--installer` | 创建 Inno Setup 安装包 |
| `--zip` | 创建 ZIP 便携版 |
| `--all` | 创建所有输出格式 |
| `--skip-clean` | 跳过清理步骤 |

## 📋 前置要求

### 必需
- Python 3.8+
- pip 包管理器

### 可选
- **Inno Setup**：用于创建 Windows 安装包（如不需要安装包可忽略）

## 🛠️ 配置文件

### PyInstaller 配置
编辑 `PyMeshGen.spec` 文件自定义：
- 包含的模块和数据文件
- 图标
- 版本信息

### Inno Setup 配置
编辑 `PyMeshGen.iss` 文件自定义：
- 安装程序界面
- 安装目录
- 快捷方式

## ⚠️ 注意事项

1. **打包时间**：首次打包约需 15-20 分钟（依赖大量科学计算库）
2. **文件大小**：可执行文件约 2-3GB（包含所有依赖）
3. **Inno Setup**：如未安装，脚本会自动跳过安装包创建

## 📖 详细文档

- PyInstaller 文档：https://pyinstaller.org/
- Inno Setup 文档：https://jrsoftware.org/ishelp/
