# PyMeshGen GUI 模块

PyMeshGen GUI是基于PyQt5和VTK开发的图形化用户界面，为PyMeshGen网格生成工具提供直观的可视化操作界面。

## 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [模块架构](#模块架构)
- [核心功能](#核心功能)
- [使用指南](#使用指南)
- [技术实现](#技术实现)
- [快捷键](#快捷键)
- [常见问题](#常见问题)

## 概述

### 设计理念

PyMeshGen GUI旨在降低网格生成工具的使用门槛，通过图形化界面提供：

- **直观的参数配置**：无需编辑JSON配置文件，通过界面直接调整参数
- **实时可视化反馈**：实时查看网格生成过程和结果
- **高效的工作流程**：支持工程管理、参数模板、批量操作等功能
- **专业的交互体验**：参考主流CAE软件设计，符合工程师使用习惯

### 主要特性

- 基于PyQt5的跨平台图形界面（Windows/Linux/macOS）
- 基于VTK的高性能3D网格渲染引擎
- Ribbon风格功能区，操作直观高效
- 支持多种网格渲染模式（实体、线框、混合、点云）
- 完整的工程管理功能（新建、打开、保存）
- 智能参数管理和配置导入导出
- 丰富的快捷键和交互方式

## 快速开始

### 启动GUI

```bash
python start_gui.py
```

### 基本工作流程

**网格生成流程：**
1. **导入CAS网格**：文件 → 导入网格，选择.cas格式文件
2. **提取边界**：几何 → 提取边界，提取边界网格及部件信息
3. **配置参数**：配置 → 部件参数，设置各部件的网格生成参数
4. **生成网格**：网格 → 生成（或按F5键）
5. **查看结果**：使用鼠标交互查看网格，按1-3键切换渲染模式
6. **导出网格**：文件 → 导出网格，保存为VTK格式

**几何模型流程：**
1. **导入几何模型**：文件 → 导入几何，选择STEP/IGES/STL格式文件
2. **查看几何**：使用鼠标交互查看几何模型，按1-3键切换渲染模式
3. **导出几何**：文件 → 导出几何，保存为STEP/IGES/STL格式

## 模块架构

```
gui/
├── __init__.py              # 模块初始化
├── gui_main.py             # 主窗口，负责整体布局和事件处理
├── gui_base.py             # 基础组件类（状态栏、信息输出、对话框等）
├── ribbon_widget.py        # Ribbon风格功能区
├── mesh_display.py         # 网格显示区域（基于VTK）
├── config_manager.py       # 配置管理
├── file_operations.py      # 文件操作
├── part_params_dialog.py   # 部件参数对话框
├── icon_manager.py         # 图标管理
├── ui_utils.py             # UI工具函数和样式定义
├── import_thread.py        # 异步几何导入线程
└── model_tree.py          # 模型树控件
```

### 模块说明

#### gui_main.py

主窗口类`PyMeshGenGUI`，负责：
- 窗口布局和初始化
- 事件处理和回调函数
- 各功能模块的协调

#### gui_base.py

提供基础UI组件：
- `StatusBar`：状态栏
- `InfoOutput`：信息输出窗口
- `DialogBase`：对话框基类
- `ConfigDialog`：配置编辑对话框
- `PartListWidget`：部件列表组件

#### ribbon_widget.py

Ribbon风格功能区：
- `RibbonWidget`：主功能区
- `RibbonTabBar`：标签栏
- `RibbonGroup`：功能分组
- 支持可折叠设计

#### mesh_display.py

网格显示模块：
- `MeshDisplayArea`：网格显示区域
- 基于VTK的高性能渲染
- 支持多种渲染模式
- 边界可视化
- 交互功能（缩放、平移、旋转）

#### config_manager.py

配置管理模块：
- `ConfigManager`：配置管理器
- 参数对象与配置字典的转换
- 配置文件的保存和加载

#### part_params_dialog.py

部件参数对话框：
- `PartParamsDialog`：部件参数设置对话框
- 支持多部件参数编辑
- 参数验证

#### ui_utils.py

UI工具函数：
- `UIStyles`：样式定义
- `LayoutConfig`：布局配置
- 各种辅助函数

#### import_thread.py

异步导入线程：
- `GeometryImportThread`：几何文件异步导入
- 避免界面冻结
- 支持进度反馈

#### model_tree.py

模型树控件：
- `ModelTreeWidget`：几何模型树形显示
- 显示几何元素详细信息
- 支持交互式选择和查看

## 核心功能

### 1. 主窗口布局

采用三栏布局设计：

- **左侧面板**：部件列表和属性面板
- **中间区域**：网格显示区域
- **右侧面板**：信息输出区域
- **顶部功能区**：Ribbon风格工具栏

### 2. Ribbon功能区

包含6个标签页：

- **文件**：新建、打开、保存、导入网格、导入几何、导出网格、导出几何
- **几何**：导入网格、提取边界
- **视图**：重置、适应、缩放、渲染模式切换
- **配置**：全局参数、部件参数、配置导入导出
- **网格**：生成、显示、清空、质量检查、平滑、优化
- **帮助**：用户手册、快速入门、快捷键、关于

### 3. 网格与几何显示

基于VTK的高性能渲染引擎：

- **网格渲染模式**：实体、线框、实体+线框
- **几何渲染模式**：实体、线框、实体+线框
  - 线框模式显示原始几何曲线（非VTK转换线框）
  - 实体+线框模式同时显示实体表面和原始几何边
- **边界可视化**：根据边界类型自动着色
- **交互功能**：鼠标拖拽旋转、滚轮缩放、中键平移
- **坐标轴显示**：左下角固定显示3D坐标轴
- **模型树显示**：树形展示几何元素（顶点、边、面、体）及其详细信息

### 4. 配置管理

- **参数对象转换**：Parameters对象与JSON配置字典的双向转换
- **配置文件管理**：支持配置文件的保存和加载
- **临时文件处理**：自动创建和清理临时配置文件

### 5. 工程管理

- **新建工程**：清空所有数据，准备新的网格生成任务
- **打开工程**：加载已保存的工程文件（.pymg格式）
- **保存工程**：保存当前工程的所有配置和网格数据

## 使用指南

### 导入CAS网格

1. 点击"文件" → "导入网格"
2. 选择Fluent .cas格式文件
3. 系统自动解析几何边界和部件信息
4. 在部件列表中显示所有部件

### 导入几何模型

1. 点击"文件" → "导入几何"
2. 选择几何文件格式（STEP/IGES/STL）
3. 系统异步导入几何模型，显示进度条
4. 导入完成后在模型树中显示几何元素（顶点、边、面、体）
5. 支持查看几何元素的详细信息（坐标、长度、面积等）

### 配置参数

#### 全局参数

1. 点击"配置" → "全局参数"
2. 设置调试级别、网格类型、可视化选项等
3. 点击"确定"保存

#### 部件参数

1. 点击"配置" → "部件参数"
2. 从下拉框选择要编辑的部件
3. 设置以下参数：
   - 最大网格尺寸（max_size）
   - 棱柱层开关（PRISM_SWITCH）：wall/off/match
   - 第一层高度（first_height）
   - 增长率（growth_rate）
   - 最大层数（max_layers）
   - 完整层数（full_layers）
   - 多方向选项（multi_direction）
4. 点击"保存"应用参数

### 生成网格

1. 配置完成后，点击"网格" → "生成"或按F5键
2. 观察信息输出窗口的生成进度
3. 生成完成后自动显示网格

### 查看网格和几何

- **旋转视角**：鼠标左键拖拽
- **缩放**：鼠标滚轮
- **平移**：鼠标中键拖拽
- **重置视图**：按R键或点击"视图" → "重置"
- **适应视图**：按F键或点击"视图" → "适应"
- **切换渲染模式**：
  - 按1键：实体模式
  - 按2键：线框模式（几何模型显示原始几何曲线）
  - 按3键：实体+线框模式（几何模型同时显示实体表面和原始几何边）
- **切换边界显示**：按O键

### 导出网格

1. 点击"文件" → "导出网格"
2. 选择输出文件路径和格式（VTK）
3. 点击"保存"

### 导出几何

1. 点击"文件" → "导出几何"
2. 选择输出文件路径和格式（STEP/IGES/STL）
3. 点击"保存"

## 技术实现

### VTK与PyQt5集成

使用`vtk.qt.QVTKRenderWindowInteractor`实现VTK与PyQt5的无缝集成：

```python
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

self.frame = QVTKRenderWindowInteractor(parent)
self.renderer = vtk.vtkRenderer()
self.render_window = self.frame.GetRenderWindow()
self.render_window.AddRenderer(self.renderer)
```

### 大规模网格与几何渲染

采用多种优化技术提升渲染性能：

- **LOD技术**：根据视距自动调整网格细节
- **异步渲染**：将网格渲染放在独立线程中
- **增量渲染**：只渲染可见区域的网格
- **渲染模式优化**：线框模式比实体模式渲染更快
- **几何边缘渲染**：使用原始几何曲线而非VTK转换线框，提高显示准确性
- **批量渲染**：对多个几何元素进行批量渲染，减少渲染调用次数
- **延迟加载**：大型几何模型采用延迟加载策略，避免界面冻结

### 异步几何导入

使用独立线程进行几何文件导入，避免界面冻结：

```python
from gui.import_thread import GeometryImportThread

self.import_thread = GeometryImportThread(file_path, format_type)
self.import_thread.progress_updated.connect(self.update_progress)
self.import_thread.finished.connect(self.on_import_finished)
self.import_thread.start()
```

支持导入格式：STEP、IGES、STL

### 内存管理

显式清理VTK对象，避免内存泄漏：

```python
def cleanup(self):
    if self.renderer and self.boundary_actors:
        for actor in self.boundary_actors:
            self.renderer.RemoveActor(actor)
        self.boundary_actors.clear()

    if self.renderer and self.mesh_actor:
        self.renderer.RemoveActor(self.mesh_actor)
        self.mesh_actor = None
```

### 跨平台兼容

- 使用系统默认字体
- 使用SVG矢量图标
- 根据操作系统调整界面布局和样式
- 使用PyQt5样式表统一界面风格

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| Ctrl+N | 新建工程 |
| Ctrl+O | 打开工程 |
| Ctrl+S | 保存工程 |
| Ctrl+I | 导入网格 |
| Ctrl+E | 导出网格 |
| Ctrl+R | 折叠/展开功能区 |
| R | 重置视图 |
| F | 适应视图 |
| + | 放大视图 |
| - | 缩小视图 |
| 1 | 实体模式 |
| 2 | 线框模式 |
| 3 | 实体+线框模式 |
| O | 切换边界显示 |
| F5 | 生成网格 |
| F6 | 显示网格 |

## 常见问题

### Q: GUI启动失败，提示缺少模块

**A**: 确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

主要依赖：
- PyQt5 >= 5.15.0
- vtk >= 9.5.0
- matplotlib >= 3.9.4
- numpy >= 1.26.0

### Q: 网格显示卡顿

**A**: 尝试以下优化方法：
1. 切换到线框模式（按2键）
2. 减少显示的网格规模
3. 关闭边界显示（按O键）

### Q: 如何自定义界面样式

**A**: 修改`ui_utils.py`中的`UIStyles`类，定义自定义样式表。

### Q: 支持哪些网格格式

**A**: 目前支持：
- 输入：Fluent .cas格式
- 输出：VTK .vtk格式

### Q: 支持哪些几何格式

**A**: 目前支持：
- 输入：STEP (.stp/.step)、IGES (.igs/.iges)、STL (.stl)
- 输出：STEP (.stp/.step)、IGES (.igs/.iges)、STL (.stl)

### Q: 几何导入时界面卡顿

**A**: 几何导入采用异步处理，大型文件可能需要较长时间：
1. 观察进度条显示的导入进度
2. 导入完成后会自动显示几何模型
3. 可在模型树中查看导入的几何元素

### Q: 如何批量生成网格

**A**: 建议使用命令行版本进行批量处理：
```bash
python PyMeshGen.py --case "./config/case1.json"
python PyMeshGen.py --case "./config/case2.json"
```

## 开发指南

### 添加新功能

1. 在相应模块中添加新功能
2. 在`ribbon_widget.py`中添加按钮
3. 在`gui_main.py`中实现回调函数
4. 更新文档和快捷键列表

### 修改界面样式

编辑`ui_utils.py`中的`UIStyles`类：

```python
class UIStyles:
    MAIN_WINDOW_STYLESHEET = """
    QMainWindow {
        background-color: #f5f5f5;
    }
    """
```

### 调试技巧

1. 使用PyQt5的调试工具：
```python
from PyQt5.QtCore import Qt
self.setAttribute(Qt.WA_NativeWindow)
```

2. 启用详细日志输出：
```python
self.params.debug_level = 2
```

## 相关链接

- [PyMeshGen主项目](../../README_zh.md)
- [用户手册](../../docs/UserGuide.md)
- [PyMeshGen GitHub](https://github.com/cfd-dev/PyMeshGen)

## 许可证

GPLv2+

## 联系方式

项目发起人：cfd_dev <cfd_dev@126.com>
