# Tecplot PLT 文件导出模块详细设计文档

## 1. 概述

### 1.1 模块定位

Tecplot PLT 导出模块是 PyMeshGen 网格生成系统的**输出接口组件**，负责将内部网格数据结构转换为 Tecplot 软件可读的 PLT 格式文件。该模块位于 `fileIO/tecplot_io.py`，通过 GUI 界面的"导出网格"功能触发。

### 1.2 功能范围

| 功能项 | 说明 |
|--------|------|
| 2D 非结构网格导出 | 使用 `FEPolygon` ZoneType |
| 3D 非结构网格导出 | 使用 `FEPolyhedron` ZoneType |
| 3D 边界区域输出 | 使用 `FEQUADRILATERAL` ZoneType，三角形面自动转换 |
| 标量场数据导出 | 支持任意数量节点标量字段 |
| 多数据源支持 | 网格字典、`Unstructured_Grid` 对象、Fluent `.cas` 文件 |

### 1.3 技术栈

- **语言**: Python 3
- **核心依赖**: `numpy`
- **可选依赖**: `fileIO.read_cas`（Fluent 导出）、`data_structure.mesh_reconstruction`（网格重建）

---

## 2. Tecplot PLT 文件格式

### 2.1 文件结构

PLT 文件由 **文件头** + **一个或多个 ZONE** 组成：

```
TITLE = "文件名 exported from PyMeshGen"
VARIABLES = "X", "Y", "Z"     ← 2D 时只有 "X", "Y"
ZONE
ZoneType = FEPolyhedron       ← 或 FEPolygon / FEQUADRILATERAL
Nodes    = <节点数>
Faces    = <面数>
Elements = <单元数>
Datapacking = BLOCK
TotalNumFaceNodes = <值>       ← 仅 3D 主区域
NumConnectedBoundaryFaces = 0  ← 仅 3D 主区域
TotalNumBoundaryConnections = 0 ← 仅 3D 主区域
<节点坐标数据>
<标量场数据>
<拓扑数据>
```

### 2.2 ZoneType 选择规则

| 场景 | ZoneType | 说明 |
|------|----------|------|
| 2D 主区域 | `FEPolygon` | 非结构三角形/四边形网格 |
| 3D 主区域 | `FEPolyhedron` | 非结构四面体/六面体/棱柱/棱锥网格 |
| 3D 边界区域 | `FEQUADRILATERAL` | 四边形面片；三角形面通过**重复第 4 个节点**转换 |

### 2.3 数据组织方式

- **Datapacking = BLOCK**: 每个变量独立成块，按变量顺序排列
- **索引体系**: 全局 1-based（内部计算用 0-based，写入时 +1）
- **行宽限制**: 每行最多 5 个数值，换行符分隔

### 2.4 拓扑数据块（仅 FEPolygon / FEPolyhedron）

拓扑数据按以下顺序写入：

| 数据项 | 适用条件 | 说明 |
|--------|----------|------|
| FaceNodeNumber | 3D 仅 | 每个面的节点数 |
| FaceNodesLink | 全部 | 每个面包含的节点 ID（1-based） |
| LeftCell | 全部 | 每个面的左侧单元 ID（1-based） |
| RightCell | 全部 | 每个面的右侧单元 ID（1-based） |
| FaceElementLink | 全部 | 每个单元包含的面 ID 列表 |

---

## 3. 架构设计

### 3.1 模块层次结构

```
┌─────────────────────────────────────────────────┐
│                 GUI 层                           │
│  ribbon_widget.py (导出按钮)                     │
│  mesh_operations.py (export_mesh / _export_plt)  │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│             公开 API 层                           │
│  export_mesh_to_plt()         ← 网格字典导出      │
│  export_unstructured_grid_to_plt() ← 对象导出     │
│  export_from_cas()            ← Fluent 文件导出   │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│           数据提取层（私有）                       │
│  _extract_from_grid()                           │
│  _extract_simplices_from_grid()                 │
│  _extract_simplices_from_unstructured_grid()    │
│  _build_edge_index_from_simplices()             │
│  _extract_boundary_zones_from_unstructured_grid()│
│  _extract_boundary_zones_from_grid_dict()       │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│           文件写入层（私有）                       │
│  _write_plt_file()          ← 主区域写入          │
│  _write_node_data()         ← 节点/标量数据       │
│  _build_topology()          ← 拓扑构建            │
│  _build_edge_to_face_map()  ← 边-面映射           │
│  _assign_cells_to_faces()   ← 单元分配            │
│  _write_topology()          ← 拓扑数据写入        │
│  _write_boundary_zones()    ← 边界区域写入        │
│  _write_single_boundary_zone()                   │
│  _collect_boundary_nodes()  ← 节点收集            │
│  _write_boundary_node_coords()                   │
│  _write_boundary_face_connectivity()             │
└─────────────────────────────────────────────────┘
```

### 3.2 数据流图

```
[网格数据] ──┐
             │
   ┌─────────▼──────────┐
   │  export_mesh_to_plt │  ← 入口函数
   │  或                │
   │  export_unstructured│
   │  _grid_to_plt       │
   └─────────┬──────────┘
             │
   ┌─────────▼──────────┐
   │  提取节点坐标        │
   │  提取单元连接        │  ← _extract_* 系列函数
   │  构建边索引          │
   │  提取边界区域        │
   └─────────┬──────────┘
             │
   ┌─────────▼──────────┐
   │  判断 2D / 3D       │
   │  决定是否输出边界    │
   └─────────┬──────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
 ┌────────┐   ┌────────────┐
 │ 2D/无  │   │ 3D 有边界   │
 │ 边界   │   │            │
 └───┬────┘   └─────┬──────┘
     │              │
     │        ┌─────▼──────────┐
     │        │ 写入边界 Zones  │ ← _write_boundary_zones()
     │        │ (FEQUADRILATERAL)│
     │        └─────┬──────────┘
     │              │
     │        ┌─────▼──────────┐
     │        │ 追加写入主区域   │ ← _write_plt_file(append=True)
     │        │ (FEPolyhedron)  │
     │        └────────────────┘
     │
     ▼
┌─────────────────────┐
│ 写入主区域 Zone      │ ← _write_plt_file()
│ (FEPolygon)         │
└─────────────────────┘
```

---

## 4. 核心算法设计

### 4.1 维度判断逻辑

维度判断采用**优先级策略**：

```python
# 优先级 1：显式声明
if grid is not None and "dimension" in grid:
    dimension = grid["dimension"]
# 优先级 2：从节点坐标推断
else:
    dimension = nodes.shape[1] if nodes.ndim == 2 else 2

# 最终判定
is_3d = dimension == 3
```

**设计理由**：
- 2D 网格在 CGNS/Fluent 中可能存储为 3D 坐标（Z=0），显式声明避免误判
- 节点坐标推断作为后备方案

### 4.2 边-单元拓扑构建

#### 4.2.1 算法流程

```
输入：单元连接列表 simplices，边索引 edge_index
输出：face_to_nodes, left_cell, right_cell

1. 构建 face_to_nodes 列表
   FOR each edge in edge_index:
     face_to_nodes.append([n1+1, n2+1])  ← 转换为 1-based

2. 构建边到面的映射 edge_to_face
   FOR each face_idx in range(num_faces):
     edge_key = (min(e1, e2), max(e1, e2))  ← 规范化边
     edge_to_face[edge_key] = face_idx

3. 分配单元到面的 LeftCell/RightCell
   FOR each cell_idx, simplex in enumerate(simplices):
     FOR each edge in simplex (按顺序):
       edge_key = (min(p1, p2), max(p1, p2))
       face_idx = edge_to_face.get(edge_key)
       IF face_idx found:
         IF left_cell[face_idx] == 0:
           left_cell[face_idx] = cell_idx + 1
         ELSE:
           right_cell[face_idx] = cell_idx + 1
```

#### 4.2.2 时间复杂度

| 步骤 | 时间复杂度 | 说明 |
|------|------------|------|
| 构建 face_to_nodes | O(F) | F = 面数 |
| 构建 edge_to_face | O(F) | 哈希表插入 |
| 分配单元到面 | O(C × N) | C = 单元数，N = 每单元边数（3 或 4） |

总体复杂度：**O(F + C)**，线性时间

### 4.3 边界区域节点重映射

3D 边界区域使用**局部节点索引**，仅包含边界面上的节点：

```
输入：边界面的节点列表（1-based 或 0-based）
输出：边界区域 Zone 的节点坐标和单元连接

1. 收集所有边界节点
   boundary_node_set = {n - node_base for each face in faces_data}
   boundary_nodes_list = sorted(boundary_node_set)

2. 构建全局 → 局部索引映射
   node_index_map = {old_idx: new_idx + 1 for new_idx, old_idx in enumerate(boundary_nodes_list)}

3. 写入节点坐标（按 boundary_nodes_list 顺序）
   FOR node_idx in boundary_nodes_list:
     write(nodes[node_idx])

4. 写入单元连接（使用 node_index_map 转换）
   FOR face in faces_data:
     n1 = node_index_map[face_nodes[0] - node_base]
     n2 = node_index_map[face_nodes[1] - node_base]
     n3 = node_index_map[face_nodes[2] - node_base]
     n4 = node_index_map[face_nodes[3] - node_base] if len >= 4 else n3
     write(n1, n2, n3, n4)
```

**关键设计**：
- `_node_base` 标记：区分 1-based（grid 字典）和 0-based（Unstructured_Grid）
- 三角形面转换：重复第 4 个节点 `(n1, n2, n3, n3)`

---

## 5. GUI 集成设计

### 5.1 用户交互流程

```
┌─────────────────────────────────────────────────────────────┐
│                        用户操作                              │
│                                                             │
│  1. 点击 "导出网格" 按钮 (Ctrl+E)                           │
│     ↓                                                       │
│  2. 弹出 QFileDialog.getSaveFileName()                      │
│     文件过滤器: "*.vtk *.stl *.obj *.msh *.ply *.plt"       │
│     ↓                                                       │
│  3. 用户选择保存路径，输入文件名（如 mesh.plt）              │
│     ↓                                                       │
│  4. 点击"保存"                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     mesh_operations.py                       │
│                                                             │
│  export_mesh() 方法:                                        │
│    1. 检查 self.gui.current_mesh 是否存在                   │
│    2. 弹出文件保存对话框                                    │
│    3. 检测文件扩展名:                                       │
│       IF .plt:                                              │
│         调用 _export_plt_file(file_path)                    │
│       ELSE:                                                 │
│         调用其他格式导出方法                                │
│                                                             │
│  _export_plt_file(file_path) 方法:                          │
│    1. 判断网格数据类型:                                     │
│       IF isinstance(mesh_data, dict):                       │
│         调用 export_mesh_to_plt(grid=mesh_data, ...)        │
│       ELIF hasattr(mesh_data, 'export_to_plt'):             │
│         调用 mesh_data.export_to_plt(output_path, ...)      │
│       ELSE:                                                 │
│         抛出 ValueError("不支持的网格对象格式")             │
│    2. 显示成功/失败消息                                     │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 GUI 组件关联

| 组件 | 文件 | 职责 |
|------|------|------|
| 导出按钮 | `gui/ribbon_widget.py:207` | 定义"导出网格"按钮及其图标、快捷键 |
| 回调绑定 | `gui/ribbon_widget.py:431` | 将按钮点击事件连接到 `mesh_operations.export_mesh()` |
| 导出逻辑 | `gui/mesh_operations.py:125-188` | 文件对话框、格式检测、调用底层导出函数 |

### 5.3 错误处理

```python
def export_mesh(self):
    if not self.gui.current_mesh:
        QMessageBox.warning(self.gui, "警告", "没有可导出的网格")
        return

def _export_plt_file(self, file_path):
    try:
        # ... 导出逻辑
    except Exception as e:
        QMessageBox.critical(self.gui, "错误", f"导出失败: {str(e)}")
```

---

## 6. 函数接口规范

### 6.1 公开 API

#### `export_mesh_to_plt()`

```python
def export_mesh_to_plt(
    grid: Optional[Dict] = None,
    nodes: Optional[np.ndarray] = None,
    faces: Optional[List[Dict]] = None,
    simplices: Optional[np.ndarray] = None,
    edge_index: Optional[np.ndarray] = None,
    scalars: Optional[Dict[str, np.ndarray]] = None,
    output_path: str = "output.plt",
    title: str = "Mesh Data",
) -> str:
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `grid` | Dict | 否 | PyMeshGen 网格数据字典 |
| `nodes` | np.ndarray | 条件 | 节点坐标 `[num_nodes, 2/3]` |
| `faces` | List[Dict] | 条件 | 面列表，每个面含 `"nodes"` 字段 |
| `simplices` | np.ndarray | 条件 | 单元连接 `[num_cells, 3/4]` |
| `edge_index` | np.ndarray | 条件 | 边索引 `[2, num_edges]` |
| `scalars` | Dict[str, np.ndarray] | 否 | 标量场字典 |
| `output_path` | str | 是 | 输出文件路径 |
| `title` | str | 否 | 文件标题 |

**调用约束**：
- 必须提供 `grid`，**或**同时提供 `nodes` 和 `faces/edge_index`
- 2D 网格：`nodes.shape[1] == 2`
- 3D 网格：`nodes.shape[1] == 3`

#### `export_unstructured_grid_to_plt()`

```python
def export_unstructured_grid_to_plt(
    unstructured_grid,
    output_path: str,
    title: str = "Mesh Data",
    scalars: Optional[Dict[str, np.ndarray]] = None,
) -> str:
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `unstructured_grid` | Unstructured_Grid | 是 | 网格对象 |
| `output_path` | str | 是 | 输出文件路径 |
| `title` | str | 否 | 文件标题 |
| `scalars` | Dict[str, np.ndarray] | 否 | 标量场字典 |

#### `export_from_cas()`

```python
def export_from_cas(
    cas_file: str,
    output_path: str,
    scalars: Optional[Dict[str, np.ndarray]] = None,
) -> str:
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `cas_file` | str | 是 | Fluent `.cas` 文件路径 |
| `output_path` | str | 是 | 输出 PLT 文件路径 |
| `scalars` | Dict[str, np.ndarray] | 否 | 标量场字典 |

### 6.2 内部函数

| 函数名 | 职责 | 输入 | 输出 |
|--------|------|------|------|
| `_extract_from_grid()` | 从网格字典提取数据 | `grid: Dict` | `(nodes, faces, simplices, edge_index)` |
| `_extract_simplices_from_grid()` | 提取单元连接 | `grid, all_faces` | `simplices_list: List` |
| `_extract_simplices_from_unstructured_grid()` | 从对象提取单元 | `unstructured_grid` | `simplices: np.ndarray` |
| `_build_edge_index_from_simplices()` | 构建边索引 | `simplices` | `edge_index: np.ndarray` |
| `_extract_boundary_zones_from_unstructured_grid()` | 从对象提取边界 | `unstructured_grid` | `boundary_zones: List[Dict]` |
| `_extract_boundary_zones_from_grid_dict()` | 从字典提取边界 | `grid: Dict` | `boundary_zones: List[Dict]` |
| `_write_plt_file()` | 写入主区域 | 节点、单元、标量等 | 写入文件 |
| `_write_node_data()` | 写入节点/标量 | 文件句柄、节点、标量 | 写入文件 |
| `_build_topology()` | 构建拓扑关系 | 面、边、单元 | `(face_to_nodes, left_cell, right_cell, total)` |
| `_build_edge_to_face_map()` | 构建边-面映射 | `edge_index, face_to_nodes` | `edge_to_face: Dict` |
| `_assign_cells_to_faces()` | 分配单元到面 | `simplices, edge_to_face` | 更新 `left_cell, right_cell` |
| `_write_topology()` | 写入拓扑数据 | 文件句柄、拓扑数据 | 写入文件 |
| `_write_boundary_zones()` | 写入边界区域 | 节点、边界数据 | 追加到文件 |
| `_write_single_boundary_zone()` | 写入单个边界 Zone | 文件句柄、区域数据 | 写入文件 |
| `_collect_boundary_nodes()` | 收集边界节点 | `faces_data, node_base` | `boundary_nodes_list: List[int]` |
| `_write_boundary_node_coords()` | 写入边界节点坐标 | 文件句柄、节点数据 | 写入文件 |
| `_write_boundary_face_connectivity()` | 写入边界单元连接 | 文件句柄、面数据 | 写入文件 |

---

## 7. 数据结构定义

### 7.1 PyMeshGen 网格字典

```python
grid: Dict = {
    "nodes": [
        {"coords": (x, y, z)},  # 或 [x, y, z]
        ...
    ],
    "cells": [
        {"nodes": [n1, n2, n3, n4]},  # 1-based 索引
        ...
    ],
    "zones": {
        "zone_name": {
            "type": "faces",
            "bc_type": "wall" | "internal" | "inflow" | ...,
            "data": [
                {"nodes": [n1, n2, n3]},  # 1-based 索引
                ...
            ]
        }
    },
    "dimension": 2 | 3
}
```

### 7.2 Unstructured_Grid 对象

```python
class Unstructured_Grid:
    node_coords: List[List[float]]  # 节点坐标列表
    cell_container: List[Cell]       # 单元容器
    dimension: int                   # 网格维度
    boundary_info: Dict[str, Dict]   # 边界区域信息
    parts_info: Dict[str, Dict]      # 部件信息（备选）
```

### 7.3 边界区域数据格式

```python
boundary_zone: Dict = {
    'part_name': str,           # 区域名称（如 "wall"）
    'bc_type': str,             # 边界条件类型
    'data': List[Dict | List],  # 面数据
    '_node_base': int           # 0（0-based）或 1（1-based）
}
```

---

## 8. 测试策略

### 8.1 测试覆盖矩阵

| 测试类 | 测试用例 | 覆盖场景 |
|--------|----------|----------|
| TestMeshToPLT | `test_export_simple_2d_mesh` | 2D 三角形网格 + 标量场 |
| TestMeshToPLT | `test_export_without_scalars` | 无标量场导出 |
| TestMeshToPLT | `test_export_with_multiple_scalars` | 多标量场导出 |
| TestMeshToPLT | `test_3d_mesh_export` | 3D 四面体 + 边界区域 |
| TestPLTFileFormat | `test_plt_file_format_valid` | 文件格式验证 |
| TestErrorHandling | `test_invalid_input_raises_error` | 异常处理 |
| TestExtractFromGrid | `test_extract_simple_grid` | 简单网格提取 |
| TestExtractFromGrid | `test_extract_mixed_faces` | 混合面类型提取 |
| TestExtractFromGrid | `test_export_from_grid_dict` | 网格字典导出 |
| TestRealFileExport | `test_export_from_cas_file` | 2D Fluent 文件导出 |
| TestRealFileExport | `test_export_semisphere_3d_cas` | 3D 混合网格导出 |
| TestGUIExportWorkflow | `test_gui_export_2d_mesh` | GUI 流程 2D 导出 |
| TestGUIExportWorkflow | `test_gui_export_3d_mesh` | GUI 流程 3D 导出 |
| TestGUIExportWorkflow | `test_gui_export_cgns_mesh` | GUI 流程 CGNS 导出 |

### 8.2 测试数据

| 文件 | 类型 | 节点数 | 单元数 |
|------|------|--------|--------|
| `naca0012-tri-coarse.cas` | 2D 三角形 | 1,362 | ~2,600 |
| `semisphere-hybrid.cas` | 3D 混合网格 | 6,291 | ~30,000 |
| `grid_chnt-1_coarse.cgns` | 3D CGNS | 221,032 | 349,053 |

### 8.3 验证点

1. **文件存在性**: `output_path.exists()`
2. **文件头格式**: 包含 `TITLE`, `VARIABLES`, `ZONE`
3. **ZoneType**: 2D → `FEPolygon`，3D → `FEPolyhedron`
4. **节点/面/单元数**: 与实际数据一致
5. **变量名**: 2D → `"X", "Y"`，3D → `"X", "Y", "Z"`
6. **边界区域**: 3D 网格包含 `FEQUADRILATERAL` Zone
7. **文件大小**: 大于预期阈值

---

## 9. 性能优化

### 9.1 大数据集处理

| 优化点 | 策略 |
|--------|------|
| 节点坐标写入 | 每行 5 个数值，减少 I/O 次数 |
| 拓扑构建 | 使用哈希表 `edge_to_face`，O(1) 查找 |
| 边界节点映射 | 预计算 `node_index_map`，避免重复查找 |

### 9.2 内存管理

- **流式写入**: 不构建完整文件内容字符串，直接逐块写入文件
- **按需转换**: `simplices` 和 `edge_index` 仅在需要时转换为 numpy 数组

### 9.3 实测性能

| 测试文件 | 节点数 | 导出时间 | 文件大小 |
|----------|--------|----------|----------|
| naca0012-tri-coarse | 1,362 | <0.1s | 151 KB |
| semisphere-hybrid | 6,291 | <0.3s | 1,016 KB |
| grid_chnt-1_coarse | 221,032 | ~37s | 56,828 KB |

---

## 10. 已知限制与未来扩展

### 10.1 当前限制

| 限制项 | 说明 |
|--------|------|
| 高阶单元 | 仅支持线性单元（三角形、四边形、四面体等） |
| 多 Zone 主区域 | 不支持输出多个主区域 Zone |
| 标量场插值 | 标量数据必须与节点一一对应，不支持单元中心数据 |
| 并行写入 | 文件写入为单线程，大数据集较慢 |

### 10.2 未来扩展方向

1. **高阶单元支持**: 输出 `FEQUADRILATERAL` / `FETETRAHEDRON` 等高阶 ZoneType
2. **多 Block 输出**: 支持将不同区域输出为独立 Zone
3. **时间序列**: 支持瞬态数据的多时间步输出
4. **并行 I/O**: 使用多线程/多进程加速大数据集写入
5. **二进制格式**: 支持 Tecplot 二进制格式（比文本格式小 5-10 倍）

---

## 11. 故障排除

### 11.1 常见错误

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| `Elements or E must be greater than 0` | 单元连接未正确提取 | 检查 `_extract_simplices_*` 函数逻辑 |
| `节点数据为空` | `nodes` 参数为 None 或空数组 | 确保网格数据已正确加载 |
| `AttributeError: 'Unstructured_Grid' object has no attribute 'get'` | 函数参数类型错误 | 使用 `_extract_boundary_zones_from_unstructured_grid` 而非 `_extract_boundary_zones_from_grid_dict` |

### 11.2 调试技巧

1. **检查 PLT 文件头部**: 确认 `Nodes`, `Faces`, `Elements` 值是否符合预期
2. **验证拓扑数据**: 检查 `LeftCell` / `RightCell` 是否覆盖所有面
3. **比对节点数**: 边界区域节点数应 ≤ 主区域节点数

---

## 附录 A: 代码文件清单

| 文件路径 | 行数 | 职责 |
|----------|------|------|
| `fileIO/tecplot_io.py` | ~807 | PLT 文件导出核心实现 |
| `data_structure/unstructured_grid.py` | ~850 | `Unstructured_Grid` 类定义 |
| `gui/ribbon_widget.py` | ~450 | GUI 导出按钮定义 |
| `gui/mesh_operations.py` | ~190 | 导出流程控制 |
| `unittests/test_mesh_to_plt.py` | ~814 | 单元测试 |

## 附录 B: 版本历史

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0 | - | 初始版本，支持 2D/3D 网格导出 |
| v1.1 | - | 重构 `_build_topology`，提取子函数 |
| v1.2 | - | 重构 `_write_boundary_zones`，拆分边界写入逻辑 |
| v1.3 | - | 修复 CGNS 文件导出，支持 section 数据结构 |
| v1.4 | 2026-04-07 | 添加 CGNS GUI 导出测试，代码优化 |
