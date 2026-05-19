# sfmesh 基础几何体曲面网格生成模块详细设计文档

---

## 目录

- [1. 概述](#1-概述)
- [2. 模块架构](#2-模块架构)
- [3. 核心数据结构](#3-核心数据结构)
- [4. 核心算法 — 阵面推进法（AFM）](#4-核心算法--阵面推进法afm)
- [5. 2D 阵面推进流水线](#5-2d-阵面推进流水线)
- [6. 基础几何体网格生成](#6-基础几何体网格生成)
- [7. CAD 几何驱动网格生成](#7-cad-几何驱动网格生成)
- [8. 曲面几何操作](#8-曲面几何操作)
- [9. 尺寸场设计](#9-尺寸场设计)
- [10. 几何工具模块](#10-几何工具模块)
- [11. 网格质量评估](#11-网格质量评估)
- [12. 对外接口与使用示例](#12-对外接口与使用示例)
- [13. 已知限制与扩展方向](#13-已知限制与扩展方向)
- [附录 A: 文件清单](#附录-a-文件清单)

---

## 1. 概述

### 1.1 模块名称

`sfmesh` — Surface Mesh Generation Module（曲面网格生成模块）

### 1.2 功能定位

`sfmesh` 模块负责从三维几何模型生成三角形曲面网格，服务于 CFD/FEA 前处理流程。模块支持两条独立的网格生成路径：

| 路径 | 入口 | 几何来源 | 网格方法 |
|------|------|----------|----------|
| CAD 文件驱动 | `generate_surface_mesh_from_file()` | IGES/STEP 文件 | 3D AFM（`SurfaceMeshGenerator`） |
| 基础几何体驱动 | `generate_cube_mesh()` / `generate_cylinder_mesh()` / `generate_rectangle_mesh()` | 参数化定义 | 2D AFM 流水线（`pipeline_2d`） |

两条路径共享底层数据结构（`NodeElement3D`、`SurfaceTriangle`、`SurfaceFront`）和质量评估工具（`mesh_quality`），但采用不同的阵面推进策略：

- **3D AFM**：直接在参数曲面上推进阵面，节点通过 `SurfaceGeometry` 投影到几何表面，适用于任意复杂曲面。
- **2D AFM 流水线**：将平面面片投影到 2D 空间，复用已有的 `Adfront2` 二维阵面推进引擎和 `QuadtreeSizing` 尺寸场，再映射回 3D 坐标。适用于基础几何体的平面和可展开曲面。

### 1.3 设计目标

1. **模块化**：按职责拆分为独立文件，OCC 操作、计算几何、AFM 流水线、公开 API 各司其职。
2. **复用已有基础设施**：2D 路径复用 `data_structure.front2d.Front`、`adfront2.Adfont2`、`meshsize.QuadtreeSizing`、`optimize.edge_swap_delaunay` 等模块，避免重复实现。
3. **跨面节点一致性**：通过坐标哈希去重确保相邻面共享边界的节点一致。
4. **质量保障**：统一的质量因子公式 `quality = 4√3 · area / (a² + b² + c²)`，值域 [0, 1]，等边三角形为 1.0。

---

## 2. 模块架构

### 2.1 文件结构与职责划分

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | 32 | 包初始化，导出公开 API |
| `surface_front.py` | 449 | 核心数据结构：`NodeElement3D`、`SurfaceTriangle`、`SurfaceFront` |
| `surface_geometry.py` | 377 | OCC 曲面几何操作：点投影、法向量、曲率、理想点计算 |
| `sizing_field.py` | 384 | 尺寸场控制：均匀、曲率自适应、梯度限制、特征邻近 |
| `mesh_quality.py` | 429 | 网格质量评估与几何相交检测 |
| `surface_mesh.py` | 745 | CAD 文件驱动的 3D AFM 网格生成器 |
| `primitives.py` | 1303 | 基础几何体网格生成入口：长方体、圆柱体、矩形、球体、椭球体 |
| `occ_utils.py` | 381 | OCC 辅助函数：面提取、包围盒、面分类 |
| `geom_utils.py` | 90 | 纯数值计算几何：2D 投影、线段相交、点在三角形内 |
| `pipeline_2d.py` | 657 | 2D AFM 流水线：离散化、阵面创建、AFM 运行、坐标映射、统一圆柱网格 |

### 2.2 模块依赖关系图

```
┌─────────────────────────────────────────────────────────┐
│                    __init__.py (公开 API)                │
└───────────┬──────────────────────────┬──────────────────┘
            │                          │
            ▼                          ▼
┌───────────────────────┐   ┌──────────────────────────┐
│   surface_mesh.py     │   │     primitives.py         │
│   (CAD 3D AFM)        │   │   (基础几何体入口)         │
└───┬───┬───┬───┬───────┘   └──┬───┬───┬───────────────┘
    │   │   │   │              │   │   │
    ▼   │   │   │              ▼   │   │
┌───────┤   │   │        ┌─────┤   │   │
│surface│   │   │        │occ_ │   │   │
│_front │   │   │        │utils│   │   │
│       │   │   │        └─────┘   │   │
└───────┘   │   │                  │   │
            ▼   │                  ▼   │
    ┌───────────┤          ┌───────────┤
    │surface_   │          │geom_      │
    │geometry   │          │utils      │
    └───────────┘          └───────────┘
            │                      │
            ▼                      │
    ┌───────────┐                  │
    │sizing_    │                  │
    │field      │                  │
    └───────────┘                  │
            │                      │
            ▼                      │
    ┌───────────┐                  │
    │mesh_      │                  │
    │quality    │                  │
    └───────────┘                  │
                                   ▼
                           ┌──────────────┐
                           │pipeline_2d   │
                           └──┬───┬───┬───┘
                              │   │   │
                              ▼   │   │
                     ┌────────────┤   │
                     │data_       │   │
                     │structure/  │   │
                     │front2d     │   │
                     └────────────┘   │
                                      ▼
                             ┌────────────────┐
                             │adfront2 +      │
                             │meshsize +      │
                             │optimize        │
                             └────────────────┘
```

### 2.3 外部依赖

| 外部模块 | 用途 | 被引用文件 |
|----------|------|-----------|
| `OCC.Core.*` (OpenCASCADE) | BRep 几何内核、曲面求值、面分类 | `surface_geometry`, `occ_utils`, `primitives`, `surface_mesh` |
| `data_structure.front2d.Front` | 2D 阵面数据结构 | `pipeline_2d` |
| `data_structure.basic_elements.NodeElementALM` | 2D 节点元素 | `pipeline_2d` |
| `data_structure.rtree_space` | R-tree 空间索引 | `primitives`, `surface_mesh` |
| `meshsize.meshsize.QuadtreeSizing` | 四叉树尺寸场 | `pipeline_2d` |
| `adfront2.adfront2.Adfont2` | 二维阵面推进引擎 | `pipeline_2d` |
| `optimize.optimize` | 边交换（含边界边保护）、Laplacian 光滑 | `pipeline_2d` |
| `fileIO.occ_loader` | OCC 初始化 | `surface_geometry`, `occ_utils`, `primitives` |
| `fileIO.geometry_io` | IGES/STEP 文件读取 | `surface_mesh` |
| `utils.message` | 日志输出 | `primitives`, `surface_mesh` |

---

## 3. 核心数据结构

### 3.1 NodeElement3D

**文件位置**: `sfmesh/surface_front.py`

**功能描述**: 三维网格节点，存储坐标、法向量、参数坐标等信息。使用 `__slots__` 优化内存。

| 成员 | 类型 | 说明 |
|------|------|------|
| `coords` | `Tuple[float, float, float]` | 三维坐标 |
| `idx` | `int` | 节点索引 |
| `bc_type` | `str` | 边界条件类型 |
| `part_name` | `str` | 所属部件名称 |
| `surface` | `TopoDS_Face` | 所在 OCC 面 |
| `uv_params` | `Tuple[float, float]` | 参数坐标 (u, v) |
| `hash` | `int` | 坐标哈希值（8 位小数精度），用于节点去重 |
| `normal` | `Tuple` | 表面法向量 |
| `bbox` | `Tuple` | 退化包围盒（点） |

**设计说明**:
- `hash` 由 `hash(tuple(f"{0.0 if round(c, 8) == 0 else round(c, 8):.8f}" for c in coords))` 计算。先对坐标做 `round(c, 8)` 归一化精度，再将 `-0.0` 统一为 `0.0`，避免 `math.sin(2π) = -2.4e-16` 格式化为 `"-0.00000000"` 导致的哈希不一致。
- 支持 `__eq__` 和 `__hash__`，可直接用于 `set` 和 `dict` 进行节点去重。

### 3.2 SurfaceTriangle

**文件位置**: `sfmesh/surface_front.py`

**功能描述**: 三角形网格元素，构造时即时计算法向量、面积、质量因子和包围盒。

| 成员 | 类型 | 说明 |
|------|------|------|
| `nodes` | `List[NodeElement3D]` | 三个节点 |
| `node_ids` | `Tuple[int, int, int]` | 节点索引 |
| `idx` | `int` | 三角形索引 |
| `surface` | `TopoDS_Face` | 所在 OCC 面 |
| `normal` | `np.ndarray` | 单位法向量（叉积计算） |
| `area` | `float` | 面积（叉积模的一半） |
| `quality` | `float` | 形状质量因子 [0, 1] |
| `bbox` | `Tuple` | 轴对齐包围盒 |

**质量因子公式**:

```
quality = 4√3 · area / (a² + b² + c²)
```

其中 `a`, `b`, `c` 为三角形三条边长。等边三角形 quality = 1.0，退化三角形 quality → 0。

### 3.3 SurfaceFront

**文件位置**: `sfmesh/surface_front.py`

**功能描述**: 阵面推进法的阵面边，支持优先队列排序。构造时计算几何属性（长度、中心、方向、法向量、切向法向量）。

| 成员 | 类型 | 说明 |
|------|------|------|
| `node_elems` | `List[NodeElement3D]` | 两个端点 |
| `center` | `Tuple` | 边中点坐标 |
| `length` | `float` | 边长 |
| `direction` | `Tuple` | 单位方向向量 |
| `normal` | `Tuple` | 表面法向量（两端点法向平均） |
| `tangent_normal` | `Tuple` | 切平面内的推进方向（`cross(direction, normal)`） |
| `al` | `float` | 搜索半径系数（初始 3.0，失败时 ×1.2） |
| `priority` | `int` | 优先级标志 |

**排序规则**: `__lt__` 方法实现——高优先级优先，同优先级短边优先。这确保最短的阵面最先被处理，生成更均匀的网格。

**设计说明**:
- `tangent_normal` 是阵面推进的关键——它定义了新节点应出现的方向（垂直于边且在切平面内）。
- 对于 CCW 排列的边界，左手法向量指向区域内部，确保 AFM 向内填充。

### 3.4 PrimitiveMeshResult

**文件位置**: `sfmesh/primitives.py`

**功能描述**: 基础几何体网格生成的结果容器。

| 成员 | 类型 | 说明 |
|------|------|------|
| `triangles` | `List[SurfaceTriangle]` | 所有三角形 |
| `nodes` | `List[NodeElement3D]` | 所有节点（已去重） |
| `face_map` | `Dict[int, List[SurfaceTriangle]]` | 面索引 → 三角形列表映射 |
| `face_types` | `Dict[int, str]` | 面索引 → 类型名称映射 |
| `num_faces` | `int` | 面的总数 |

---

## 4. 核心算法 — 阵面推进法（AFM）

### 4.1 算法总览

阵面推进法（Advancing Front Method）从几何边界出发，逐个生成三角形，向区域内部推进，直到整个面被填满。

```
┌──────────────────────────────┐
│  提取边界，创建初始阵面        │
│  推入最小堆                   │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  弹出最短阵面                 │◄─────────┐
└──────────────┬───────────────┘          │
               ▼                          │
┌──────────────────────────────┐          │
│  计算局部间距 spacing         │          │
│  计算理想点 ideal_point       │          │
└──────────────┬───────────────┘          │
               ▼                          │
┌──────────────────────────────┐          │
│  R-tree 搜索候选节点          │          │
│  radius = al × spacing       │          │
└──────────────┬───────────────┘          │
               ▼                          │
┌──────────────────────────────┐          │
│  评分 + 过滤                  │          │
│  - 侧向检查                   │          │
│  - 退化三角形排除             │          │
│  - 重复三角形排除             │          │
│  - 相交检测                   │          │
└──────────────┬───────────────┘          │
               ▼                          │
         选中节点?                         │
          ┌──┴──┐                         │
          │ No  │ Yes                     │
          ▼     ▼                         │
    al ×= 1.2  创建三角形                  │
    重新入堆   创建子阵面                   │
          │    更新空间索引                 │
          │         │                     │
          └─────────┴─────────────────────┘
```

### 4.2 边界提取与初始阵面创建

**文件位置**: `sfmesh/surface_front.py` — `create_initial_fronts_from_surface()`

**算法步骤**:

1. 使用 `TopExp_Explorer` 遍历 OCC 面的 WIRE 和 EDGE
2. 对每条 EDGE，通过 `BRep_Tool.Curve` 提取参数曲线
3. 按 `spacing` 均匀离散化曲线：`num_points = max(2, int((last - first) / spacing))`
4. 将采样点投影到曲面获取 UV 和法向量
5. 按坐标字符串 `f"{c:.6f}"` 去重节点
6. 在相邻节点间创建 `SurfaceFront` 对象

### 4.3 理想点计算

**文件位置**: `sfmesh/surface_geometry.py` — `SurfaceGeometry.compute_ideal_point_on_surface()`

理想点是新三角形第三个顶点的最佳候选位置，位于阵面法向的切平面内，距离为 `d`：

```
ideal_point_3d = center + d × tangent_normal
```

其中 `d = quality_factor × √(spacing × front_length)`，夹在 `[0.5×spacing, 2.0×spacing]` 范围内。

由于 `ideal_point_3d` 一般不在曲面上，需要迭代投影回曲面：

1. 计算 3D 理想点
2. 投影到参数空间 (u, v)
3. 在 (u, v) 处求值 3D 坐标
4. 若收敛（距离 < 容差），返回；否则重复（最多 10 次）

### 4.4 候选节点搜索与评分

**搜索**: 以阵面中心为查询点，`al × spacing` 为半径，通过 R-tree 查找附近的已有节点。

**评分算法** (`_select_best_node_afm`):

1. **过滤**:
   - 排除阵面自身的两个端点
   - 排除与理想点异侧的候选（侧向检查）
   - 排除退化三角形（高度 < 边长的 5%）
   - 排除已存在的重复三角形（frozenset 哈希）

2. **评分**: `score = quality × distance_factor`
   - `distance_factor = 1.0 / (1.0 + dist / front_len)`
   - 距离理想点越近，得分越高

3. **边界节点优先**: 位于边界上的节点获得额外优先级，促进阵面去重。

4. **回退到理想点**: 若无合适已有节点，则创建理想点作为新节点（quality 折扣 0.8）。

### 4.5 相交检测

**文件位置**: `sfmesh/primitives.py` — `_check_intersection_afm()`

RTree 加速的相交检测流程：

1. 计算候选三角形的 AABB，查询 R-tree 获取重叠的已有三角形
2. 跳过共享 2+ 顶点的合法相邻三角形
3. 对每条新边执行 `check_edge_triangle_intersection()`（射线-平面相交 + 重心坐标判定）
4. 对共享 1 顶点的三角形执行共面重叠检测（`_are_coplanar_triangles_overlapping`）

### 4.6 阵面更新与去重

创建三角形后：

1. **移除已消耗阵面**: 从 `front_hash_set` 中删除
2. **创建子阵面**: 新三角形产生两条新边，各创建一个 `SurfaceFront`
3. **去重**: 若新阵面的哈希已存在于 `front_hash_set`，说明该边已被另一三角形占用（闭合），移除旧阵面而非添加新的

```python
for new_front in [new_front1, new_front2]:
    if new_front.hash in front_hash_set:
        front_hash_set.discard(new_front.hash)  # 边闭合，移除
    else:
        front_hash_set.add(new_front.hash)
        heapq.heappush(front_list, new_front)
```

### 4.7 搜索半径自适应扩展

当无法为阵面找到合适节点时：

```python
base_front.al *= 1.2
if base_front.al < 20.0:
    heapq.heappush(front_list, base_front)
```

`al` 从初始值 3.0 开始，每次失败扩大 1.2 倍，上限 20.0。这允许在网格稀疏区域逐步扩大搜索范围，同时避免无限循环。

---

## 5. 2D 阵面推进流水线

### 5.1 设计目的

**文件位置**: `sfmesh/pipeline_2d.py`

对于基础几何体的平面面片和可展开曲面，直接在 2D 空间运行 AFM 比 3D AFM 更高效、更稳定。2D 流水线复用项目已有的 `Adfront2` 引擎和 `QuadtreeSizing` 尺寸场，避免重复实现。

**核心优势**:
- 复用成熟基础设施（`Adfront2`、`QuadtreeSizing`、`edge_swap_delaunay`、`laplacian_smooth`）
- 2D 空间无投影误差，网格质量更高
- 边交换 + Laplacian 光滑后处理进一步提升质量

### 5.2 流水线架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  边界离散化   │────▶│ Front 创建    │────▶│ QuadtreeSizing│
│  (2D 点列表)  │     │ (NodeElementALM)│   │  (尺寸场)     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  2D→3D 映射  │◀────│ 曲面投影     │◀────│ Adfront2     │
│  (坐标变换)   │     │ (节点拉回曲面) │     │ (AFM 引擎)   │
└──────────────┘     └──────┬───────┘     └──────────────┘
        ▲                   │
        │                   ▼
┌───────┴───────┐   ┌──────────────┐
│ edge_swap_    │   │ Laplacian    │
│ delaunay      │   │ 光滑 (3 次)   │
│ (边交换优化)   │   │              │
└───────────────┘   └──────────────┘
```

### 5.3 边界离散化

**`_discretize_edge_2d(start_2d, end_2d, spacing)`**

将 2D 线段均匀离散化，段数 `n = max(2, round(length / spacing) + 1)`，返回 `n + 1` 个点。

**`_discretize_circle_2d(center, radius, n_segments)`**

将圆离散化为 `n_segments` 个点，CCW 排列，用于圆盘面边界。

**`_create_fronts_from_2d_edges(edge_points_2d, face_name)`**

从 2D 边界点列表创建 `Front` 对象。每个 `Front` 包含两个 `NodeElementALM` 节点（Z=0），边界条件为 `"BCWall"`。

### 5.4 QuadtreeSizing + Adfront2 集成

**`_run_afm_2d_pipeline(all_fronts, spacing, face_size)`**

核心 2D AFM 流水线，关键设计：

1. **`_PaddedSizingField`**: 继承 `QuadtreeSizing`，扩展包围盒 `max(face_size × 0.5, 5 × spacing)`，并为 `spacing_at()` 提供安全回退（超出范围时返回 `global_spacing`），避免边界附近的 `ValueError`。

2. **`Adfront2` 配置**: 使用 `_ParamObj` 提供 `debug_level=0` 和 `mesh_type=1`。

3. **主循环**: 最多 `max(50000, (face_size / spacing)² × 15)` 步，每步执行 `add_new_point` → `search_candidates` → `select_point` → `update_data`。

4. **后处理**: `edge_swap_delaunay` 改善 Delaunay 性，`laplacian_smooth(num_iter=3)` 平滑节点位置。

### 5.5 后处理优化

**`_post_process_surface_mesh(triangles, nodes, num_smooth_iter, fixed_node_ids, node_projectors)`**

对合并后的曲面网格执行三步后处理：

1. **边交换** (`edge_swap_delaunay`): 检查每条内部边，若交换后可改善局部 Delaunay 性则执行交换。受 `boundary_nodes_list` 保护：若公共边两端点均为边界节点则跳过交换；若 `c` 或 `d` 中有边界节点则不创建新边界边。

2. **Laplacian 光滑** (`laplacian_smooth`): 将每个内部节点移至其邻域重心，执行 3 次迭代。`boundary_set` 中的节点（拓扑边界 + `fixed_node_ids` 几何边界）保持不动。

3. **曲面投影** (`node_projectors`): 光滑后，对每个非边界节点调用其对应的投影函数，将节点拉回几何面。投影函数在合并阶段按面分配：

| 面 | 投影函数 | 说明 |
|----|----------|------|
| 底面 (z=z₀) | `(x, y, z) → (x, y, z₀)` | 投影到平面 |
| 顶面 (z=z₁) | `(x, y, z) → (x, y, z₁)` | 投影到平面 |
| 侧面 (圆柱) | `(x, y, z) → (cx + r·dx/d, cy + r·dy/d, z)` | 投影到柱面，`d = √(dx²+dy²)` |

投影确保 Laplacian 光滑不会导致节点漂离几何面，保持几何保真度。

### 5.6 2D→3D 坐标映射

**`_unstr_grid_to_3d(unstr_grid, map_to_3d, normal_3d, normal_func)`**

将 `Adfront2` 生成的 2D `Unstructured_Grid` 转换为 3D 网格：

1. 遍历所有节点，通过 `map_to_3d(x2d, y2d)` 映射到 3D 坐标
2. 法向量可通过 `normal_func(x2d, y2d)` 动态计算（如圆柱面径向法向），或使用固定 `normal_3d`
3. 遍历所有单元格，创建 `SurfaceTriangle`

**`_mesh_face_2d_pipeline(corners_3d, spacing, face_name)`**

平面四边形面的完整 2D 流水线入口：

1. 计算面法向量，确定常量轴（法向最大分量方向）
2. 将 4 个角点投影到活跃的 2D 轴
3. 离散化 4 条边 → 创建 Front → 运行 AFM
4. 通过逆投影映射回 3D

### 5.7 圆柱侧面展开

**`_mesh_lateral_cylinder_2d(base_center, radius, height, spacing, face_name)`**

圆柱侧面是可展开曲面，展开策略：

1. 将侧面展开为 `(s, z)` 平面上的矩形 `[0, L] × [z0, z1]`，其中 `L = 2πr` 为周长
2. 在矩形上运行 2D AFM
3. 映射回 3D 圆柱坐标：

```
θ = s / r
x = cx + r · cos(θ)
y = cy + r · sin(θ)
z = z
```

法向量为径向向外：`(cos(θ), sin(θ), 0)`

**设计说明**: 展开矩形的左右边界 `s=0` 和 `s=L` 代表同一条物理母线，映射回 3D 后坐标相同，由调用方的节点哈希去重自动处理。

---

## 6. 基础几何体网格生成

### 6.1 逐面策略

**文件位置**: `sfmesh/primitives.py`

基础几何体采用逐面独立网格生成 + 跨面节点去重的策略：

1. 定义几何体的面（角点、边界）
2. 对每个面独立生成 2D AFM 网格
3. 合并所有面的结果，通过坐标哈希去重共享边界的节点

### 6.2 长方体网格生成

**`generate_cube_mesh(corner1, corner2, spacing, output_vtk)`**

定义 8 个顶点和 6 个面，CCW 绕序确保 Front 左手法向量指向内部：

```python
v = [(x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0),  # 底面
     (x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (x0,y1,z1)]   # 顶面

face_defs = [
    ("bottom", [v[0], v[1], v[2], v[3]]),  # CCW in XY
    ("top",    [v[4], v[5], v[6], v[7]]),  # CCW in XY
    ("front",  [v[0], v[1], v[5], v[4]]),  # CCW in XZ
    ("back",   [v[3], v[2], v[6], v[7]]),  # CCW in XZ
    ("right",  [v[1], v[2], v[6], v[5]]),  # CCW in YZ
    ("left",   [v[0], v[3], v[7], v[4]]),  # CCW in YZ
]
```

每个面通过 `_mesh_face_2d_pipeline` 生成网格。

### 6.3 圆柱体网格生成

**`generate_cylinder_mesh(base_center, radius, height, spacing, output_vtk)`**

圆柱体采用统一网格生成策略，通过 `_mesh_cylinder_unified()` 确保端面和柱面共享边界节点。

**`_mesh_cylinder_unified(base_center, radius, height, spacing)`**（`pipeline_2d.py`）

统一圆柱网格生成的核心函数，解决逐面独立生成时边界节点不一致的问题。

**算法流程**:

1. **统一离散化圆边界**: `n_segments = max(6, round(2πr / spacing))`，生成圆周上的点序列 `circle_pts`
2. **端面网格**: 底面和顶面共用同一组 `circle_pts_2d` 作为边界，通过 `_mesh_disk_2d(boundary_pts_2d=...)` 传入
3. **侧面网格**: 将圆边界转换为展开坐标 `(s, z)`，添加接缝绕回点 `(L, z)` 保证闭合，通过 `_mesh_lateral_cylinder_2d(boundary_pts_2d=...)` 传入
4. **节点合并**: 遍历三个面的节点，通过坐标哈希去重，记录每个节点所属面（用于曲面投影）
5. **几何边界识别**: 对每个面独立统计边出现次数，仅出现 1 次的边为几何边界边，其端点加入 `geometric_boundary_ids`
6. **后处理**: 调用 `_post_process_surface_mesh`，传入 `fixed_node_ids`（几何边界节点固定）和 `node_projectors`（光滑后投影回曲面）

**关键设计**:

- **接缝闭合**: 侧面展开矩形的 `s=0` 和 `s=L` 代表同一条母线。添加绕回点 `(L, z)` 后，哈希归一化自动将其与起点 `(0, z)` 去重，实现无缝闭合。
- **边界节点共享**: 端面和侧面使用相同的圆边界离散点，合并后共享同一组节点对象，确保拓扑一致。
- **几何边界保护**: `fixed_node_ids` 中的节点在 Laplacian 光滑中保持不动，边界边在边交换中受保护不被交换。

**三个面的网格策略**:

| 面 | 方法 | 说明 |
|----|------|------|
| 底面 | `_mesh_disk_2d(normal_z=-1.0, boundary_pts_2d=...)` | 使用统一圆边界 + 2D AFM |
| 顶面 | `_mesh_disk_2d(normal_z=+1.0, boundary_pts_2d=...)` | 使用统一圆边界 + 2D AFM |
| 侧面 | `_mesh_lateral_cylinder_2d(boundary_pts_2d=...)` | 展开矩形 + 统一边界 + 2D AFM + 圆柱映射 |

### 6.4 矩形网格生成

**`generate_rectangle_mesh(corner1, corner2, spacing, output_vtk)`**

单面网格生成，直接调用 `_mesh_face_2d_pipeline`。需要至少两个方向有非零长度。

### 6.5 跨面节点去重

合并多面结果时，使用 `node_hash_to_global_idx` 字典实现坐标哈希去重：

```python
for node in nodes:
    h = node.hash  # 坐标哈希（含 -0.0 归一化）
    if h not in node_hash_to_global_idx:
        node_hash_to_global_idx[h] = global_idx
        node.idx = global_idx
        all_nodes.append(node)
        global_idx += 1
    node_to_global[id(node)] = node_hash_to_global_idx[h]
```

这确保共享边界的节点在合并后只保留一份，拓扑一致。

**圆柱体的统一去重**（`_mesh_cylinder_unified`）:

圆柱体的端面和侧面共享圆形边界。统一离散化后，端面的圆边界节点与侧面展开坐标的接缝节点通过哈希归一化自动去重：

- 圆边界节点坐标 `(x, y, 0)` 与侧面底边展开坐标 `(s, z₀)` 映射回 3D 后相同
- 接缝点 `(L, z)` 与起点 `(0, z)` 映射回 3D 后坐标相同，哈希一致

合并时同时记录每个节点所属的面（`node_face_map`），用于后处理时的曲面投影。

### 6.6 参数化网格回退方案

**`_mesh_face_parametric(face, spacing, node_id_offset)`**

当 AFM 失败时（如复杂曲面），回退到参数化网格：

1. 在参数空间 `(u, v)` 生成均匀网格
2. 通过 `BRepClass_FaceClassifier` 过滤面外点
3. 每个网格单元产生两个三角形（结构化剖分）
4. 节点投影到几何曲面确保保真度

此方法快速但质量不如 AFM，仅作为备选方案。

---

## 7. CAD 几何驱动网格生成

### 7.1 SurfaceMeshGenerator

**文件位置**: `sfmesh/surface_mesh.py`

通用的 3D AFM 网格生成器，直接在 OCC 参数曲面上推进阵面。

**与 2D 流水线的区别**:

| 特性 | 3D AFM (`SurfaceMeshGenerator`) | 2D 流水线 (`pipeline_2d`) |
|------|--------------------------------|--------------------------|
| 工作空间 | 3D 参数曲面 | 2D 平面 |
| 几何保真 | 节点实时投影到曲面 | 映射回 3D + 光滑后曲面投影 |
| 尺寸场 | `SurfaceSizingField`（曲率自适应） | `QuadtreeSizing`（四叉树） |
| 优化 | 无后处理 | 边交换 + Laplacian 光滑 + 曲面投影 |
| 适用场景 | 任意复杂曲面 | 平面/可展开曲面 |
| 空间索引 | R-tree | Adfront2 内置 |

**主循环**: 每 100 次迭代输出统计信息（节点数、三角形数、阵面数）。

### 7.2 文件驱动入口

**`generate_surface_mesh_from_file(filename, global_spacing, output_vtk)`**

1. 通过 `fileIO.geometry_io` 加载 IGES/STEP 文件
2. 提取所有 `TopoDS_Face`
3. 对每个面独立调用 `SurfaceMeshGenerator`
4. 合并结果，可选导出 VTK

### 7.3 两种路径对比

| 维度 | CAD 驱动 | 基础几何体驱动 |
|------|----------|---------------|
| 几何来源 | IGES/STEP 文件 | 参数化定义 |
| 面提取 | OCC `TopExp_Explorer` | 手动顶点定义 |
| AFM 实现 | `SurfaceMeshGenerator`（3D） | `_mesh_face_2d_pipeline`（2D） |
| 公开 API | `generate_surface_mesh_from_file()` | `generate_cube_mesh()` 等 |
| 输出格式 | `Unstructured_Grid` 或 VTK | `PrimitiveMeshResult` |

---

## 8. 曲面几何操作

### 8.1 SurfaceGeometry

**文件位置**: `sfmesh/surface_geometry.py`

封装所有 OpenCASCADE 曲面操作，提供统一的 Python 接口。

**关键方法**:

| 方法 | 功能 | OCC 底层 |
|------|------|----------|
| `project_point_to_surface()` | 3D 点 → 参数 UV | `GeomAPI_ProjectPointOnSurf` |
| `evaluate_point(u, v)` | 参数 UV → 3D 坐标 | `BRepAdaptor_Surface.Value` |
| `get_surface_normal(u, v)` | 计算表面法向量 | `GeomLProp_SLProps` (1 阶) |
| `get_surface_curvature(u, v)` | 计算平均/高斯/主曲率 | `GeomLProp_SLProps` (2 阶) |
| `compute_ideal_point_on_surface()` | 迭代投影理想点到曲面 | 多步投影 |
| `get_surface_area()` | 数值积分求面积 | 中点法则 |

**设计说明**: `project_point_to_surface` 失败时回退到参数域中心 (`_estimate_uv`)，确保鲁棒性。

---

## 9. 尺寸场设计

### 9.1 SurfaceSizingField

**文件位置**: `sfmesh/sizing_field.py`

控制网格元素尺寸，支持多种自适应策略。

| 策略 | 方法 | 说明 |
|------|------|------|
| 均匀 | `spacing_at()` → `global_spacing` | 全局统一尺寸 |
| 曲率自适应 | `_compute_curvature_based_spacing()` | `spacing = curvature_factor / max_curvature` |
| 梯度限制 | `get_gradient_spacing()` | 限制相邻节点尺寸变化率 |
| 局部指定 | `set_local_spacing()` | 用户指定点的局部尺寸 |

**理想点距离**: `d = quality_factor × √(spacing × front_length)`，夹在 `[0.5×spacing, 2.0×spacing]`。

### 9.2 AdaptiveSizingField

继承 `SurfaceSizingField`，增加特征邻近自适应：

- `add_feature_point()`: 点特征附近加密
- `add_feature_line()`: 线特征附近加密
- 特征间距: `spacing = global_spacing × ratio × (dist / radius + 0.1)`

---

## 10. 几何工具模块

### 10.1 OCC 工具函数

**文件位置**: `sfmesh/occ_utils.py`

| 函数 | 功能 |
|------|------|
| `_extract_faces(shape)` | 从 OCC 形状提取所有面 |
| `_get_face_bbox(face)` | 获取面的 AABB |
| `_get_face_bbox_center(face)` | 获取面的包围盒中心 |
| `_classify_cube_faces(faces, corner1, corner2)` | 按坐标分类长方体 6 个面 |
| `_classify_cylinder_faces(faces, base_z, top_z)` | 按曲面类型分类圆柱体 3 个面 |
| `_is_point_in_face(u, v, face)` | 判断参数点是否在面内 |
| `_is_planar_face(face)` | 判断面是否为平面 |

### 10.2 计算几何函数

**文件位置**: `sfmesh/geom_utils.py`

纯数值计算，无 OCC 依赖，可独立测试。

| 函数 | 功能 | 算法 |
|------|------|------|
| `_project_to_2d(points, normal)` | 3D→2D 投影 | 消除法向最大分量轴 |
| `_segments_intersect_2d(a1, a2, b1, b2)` | 2D 线段相交 | 叉积方向 + AABB 排斥 |
| `_point_in_triangle_2d(p, t0, t1, t2)` | 点在三角形内 | 重心符号法 |
| `_are_coplanar_triangles_overlapping(...)` | 共面三角形重叠 | 2D 投影 + 边相交 + 包含测试 |

---

## 11. 网格质量评估

### 11.1 质量指标

**文件位置**: `sfmesh/mesh_quality.py` — `SurfaceMeshQuality`（静态方法类）

| 指标 | 方法 | 公式/说明 |
|------|------|-----------|
| 形状质量 | `triangle_quality()` | `4√3 · area / (a² + b² + c²)` |
| 长宽比 | `triangle_aspect_ratio()` | `max_edge / (2√3 · area / max_edge)` |
| 最小角 | `triangle_min_angle()` | 三个内角的最小值（度） |
| 最大角 | `triangle_max_angle()` | 三个内角的最大值（度） |
| 翘曲 | `triangle_warpage()` | 三角形恒为平面，始终返回 0.0 |
| Jacobian | `triangle_jacobian()` | `||cross(v1, v2)||` |

**全网格统计** (`evaluate_mesh`): 计算质量、长宽比、角度的均值/最小值/最大值/标准差，统计低质量三角形（quality < 0.3）数量。

### 11.2 相交检测算法

**三角形-三角形相交** (`check_triangle_intersection`): 分离轴测试——检查一个三角形的所有顶点是否在另一个三角形平面的同侧。

**边-三角形相交** (`check_edge_triangle_intersection`): 射线-平面相交——计算边与三角形平面的交点，用重心坐标判定交点是否在三角形内。

---

## 12. 对外接口与使用示例

### 12.1 公开 API

通过 `sfmesh/__init__.py` 导出：

```python
from sfmesh import (
    # 基础几何体
    generate_cube_mesh,
    generate_cylinder_mesh,
    generate_rectangle_mesh,
    generate_sphere_mesh,
    generate_ellipsoid_mesh,
    PrimitiveMeshResult,
    # CAD 文件驱动
    SurfaceMeshGenerator,
    # 数据结构
    SurfaceFront, NodeElement3D, SurfaceTriangle,
    # 几何与尺寸
    SurfaceGeometry, SurfaceSizingField,
    # 质量评估
    SurfaceMeshQuality,
)
```

### 12.2 使用示例

**生成长方体网格**:

```python
from sfmesh import generate_cube_mesh

result = generate_cube_mesh(
    corner1=(0.0, 0.0, 0.0),
    corner2=(2.0, 3.0, 4.0),
    spacing=0.5,
    output_vtk="cube_mesh.vtk",
)

print(f"三角形数: {len(result.triangles)}")
print(f"节点数: {len(result.nodes)}")
print(f"面数: {result.num_faces}")
```

**生成圆柱体网格**:

```python
from sfmesh import generate_cylinder_mesh

result = generate_cylinder_mesh(
    base_center=(0.0, 0.0, 0.0),
    radius=1.0,
    height=2.0,
    spacing=0.3,
)
```

**从 IGES 文件生成网格**:

```python
from sfmesh.surface_mesh import generate_surface_mesh_from_file

generate_surface_mesh_from_file(
    filename="model.iges",
    global_spacing=0.5,
    output_vtk="output.vtk",
)
```

**质量评估**:

```python
from sfmesh import SurfaceMeshQuality

stats = SurfaceMeshQuality.evaluate_mesh(result.triangles)
print(f"平均质量: {stats['quality_mean']:.3f}")
print(f"低质量三角形: {stats['quality_poor_count']}")
```

---

## 13. 已知限制与扩展方向

### 13.1 已知限制

1. **2D 流水线仅支持平面和可展开曲面**: 圆柱侧面通过展开近似，球面等不可展开曲面必须使用 3D AFM。
2. **共面重叠检测为保守估计**: `_are_coplanar_triangles_overlapping` 使用 2D 投影 + 边相交，在极端退化情况下可能漏判。
3. **搜索半径扩展上限固定**: `al < 20.0` 的硬编码上限在极大尺寸差异的场景下可能不足。
4. **无并行化**: 逐面网格生成是串行的，多面之间无并行。
5. **球体/椭球体依赖 OCC**: `generate_sphere_mesh` 和 `generate_ellipsoid_mesh` 使用参数化网格回退方案，质量不如 AFM。

### 13.2 扩展方向

- **支持更多基础几何体**: 圆锥体、棱锥体、环面等
- **3D AFM 后处理**: 为 `SurfaceMeshGenerator`（CAD 路径）添加边交换和 Laplacian 光滑（基础几何体路径已实现）
- **并行面网格生成**: 多面独立生成时可并行化
- **自适应加密**: 基于曲率或误差估计的局部加密
- **四边形网格**: 扩展到四边形或混合网格生成

---

## 附录 A: 文件清单

| 文件名 | 行数 | 功能摘要 |
|--------|------|----------|
| `__init__.py` | 32 | 包初始化，导出公开 API（含球体/椭球体） |
| `surface_front.py` | 449 | 核心数据结构：NodeElement3D、SurfaceTriangle、SurfaceFront |
| `surface_geometry.py` | 377 | OCC 曲面几何操作封装 |
| `sizing_field.py` | 384 | 尺寸场控制（均匀/曲率/梯度/特征） |
| `mesh_quality.py` | 429 | 网格质量评估与相交检测 |
| `surface_mesh.py` | 745 | CAD 文件驱动的 3D AFM 网格生成器 |
| `primitives.py` | 1303 | 基础几何体网格生成入口（长方体/圆柱体/矩形/球体/椭球体） |
| `occ_utils.py` | 381 | OCC 辅助函数 |
| `geom_utils.py` | 90 | 纯数值计算几何函数 |
| `pipeline_2d.py` | 657 | 2D AFM 流水线 + 统一圆柱网格 + 曲面投影 |
| **合计** | **4847** | |

---

*版本: 1.1*
*日期: 2026-05-19*
*作者: Claude Code*
