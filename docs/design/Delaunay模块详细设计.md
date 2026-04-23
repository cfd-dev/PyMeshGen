# Delaunay 模块详细设计文档

## 目录

- [1. 概述](#1-概述)
- [2. 模块定位与总体架构](#2-模块定位与总体架构)
- [3. 目录结构与职责划分](#3-目录结构与职责划分)
- [4. 对外接口与调用关系](#4-对外接口与调用关系)
- [5. 边界输入归一化设计](#5-边界输入归一化设计)
- [6. 核心数据结构设计](#6-核心数据结构设计)
- [7. 几何谓词与数值稳健性](#7-几何谓词与数值稳健性)
- [8. Bowyer-Watson 核心算法设计](#8-bowyer-watson-核心算法设计)
- [9. Triangle 后端设计](#9-triangle-后端设计)
- [10. 后处理与验证设计](#10-后处理与验证设计)
- [11. 与主流程的集成关系](#11-与主流程的集成关系)
- [12. 配置参数与行为控制](#12-配置参数与行为控制)
- [13. 扩展点与维护建议](#13-扩展点与维护建议)
- [14. 已知限制与风险](#14-已知限制与风险)

---

## 1. 概述

`delaunay\` 模块是 PyMeshGen 在 `mesh_type == 4` 时的二维三角剖分核心实现，负责将初始边界阵面和尺寸场转换为满足边界约束、孔洞约束和基本拓扑要求的三角网格。

当前模块支持两条后端路径：

1. **Bowyer-Watson / Gmsh 风格后端**
   - 主实现文件：`delaunay\bw_core_stable.py`
   - 适用于当前项目的主 Delaunay 路径
   - 支持受保护边界边、孔洞清理、局部恢复、Laplacian 平滑、拓扑清理
2. **Triangle 后端**
   - 主实现文件：`delaunay\triangle_backend.py`
   - 通过 Triangle 可执行文件完成 PSLG 三角剖分
   - 作为 mesh_type=4 的替代后端，尤其用于带边界层时的内区填充

模块设计目标：

- 输入保持与项目已有 `front2d` / `QuadtreeSizing` 兼容
- 输出保持为统一数组三元组：`(points, simplices, boundary_mask)`
- 将“边界归一化”“核心剖分”“后处理修复”“测试验证”解耦
- 在不牺牲鲁棒性的前提下尽量贴近 Gmsh 的二维 Bowyer-Watson 工作流

---

## 2. 模块定位与总体架构

### 2.1 在项目中的位置

`delaunay\` 并不是独立入口，而是 `core.py` 的一个子系统。其典型调用链如下：

```text
core.generate_mesh()
  -> QuadtreeSizing(...)                    # 构建尺寸场
  -> create_bowyer_watson_mesh(...)         # delaunay 公共入口
      -> bw_utils._build_boundary_input()   # front -> 点/边/孔洞
      -> backend dispatch
         -> BowyerWatsonMeshGenerator
         -> create_triangle_mesh
  -> core._recover_delaunay_boundary_edges()  # 轻量边翻转补恢复（非 Triangle）
  -> Unstructured_Grid.from_cells()           # 统一转网格对象
```

### 2.2 分层结构

```text
公共入口层
├── __init__.py
└── bw_utils.py

算法核心层
├── bw_core_stable.py
├── bw_cavity.py
├── bw_types.py
└── bw_predicates.py

替代后端层
└── triangle_backend.py

后处理 / 验证层
├── postprocess.py
└── validation.py
```

---

## 3. 目录结构与职责划分

| 文件 | 角色 | 主要职责 |
| --- | --- | --- |
| `__init__.py` | 包入口 | 暴露 `BowyerWatsonMeshGenerator`、`MTri3`、`create_bowyer_watson_mesh` |
| `bw_utils.py` | 输入归一化与后端分发 | 从 `front` 提取边界点、边界边、外边界和孔洞；按配置选择 Bowyer-Watson 或 Triangle |
| `bw_core_stable.py` | 主算法实现 | Gmsh 风格 Bowyer-Watson、边界恢复、孔洞/域清理、拓扑清理 |
| `bw_cavity.py` | 空腔搜索与重连 | Cavity 搜索、star-shaped 校验、插点后重连、失败恢复 |
| `bw_predicates.py` | 几何谓词 | `orient2d`、`incircle`、外接圆、各向异性外接圆辅助 |
| `bw_types.py` | Gmsh 风格数据结构 | `MTri3`、`EdgeXFace`、`TriangulationState`、邻接构建、cavity shell 收集 |
| `triangle_backend.py` | Triangle 包装层 | 构建 Triangle 可执行文件、写 PSLG `.poly`、解析 `.node/.ele`、内部点采样 |
| `postprocess.py` | 轻量后处理 | 三角数组上的边翻转恢复、拓扑快速检查 |
| `validation.py` | 测试验证工具 | 边界恢复、孔洞清理、拓扑洁净检查 |

设计上的关键取舍：

- **`bw_utils.py` 只处理输入归一化，不承载算法细节**
- **`bw_core_stable.py` 承载实际网格生成状态机**
- **`postprocess.py` 保持数组级实现，便于在 `core.py` 中复用**
- **`validation.py` 只面向测试与验收，不参与生产运行链**

---

## 4. 对外接口与调用关系

### 4.1 包级公开接口

`delaunay\__init__.py` 当前公开：

- `BowyerWatsonMeshGenerator`
- `MTri3`
- `create_bowyer_watson_mesh`

其中真正建议业务层使用的是：

```python
from delaunay import create_bowyer_watson_mesh
```

### 4.2 公共入口 `create_bowyer_watson_mesh`

定义于 `delaunay\bw_utils.py`，职责是：

1. 接收 `boundary_front` 和 `QuadtreeSizing`
2. 归一化边界输入为 `BoundaryInput`
3. 自动识别孔洞与外边界
4. 根据 `backend` 选择：
   - `triangle`
   - `bowyer_watson` -> `BowyerWatsonMeshGenerator`
5. 统一返回：
    - `points: np.ndarray`
    - `simplices: np.ndarray`
    - `boundary_mask: np.ndarray`

### 4.3 核心设计原则

- 上层不需要知道 Bowyer-Watson 内部拓扑如何组织，只关心统一数组输出
- 后端切换不改变返回格式
- 外边界 / 孔洞识别在进入核心算法前完成，避免后端重复做 front 级解析

---

## 5. 边界输入归一化设计

### 5.1 `BoundaryInput`

`BoundaryInput` 是 `bw_utils.py` 中的不可变数据载体：

```text
BoundaryInput
├── boundary_points   # 去重后的边界点坐标
├── boundary_edges    # 边界边索引对
├── holes             # 孔洞环列表
└── outer_boundary    # 主外边界点环
```

### 5.2 归一化流程

```text
boundary_front
  -> _build_front_graph()
  -> _trace_boundary_loops()
  -> _classify_boundary_loops()
  -> _extract_boundary_points_and_edges()
  -> BoundaryInput
```

### 5.3 关键处理点

1. **front 图重建**
   - 以 `node.hash` 为图节点
   - 以阵面为无向边
2. **边界环追踪**
   - 从 front 图中提取闭合 loop
3. **外边界 / 孔洞判别**
   - 基于面积与重心包含关系
4. **索引稳定化**
   - 所有边界边在进入核心算法前重映射到连续索引

这样做的好处是：

- 算法核心不需要理解 `front` 对象
- Triangle 后端与 Bowyer-Watson 后端可共享同一边界输入
- 测试与调试时可以直接输出边界点/边，不依赖原始 front 结构

---

## 6. 核心数据结构设计

### 6.1 `MTri3`（Gmsh 风格主结构）

`bw_types.py` 中的 `MTri3` 是当前主路径的核心结构：

- `vertices`
- `neighbors`
- `circumcenter`
- `circumradius`
- `deleted`
- `idx`
- `quality`

其设计重点是：

1. **懒删除**
   - 空腔搜索、边翻转、局部修复时避免频繁从容器中物理移除
2. **显式邻接**
   - Cavity 搜索和边恢复可以直接走邻接关系
3. **缓存几何量**
   - 避免重复计算外接圆与质量

### 6.1.1 `TriangulationState`（常驻拓扑索引层）

`TriangulationState` 是在 `MTri3` 之上的常驻索引缓存，用于替代反复现建现用的临时 edge map / vertex map：

- `triangle_by_id`
- `edge_to_tris`
- `vertex_to_tris`
- `vertex_to_neighbors`

其职责不是替代三角形对象，而是统一维护：

1. **邻接重建**
   - 用一次边遍历同时完成 `neighbors` 回填和 edge index 构建
2. **高频查询**
   - 查询边是否存在、某边 incident triangles、某顶点 incident triangles、某顶点邻点集合
3. **懒删除压缩**
   - 在需要时统一物理压缩 deleted triangles，并刷新索引

当前 Gmsh Bowyer-Watson 主路径已优先使用这层索引来支撑：

- refinement 队列重建
- 受保护边存在性检查
- edge use count / incident triangle 查询
- boundary vertex 邻域查询

### 6.2 `EdgeXFace`

`EdgeXFace` 表示“边 - 面”关系，主要用于：

- 空腔 shell 收集
- 约束边标识
- 局部重连时的边界表示

### 6.3 邻接构建

当前邻接构建分成两层：

1. `build_adjacency_from_triangles()`
   - 负责轻量级的三角形邻接回填
   - 适合高频、只需要 `neighbors` 的局部重建
2. `TriangulationState`
   - 负责显式构建 `edge_to_tris`、`vertex_to_tris`、`vertex_to_neighbors`
   - 只在需要拓扑索引的热点路径按需刷新

这套分层是为了兼顾性能与一致性，几乎所有局部操作都依赖它：

- cavity 扩张
- 边翻转
- 边界恢复
- 局部 strip 搜索

这样设计的好处是：

1. **减少重复全表扫描**
2. **让轻量邻接与重型拓扑索引各自承担合适的成本**
3. **便于后续继续向稳定 id / 更扁平的索引表示演进**

---

## 7. 几何谓词与数值稳健性

`bw_predicates.py` 提供了 Bowyer-Watson 所需的几何基础：

- `orient2d` / `orient2d_fast`
- `incircle` / `incircle_fast`
- `circumcenter_precise`
- `compute_circumcircle`
- `point_in_circumcircle_robust`

### 7.1 为什么单独拆文件

原因有三：

1. **算法逻辑与几何数值分离**
2. **可以在不同实现中复用**
3. **便于将来替换更强的鲁棒谓词实现**

### 7.2 稳健性策略

- 外接圆判断优先使用鲁棒 `incircle`
- 点定位与朝向判断使用 `orient2d_fast`
- 外接圆心使用高精度求解
- 对极小面积 / 共线情况做显式防守

---

## 8. Bowyer-Watson 核心算法设计

### 8.1 `BowyerWatsonMeshGenerator`

这是当前 `backend="bowyer_watson"` 的唯一实现，设计上更接近 Gmsh `meshGFaceDelaunayInsertion.cpp` 的二维思路。

主入口阶段顺序：

```text
阶段 1: 初始三角剖分
阶段 2: Gmsh 风格迭代插点
阶段 2.5: CDT 边界恢复
阶段 2.6~2.95: 孔洞 / 域外清理 + 再恢复
阶段 3: Laplacian 平滑
阶段 3.5: 重叠/重复/退化/交叉清理
输出压缩与导出
```

### 8.2 初始三角剖分

初始剖分步骤：

1. 创建超级三角形
2. 按顺序插入所有边界点
3. 使用 cavity 搜索删除被新点破坏的三角形
4. 用 shell 边与新点重新连接
5. 删除含超级顶点的三角形
6. 立即恢复初始剖分中缺失的受保护边

这一步的目标不是得到最终高质量网格，而是建立一个：

- 覆盖整个域
- 邻接正确
- 边界点完整
- 可继续细化的初始三角网

### 8.3 迭代细化策略

细化主循环 `_insert_points_iteratively()` 的核心元素：

1. **优先级队列**
   - 按外接圆半径排序，优先处理“坏三角形”
2. **尺寸场驱动**
   - `target_size = sizing_system.spacing_at(tri_center)`
3. **质量阈值**
   - 边界区与内区使用不同阈值
4. **早停策略**
   - 定期全量检查剩余不满足要求的单元数量
5. **压缩与重建**
   - deleted 三角形积累到一定规模后，压缩容器并重建队列

### 8.4 细化点选择

候选点选择由 `_select_refinement_point()` 完成：

1. 若是靠近受保护边的低质量三角形，优先尝试 **off-center**
2. 否则默认使用 **circumcenter**
3. 若候选点不满足域内/孔洞/最小间距要求，则拒绝

这一设计的目的：

- 对边界附近使用更保守的插点，减少边界恢复压力
- 对内部维持 Delaunay 细化效率

### 8.5 Cavity 搜索与重连

`bw_cavity.py` 实现空腔算法：

```text
start_tri
  -> recur_find_cavity()
      -> 沿邻接递归扩张
      -> 碰到受保护边时停止
      -> 收集 cavity triangles
      -> 收集 shell edges
  -> insert_vertex()
      -> 新点连接 shell
      -> 建立新三角形
      -> 更新邻接
```

核心约束：

- 受保护边不能跨越
- shell 必须能包围 cavity
- 失败时必须恢复 deleted 标记和临时点

### 8.6 失败回滚

`_rollback_failed_cavity_insertion()` 与 `_insert_refinement_point()` 负责：

- 回滚 deleted 三角形
- 删除失败插入产生的临时点
- 防止留下“孤儿点”“空洞”或不完整局部拓扑

这是当前实现区别于早期版本的重要稳健性增强点。

### 8.7 边界恢复

模块中存在两层边界恢复：

1. **核心恢复**：`bw_core_stable.py`
   - `BowyerWatsonMeshGenerator._constrained_delaunay_triangulation()`
   - 面向受保护边执行 swap-first 的精确恢复
2. **轻量恢复**：`postprocess.py`
   - `recover_boundary_edges_by_swaps()`
   - 在 `core.py` 中作为数组级补恢复使用

设计目的：

- 核心算法内部尽量保证精确约束边
- 上层在最终三角数组层面还有一次便宜的补救机会

### 8.8 孔洞与域清理

Gmsh 主路径中的孔洞处理顺序非常关键：

1. 先完整细化
2. 再恢复边界
3. 再清理孔洞与域外三角形
4. 必要时再次做 CDT 恢复

这样做的原因是：

- 若过早删洞，局部 cavity 不完整，边界恢复会更困难
- 若清理后不再恢复，某些真实边界边可能被再次丢失

### 8.9 拓扑清理

最终输出前，Gmsh 主路径会执行：

- 重叠三角形清理
- 重复三角形清理
- 严格相交三角形清理
- 退化三角形清理
- 被隔离边界点修复
- 孔洞侧 boundary-fan 清理

这一步是“结果正确性”的最后保障层。

### 8.10 多边界环场景

当前实现已经显式支持通过边界连通分量区分不同边界环：

- 对**同一边界环上的 boundary fan 三角形**可放宽尺寸细化
- 对**跨外边界/孔洞边界的桥接三角形**仍允许内部细化

这一点对环域类算例（如 `quad_quad`）尤其关键。

---

## 9. Triangle 后端设计

`triangle_backend.py` 为 Jonathan Shewchuk 的 Triangle 提供本地包装。

### 9.1 设计目标

- 不修改第三方源码
- 使用 CLI 而非 DLL/ctypes 桥接
- 继续复用项目已有尺寸场和边界 front 体系

### 9.2 工作流

```text
QuadtreeSizing
  -> 内部点采样
  -> 写 mesh.poly
  -> 调用 triangle.exe
  -> 解析 mesh.1.node / mesh.1.ele
  -> 返回 (points, simplices, boundary_mask)
```

### 9.3 内部点采样策略

支持两种策略：

1. `cartesian`
   - 叶节点中心 + 四分点
   - 保持原始网格点云风格
2. `equilateral`
   - 近似三角晶格采样
   - 更偏向等边三角形

### 9.4 关键实现点

- `_ensure_triangle_exe()`：必要时自动构建可执行文件
- `_write_poly_file()`：输出 PSLG
- `_triangle_switches()`：统一 Triangle 参数
- `_parse_node_file()` / `_parse_ele_file()`：解析 Triangle 输出

Triangle 后端本质上是：

- **边界与尺寸场仍由 PyMeshGen 控制**
- **三角剖分核心委托给 Triangle**

---

## 10. 后处理与验证设计

### 10.1 `postprocess.py`

提供两类轻量工具：

1. `recover_boundary_edges_by_swaps()`
   - 对数组级三角形做边翻转恢复
2. `is_topology_valid()`
   - 检查：
     - 是否存在超过 2 个单元共边
     - 网格连通性
     - 严格边交叉

它的定位是：

- 不依赖 `MTri3`
- 便于 `core.py` 在统一结果层面做补检查

### 10.2 `validation.py`

主要面向单元测试和验收检查：

- `check_boundary_edges()`
- `check_hole_cleanup()`
- `check_topology_clean()`

核心思路：

- 重新解析输入 CAS 边界
- 将输入边界与输出网格建立节点映射
- 检查边界是否恢复、孔洞是否为空、是否存在严格交叉

这部分代码不参与正式生成流程，但对于回归测试非常关键。

---

## 11. 与主流程的集成关系

### 11.1 `core.py` 中的集成

`core.py` 里 mesh_type=4 的职责分工如下：

1. 读取输入网格并构造初始 `front`
2. 构造 `QuadtreeSizing`
3. 决定是否先生成边界层
4. 调用 `create_bowyer_watson_mesh()`
5. 对非 Triangle 后端做轻量边翻转恢复
6. 转成 `Unstructured_Grid`

### 11.2 与尺寸场的关系

`delaunay\` 本身不生成尺寸场，只消费 `QuadtreeSizing`：

- Bowyer-Watson 主路径通过 `spacing_at()` 获取局部目标尺寸
- Triangle 后端通过 quadtree 叶节点采样内部点

### 11.3 与边界层的关系

- 无边界层：`front_heap` 直接进入 Delaunay
- 有边界层：边界层先生成，内区 front 更新后再进入 Delaunay / Triangle

因此 `delaunay\` 只负责“当前 front 所围成区域”的三角剖分，不负责边界层推进。

---

## 12. 配置参数与行为控制

与 `delaunay\` 强相关的参数主要有：

| 参数 | 来源 | 作用 |
| --- | --- | --- |
| `mesh_type` | `Parameters` | `4` 时启用 Delaunay 主流程 |
| `delaunay_backend` | `Parameters` / case JSON | 选择 `bowyer_watson` 或 `triangle` |
| `triangle_point_strategy` | `Parameters` / case JSON | Triangle 内部点采样策略 |
| `sizing_decay` | `Parameters` / case JSON | 控制 `QuadtreeSizing` 尺寸场衰减 |
| `smoothing_iterations` | 调用参数 | 控制 Bowyer-Watson 输出前的 Laplacian 平滑 |
| `target_triangle_count` | 调用参数 | 限制目标细化规模 |
| `seed` | 调用参数 | 控制可重复性 |

行为上的补充规则：

- 带边界层时，上层可能强制切换到 Triangle 后端

---

## 13. 扩展点与维护建议

### 13.1 适合新增功能的部位

1. **新增边界输入处理**
   - 优先改 `bw_utils.py`
2. **新增几何谓词或更高精度策略**
   - 改 `bw_predicates.py`
3. **新增局部重连 / cavity 规则**
   - 改 `bw_cavity.py`
4. **新增后端**
   - 复用 `create_bowyer_watson_mesh()` 的 backend dispatch
5. **新增结果验证**
   - 改 `validation.py`

### 13.2 维护建议

- 不要在 `bw_utils.py` 中重新塞入核心算法逻辑
- 不要让 `core.py` 重新承担边界解析和三角数组后处理细节
- 修改 `bw_core_stable.py` 时优先保持入口编排与局部算法分离
- 任何对细化规则、边界恢复规则的修改，都应至少回归：
  - `test_anw_bowyer_watson`
  - `test_naca0012_bowyer_watson`
  - `test_quad_quad_bowyer_watson`
  - 相关带边界层用例

---

## 14. 已知限制与风险

1. **`bw_core_stable.py` 体量仍然偏大**
   - 入口流程已拆小，但边界恢复相关 helper 仍然密集
2. **运行时间对复杂算例较敏感**
   - 如 `NACA0012` 一类 case 的运行时间仍可能靠近测试阈值
3. **边界恢复与局部清理高度耦合**
   - 修改恢复顺序时容易影响孔洞清理和拓扑洁净
4. **Triangle 后端依赖本地编译环境**
   - 首次构建需要可用的 Visual Studio C 工具链
5. **二维实现优先**
   - 当前设计主要服务于二维三角网格场景，不直接覆盖三维 Delaunay

---

## 总结

`delaunay\` 模块的核心设计思想可以概括为：

1. **输入统一**：front 先归一化成点/边/孔洞
2. **后端可切换**：Bowyer-Watson 与 Triangle 共用同一上层接口
3. **核心算法分层**：几何谓词、cavity、数据结构、后处理分开实现
4. **结果优先**：边界恢复、孔洞清理、拓扑清理构成多层保障
5. **集成友好**：始终以 `(points, simplices, boundary_mask)` 作为对上层的稳定契约

这套设计使得模块既能服务于当前项目的工程化网格生成，又保留了继续对齐 Gmsh / Triangle / 本项目自定义策略的扩展空间。
