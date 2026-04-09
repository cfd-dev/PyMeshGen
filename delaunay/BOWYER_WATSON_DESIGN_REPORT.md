# PyMeshGen Bowyer-Watson 实现详细设计报告

## 1. 概述

本文档详细分析 PyMeshGen 项目中 Bowyer-Watson Delaunay 三角剖分算法的实现原理、数据结构、核心流程和优化策略。

**源码位置：** `delaunay/bw_core.py`（主实现，1318 行）、`delaunay/helpers.py`（辅助函数，187 行）

**设计目标：**
- 以离散边界网格作为输入
- 使用 QuadtreeSizing 尺寸场控制网格尺寸
- 支持自动/手动孔洞处理
- 边界边保护与恢复
- 增量式点插入（避免全量重剖分）

---

## 2. 核心算法原理

### 2.1 Bowyer-Watson 算法基本原理

本项目采用经典的增量插入法（Incremental Insertion）构建 Delaunay 三角网：

1. **超级三角形**：创建包含所有边界点的初始超级三角形
2. **逐点插入**：逐个插入边界点，维护 Delaunay 性质
3. **空腔搜索（Cavity Search）**：找到所有外接圆包含新顶点的三角形
4. **空腔删除**：删除这些违反 Delaunay 条件的三角形
5. **重新连接**：将新顶点与空腔边界上的所有边连接
6. **迭代加密**：基于质量度量迭代插入内部点，直至满足尺寸和质量要求

### 2.2 算法变体与扩展

本项目在传统 Bowyer-Watson 基础上增加了：
- **保护边机制**：边界边不被破坏（类似 Gmsh 的 `internalEdges`）
- **前端方法变体（Frontal Method）**：在活跃边中垂线上找最优插入点
- **边界边恢复**：通过边翻转或中点插入恢复丢失的边界边
- **Laplacian 平滑**：度量空间加权平均，质量接受准则

---

## 3. 核心数据结构

### 3.1 Triangle - 三角形单元类

```python
class Triangle:
    __slots__ = [
        'vertices',           # 排序后的顶点索引元组 (v0, v1, v2)
        'circumcenter',       # 外接圆圆心 (缓存)
        'circumradius',       # 外接圆半径 (缓存)
        'idx',                # 三角形索引
        'circumcircle_valid', # 外接圆缓存有效性标志
        'quality',            # 质量度量 (2 * r_inscribed / r_circumscribed)
        'quality_valid',      # 质量缓存有效性标志
        'circumcircle_bbox',  # 外接圆包围盒 (用于快速剔除)
        'neighbors',          # 邻接三角形列表 [tri0, tri1, tri2]
    ]
```

**关键设计：**
- `__slots__` 减少内存开销（每实例节省约 40% 内存）
- 顶点索引按升序存储，便于去重和比较 (`tuple(sorted([p1, p2, p3]))`)
- `neighbors[i]` 对应 `vertices[i]->vertices[(i+1)%3]` 这条边的邻接三角形
- 外接圆和质量计算结果缓存，避免重复计算

### 3.2 BowyerWatsonMeshGenerator - 主生成器类

```python
class BowyerWatsonMeshGenerator:
    # 输入参数
    original_points: np.ndarray          # 原始边界点 (N, 2)
    protected_edges: Set[FrozenSet]      # 受保护边界边集合
    sizing_system: QuadtreeSizing        # 尺寸场控制系统
    max_edge_length: float               # 全局最大边长
    smoothing_iterations: int            # Laplacian 平滑迭代次数
    holes: List[np.ndarray]              # 孔洞边界列表

    # 工作状态变量
    points: np.ndarray                   # 当前所有点（含插入点）
    triangles: List[Triangle]            # 当前三角形列表
    boundary_mask: np.ndarray            # 边界点掩码
    boundary_count: int                  # 边界点数量
    _kdtree: KDTree                      # KD 树（用于最近邻查询）
```

**关键设计：**
- 使用 `KDTree` 加速最近邻查询（点间距检查）
- `boundary_mask` 区分边界点和内部插入点
- `protected_edges` 使用 `frozenset` 确保无向边唯一性

### 3.3 辅助数据结构

**边界环提取（helpers.py）：**
```python
def _extract_boundary_loops_from_fronts(boundary_front):
    # 邻接表
    adjacency: Dict[int, List[int]]    # 节点哈希 → 相邻节点列表
    node_coords: Dict[int, np.ndarray] # 节点哈希 → 坐标

    # 返回值
    outer_loops: List[np.ndarray]      # 外边界环（逆时针）
    hole_loops: List[np.ndarray]       # 孔洞环（顺时针）
```

---

## 4. 核心算法流程

### 4.1 主入口：generate_mesh

```python
def generate_mesh(self, target_triangle_count: Optional[int] = None):
    """
    阶段 1/3: 初始三角剖分 (_triangulate)
    阶段 2/3: 迭代插入内部点 (_insert_points_iteratively)
    阶段 2.5/3: 清理孔洞内三角形 (_remove_hole_triangles)
    阶段 2.6/3: 恢复边界边 (_recover_boundary_edges)
    阶段 3/3: Laplacian 平滑 (可选)
    """
```

**执行流程：**

```
┌─────────────────────────────────────────────────┐
│ 阶段 1/3: 初始三角剖分                            │
│    - _triangulate()                             │
│    - 创建超级三角形                              │
│    - 逐点插入边界点                              │
│    - 删除超级三角形                              │
│    - 构建邻接关系 (_build_adjacency)             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 阶段 2/3: 迭代插入内部点                          │
│    ┌─────────────────────────────────────────┐  │
│    │ 2.1 从优先级队列取出最差三角形           │  │
│    │     heapq.heappop(heap)                 │  │
│    └────────────────┬────────────────────────┘  │
│                     │                            │
│                     ▼                            │
│    ┌─────────────────────────────────────────┐  │
│    │ 2.2 检查终止条件                         │  │
│    │     - 点数 > 100000                     │  │
│    │     - 迭代 > 50000                      │  │
│    │     - 达到目标三角形数                   │  │
│    │     - 堆为空（所有三角形满足要求）       │  │
│    └────────────────┬────────────────────────┘  │
│                     │                            │
│                     ▼                            │
│    ┌─────────────────────────────────────────┐  │
│    │ 2.3 计算插入点                           │  │
│    │     - _optimal_point_frontal() [优先]   │  │
│    │     - circumcenter [回退]               │  │
│    │     - 重心坐标随机采样 [超出范围时]     │  │
│    └────────────────┬────────────────────────┘  │
│                     │                            │
│                     ▼                            │
│    ┌─────────────────────────────────────────┐  │
│    │ 2.4 验证插入点                           │  │
│    │     - KDTree 最近邻查询                 │  │
│    │     - _validate_new_point()             │  │
│    └────────────────┬────────────────────────┘  │
│                     │                            │
│                     ▼                            │
│    ┌─────────────────────────────────────────┐  │
│    │ 2.5 插入点                               │  │
│    │     - _insert_point_incremental()       │  │
│    │     - 更新优先级队列 (每 50 次迭代)     │  │
│    └─────────────────────────────────────────┘  │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 阶段 2.5/3: 清理孔洞内三角形                      │
│    - 修复孔洞多边形方向                          │
│    - 删除质心在孔洞内的三角形                    │
│    - 删除顶点在孔洞内的三角形                    │
│    - 删除孔洞内孤立节点                          │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 阶段 2.6/3: 恢复边界边                            │
│    - 检测丢失的边界边                            │
│    - 边翻转恢复 (_recover_edge_by_flipping)      │
│    - 中点插入恢复 (_insert_midpoint_for_edge)    │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 阶段 3/3: Laplacian 平滑 (可选)                   │
│    - 度量空间加权平均                            │
│    - 逐步衰减因子 (FACTOR /= 1.4)               │
│    - 移动接受准则（质量不下降）                  │
└─────────────────────────────────────────────────┘
```

### 4.2 初始三角剖分：_triangulate

```python
def _triangulate(self) -> List[Triangle]:
    """
    1. 创建超级三角形（包含所有点的包围盒 × 10）
    2. 对每个边界点 i:
       a. 找到所有外接圆包含点 i 的三角形 (bad_triangles)
       b. 收集 Cavity 边界边 (shell edges)
          - 统计每条边出现在 bad_triangles 中的次数
          - 出现 1 次的是 shell 边界边
       c. 删除 bad_triangles
       d. 用点 i 与 shell 边创建新三角形
    3. 删除包含超级三角形顶点的三角形
    4. 构建邻接关系 (_build_adjacency)
    """
```

**时间复杂度：** O(n²) 最坏情况（n 为边界点数）

### 4.3 增量式点插入：_insert_point_incremental

```python
def _insert_point_incremental(self, point_idx: int, triangles: List[Triangle]):
    """
    1. 找到所有外接圆包含新点的三角形 (bad_triangles)
    2. 递归查找 Cavity 边界 (_find_cavity_with_protection)
       - 遇到保护边时停止递归，加入 shell
       - 使用显式邻接关系 (O(1) 查找)
    3. 保存旧 Cavity 三角形用于星形性验证
    4. 删除 Cavity 内三角形
    5. 用新点与 shell 边创建新三角形
    6. 增量更新邻接关系 (_update_adjacency_after_insertion)
    """
```

### 4.4 递归空腔搜索：_find_cavity_with_protection

```python
def _find_cavity_with_protection(self, start_tri, point_idx, all_triangles,
                                 bad_tri_ids, shell_edges, cavity_set):
    """
    递归逻辑：
    1. 将当前三角形加入 cavity_set
    2. 遍历三条边 (i = 0, 1, 2):
       a. 检查是否是受保护边界边
          - 是 → 加入 shell_edges，不递归
       b. 使用显式邻接关系查找邻居 (O(1))
          neighbor = tri.neighbors[i]
       c. 如果邻居不存在 → 加入 shell_edges
       d. 如果邻居在 bad_tri_ids 中 → 递归处理
       e. 否则 → 加入 shell_edges
    """
```

### 4.5 迭代插点主循环：_insert_points_iteratively

```python
def _insert_points_iteratively(self, target_triangle_count: Optional[int] = None):
    """
    终止条件：
    - 点数 > 100000
    - 迭代 > 50000
    - 达到目标三角形数
    - 堆为空（所有三角形满足要求）
    - 连续失败 > 500 次

    优化策略：
    P1-1: 优先级队列（堆）优化最差三角形查找
    P1-2: 动态 KD 树更新（点数变化 > 10% 或每 100 次迭代重建）
    P1-3: 增强点间距检查（边长 + 角度综合验证）
    P2-1: 前端方法最优插入点计算
    """
```

### 4.6 插入点计算

**三种策略（优先级递减）：**

```python
def _compute_insertion_point(self, tri, ...):
    """
    1. _optimal_point_frontal() [优先]
       - 找到最长边作为活跃边
       - 计算活跃边中点
       - 在中垂线上找最优位置（接近等边三角形）

    2. circumcenter [回退]
       - 外接圆圆心

    3. 重心坐标随机采样 [超出范围时]
       - r1, r2 = random()
       - if r1 + r2 > 1: r1, r2 = 1-r1, 1-r2
       - point = p1 + r1*(p2-p1) + r2*(p3-p1)
    """
```

---

## 5. 关键几何计算

### 5.1 外接圆计算

```python
def _compute_circumcircle(self, tri: Triangle):
    """
    求解方程组（行列式法）：
    d = 2 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))

    if |d| < 1e-12:  # 退化三角形
        center = 重心
        radius = 到重心的距离
    else:
        ux = ((ax²+ay²)*(by-cy) + (bx²+by²)*(cy-ay) + (cx²+cy²)*(ay-by)) / d
        uy = ((ax²+ay²)*(cx-bx) + (bx²+by²)*(ax-cx) + (cx²+cy²)*(bx-ax)) / d
        center = [ux, uy]
        radius = |p1 - center|
    """
```

**缓存策略：** 计算结果缓存到 `Triangle` 对象，设置 `circumcircle_valid = True`

### 5.2 点在圆内测试（Robust Predicates）

```python
def _robust_incircle(self, ax, ay, bx, by, cx, cy, px, py):
    """
    Shewchuk 的 Robust Predicates（简化版）

    构造增广矩阵行列式：
    | ax-px  ay-py  (ax-px)²+(ay-py)² |
    | bx-px  by-py  (bx-px)²+(by-py)² |
    | cx-px  cy-py  (cx-px)²+(cy-py)² |

    det = adx*(bdy*clift - cdy*blift) - ady*(bdx*clift - cdx*blift) + alift*(bdx*cdy - cdx*bdy)

    orient = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)

    返回 det * orient
      > 0: 点在圆内
      = 0: 点在圆上
      < 0: 点在圆外
    """
```

### 5.3 三角形质量计算

```python
def _compute_triangle_quality(self, tri: Triangle):
    """
    质量度量：q = 2 * r_inscribed / r_circumscribed

    r_inscribed = Area / s  (s = 半周长)
    r_circumscribed = circumradius

    范围：[0, 1]，1 为等边三角形
    """
```

### 5.4 星形性验证

```python
def _validate_star_shaped(self, shell_edges, new_point_idx, old_cavity_tris):
    """
    参考 Gmsh 体积守恒检查：
    |oldArea - newArea| < EPS * oldArea

    EPS = 1e-10
    确保插入点不会产生重叠或空洞
    """
```

---

## 6. 边界边恢复

### 6.1 边界恢复策略

```python
def _recover_boundary_edges(self) -> int:
    """
    1. 检测丢失的边界边
       - 遍历所有 protected_edges
       - 检查是否有三角形包含该边的两个顶点

    2. 对每条丢失的边：
       a. 优先尝试边翻转 (_recover_edge_by_flipping)
       b. 翻转失败则中点插入 (_insert_midpoint_for_edge)
    """
```

### 6.2 边翻转恢复

```python
def _recover_edge_by_flipping(self, v1: int, v2: int, max_iter: int = 500):
    """
    迭代查找与目标边 (v1, v2) 相交的网格边：
    1. _find_intersecting_edge(v1, v2)
       - 使用叉积判断线段相交
       - 返回相交的网格边 (a_idx, b_idx)

    2. _flip_edge(a_idx, b_idx)
       - 检查四边形是否为凸四边形
       - 删除两个旧三角形
       - 创建两个新三角形
       - 重建邻接关系

    3. 重复直至 (v1, v2) 成为三角形边或达到最大迭代次数
    """
```

### 6.3 中点插入恢复（回退策略）

```python
def _insert_midpoint_for_edge(self, v1: int, v2: int):
    """
    1. 计算边中点
    2. KDTree 检查最小距离
    3. 插入中点 (_insert_point_incremental)
    4. 重建邻接关系
    """
```

---

## 7. 孔洞处理

### 7.1 孔洞清理流程

```python
def _remove_hole_triangles(self) -> int:
    """
    1. 修复孔洞多边形方向
       - is_polygon_clockwise() 检测
       - 顺时针 → 逆时针（反转）

    2. 删除孔洞内三角形
       - 检查质心是否在孔洞内 (point_in_polygon)
       - 检查顶点是否在孔洞内（排除边界点）

    3. 删除孔洞内孤立节点
       - 收集所有使用的节点
       - 找出未使用的节点（orphan）
       - 删除在孔洞内的孤立节点
       - 重建索引映射
    """
```

### 7.2 边界环提取

```python
def _extract_boundary_loops_from_fronts(boundary_front):
    """
    1. 构建邻接表
       - 遍历所有 Front 边
       - 节点哈希 → 相邻节点列表

    2. 追踪连续边形成环
       - 从未访问节点开始
       - 沿邻接表追踪直到回到起点

    3. 区分外边界和孔洞
       - 计算每个环的质心和面积
       - 如果质心在另一个更大的环内 → 孔洞
       - 否则 → 外边界
    """
```

---

## 8. Laplacian 平滑

### 8.1 平滑策略

```python
def _laplacian_smoothing(self, iterations: int = 3, alpha: float = 0.5):
    """
    参考 Gmsh laplaceSmoothing：

    1. 度量加权平均
       - 权重 = 1 / distance（距离越近权重越大）
       - target_pos = Σ(neighbor_pos * weight) / Σ(weight)

    2. 逐步衰减因子
       - FACTOR = 1.0
       - 每次尝试失败：FACTOR /= 1.4
       - 最多尝试 5 次

    3. 移动接受准则
       - trial_quality >= best_quality * 0.95
       - 允许 5% 的质量损失

    4. 边界约束
       - 边界点保持不动
       - 内部点约束在边界包围盒内（留 margin）
    """
```

---

## 9. 性能优化策略

### 9.1 数据结构优化

| 优化技术 | 实现方式 | 效果 |
|---------|---------|------|
| **`__slots__`** | Triangle 类使用 `__slots__` | 减少约 40% 内存 |
| **顶点索引排序** | `tuple(sorted([p1, p2, p3]))` | 便于去重和比较 |
| **外接圆缓存** | `circumcircle_valid` 标志 | 避免重复计算 |
| **显式邻接关系** | `neighbors[3]` 数组 | Cavity 查找从 O(n) 降为 O(1) |
| **优先级队列** | `heapq` 最小堆 | O(log n) 访问最差三角形 |
| **KDTree** | `scipy.spatial.KDTree` | O(log n) 最近邻查询 |
| **动态 KD 树更新** | 点数变化 > 10% 重建 | 平衡准确性和性能 |

### 9.2 计算优化

```python
1. 缓存策略：
   - circumcircle_valid / quality_valid 标志
   - 计算结果缓存到 Triangle 对象

2. 提前终止：
   - 点数 > 100000
   - 迭代 > 50000
   - 连续失败 > 500 次
   - 堆为空

3. 增量更新：
   - _update_adjacency_after_insertion() 只更新受影响部分
   - 优先级队列每 50 次迭代重建（而非每次）

4. 边界框快速剔除：
   - circumcircle_bbox 用于快速判断点是否可能在圆内
```

---

## 10. 关键参数

| 参数 | 含义 | 默认值 | 影响 |
|-----|------|--------|------|
| **max_edge_length** | 全局最大边长 | `None` | 网格密度 |
| **target_triangle_count** | 目标三角形数 | `None` | 网格分辨率 |
| **smoothing_iterations** | 平滑迭代次数 | `0` | 网格平滑度 |
| **seed** | 随机种子 | `None` | 可重复性 |
| **holes** | 孔洞边界 | `[]` | 孔洞处理 |
| **max_iterations** | 最大迭代次数 | `50000` | 计算复杂度上限 |
| **max_points** | 最大节点数 | `100000` | 内存上限 |
| **min_dist_threshold** | 最小点间距 | `0.01 * avg_edge` | 网格密度下限 |
| **max_consecutive_failures** | 最大连续失败次数 | `500` | 终止条件 |

---

## 11. 算法复杂度分析

| 阶段 | 时间复杂度 | 空间复杂度 | 说明 |
|-----|-----------|-----------|------|
| **初始剖分** | O(n²) 最坏 | O(n) | n 为边界点数 |
| **邻接构建** | O(n log n) | O(n) | 边映射排序 |
| **点插入** | O(N · k) 均摊 | O(N) | N 为插入点数，k 为平均 Cavity 大小 |
| **Cavity 搜索** | O(k) | O(k) | k 为空腔大小 |
| **边界恢复** | O(m · max_iter) | O(1) | m 为丢失边数 |
| **孔洞清理** | O(N · h) | O(h) | h 为孔洞数 |
| **Laplacian 平滑** | O(iter · N) | O(N) | iter 为迭代次数 |
| **总体** | O(N²) 最坏 | O(N) | N 为最终节点数 |

---

## 12. 代码模块结构

```
delaunay/
├── __init__.py                    # 包入口，导出公共 API
├── bw_core.py (1318 行)           # 核心实现
│   ├── Triangle                   # 三角形数据结构
│   └── BowyerWatsonMeshGenerator  # 主生成器类
│       ├── 外接圆计算
│       │   ├── _compute_circumcircle
│       │   ├── _point_in_circumcircle
│       │   └── _robust_incircle
│       ├── 质量计算
│       │   ├── _compute_triangle_quality
│       │   └── _get_target_size_for_triangle
│       ├── 三角剖分
│       │   ├── _create_super_triangle
│       │   ├── _triangulate
│       │   ├── _insert_point_incremental
│       │   └── _find_cavity_with_protection
│       ├── 邻接关系
│       │   ├── _build_adjacency
│       │   └── _update_adjacency_after_insertion
│       ├── 迭代插点
│       │   ├── _insert_points_iteratively
│       │   ├── _compute_insertion_point
│       │   ├── _optimal_point_frontal
│       │   └── _validate_new_point
│       ├── 平滑
│       │   ├── _laplacian_smoothing
│       │   ├── _compute_vertex_quality
│       │   └── _compute_vertex_quality_at
│       ├── 孔洞处理
│       │   └── _remove_hole_triangles
│       ├── 边界恢复
│       │   ├── _recover_boundary_edges
│       │   ├── _recover_single_edge
│       │   ├── _recover_edge_by_flipping
│       │   ├── _find_intersecting_edge
│       │   ├── _flip_edge
│       │   └── _insert_midpoint_for_edge
│       └── 公共入口
│           └── generate_mesh
│
├── helpers.py (187 行)            # 辅助函数
│   ├── _extract_boundary_loops_from_fronts
│   └── create_bowyer_watson_mesh
│
└── bowyer_watson.py               # 向后兼容模块（重新导出）
```

---

## 13. 测试覆盖

### 13.1 单元测试 (`unittests/test_bowyer_watson.py`, 1304 行)

| 测试类 | 覆盖内容 |
|--------|----------|
| `TestBowyerWatsonBasic` | 基础功能：正方形、圆形、尺寸场集成 |
| `TestBowyerWatsonQuality` | 网格质量评估 |
| `TestBowyerWatsonCASFiles` | 实际算例（quad_quad, NACA0012, 30P30N） |
| `TestBowyerWatsonEdgeCases` | 边界条件：最小边界、平滑次数、随机种子、凹多边形 |
| `TestBowyerWatsonIntegration` | 与 core.py 集成测试 |
| `TestBowyerWatsonJSONConfig` | JSON 配置测试（多种翼型，含/不含边界层） |

### 13.2 改进验证 (`test_bw_core_improvements.py`, 307 行)

| 测试函数 | 验证内容 |
|---------|----------|
| `test_basic_square` | 基本正方形 + 邻接关系验证 |
| `test_circle` | 圆形边界 + Robust Predicates + 优先级队列 |
| `test_hole` | 带孔洞网格 + 星形性验证 |
| `test_complex_boundary` | L 形边界 + 前端方法 |
| `performance_comparison` | 性能对比（优先级队列优化） |
| `test_adjacency_consistency` | 邻接关系双向性验证 |

---

## 14. 总结

### 14.1 核心优势

✅ **增量式插点**：避免全量重剖分，提高效率
✅ **显式邻接关系**：Cavity 查找从 O(n) 降为 O(1)
✅ **优先级队列优化**：O(log n) 访问最差三角形
✅ **边界边保护**：类似 Gmsh 的 `internalEdges` 机制
✅ **边界恢复**：边翻转 + 中点插入双重策略
✅ **鲁棒谓词**：Shewchuk 的 Robust Predicates（简化版）
✅ **前端方法**：最优插入点计算提高网格质量
✅ **Laplacian 平滑**：度量加权 + 质量接受准则
✅ **孔洞处理**：自动检测 + 方向修复 + 完整清理

### 14.2 关键创新

🔹 **动态 KD 树更新**：点数变化 > 10% 重建，平衡准确性和性能
🔹 **三重插入点策略**：前端方法 → 外接圆圆心 → 重心随机采样
🔹 **增强点验证**：边长 + 角度 + 距离综合检查
🔹 **边界环自动提取**：从 Front 对象自动识别外边界和孔洞

### 14.3 适用场景

- 二维平面区域自动网格划分
- 复杂多边形边界网格生成
- 带孔洞区域网格生成
- 自适应网格细化（AMR）
- 翼型网格生成（NACA, RAE 等）

---

**文档版本：** v1.0
**分析日期：** 2026年4月10日
**源码版本：** PyMeshGen delaunay/bw_core.py (1318 行)
