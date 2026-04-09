# Bowyer-Watson 算法详细设计文档

## 一、算法概述

Bowyer-Watson 算法是一种基于 **Delaunay 准则** 的增量式网格生成算法。其核心思想是：
1. 从一个包含所有点的"超级三角形"开始
2. 逐点插入新顶点，每次插入时删除新点外接圆内的三角形（违反 Delaunay 准则）
3. 用新点与 Cavity 边界重新三角化
4. 重复插入直到满足质量/尺寸要求

该算法由参考代码（Gmsh `delaunay/ref/`）扩展，增加了以下特性：
- **边界边保护**：在 Cavity 查找中阻止递归破坏约束边
- **边界边恢复**：通过边翻转或中点插入恢复丢失的边界边
- **尺寸场控制**：集成 QuadtreeSizing 实现自适应网格密度
- **孔洞处理**：支持手动和自动孔洞检测与清理
- **迭代插点**：以外接圆圆心为策略的智能点插入

---

## 二、核心数据结构设计

### 2.1 Triangle 类

**设计目标**：三角形单元的高效表示，带计算缓存避免重复计算。

```python
class Triangle:
    __slots__ = [
        'vertices',           # 顶点索引元组（升序存储）
        'circumcenter',       # 外接圆圆心 (numpy.ndarray)
        'circumradius',       # 外接圆半径 (float)
        'idx',                # 三角形索引（可选）
        'circumcircle_valid', # 外接圆缓存是否有效
        'quality',            # 三角形质量（2*r_inscribed/r_circumscribed）
        'quality_valid',      # 质量缓存是否有效
        'circumcircle_bbox',  # 外接圆包围盒（用于空间加速）
    ]
```

**关键设计决策**：
- 顶点按升序存储：便于去重和比较 (`__eq__`, `__hash__`)
- 使用 `__slots__`：减少内存占用
- 缓存外接圆和质量：避免重复计算（参考 Gmsh 的 `MTri3::circum_radius`）

**参考代码对比**：
- Gmsh 使用 `MTri3` 包装 `MTriangle`，存储 `neigh[3]` 邻接指针
- Python 实现使用全局搜索查找邻接三角形（无显式邻接关系存储）

---

### 2.2 BowyerWatsonMeshGenerator 类

**主要成员变量**：

| 变量 | 类型 | 说明 |
|------|------|------|
| `original_points` | `np.ndarray` | 原始边界点（不变） |
| `points` | `np.ndarray` | 当前所有点（含内部点） |
| `triangles` | `List[Triangle]` | 当前三角形集合 |
| `protected_edges` | `set[frozenset]` | 受保护的边界边集合 |
| `sizing_system` | `QuadtreeSizing` | 尺寸场控制系统 |
| `max_edge_length` | `float` | 全局最大边长 |
| `holes` | `List[np.ndarray]` | 孔洞边界列表 |
| `boundary_mask` | `np.ndarray` | 边界点标记掩码 |
| `boundary_count` | `int` | 边界点数量 |
| `_kdtree` | `KDTree` | 用于最近邻查询的 KD 树 |

---

## 三、算法核心模块

### 3.1 外接圆计算（Circumcircle）

**算法原理**：

对于三角形顶点 `A(ax,ay)`, `B(bx,by)`, `C(cx,cy)`，外接圆圆心 `(ux, uy)` 由以下公式计算：

```
d = 2 * [ax(by - cy) + bx(cy - ay) + cx(ay - by)]

ux = [(ax² + ay²)(by - cy) + (bx² + by²)(cy - ay) + (cx² + cy²)(ay - by)] / d
uy = [(ax² + ay²)(cx - bx) + (bx² + by²)(ax - cx) + (cx² + cy²)(bx - ax)] / d
```

**退化情况处理**：
- 当 `|d| < 1e-12` 时，三角形近似共线，使用质心作为圆心

**缓存策略**：
```python
def _compute_circumcircle(self, tri: Triangle):
    if tri.circumcircle_valid:
        return tri.circumcenter, tri.circumradius
    
    # ... 计算逻辑 ...
    
    tri.circumcenter = center
    tri.circumradius = radius
    tri.circumcircle_valid = True
    tri.circumcircle_bbox = (cx - r, cy - r, cx + r, cy + r)  # 包围盒用于加速
```

**参考代码对比**：

Gmsh 提供多种外接圆计算模式：
1. **参数空间外接圆** (`circUV`)：在 2D 参数 (u,v) 空间中计算
2. **各向异性度量外接圆** (`circumCenterMetric`)：考虑度量张量 `M = [a,b;b,d]` 的距离
3. **3D 空间外接圆** (`circumCenterXYZ`)：直接在物理空间计算

Python 实现仅使用标准欧氏距离，适用于各向同性网格。

---

### 3.2 Delaunay 准则判断

**判断逻辑**：
```python
def _point_in_circumcircle(self, point, tri):
    distance = ||point - tri.circumcenter||
    return distance < tri.circumradius * (1.0 + 1e-10)  # 容差
```

**容差设计**：
- 使用 `1e-10` 容差避免浮点误差导致的边界情况错误判断
- 参考代码使用 **Robust Predicates** (`orient2d`, `incircle`) 实现精确判断

**参考代码的精确谓词**：
```cpp
double result = robustPredicates::incircle(pa, pb, pc, param) *
                robustPredicates::orient2d(pa, pb, pc);
return (result > 0) ? 1 : 0;
```

> **改进建议**：Python 实现可引入 `robust-predicates` 库提高数值稳定性

---

### 3.3 超级三角形创建

**算法流程**：
```
1. 计算点集的包围盒：[min_x, max_x] × [min_y, max_y]
2. 计算扩展量：delta = max(max_x-min_x, max_y-min_y) * 10
3. 创建三个顶点：
   P1 = (min_x - delta, min_y - delta)
   P2 = (max_x + delta, min_y - delta)
   P3 = ((min_x+max_x)/2, max_y + 3*delta)
4. 将超级三角形顶点追加到 points 数组末尾
```

**设计考虑**：
- 超级三角形必须足够大以包含所有点
- 顶点索引使用 `len(points)`, `len(points)+1`, `len(points)+2`
- 剖分完成后删除这三个虚拟顶点

---

### 3.4 Cavity 查找算法（核心）

**Cavity 定义**：
- 所有外接圆包含新插入点的三角形集合
- 这些三角形违反 Delaunay 准则，需要被删除和重三角化

**递归算法** `_find_cavity_with_protection`：

```
输入：start_tri（起始三角形）, point_idx（新点索引）
输出：shell_edges（Cavity 边界边）, cavity_set（Cavity 三角形集合）

算法流程：
1. 如果 start_tri 已处理过或不在 bad_tri_ids 中，返回
2. 将 start_tri 加入 cavity_set
3. 对 start_tri 的三条边：
   a. 如果是受保护边界边：
      - 加入 shell_edges
      - 不继续递归（保护该边）
   b. 否则查找相邻三角形 neighbor：
      - 如果 neighbor 不存在（边界边）：加入 shell_edges
      - 如果 neighbor 在 bad_tri_ids 中：递归处理 neighbor
      - 否则：加入 shell_edges（Cavity 边界）
```

**关键设计**：
- **边界保护**：`_is_protected_edge()` 检查边是否在 `protected_edges` 集合中
- **递归终止条件**：
  1. 遇到保护边
  2. 遇到网格边界
  3. 遇到不违反 Delaunay 准则的三角形

**参考代码对比**：

Gmsh 的 `recurFindCavityAniso` 增加了：
1. **内部边保护**：`data.internalEdges` 包含嵌入边（曲线离散后的边）
2. **各向异性判断**：使用度量空间中的距离判断 Delaunay 准则
3. **Euler 公式验证**：`shell.size() == cavity.size() + 2`

---

### 3.5 增量式点插入

**算法流程** `_insert_point_incremental`：

```
输入：point_idx（新点索引）, triangles（当前三角形集合）
输出：更新后的 triangles

1. 获取新点坐标 point = points[point_idx]
2. 查找所有外接圆包含 point 的三角形 → bad_triangles
3. 如果 bad_triangles 为空，直接返回（点在 Delaunay 区域外）
4. 调用 _find_cavity_with_protection 查找 Cavity：
   - shell_edges：Cavity 边界边
   - cavity_set：需要删除的三角形
5. 从 triangles 中删除 cavity_set 中的所有三角形
6. 对 shell_edges 中的每条边 (v1, v2)：
   - 创建新三角形 Triangle(v1, v2, point_idx)
   - 计算并缓存外接圆
   - 添加到 triangles
7. 返回更新后的 triangles
```

**参考代码的额外验证**：

Gmsh 的 `insertVertexB` 增加了：
1. **星形性验证**：新旧 Cavity 面积守恒 `|oldVolume - newVolume| < EPS`
2. **点间距检查**：新点到边的距离不能过小 `d > LL * 0.5`
3. **质量检查**：新三角形不能有太小的边或太钝的角

---

### 3.6 迭代插点主循环

**策略**：以外接圆圆心为插入点，迭代改进网格质量

**算法流程** `_insert_points_iteratively`：

```
初始化：
  - 设置终止条件（目标三角形数、最大迭代、质量达标）
  - 构建 KD 树用于最近邻查询
  - 计算最小距离阈值 min_dist_threshold

循环直到满足终止条件：
  1. 遍历所有三角形，查找"最差"三角形：
     - 如果 max_edge > target_size * 1.1 → 需要细分
     - 如果 quality < 0.3（有尺寸场）或 < 0.5（无尺寸场）→ 需要细分
     - 选择质量最差的三角形作为 worst_triangle
  
  2. 如果无需改进，退出循环
  
  3. 计算新点位置：
     - 默认：worst_triangle 的外接圆圆心
     - 如果圆心超出边界包围盒：使用重心坐标随机采样
  
  4. 距离检查：
     - 使用 KD 树查询新点到最近已有点的距离
     - 如果 min_dist > min_dist_threshold：接受新点
     - 否则：标记该三角形为失败，跳过
  
  5. 插入新点：
     - 追加到 points 数组
     - 调用 _insert_point_incremental 更新三角剖分
  
  6. 失败处理：
     - 连续失败超过 500 次 → 退出循环
```

**终止条件**（满足任一即停止）：
1. 达到目标三角形数量
2. 所有三角形满足尺寸和质量要求
3. 节点数超过 100,000
4. 迭代次数超过 50,000
5. 连续 500 次插入失败

**参考代码对比**：

Gmsh 的 `bowyerWatson` 使用不同的策略：
1. **优先级队列**：`std::set<MTri3*, compareTri3Ptr>` 按外接圆半径排序
2. **始终处理最差三角形**：`worst = *AllTris.begin()`
3. **各向异性度量**：考虑曲面的一阶导数构建度量张量
4. **前端方法变体** (`bowyerWatsonFrontal`)：在活跃边的中垂线上找最优插入点

---

### 3.7 三角形质量计算

**质量度量**：`quality = 2 * r_inscribed / r_circumscribed`

```
输入：三角形顶点 P1, P2, P3

1. 计算边长：a = ||P2-P1||, b = ||P3-P2||, c = ||P1-P3||
2. 计算半周长：s = (a + b + c) / 2
3. 计算面积（Heron 公式）：
   area = sqrt[s * (s-a) * (s-b) * (s-c)]
4. 计算内切圆半径：r_inscribed = area / s
5. 计算外接圆半径：r_circumscribed（已缓存）
6. 质量：quality = min(2 * r_inscribed / r_circumscribed, 1.0)
```

**质量范围**：
- `1.0`：完美等边三角形
- `0.0`：退化三角形（面积为零）
- 通常要求 `quality > 0.3`（有尺寸场）或 `> 0.5`（无尺寸场）

**参考代码的质量度量**：

Gmsh 使用多种度量方式：
1. **形状质量度量**：`gammaShapeMeasure()` 返回 [0,1]
2. **归一化外接圆半径**：`circum_radius / lc`（越小越好）
3. **各向异性质量**：考虑背景网格方向性的无穷范数

---

### 3.8 边界边恢复

**问题**：Delaunay 三角剖分可能破坏约束边界边

**解决方案**：两阶段恢复策略

#### 阶段 1：边翻转恢复

**算法流程** `_recover_edge_by_flipping`：

```
输入：需要恢复的边界边 (v1, v2)

循环最多 500 次：
  1. 查找与线段 (v1, v2) 严格相交的三角形边 (a, b)
  2. 如果没有相交边，检查 (v1, v2) 是否已存在：
     - 是 → 恢复成功
     - 否 → 恢复失败（可能被其他边阻挡）
  3. 尝试翻转边 (a, b)：
     - 查找共享边 (a, b) 的两个三角形 t1, t2
     - 检查四边形是否为凸四边形
     - 如果是：删除 t1, t2，创建新三角形
     - 如果否：翻转失败，回退到中点插入
```

**边翻转的几何条件**：
- 四边形 `(n1, a, n2, b)` 必须是凸四边形
- 使用叉积判断凸性：所有叉积同号

#### 阶段 2：中点插入恢复

**当边翻转失败时**（非凸四边形）：

```
1. 计算边界边 (v1, v2) 的中点
2. 检查中点是否与已有点过于接近（KD 树查询）
3. 如果可接受：
   - 追加中点到 points
   - 调用 _insert_point_incremental 重新三角化
4. 否则：恢复失败
```

**参考代码对比**：

Gmsh 的 `recoverEdgeBySwaps` 使用类似策略：
1. 查找与目标边相交的边
2. 使用 `intersection_segments()` 判断相交
3. 使用 `diffend()` 避免端点重合的无效翻转
4. 循环直到无法再翻转

---

### 3.9 孔洞处理

**算法流程** `_remove_hole_triangles`：

```
输入：holes（孔洞边界列表）

1. 孔洞方向修正：
   - 如果孔洞是顺时针 → 反转为逆时针
   - 确保所有孔洞使用统一方向

2. 删除孔洞内的三角形：
   对每个三角形：
     a. 计算质心 centroid
     b. 如果 centroid 在任一孔洞内 → 标记删除
     c. 如果质心不在孔洞内，检查顶点：
        - 对内部顶点（非边界点）：如果在孔洞内 → 标记删除

3. 反向删除标记的三角形（保持索引有效）

4. 删除孤立节点：
   - 收集所有三角形使用的节点
   - 找到未使用的节点（orphan_nodes）
   - 删除在孔洞内的孤立内部节点
   - 更新所有三角形的顶点索引（index_map 重映射）
```

**点在多边形内判断**：
- 使用 `point_in_polygon()` 函数（射线法或 winding number 法）
- 参考代码使用 **ANN kdtree** 进行距离判断

**参考代码的 3D 孔洞雕刻**：

Gmsh 的 `carveHole` 针对 3D 网格：
1. 构建 carving 表面的顶点集合
2. 使用 ANN kdtree 进行最近邻搜索
3. 删除距离表面 < threshold 的所有体单元
4. 生成孔洞的离散边界（暴露面）

---

### 3.10 Laplacian 平滑

**算法流程** `_laplacian_smoothing`：

```
输入：iterations（迭代次数）, alpha（平滑因子，默认 0.5）

对每次迭代：
  1. 构建邻接关系：
     neighbor_dict[v] = {与 v 相邻的所有顶点}
  
  2. 对每个顶点 v：
     a. 如果 v 是边界点 → 跳过
     b. 计算邻居中心：center = mean(neighbor_positions)
     c. 更新位置：new_pos = v + alpha * (center - v)
     d. 约束在边界包围盒内：clip(new_pos, bounds)
  
  3. 更新所有点位置
```

**关键设计**：
- 边界点保持不动
- 内部点约束在原始边界包围盒内（margin = 1e-6）
- 默认关闭（`smoothing_iterations = 0`）

**参考代码的改进**：

Gmsh 的 `laplaceSmoothing` 增加了：
1. **度量空间加权**：使用度量张量计算距离权重
2. **逐步衰减因子**：`FACTOR = 1.0, 1/1.4, 1/1.4², ...`
3. **移动接受准则**：
   - 新位置的相邻单元总面积不能显著减小
   - 新位置的最小形状质量不能降低

---

## 四、完整算法流程

### 4.1 主入口 `generate_mesh`

```
输入：target_triangle_count（目标三角形数量）
输出：(points, simplices, boundary_mask)

阶段 1/3：初始三角剖分
  1. 初始化 points = original_points
  2. 设置 boundary_mask
  3. 调用 _triangulate()：
     - 创建超级三角形
     - 逐点插入所有边界点
     - 删除超级三角形
  4. 记录边界点数量

阶段 2/3：迭代插入内部点
  1. 调用 _insert_points_iteratively()
  2. 如果启用孔洞：调用 _remove_hole_triangles()
  3. 调用 _recover_boundary_edges() 恢复边界边

阶段 3/3：Laplacian 平滑（可选）
  1. 如果 smoothing_iterations > 0：
     - 重新设置 boundary_mask
     - 调用 _laplacian_smoothing()
  2. 否则跳过

返回结果：
  - points：所有点坐标 (N, 2)
  - simplices：三角形索引数组 (M, 3)
  - boundary_mask：边界点标记 (N,)
```

### 4.2 数据流图

```
boundary_points ──┐
boundary_edges ───┼──► BowyerWatsonMeshGenerator ──► generate_mesh()
sizing_system ────┤                                    │
holes ────────────┘                                    ▼
                                              (points, simplices, mask)
```

---

## 五、算法复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| 外接圆计算 | O(1) | O(1) | 带缓存，只计算一次 |
| Delaunay 判断 | O(1) | O(1) | 使用缓存的外接圆 |
| Cavity 查找 | O(k) | O(k) | k 为 Cavity 大小 |
| 单点插入 | O(n) | O(1) | n 为三角形总数（邻接搜索） |
| 迭代插点 | O(I·n) | O(n) | I 为迭代次数 |
| 边界边恢复 | O(E·n) | O(1) | E 为丢失边数 |
| 孔洞处理 | O(H·n) | O(n) | H 为孔洞数 |
| Laplacian 平滑 | O(I·n) | O(n) | I 为迭代次数 |

**总复杂度**：O(I·n + E·n + H·n)，其中 I 通常远小于 n

---

## 六、数值稳定性与鲁棒性

### 6.1 当前实现的问题

1. **外接圆计算的浮点误差**：
   - 使用标准公式，在退化三角形（接近共线）时可能不稳定
   - 容差 `1e-10` 可能不够保守

2. **邻接三角形搜索**：
   - `_find_neighbor_triangle` 使用线性扫描 O(n)
   - 参考代码使用显式邻接指针 O(1)

3. **点位置判断**：
   - 未使用 Robust Predicates

### 6.2 参考代码的改进方案

1. **精确几何谓词**：
   ```cpp
   double result = robustPredicates::incircle(pa, pb, pc, param) *
                   robustPredicates::orient2d(pa, pb, pc);
   ```

2. **度量空间距离**：
   ```cpp
   double dist = d1 * metric[0] + 2 * d2 * metric[1] + d3 * metric[2];
   ```

3. **星形性验证**：
   ```cpp
   if (std::abs(oldVolume - newVolume) > EPS * oldVolume)
       return false;  // 非星形 Cavity，拒绝插入
   ```

---

## 七、与参考代码的关键差异

| 特性 | Python 实现 | Gmsh 参考代码 |
|------|------------|--------------|
| **邻接关系** | 全局线性搜索 | 显式 `neigh[3]` 指针 |
| **Delaunay 判断** | 欧氏距离 | 各向异性度量 + Robust Predicates |
| **Cavity 验证** | 无 | 星形性检查（体积守恒） |
| **点间距控制** | KD 树最小距离 | 边长和角度综合检查 |
| **边界保护** | `protected_edges` 集合 | `internalEdges` + 嵌入边 |
| **三角形排序** | 遍历查找最差 | `std::set` 优先级队列 |
| **尺寸场** | QuadtreeSizing | 背景网格 + 曲面导数 |
| **孔洞处理** | 点在多边形内判断 | ANN kdtree 距离阈值 |
| **平滑** | 简单 Laplacian | 度量加权 + 移动接受准则 |
| **前端方法** | 无 | `bowyerWatsonFrontal` 变体 |

---

## 八、改进建议

### 8.1 短期改进（容易实现）

1. **引入邻接关系缓存**：
   - 在 Triangle 中添加 `neighbors: List[Optional[int]]`
   - 每次插入点后更新邻接关系
   - 将 `_find_neighbor_triangle` 从 O(n) 降为 O(1)

2. **使用 Robust Predicates**：
   - 安装 `robust-predicates` Python 包
   - 替换 `_point_in_circumcircle` 中的距离判断

3. **增加星形性验证**：
   - 在 `_insert_point_incremental` 中计算新旧 Cavity 面积
   - 面积差异过大时拒绝插入

### 8.2 中期改进（需要较多工作）

1. **实现优先级队列**：
   - 使用 `heapq` 替代遍历查找最差三角形
   - 按质量或外接圆半径排序

2. **前端方法变体**：
   - 实现 `optimalPointFrontal` 在活跃边中垂线上找最优插入点
   - 提高网格质量

3. **度量空间距离**：
   - 如果集成 QuadtreeSizing 的各向异性信息
   - 使用度量张量改进 Delaunay 判断

### 8.3 长期改进（复杂功能）

1. **四边形重组**：
   - 参考 Gmsh 的 `recombineIntoQuads`
   - 使用贪心或 Blossom 算法配对三角形

2. **边界层网格**：
   - 实现 `bowyerWatsonFrontalLayers`
   - 使用无穷范数生成拉伸单元

3. **3D 扩展**：
   - 集成 TetGen 进行 3D Delaunay
   - 实现 3D 边界恢复和孔洞雕刻

---

## 九、总结

Bowyer-Watson 算法是一个优雅且实用的网格生成方法，其核心思想简单但实现细节丰富。参考代码（Gmsh）展示了工业级实现需要考虑的众多因素：

1. **数值稳定性**：Robust Predicates 避免浮点误差
2. **质量控制**：多种度量方式确保单元质量
3. **约束处理**：边界保护和恢复机制
4. **性能优化**：邻接关系、优先级队列、空间索引
5. **扩展性**：各向异性、前端方法、边界层

当前 Python 实现已经具备基本功能，但在性能、鲁棒性和质量方面还有改进空间。建议根据实际需求逐步引入上述改进。
