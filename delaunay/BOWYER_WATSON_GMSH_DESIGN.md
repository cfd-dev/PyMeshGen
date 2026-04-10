# Gmsh Bowyer-Watson 算法实现详细设计文档

## 1. 概述

本文档基于 Gmsh 开源网格生成器的 C++ 源代码，详细分析 Bowyer-Watson Delaunay 三角剖分算法的实现原理、数据结构、核心流程和优化策略。该实现主要用于二维曲面网格生成，支持各向同性与各向异性网格尺寸控制。

**源码位置：** `delaunay/ref/` 目录

---

## 2. 核心算法原理

### 2.1 Bowyer-Watson 算法基本原理

Bowyer-Watson 算法是一种增量插入法（Incremental Insertion）构建 Delaunay 三角网的经典算法。其核心思想是：

1. **初始三角网**：从一个合法的初始三角剖分开始
2. **点插入循环**：逐个插入新顶点
3. **空腔搜索（Cavity Search）**：找到所有外接圆包含新顶点的三角形
4. **空腔删除**：删除这些违反 Delaunay 条件的三角形
5. **重新连接**：将新顶点与空腔边界上的所有边连接，形成新的三角形
6. **终止条件**：当所有三角形的质量满足预设标准时停止

### 2.2 各向异性 Delaunay 准则

Gmsh 的实现扩展了传统的 Bowyer-Watson 算法，支持**各向异性度量空间**（Anisotropic Metric Space）。在传统算法中，判断点是否在三角形外接圆内使用欧氏距离；而 Gmsh 使用度量张量（Metric Tensor）定义的距离：

```
d²(p, q) = (p - q)ᵀ M (p - q)
```

其中 M 是对称正定度量张量：
```
M = | a  b |
    | b  d |
```

---

## 3. 核心数据结构

### 3.1 MTri3 - 三角形包装类

```cpp
class MTri3 {
private:
    bool deleted;              // 标记三角形是否已删除
    MTriangle *base;           // 实际的三角形几何对象
    MTri3 *neigh[3];           // 三个邻居三角形指针
    double circum_radius;      // 外接圆半径（质量度量）

public:
    static int radiusNorm;     // 半径计算模式：2=欧氏, 其他=各向异性
    
    // 构造函数：计算外接圆半径
    MTri3(MTriangle *t, double lc, SMetric3 *metric, 
          bidimMeshData *data, GFace *gf);
    
    // 判断点是否在 circumcircle 内（XY平面）
    int inCircumCircle(const double *p) const;
    
    // 获取/设置状态
    bool isDeleted() const;
    void setDeleted(bool val);
    double getRadius() const;
    void forceRadius(double r);
    
    MTriangle* tri() const;
    MTri3* getNeigh(int i) const;
    void setNeigh(int i, MTri3 *n);
};
```

**关键设计：**
- `deleted` 标志用于懒删除（Lazy Deletion），避免频繁的集合操作
- `neigh[3]` 维护三角形邻接关系，支持 O(1) 的邻居访问
- `circum_radius` 作为三角形质量度量，用于优先级队列排序

### 3.2 bidimMeshData - 二维网格数据管理

```cpp
class bidimMeshData {
public:
    // 顶点参数坐标
    std::vector<double> Us;    // U方向参数
    std::vector<double> Vs;    // V方向参数
    
    // 网格尺寸
    std::vector<double> vSizes;      // 局部网格尺寸
    std::vector<double> vSizesBGM;   // 背景场网格尺寸
    
    // 顶点索引映射
    std::map<MVertex*, int> vertexToIndex;
    
    // 等价顶点映射（处理周期性边界）
    std::map<MVertex*, MVertex*> *equivalence;
    
    // 参数坐标映射
    std::map<MVertex*, SPoint2> *parametricCoordinates;
    
    // 内部边集合（不可修改的边）
    std::set<MEdge> internalEdges;
    
    // 添加顶点
    void addVertex(MVertex *v, double u, double v, double lc, double lcBGM);
    
    // 获取顶点索引
    int getIndex(MVertex *v) const;
    
    // 获取等价顶点
    MVertex* equivalent(MVertex *v) const;
};
```

**关键设计：**
- 使用数组索引而非指针直接访问顶点属性，提高缓存命中率
- 支持两种尺寸场：局部计算的 `vSizes` 和全局背景场 `vSizesBGM`
- 处理周期性边界条件的等价顶点映射

### 3.3 edgeXface - 边-面关系结构

```cpp
struct edgeXface {
    MTri3 *t1;    // 包含该边的三角形
    int i1;       // 边在三角形中的局部索引 (0,1,2)
    int ori;      // 边的方向标记
    
    // 获取边的两个顶点
    MVertex* _v(int i) const;
    
    // 比较运算符（用于排序和查找）
    bool operator<(const edgeXface &other) const;
    bool operator==(const edgeXface &other) const;
};
```

**关键设计：**
- 用于三角形连接关系的构建和维护
- 支持空腔边界（Cavity Shell）的表示

### 3.4 compareTri3Ptr - 三角形比较器

```cpp
struct compareTri3Ptr {
    bool operator()(const MTri3 *a, const MTri3 *b) const {
        // 按外接圆半径降序排列（最差的三角形在前）
        if (a->getRadius() != b->getRadius())
            return a->getRadius() > b->getRadius();
        return a < b;  // 指针地址作为次级排序键
    }
};
```

**关键设计：**
- 用于 `std::set<MTri3*, compareTri3Ptr>` 优先级队列
- 确保每次迭代都能访问到质量最差的三角形

---

## 4. 核心算法流程

### 4.1 主函数：bowyerWatson

```cpp
void bowyerWatson(GFace *gf, int MAXPNT,
                  std::map<MVertex*, MVertex*> *equivalence,
                  std::map<MVertex*, SPoint2> *parametricCoordinates)
```

**执行流程：**

```
┌─────────────────────────────────────────────────┐
│ 1. 初始化阶段                                     │
│    - 创建 bidimMeshData 数据结构                 │
│    - 构建初始三角形集合 AllTris                  │
│    - 连接三角形邻接关系                           │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 2. 主循环（while(1)）                            │
│    ┌─────────────────────────────────────────┐  │
│    │ 2.1 从 AllTris 取出半径最大的三角形 worst│  │
│    │     (AllTris.begin())                   │  │
│    └────────────────┬────────────────────────┘  │
│                     │                            │
│                     ▼                            │
│    ┌─────────────────────────────────────────┐  │
│    │ 2.2 如果 worst 已标记删除                │  │
│    │     - 释放内存                           │  │
│    │     - 从集合中移除                       │  │
│    │     - 继续下一轮                         │  │
│    └────────────────┬────────────────────────┘  │
│                     │                            │
│                     ▼                            │
│    ┌─────────────────────────────────────────┐  │
│    │ 2.3 检查终止条件                         │  │
│    │     - worst->radius < 0.5*sqrt(2)       │  │
│    │     - 顶点数 > MAXPNT                   │  │
│    │     满足则跳出循环                       │  │
│    └────────────────┬────────────────────────┘  │
│                     │                            │
│                     ▼                            │
│    ┌─────────────────────────────────────────┐  │
│    │ 2.4 计算插入点                           │  │
│    │     - circUV(): 计算 circumcenter (UV)  │  │
│    │     - buildMetric(): 计算度量张量       │  │
│    │     - circumCenterMetric(): 修正中心   │  │
│    └────────────────┬────────────────────────┘  │
│                     │                            │
│                     ▼                            │
│    ┌─────────────────────────────────────────┐  │
│    │ 2.5 插入新顶点                           │  │
│    │     - insertAPoint()                    │  │
│    │     - 更新 AllTris                      │  │
│    └─────────────────────────────────────────┘  │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 3. 后处理阶段                                     │
│    - splitElementsInBoundaryLayerIfNeeded()      │
│    - transferDataStructure(): 输出最终网格       │
└─────────────────────────────────────────────────┘
```

### 4.2 点插入函数：insertAPoint

```cpp
static bool insertAPoint(GFace *gf, 
                         std::set<MTri3*, compareTri3Ptr>::iterator it,
                         double center[2],      // 插入点参数坐标
                         double metric[3],      // 度量张量
                         bidimMeshData &data,
                         std::set<MTri3*, compareTri3Ptr> &AllTris,
                         std::set<MTri3*, compareTri3Ptr> *ActiveTris,
                         MTri3 *worst,
                         MTri3 **oneNewTriangle,
                         bool testStarShapeness)
```

**执行流程：**

```
┌─────────────────────────────────────────────────┐
│ 1. 确定搜索起点                                  │
│    - 如果提供了 worst 三角形                    │
│    - 否则从迭代器获取                           │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 2. 空腔搜索（Cavity Search）                     │
│    ┌─────────────────────────────────────────┐  │
│    │ 2.1 检查 worst 是否被新点破坏            │  │
│    │     inCircumCircleAniso()               │  │
│    └────────────────┬────────────────────────┘  │
│                     │                            │
│                     ▼                            │
│    ┌─────────────────────────────────────────┐  │
│    │ 2.2 如果是：递归查找空腔                 │  │
│    │     recurFindCavityAniso()              │  │
│    │     - 标记所有违反 Delaunay 的三角形    │  │
│    │     - 收集空腔边界边（shell）           │  │
│    └────────────────┬────────────────────────┘  │
│                     │                            │
│                     ▼                            │
│    ┌─────────────────────────────────────────┐  │
│    │ 2.3 如果否：搜索包含新点的三角形         │  │
│    │     search4Triangle()                   │  │
│    │     - 使用重心坐标法定位                 │  │
│    │     - 线段相交测试引导搜索方向          │  │
│    └────────────────┬────────────────────────┘  │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 3. 创建新顶点                                    │
│    - MFaceVertex: 在曲面上的顶点                │
│    - 插值计算局部网格尺寸                       │
│    - 添加到 bidimMeshData                       │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 4. 插入顶点并重新三角化                          │
│    insertVertexB()                              │
│    - 检查空腔是否为星形（Star-shaped）          │
│    - 检查体积守恒                              │
│    - 检查边长不过近                             │
│    - 创建新三角形并连接邻居                     │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 5. 结果处理                                      │
│    - 成功：更新 AllTris/ActiveTris              │
│    - 失败：恢复标记，回退操作                   │
└─────────────────────────────────────────────────┘
```

### 4.3 递归空腔搜索：recurFindCavityAniso

```cpp
static void recurFindCavityAniso(GFace *gf,
                                 std::list<edgeXface> &shell,
                                 std::list<MTri3*> &cavity,
                                 double *metric,    // 度量张量
                                 double *param,     // 插入点参数坐标
                                 MTri3 *t,          // 当前三角形
                                 bidimMeshData &data)
```

**递归逻辑：**

```cpp
1. 标记当前三角形为已删除: t->setDeleted(true)
2. 将当前三角形加入空腔: cavity.push_back(t)
3. 遍历三个邻居 (i = 0, 1, 2):
   a. 获取邻居: neigh = t->getNeigh(i)
   b. 如果邻居不存在 或 是内部边:
      - 将当前边加入 shell: shell.push_back(edgeXface(t, i))
   c. 如果邻居存在且未删除:
      - 检查邻居是否被新点破坏: inCircumCircleAniso()
      - 如果是: 递归处理邻居 recurFindCavityAniso(neigh)
      - 如果否: 将当前边加入 shell

// 递归结束后:
// - cavity 包含所有需要删除的三角形
// - shell 包含空腔的边界边
// - 根据 Euler 公式: shell.size() == cavity.size() + 2
```

### 4.4 顶点插入与重新连接：insertVertexB

```cpp
static int insertVertexB(std::list<edgeXface> &shell,
                         std::list<MTri3*> &cavity,
                         bool force,
                         GFace *gf,
                         MVertex *v,          // 新顶点
                         double *param,
                         MTri3 *t,
                         std::set<MTri3*, compareTri3Ptr> &allTets,
                         std::set<MTri3*, compareTri3Ptr> *activeTets,
                         bidimMeshData &data,
                         double *metric,
                         MTri3 **oneNewTriangle,
                         bool verifyStarShapeness)
```

**核心步骤：**

```
1. 验证空腔有效性:
   - cavity.size() == 1 → 返回 -1 (无法插入)
   - shell.size() != cavity.size() + 2 → 返回 -2 (Euler公式违反)

2. 计算旧空腔体积:
   oldVolume = Σ |getSurfUV(triangle)|

3. 创建新三角形:
   for 每条边界边 (v0, v1) in shell:
   - 创建新三角形: MTriangle(v0, v1, v)
   - 计算局部尺寸: lc = avg(data.vSizes[v0], vSizes[v1], vSizes[v])
   - 包装为 MTri3: new MTri3(t, lc, ...)
   - 检查点是否过近: d(v0,v) < 0.5*lc 或 d(v1,v) < 0.5*lc
   - 检查角度是否过钝: cos(angle) < -0.9999

4. 体积守恒验证:
   newVolume = Σ |getSurfUV(newTriangle)|
   if |oldVolume - newVolume| < EPS * oldVolume:
      连接新三角形邻接关系: connectTris()
      添加到 allTets 和 activeTets
      返回 1 (成功)
   else:
      恢复空腔标记
      删除新三角形
      返回 -3 (非星形空腔)
```

---

## 5. 关键几何计算

### 5.1 外接圆计算（参数空间）

```cpp
static void circumCenterMetric(double *pa, double *pb, double *pc,
                               const double *metric,
                               double *x,           // 输出: 外接圆心
                               double &Radius2)     // 输出: 外接圆半径平方
```

**数学原理：**

求解方程组：
```
(x - pa)ᵀ M (x - pa) = (x - pb)ᵀ M (x - pb) = (x - pc)ᵀ M (x - pc)
```

展开为线性方程组：
```
sys[0][0] = 2[a(pa[0]-pb[0]) + b(pa[1]-pb[1])]
sys[0][1] = 2[d(pa[1]-pb[1]) + b(pa[0]-pb[0])]
sys[1][0] = 2[a(pa[0]-pc[0]) + b(pa[1]-pc[1])]
sys[1][1] = 2[d(pa[1]-pc[1]) + b(pa[0]-pc[0])]

rhs[0] = a(pa[0]²-pb[0]²) + d(pa[1]²-pb[1]²) + 2b(pa[0]pa[1]-pb[0]pb[1])
rhs[1] = a(pa[0]²-pc[0]²) + d(pa[1]²-pc[1]²) + 2b(pa[0]pa[1]-pc[0]pc[1])

求解: sys * x = rhs
```

### 5.2 各向异性点在圆内测试

```cpp
int inCircumCircleAniso(GFace *gf, double *p1, double *p2, double *p3,
                        double *uv,      // 测试点
                        double *metric)
```

**判断逻辑：**

```cpp
1. 计算外接圆心 x 和半径平方 Radius2
2. 计算测试点到圆心的度量距离:
   d0 = x[0] - uv[0]
   d1 = x[1] - uv[1]
   d3 = d0²*a + d1²*d + 2*d0*d1*b
3. 计算容差: tolerance = computeTolerance(Radius2)
4. 返回: d3 < Radius2 - tolerance
```

**容差策略：**
```cpp
static double computeTolerance(const double radius) {
    if (radius <= 1e3) return 1e-12;
    if (radius <= 1e5) return 1e-11;
    return 1e-9;
}
```

### 5.3 度量张量构建

```cpp
void buildMetric(GFace *gf, double *uv, double *metric)
{
    // 计算曲面在该点的第一基本形式
    Pair<SVector3, SVector3> der = gf->firstDer(SPoint2(uv[0], uv[1]));
    
    metric[0] = dot(der.first(), der.first());   // E = r_u · r_u
    metric[1] = dot(der.second(), der.first());  // F = r_v · r_u
    metric[2] = dot(der.second(), der.second()); // G = r_v · r_v
}
```

**几何意义：** 度量张量捕获了参数空间到物理空间的局部变形，使得算法可以在参数空间工作，同时保证物理空间的 Delaunay 性质。

---

## 6. 初始网格构建

### 6.1 网格生成数据结构构建

```cpp
bool buildMeshGenerationDataStructures(GFace *gf, 
                                       std::set<MTri3*, compareTri3Ptr> &AllTris,
                                       bidimMeshData &data)
```

**执行流程：**

```
1. 收集所有顶点:
   - 遍历 gf->triangles 收集顶点
   - 初始化 vSizesMap 为 -1

2. 计算局部网格尺寸:
   for 每个三角形:
     for 每条边 (vi, vj):
       l = distance(vi, vj)
       vSizesMap[vi] = min(vSizesMap[vi], l)
       vSizesMap[vj] = min(vSizesMap[vj], l)

3. 处理嵌入几何:
   - 嵌入顶点: 使用 prescribedMeshSizeAtVertex()
   - 嵌入边: 添加到 internalEdges (不可修改)
   - 小边保护: 防止尺寸场污染

4. 构建 bidimMeshData:
   for 每个顶点:
     - 计算参数坐标: reparamMeshVertexOnFace()
     - 添加顶点: data.addVertex(v, u, v, lc, lcBGM)

5. 创建初始 MTri3 集合:
   for 每个三角形:
     - 计算平均尺寸: lc = avg(vSizes[v0], vSizes[v1], vSizes[v2])
     - 创建包装: AllTris.insert(new MTri3(tri, lc, ...))

6. 清空原始三角形列表: gf->triangles.clear()

7. 构建邻接关系: connectTriangles(AllTris)
```

### 6.2 三角形连接关系构建

```cpp
template <class Iterator>
static void connectTris(Iterator beg, Iterator end,
                        std::vector<edgeXface> &conn)
```

**算法：**

```
1. 收集所有边-面对:
   for 每个三角形 t:
     for i = 0, 1, 2:
       conn.push_back(edgeXface(t, i))

2. 排序（相同边相邻）:
   std::sort(conn.begin(), conn.end())

3. 匹配邻居:
   for i = 0 to conn.size()-2:
     if conn[i] == conn[i+1] 且 t1 != t2:
       conn[i].t1->setNeigh(conn[i].i1, conn[i+1].t1)
       conn[i+1].t1->setNeigh(conn[i+1].i1, conn[i].t1)
       i++  // 跳过已匹配的
```

---

## 7. 变体算法

### 7.1 Frontal Delaunay (bowyerWatsonFrontal)

**核心思想：** 使用前沿推进策略，从活动三角形边界插入新点，而非随机选择最差三角形。

**关键差异：**

```cpp
1. 维护两个集合:
   - AllTris: 所有三角形
   - ActiveTris: 活动边界上的三角形

2. 活动三角形判断:
   bool isActive(MTri3 *t, double limit_, int &active) {
     for i = 0, 1, 2:
       neigh = t->getNeigh(i)
       if (!neigh || (neigh->radius < limit_ && neigh->radius > 0)):
         active = i
         return true
     return false
   }

3. 最优点计算:
   optimalPointFrontalB():
   - 计算活动边中点
   - 沿外接圆心方向推进
   - 距离由尺寸场和几何约束决定:
     L = min(d, q)
     d = ρ̂ * √3 / 2  // 理想三角形高度
     q = 当前距离

4. 表面投影:
   - 使用曲线-曲面求交确保点在曲面上
   - intersectCurveSurface()
```

### 7.2 无限范数 Delaunay (bowyerWatsonFrontalLayers)

**适用场景：** 生成结构化边界层网格，支持四边形主导网格。

**关键特性：**

```cpp
1. 使用 L∞ 范数（无穷范数）:
   lengthInfniteNorm(p, q, quadAngle):
   - 旋转到背景网格对齐坐标系
   - 返回 max(|x1-x2|, |y1-y2|)

2. 平行四边形优化:
   optimalPointFrontalQuad():
   - 生成对齐背景场的点
   - 支持四边形重组

3. 多层推进:
   - 按层迭代 (max_layers = 4 或 10000)
   - 每层独立处理前沿
```

### 7.3 并行四边形打包 (bowyerWatsonParallelograms)

**核心思想：** 预先生成候选点集（平行四边形打包），然后批量插入。

```cpp
1. 候选点生成:
   - packingOfParallelograms() 或 Filler2D::pointInsertion2D()
   - 返回 packed (顶点集) 和 metrics (度量张量集)

2. Hilbert 排序:
   SortHilbert(packed)  // 提高缓存局部性

3. 逐点插入:
   for 每个候选点:
     - 从 AllTris 获取最差三角形
     - insertAPoint() 插入
     - 维护 oneNewTriangle 用于增量定位

4. 懒删除优化:
   if (1.0 * AllTris.size() > 2.5 * DATA.vSizes.size()):
     清理所有已删除三角形
```

---

## 8. 鲁棒性处理

### 8.1 鲁棒谓词（Robust Predicates）

使用 Shewchuk 的自适应精度浮点谓词：

```cpp
class robustPredicates {
    // 2D 方向测试（三点定向）
    double orient2d(double *pa, double *pb, double *pc);
    
    // 2D 圆内测试
    double incircle(double *pa, double *pb, double *pc, double *pd);
    
    // 3D 球内测试
    double insphere(double *pa, double *pb, double *pc, double *pd, double *pe);
    
    // 3D 方向测试
    double orient3d(double *pa, double *pb, double *pc, double *pd);
};
```

**使用方式：**
```cpp
int inCircumCircle(MTriangle *t, const double *p, const double *param,
                   bidimMeshData &data) {
    // 参数空间圆内测试
    double result = robustPredicates::incircle(pa, pb, pc, param) *
                    robustPredicates::orient2d(pa, pb, pc);
    return (result > 0) ? 1 : 0;
}
```

### 8.2 星形空腔验证

```cpp
// 体积守恒检查
double oldVolume = Σ |getSurfUV(cavity_triangle)|
double newVolume = Σ |getSurfUV(new_triangle)|

if (|oldVolume - newVolume| < EPS * oldVolume):
    // 空腔是星形的
    接受插入
else:
    // 非星形空腔，回退
    恢复标记
    return -3
```

### 8.3 退化顶点处理

```cpp
static void getDegeneratedVertices(GFace *gf, std::set<GEntity*> &degenerated) {
    for 每条边界边 e:
        if (e->getBeginVertex() == e->getEndVertex()):
            if (e->geomType() == GEntity::Unknown):
                degenerated.insert(e->getBeginVertex())
}
```

**应用场景：** 处理退化边界（如周期边界的奇点）。

### 8.4 周期性边界处理

```cpp
// 等价顶点映射
std::map<MVertex*, MVertex*> *equivalence;

// 在 transferDataStructure 中处理
void transferDataStructure(GFace *gf,
                           std::set<MTri3*, compareTri3Ptr> &AllTris,
                           bidimMeshData &data) {
    // 1. 输出所有未删除三角形
    // 2. 统一三角形定向
    // 3. 分割等价三角形
    // 4. 合并等价顶点
    computeEquivalences(gf, data.equivalence);
}
```

---

## 9. 边界恢复

### 9.1 三维边界恢复（meshGRegionBoundaryRecovery.cpp）

**使用 TetGen 的边界恢复算法：**

```cpp
int tetgenmesh::reconstructmesh(void *p, double tol) {
    // 1. 从 GRegion 收集顶点
    // 2. 构建初始 3D Delaunay: delaunayMeshIn3D()
    // 3. 重建表面网格:
    //    - 标记 FACETVERTEX
    //    - 创建 shell faces 和 subsegs
    // 4. 统一边界段: unifysegments()
    // 5. 恢复边界: recoverboundary(t)
    // 6. 挖洞: carveholes()
    // 7. 删除 Steiner 点: suppresssteinerpoints()
    // 8. Delaunay 优化: recoverdelaunay()
    // 9. 网格优化: optimizemesh()
}
```

### 9.2 二维边恢复（recoverEdges）

**使用边交换（Edge Swapping）：**

```cpp
void recoverEdges(std::vector<MTri3*> &t, std::vector<MEdge> &edges) {
    // 1. 找出缺失的边
    for 每个目标边 e:
        if e not in mesh_edges:
            edgesToRecover.push_back(e)
    
    // 2. 逐个恢复
    for 每个待恢复边 (v1, v2):
        while (recoverEdgeBySwaps(t, v1, v2, edges)):
            // 通过局部边交换恢复
}

bool recoverEdgeBySwaps(...) {
    // 找到与目标边相交的网格边
    if (intersection_segments(mesh_edge, target_edge)):
        // 执行 2-2 交换
        if (swapedge(v1, v2, v3, o, t, local_edge)):
            return true  // 继续迭代
    return false  // 无法继续
}
```

---

## 10. 网格优化

### 10.1 后处理优化（meshGFaceOptimize.cpp）

**优化策略：**

```cpp
1. 顶点重定位 (Vertex Relocation):
   - 计算最优位置（最大化最小角度）
   - 投影到曲面
   
2. 边交换 (Edge Swapping):
   - 2-2 交换：改善局部 Delaunay 性质
   - 检查质量提升
   
3. 顶点折叠 (Vertex Collapse):
   _tryToCollapseThatVertex(v1, v2):
   - 合并两个顶点到中点
   - 验证质量提升
   - 更新邻接关系
   
4. 四边形重组 (Recombination):
   RecombineTriangle:
   - 合并相邻三角形
   - 质量度量: η = min(90° - angles)
   - 跨场对齐（Cross Field Alignment）
   
5. 边界层处理:
   splitElementsInBoundaryLayerIfNeeded()
```

### 10.2 质量度量

```cpp
// 形状质量度量 (gamma)
double gammaShapeMeasure() {
    // 等边性度量: η = 4√3 * Area / (a² + b² + c²)
    // 范围: [0, 1], 1 为等边三角形
}

// 外接圆半径
circum_radius = circumradius / lc
// lc: 局部网格尺寸
// 用于判断是否需要进一步加密
```

---

## 11. 性能优化策略

### 11.1 数据结构优化

| 优化技术         | 实现方式                           | 效果                    |
| ---------------- | ---------------------------------- | ----------------------- |
| **懒删除**       | `deleted` 标记，延迟清理           | 避免频繁集合操作        |
| **索引访问**     | `bidimMeshData::getIndex()`        | 提高缓存命中率          |
| **Hilbert 排序** | `SortHilbert(packed)`              | 改善空间局部性          |
| **增量定位**     | `search4Triangle()` 从上次位置开始 | 避免全局搜索            |
| **优先级队列**   | `std::set` 按半径排序              | O(log n) 访问最差三角形 |

### 11.2 计算优化

```cpp
1. 度量张量缓存:
   - 在三角形重心处计算一次
   - 用于所有相关测试
   
2. 容差自适应:
   computeTolerance(Radius2):
   - 小圆: 1e-12
   - 中圆: 1e-11
   - 大圆: 1e-9
   
3. 提前终止:
   - 空腔大小为 1 时拒绝
   - 点过近时拒绝
   - 体积不守恒时回退

4. 批量清理:
   if (AllTris.size() > 2.5 * vSizes.size()):
       清理所有已删除三角形
```

---

## 12. 关键参数

| 参数           | 含义               | 默认值        | 影响         |
| -------------- | ------------------ | ------------- | ------------ |
| **LIMIT_**     | 目标外接圆半径阈值 | `0.5 * √2`    | 网格密度     |
| **MAXPNT**     | 最大顶点数         | 用户指定      | 计算复杂度   |
| **radiusNorm** | 半径计算模式       | `2` (欧氏)    | 质量度量方式 |
| **EPS**        | 星形验证容差       | `1e-12`       | 鲁棒性       |
| **max_layers** | 边界层层数         | `4` / `10000` | 边界层厚度   |

---

## 13. 算法复杂度分析

| 阶段         | 时间复杂度        | 空间复杂度 | 说明              |
| ------------ | ----------------- | ---------- | ----------------- |
| **初始构建** | O(n log n)        | O(n)       | n 为初始顶点数    |
| **点插入**   | O(n · log n) 均摊 | O(n)       | 每次插入 O(log n) |
| **空腔搜索** | O(k)              | O(k)       | k 为空腔大小      |
| **连接更新** | O(k log k)        | O(k)       | k 条边界边排序    |
| **总体**     | O(N · log N)      | O(N)       | N 为最终顶点数    |

**实际性能：**
- 5000 次迭代输出一次日志
- 典型曲面：数万到数十万顶点
- 内存开销：每个三角形约 100 字节

---

## 14. 总结

Gmsh 的 Bowyer-Watson 实现是一个工业级的高质量 Delaunay 网格生成器，具有以下特点：

### 14.1 核心优势

✅ **各向异性支持**：通过度量张量实现自适应网格加密  
✅ **曲面嵌入**：参数空间与物理空间的双射映射  
✅ **鲁棒性**：自适应精度谓词 + 多重验证机制  
✅ **灵活性**：多种变体算法适配不同场景  
✅ **高效性**：懒删除 + 增量定位 + Hilbert 排序  

### 14.2 关键创新

🔹 **Frontal Delaunay**：前沿推进策略生成更均匀网格  
🔹 **L∞ 范数**：支持结构化边界层网格  
🔹 **平行四边形打包**：预优化点分布  
🔹 **边界恢复**：保持几何保真度  

### 14.3 适用场景

- 复杂曲面自动网格划分
- 自适应网格细化（AMR）
- 各向异性物理场仿真网格
- 边界层网格生成
- 四边形主导网格生成

---

## 附录：核心函数调用关系图

```
bowyerWatson()
├── buildMeshGenerationDataStructures()
│   ├── setLcsInit() / setLcs()
│   ├── data.addVertex()
│   └── connectTriangles()
│       └── connectTris()
│
├── [主循环]
│   ├── circUV()
│   ├── buildMetric()
│   ├── circumCenterMetric()
│   │   └── circumCenterMetric() [求解 2x2 线性系统]
│   └── insertAPoint()
│       ├── inCircumCircleAniso()
│       │   └── circumCenterMetric()
│       ├── recurFindCavityAniso() [递归]
│       │   └── inCircumCircleAniso()
│       │   └── recurFindCavityAniso() [递归调用]
│       ├── search4Triangle()
│       │   └── invMapUV() [重心坐标定位]
│       └── insertVertexB()
│           ├── getSurfUV() [面积计算]
│           ├── connectTris()
│           └── isActive()
│
├── splitElementsInBoundaryLayerIfNeeded()
└── transferDataStructure()
    └── computeEquivalences()

---

## 7. 辅助算法详解

### 7.1 终止条件分析

```cpp
// 条件1: 网格尺寸达标
worst->getRadius() < 0.5 * std::sqrt(2.0)

// 条件2: 顶点数量超限
(int)DATA.vSizes.size() > MAXPNT
```

**阈值含义**：
- `0.5 * sqrt(2) ≈ 0.707`
- 当三角形限定圆半径小于此值时，表示局部网格已足够细

### 7.2 相关变体算法

#### `bowyerWatsonFrontal()` - 前沿算法

**文件位置**: `meshGFaceDelaunayInsertion.cpp:1289-1475`

使用 Rebay (JCP 1993) 方法，在 Voronoi 边上插入点。

#### `optimalPointFrontal()` - 最优点计算

**文件位置**: `meshGFaceDelaunayInsertion.cpp:1130-1192`

计算使三角形接近正三角形的最优插入点。

### 7.3 关键算法特点总结

| 特性          | 实现方式                                  |
| ------------- | ----------------------------------------- |
| 数据结构      | `std::set<MTri3*, compareTri3Ptr>` 最小堆 |
| Delaunay 准则 | 各向异性 `inCircumCircleAniso`            |
| 腔体查找      | 递归 `recurFindCavityAniso`               |
| 拓扑维护      | 邻接指针 `MTri3::neigh[3]`                |
| 收敛判断      | 限定圆半径阈值                            |
| 失败恢复      | 标记重置 + 半径强制置负                   |

---

## 8. 搜索与映射算法

### 8.1 搜索算法 `search4Triangle()`

**文件位置**: `meshGFaceDelaunayInsertion.cpp:847-920`

用于在 Delaunay 三角网中定位包含给定点的三角形：

```cpp
static MTri3 *search4Triangle(MTri3 *t, double pt[2], bidimMeshData &data,
                              std::set<MTri3 *, compareTri3Ptr> &AllTris,
                              double uv[2], bool force = false)
{
    // 首先检查当前三角形
    bool inside = invMapUV(t->tri(), pt, data, uv, 1.e-8);
    if(inside) return t;

    // 使用直线段相交搜索穿过三角形的边
    SPoint3 q1(pt[0], pt[1], 0);
    int ITER = 0;
    while(1) {
        // 计算当前三角形重心
        int index0 = data.getIndex(t->tri()->getVertex(0));
        int index1 = data.getIndex(t->tri()->getVertex(1));
        int index2 = data.getIndex(t->tri()->getVertex(2));
        SPoint3 q2((data.Us[index0] + data.Us[index1] + data.Us[index2]) / 3.0,
                   (data.Vs[index0] + data.Vs[index1] + data.Vs[index2]) / 3.0, 0);
        int i;
        for(i = 0; i < 3; i++) {
            int i1 = data.getIndex(t->tri()->getVertex(i == 0 ? 2 : i - 1));
            int i2 = data.getIndex(t->tri()->getVertex(i));
            SPoint3 p1(data.Us[i1], data.Vs[i1], 0);
            SPoint3 p2(data.Us[i2], data.Vs[i2], 0);
            if(intersection_segments_2(p1, p2, q1, q2)) break;
        }
        if(i >= 3) break;
        t = t->getNeigh(i);
        if(!t) break;
        inside = invMapUV(t->tri(), pt, data, uv, 1.e-8);
        if(inside) return t;
        if(ITER++ > (int)AllTris.size()) break;
    }

    // force 模式下穷举搜索
    if(force) {
        for(auto itx = AllTris.begin(); itx != AllTris.end(); ++itx) {
            if(!(*itx)->isDeleted()) {
                inside = invMapUV((*itx)->tri(), pt, data, uv, 1.e-8);
                if(inside) return *itx;
            }
        }
    }
    return nullptr;
}
```

### 8.2 逆向参数映射 `invMapUV()`

**文件位置**: `meshGFaceDelaunayInsertion.cpp:622-641`

将物理空间点映射到参数空间的三角形重心坐标：

```cpp
static bool invMapUV(MTriangle *t, double *p, bidimMeshData &data,
                     double *uv, double tol)
{
    double mat[2][2], b[2];
    int index0 = data.getIndex(t->getVertex(0));
    int index1 = data.getIndex(t->getVertex(1));
    int index2 = data.getIndex(t->getVertex(2));

    double u0 = data.Us[index0], v0 = data.Vs[index0];
    double u1 = data.Us[index1], v1 = data.Vs[index1];
    double u2 = data.Us[index2], v2 = data.Vs[index2];

    mat[0][0] = u1 - u0;
    mat[0][1] = u2 - u0;
    mat[1][0] = v1 - v0;
    mat[1][1] = v2 - v0;

    b[0] = p[0] - u0;
    b[1] = p[1] - v0;
    sys2x2(mat, b, uv);

    return uv[0] >= -tol && uv[1] >= -tol &&
           uv[0] <= 1. + tol && uv[1] <= 1. + tol &&
           1. - uv[0] - uv[1] > -tol;
}
```

### 8.3 体积计算 `getSurfUV()`

**文件位置**: `meshGFaceDelaunayInsertion.cpp:643-680`

```cpp
inline double getSurfUV(MTriangle *t, bidimMeshData &data)
{
    int index0 = data.getIndex(t->getVertex(0));
    int index1 = data.getIndex(t->getVertex(1));
    int index2 = data.getIndex(t->getVertex(2));

    double u1 = data.Us[index0], v1 = data.Vs[index0];
    double u2 = data.Us[index1], v2 = data.Vs[index1];
    double u3 = data.Us[index2], v3 = data.Vs[index2];

    return 0.5 * ((u2-u1)*(v3-v1) - (u3-u1)*(v2-v1));
}
```

---

## 9. Delaunay 准则检查

### `inCircumCircleAniso()`

**文件位置**: `meshGFaceDelaunayInsertion.cpp:280-310`

```cpp
int inCircumCircleAniso(GFace *gf, MTriangle *base, const double *uv,
                        const double *metricb, bidimMeshData &data)
{
    SPoint3 c;
    double x[2], Radius2;
    double metric[3];
    if(!metricb) {
        int index0 = data.getIndex(base->getVertex(0));
        int index1 = data.getIndex(base->getVertex(1));
        int index2 = data.getIndex(base->getVertex(2));
        double pa[2] = {(data.Us[index0]+data.Us[index1]+data.Us[index2])/3.,
                        (data.Vs[index0]+data.Vs[index1]+data.Vs[index2])/3.};
        buildMetric(gf, pa, metric);
    }
    else {
        metric[0] = metricb[0];
        metric[1] = metricb[1];
        metric[2] = metricb[2];
    }

    circumCenterMetric(base, metric, data, x, Radius2);

    const double a = metric[0], b = metric[1], d = metric[2];
    const double d0 = (x[0] - uv[0]);
    const double d1 = (x[1] - uv[1]);
    const double d3 = d0*d0*a + d1*d1*d + 2.0*d0*d1*b;

    return d3 < Radius2;
}
```

**容差策略**:
```cpp
static double computeTolerance(const double radius) {
    if (radius <= 1e3) return 1e-12;
    if (radius <= 1e5) return 1e-11;
    return 1e-9;
}
```

---

**文档版本：** v1.0
**分析日期：** 2026年4月10日
**源码版本：** Gmsh (C) 1997-2024 C. Geuzaine, J.-F. Remacle
