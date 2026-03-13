# AdFront2Hybrid 四边形网格生成详细设计文档

## 1. 概述

本文档详细说明 `backup/AdFront2Hybrid`目录中四边形网格生成算法的实现原理和流程（仅针对`use_ann=0` 的情况，即不使用人工神经网络进行新点预测）。

### 1.1 算法特点

- **方法类型**：基于阵面推进法（Advancing Front Method）的三角形/四边形混合网格生成
- **推进策略**：优先生成四边形，四边形质量不满足要求时退化为三角形
- **核心思想**：从边界阵面向内部推进，每次尝试生成一个四边形单元，若失败则生成三角形单元

### 1.2 主要类结构

```
AdFront2Hybrid (继承自 AdFront2)
├── 属性
│   ├── flag_hybrid        # 混合网格标志（1-四边形，0-三角形）
│   ├── nn_fun_quad        # 神经网络函数（use_ann=0 时不使用）
│   ├── nCells_quad        # 四边形单元计数
│   ├── countMode_tri      # 三角形生成模式计数
│   └── countMode_quad     # 四边形生成模式计数
└── 方法
    ├── AdvancingFront_Hybrid()          # 主推进循环
    ├── GenerateHybridCell()             # 生成混合单元
    ├── ADD_POINT_quad()                 # 四边形新点定义
    ├── GenerateQuads_mode1~4()          # 四种四边形生成模式
    ├── UpdateQuadCells()                # 更新四边形数据
    └── SearchExtraQuadCells()           # 搜索额外四边形
```

## 2. 算法主流程

### 2.1 推进循环（AdvancingFront_Hybrid）

```matlab
function AdvancingFront_Hybrid(this)
    while this.nFronts > 0
        1. InitShortestFront()           % 初始化最短阵面信息
        2. GetSp()                       % 计算当地网格步长
        3. GenerateHybridCell()          % 生成一个混合单元
        4. UpdateQuadCells/TriCells()    % 更新单元数据
        5. PLOT_NEW_FRONT()              % 可视化（可选）
        6. DeleteInactiveFront()         % 删除非活跃阵面
        7. Sort_AFT()                    % 阵面排序
        8. UpdateCounters()              % 更新计数器
    end
end
```

**关键步骤说明**：

1. **阵面选择**：通过 `InitShortestFront()` 选择当前最短的阵面作为活跃阵面
2. **步长计算**：根据尺寸场对象获取当前位置的网格步长 `Sp`
3. **单元生成**：调用 `GenerateHybridCell()` 尝试生成四边形，失败则生成三角形
4. **搜索范围扩展**：若找不到合适节点，以 1.2 倍系数扩大搜索范围 `al`
5. **阵面排序**：使用 `Sort_AFT()` 对阵面进行排序，优化推进顺序

### 2.2 混合单元生成流程（GenerateHybridCell）

```matlab
function GenerateHybridCell(this)
    this.flag_hybrid = 1;
    
    % 步骤 1: 生成四边形候选点（use_ann=0）
    if this.useANN == 0
        this.ADD_POINT_quad();    % 基于几何规则生成 2 个候选点
    end
    
    % 步骤 2: 判断是否退化为三角形
    dis = norm(Pbest1 - Pbest2);
    if dis < 0.2 * this.ds_base
        this.flag_hybrid = 0;     % 两点距离过小，退化为三角形
    end
    
    % 步骤 3: 根据模式生成四边形
    if this.flag_hybrid == 1
        switch this.mode
            case 1: GenerateQuads_mode1()
            case 2: GenerateQuads_mode2()
            case 3: GenerateQuads_mode3()
            case 4: GenerateQuads_mode4()
        end
    end
    
    % 步骤 4: 验证四边形是否有效
    if sum(this.node_select == -1) > 0
        this.flag_hybrid = 0;     % 四边形生成失败
    end
    
    % 步骤 5: 若四边形失败，退化为三角形
    if this.flag_hybrid == 0
        this.GenerateTri();
        while this.node_select == -1
            this.al = 2 * this.al;    % 扩大搜索范围
            this.GenerateTri();
        end
    end
end
```

## 3. 四边形生成四种模式

### 3.1 模式 1：标准四边形生成（两个新点）

**调用条件**：`mode == 1`，活跃阵面的两个端点都需要生成新点

**几何示意**：
```
    Pbest1     Pbest2
      *--------*
      |        |
      |        |
      *--------*
  node1_base  node2_base
```

**核心流程**（GenerateQuads_mode1）：

1. **构建候选点列表**：
   ```matlab
   candidateList1 = NodeCandidate(..., Pbest(1,:), al2 * Sp);
   candidateList2 = NodeCandidate(..., Pbest(2,:), al2 * Sp);
   candidateList1 = NodeCandidate_Best(..., node_best + 1, Sp);
   candidateList2 = NodeCandidate_Best(..., node_best + 2, Sp);
   ```
   - 在 `al2 * Sp` 范围内搜索现有节点作为候选
   - 优先选择新生成的点（`node_best + 1`, `node_best + 2`）

2. **相交判断准备**：
   ```matlab
   Pbest_mid = 0.5 * (Pbest1 + Pbest2);
   node_tmp1 = NodeCandidate(..., Pbest_mid, al * Sp);  % 相交判断涉及的点
   frontCandidate = FrontCandidate(AFT_stack, node_tmp1);
   faceCandidate = FrontCandidate(Grid_stack, node_tmp1);
   ```

3. **质量评估与选择**：
   ```matlab
   for i = 1:M, for j = 1:N
       quality = QualityCheckQuad(node1_base, node2_base, node2, node1, coords);
       Qp(i,j) = quality;
       
       % 惩罚已有节点（优先选择新点）
       if node1 > node_best && node2 > node_best
           Qp = coeff^2 * Qp;
       elseif node1 > node_best || node2 > node_best
           Qp = coeff * Qp;
       end
   end
   ```
   - 计算所有候选组合的质量
   - 对使用已有节点的组合施加惩罚系数（`coeff < 1`）
   - 按质量降序排序，依次尝试

4. **几何约束检查**（对每个候选组合）：
   - **相交检查**：`IsNotCrossAllMode()` 确保新边不与现有阵面/网格边相交
   - **左单元检查**：`IsLeftCell()` 确保节点在阵面左侧（逆时针方向）
   - **内部点检查**：`IsPointInCell()` 确保没有点落在新单元内部
   - **对角线检查**：`IsPointDiagnoal()` 检查对角线约束
   - **邻近检查**：`IsEdgeClose2Point()` 检查新边与点的距离

5. **质量验证**：
   ```matlab
   quality = QualityCheckQuad(...);
   if quality < epsilon
       this.node_select = [-1,-1];  % 质量太差，放弃四边形
       return;
   end
   ```

6. **标志位设置**：
   - 若选择的是新生成的点（`node_best + 1/2`），设置 `flag_best` 标志
   - 更新坐标数组 `xCoord_AFT`, `yCoord_AFT`

### 3.2 模式 2：混合模式（一个 ANN 点 + 一个新点）

**调用条件**：`mode == 2`，一侧使用 ANN 预测点，另一侧生成新点

**注意**：在 `use_ann=0` 的情况下，此模式理论上不会被调用，但代码保留以支持混合策略

**几何示意**：
```
    (ANN 点)      Pbest2
       *--------*
       |        |
       |        |
       *--------*
  node1_base  node2_base
```

**核心差异**（GenerateQuads_mode2）：
- `node_select(1) = node_ann`（ANN 预测点）
- 仅需搜索 `node_select(2)` 的候选点
- 候选点列表长度 `N`，质量矩阵为 `1×N`

### 3.3 模式 3：混合模式（一个新点 + 一个 ANN 点）

**调用条件**：`mode == 3`，与模式 2 对称

**几何示意**：
```
    Pbest1     (ANN 点)
      *--------*
      |        |
      |        |
      *--------*
  node1_base  node2_base
```

**核心差异**（GenerateQuads_mode3）：
- `node_select(2) = node_ann`（ANN 预测点）
- 仅需搜索 `node_select(1)` 的候选点

### 3.4 模式 4：纯 ANN 模式（两个 ANN 点）

**调用条件**：`mode == 4`，两侧都使用 ANN 预测点

**注意**：在 `use_ann=0` 的情况下，此模式不会被调用

**核心差异**（GenerateQuads_mode4）：
- `node_select = node_ann`（两个点都是 ANN 预测）
- 不需要搜索候选点，直接进行几何验证
- 仅需检查相交、对角线、邻近等约束

## 4. 新点定义（ADD_POINT_quad）

### 4.1 算法原理

基于阵面端点的几何邻域信息，按照角度和距离规则生成两个新点。

**输入**：
- `node1`, `node2`：活跃阵面的两个端点
- `neighborNode1`, `neighborNode2`：端点的相邻节点
- `Sp`：当地网格步长

**输出**：
- `Pbest = [P1; P2]`：两个新点的坐标

### 4.2 计算步骤

**步骤 1：计算基向量与距离**
```matlab
A = [xCoord(node1), yCoord(node1)];
B = [xCoord(node2), yCoord(node2)];
C = [xCoord(neighborNode1(1)), yCoord(neighborNode1(1))];
D = [xCoord(neighborNode2(1)), yCoord(neighborNode2(1))];

AB = B - A;
AC = C - A;
BD = D - B;

ds_AB = norm(AB);
ds_AC = norm(AC);
ds_BD = norm(BD);
```

**步骤 2：确定推进距离**
```matlab
sp_method = 1;  % 三种方法可选

if sp_method == 1
    d1 = Sp; d2 = Sp;                    % 直接使用当地步长
elseif sp_method == 2
    d1 = min(1.25*ds_AB, max(0.8*ds_AB, Sp));  % 受阵面长度约束
elseif sp_method == 3
    d1 = min(1.25*ds_AB, max(0.8*ds_AB, Sp, dref)); % 综合约束
end
```

**步骤 3：角度判断与新点生成**

对于 `Pbest1`：
```matlab
theta = AnglePoints(A, B, C);  % 计算∠ABC
theta_d = theta * 180 / pi;

if theta_d < theta1 (120°)
    % 角度过小，直接使用相邻点
    Pbest1 = C;
elseif theta1 <= theta_d < theta2 (200°)
    % 角度适中，沿角平分线方向推进
    base_vec = AB / ds_AB;
    rotationMatrix = [cos(theta/2), -sin(theta/2); 
                      sin(theta/2), cos(theta/2)];
    AP = rotationMatrix * base_vec';
    Pbest1 = A + AP' * d1;
else % theta_d >= theta2
    % 角度过大，沿法向推进
    normal = normal_vector(node1, node2, xCoord, yCoord);
    Pbest1 = A + normal * d1;
end
```

对于 `Pbest2`，采用对称逻辑，基于点 B 和相邻点 D 计算。

### 4.3 角度阈值说明

| 角度范围 | 处理方式 | 说明 |
|---------|---------|------|
| θ < 120° | 使用相邻点 | 阵面端点处角度较尖锐，直接使用已有节点 |
| 120° ≤ θ < 200° | 角平分线推进 | 角度适中，沿角平分线方向生成新点 |
| θ ≥ 200° | 法向推进 | 角度过大（接近平角），沿阵面法向生成 |

## 5. 数据结构更新

### 5.1 四边形单元更新（UpdateQuadCells）

```matlab
function UpdateQuadCells(this)
    this.nCells = this.nCells + 1;
    
    % 更新四边形拓扑
    this.Update_AFT_INFO_quad(node1_base, node2_base, 
                               node_select(2), node_select(1));
    
    % 搜索额外的四边形/三角形
    this.SearchExtraQuadCells();
end
```

### 5.2 阵面信息更新（Update_AFT_INFO_quad）

**功能**：更新四边形单元及其四条边的阵面状态

**核心逻辑**：
```matlab
function Update_AFT_INFO_quad(this, node1, node2, node3, node4)
    % 1. 凸四边形检查
    flagConvexPoly = IsConvexPloygon(node1, node2, node3, node4, coords);
    if flagConvexPoly == 0
        this.nCells = this.nCells - 1;  % 非凸，撤销单元
        return;
    end
    
    % 2. 记录单元拓扑
    this.cellNodeTopo{this.nCells} = [node1, node2, node3, node4];
    this.nCells_quad = this.nCells_quad + 1;
    
    % 3. 更新四条边的阵面信息
    this.AFT_stack = UpdateOneFront(node1, node2, node3, node4, ...);
    this.AFT_stack = UpdateOneFront(node2, node3, node4, node1, ...);
    this.AFT_stack = UpdateOneFront(node3, node4, node1, node2, ...);
    this.AFT_stack = UpdateOneFront(node4, node1, node2, node3, ...);
end
```

**UpdateOneFront 函数逻辑**：
```matlab
function AFT_stack = UpdateOneFront(node1, node2, node3, node4, ...)
    dist = DISTANCE(node1, node2, xCoord, yCoord);
    
    % 判断左右单元属性
    flag1 = IsLeftCell(node1, node2, node3, coords);  % node3 是否在左侧
    flag2 = IsLeftCell(node1, node2, node4, coords);  % node4 是否在左侧
    
    % 检查阵面是否已存在
    [row, direction] = FrontExist(node1, node2, AFT_stack);
    
    if (flag1 == 1 && flag2 == 1)  % 左单元
        if (row ~= -1)  % 阵面已存在
            if (direction == 1)
                AFT_stack(row, 3) = nCells;  % 更新左单元编号
            else
                AFT_stack(row, 4) = nCells;  % 更新右单元编号
            end
        else  % 阵面不存在，创建新阵面（反向存储）
            AFT_stack(end+1,:) = [node2, node1, -1, nCells, dist, nFronts+1, 2];
        end
    else  % 右单元
        if (row ~= -1)
            if (direction == 1)
                AFT_stack(row, 4) = nCells;  % 更新右单元编号
            else
                AFT_stack(row, 3) = nCells;  % 更新左单元编号
            end
        else
            AFT_stack(end+1,:) = [node2, node1, nCells, -1, dist, nFronts+1, 2];
        end
    end
end
```

**阵面数据结构** `AFT_stack`：
```
[node1, node2, leftCell, rightCell, distance, frontIndex, type]
- node1, node2: 阵面端点编号
- leftCell: 左侧单元编号（-1 表示边界）
- rightCell: 右侧单元编号（-1 表示边界）
- distance: 阵面长度
- frontIndex: 阵面索引
- type: 阵面类型（2 表示内部阵面）
```

### 5.3 额外单元搜索（SearchExtraQuadCells）

**目的**：在生成主四边形后，检查是否可以形成额外的四边形或三角形单元

**触发条件**：`flag_best(1)==0 || flag_best(2)==0`（即至少有一个点是已有节点）

**核心思想**：
1. 找到新节点的邻点（通过阵面连接关系）
2. 检查邻点的邻点之间是否相连
3. 若形成闭合回路，则构成新单元

**搜索逻辑**：
```matlab
neighbor1 = NeighborNodes(node_select(1), AFT_stack, -1);
neighbor1 = [node1_base, neighbor1];
neighbor1 = unique(neighbor1);

% 遍历邻点
for i = 1:length(neighbor1)
    neighborNode = neighbor1(i);
    
    % 找到邻点所在的阵面
    for j = 1:size(AFT_stack,1)
        if AFT_stack(j,1) == neighborNode
            neighborNodeOfNeighbor = AFT_stack(j,2);
            
            % 检查邻点的邻点是否也相邻
            if find(neighbor1 == neighborNodeOfNeighbor)
                % 形成三角形
                new_cell(end+1,:) = [node_select(1), neighborNode, 
                                      neighborNodeOfNeighbor, -11];
            end
            
            % 继续搜索邻点的邻点的邻点
            for k = 1:size(AFT_stack,1)
                if AFT_stack(k,1) == neighborNodeOfNeighbor
                    if find(neighbor1 == AFT_stack(k,2))
                        % 形成四边形
                        new_cell(end+1,:) = [node_select(1), neighborNode, 
                                              neighborNodeOfNeighbor, AFT_stack(k,2)];
                    end
                end
            end
        end
    end
end
```

**后处理步骤**：
1. **去无效单元**：移除节点数少于 3 的单元
2. **去重复单元**：移除重复的四边形和三角形
3. **去冲突单元**：若四边形和三角形共用 3 个节点，优先保留三角形
4. **去已有单元**：检查是否已存在于 `cellNodeTopo`
5. **质量检查**：验证四边形质量是否满足 `epsilon` 阈值
6. **方向检查**：通过叉积确保单元节点顺序为逆时针
7. **更新数据结构**：调用 `Update_AFT_INFO_quad` 或 `Update_AFT_INFO_TRI`

## 6. 关键几何检查函数

### 6.1 质量检查（QualityCheckQuad）

**功能**：计算四边形质量指标

**输入**：四个节点编号、坐标数组

**输出**：质量值（0~1，越大越好）

**阈值**：`epsilon`（通常取 0.5）

### 6.2 相交检查（IsNotCrossAllMode）

**功能**：检查新边是否与现有阵面或网格边相交

**模式**：
- `mode=1`：标准四边形模式，检查所有四条新边
- `mode=2,3`：混合模式，检查部分新边
- `mode=4`：纯 ANN 模式，仅检查连接边

**检查对象**：
- 活跃阵面（`AFT_stack`）
- 已生成网格边（`Grid_stack`）

### 6.3 左单元检查（IsLeftCell）

**功能**：判断点是否在阵面左侧（逆时针方向）

**数学原理**：通过向量叉积判断
```matlab
cross_product = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
if cross_product > 0
    flagLeftCell = 1;  % 在左侧
else
    flagLeftCell = 0;  % 不在左侧
end
```

### 6.4 内部点检查（IsPointInCell）

**功能**：检查是否有任何点落在新单元内部

**方法**：点在多边形内判断（射线法或面积法）

### 6.5 对角线检查（IsPointDiagnoal）

**功能**：检查对角线约束，避免形成退化单元

### 6.6 邻近检查（IsEdgeClose2Point）

**功能**：检查新边与点的距离是否过近

**阈值**：`proximity_tol * Sp`（通常取 0.5 倍步长）

## 7. 算法参数说明

### 7.1 核心参数

| 参数名 | 默认值 | 说明 |
|-------|-------|------|
| `al` | 0.8 | 候选点搜索范围系数（四边形） |
| `al` (三角形) | 3.0 | 候选点搜索范围系数（三角形） |
| `epsilon` | 0.5 | 四边形质量阈值 |
| `coeff` | 0.8 | 已有节点的惩罚系数 |
| `theta1` | 120° | 角度判断阈值 1 |
| `theta2` | 200° | 角度判断阈值 2 |
| `al2` | 0.8 | 候选点初始搜索系数 |

### 7.2 扩展系数

- **搜索范围扩展**：当找不到合适节点时，`al = 1.2 * al`
- **候选点范围扩展**：`al2 = 2.0 * al2`，最大到 3.2

## 8. 鲁棒性措施

### 8.1 四边形退化策略

```matlab
if quality < epsilon
    this.flag_hybrid = 0;  % 四边形质量太差
    this.GenerateTri();    % 退化为三角形
end
```

### 8.2 搜索范围动态扩展

```matlab
while sum(this.node_select == -1) == 2
    this.al = 1.2 * this.al;
    this.GenerateHybridCell();
    disp('Expand searching range!');
end
```

### 8.3 阵面回退机制（Python 版本）

```python
def defer_base_front_for_retry(self):
    """将当前活跃前沿回退到堆中，优先尝试其他前沿"""
    retry_count = self.front_retry_counter.get(front_hash, 0)
    if retry_count >= self.max_front_retries_for_quad:
        return False
    
    self.front_retry_counter[front_hash] = retry_count + 1
    heapq.heappush(self.front_list, self.base_front)
    return True
```

### 8.4 阵面排序

```matlab
this.AFT_stack = Sort_AFT(this.AFT_stack);
```

- 按阵面长度排序，优先处理短阵面
- 避免长阵面导致后续网格质量下降

### 8.5 非活跃阵面删除

```matlab
[this.AFT_stack, this.Grid_stack] = DeleteInactiveFront(...);
```

- 定期清理两侧都有单元的阵面
- 减少数据结构规模，提高效率

## 9. 流程图

### 9.1 主推进流程

```
开始
  ↓
选择最短阵面 (InitShortestFront)
  ↓
计算当地步长 (GetSp)
  ↓
生成混合单元 (GenerateHybridCell)
  ├─ 生成 Pbest 点 (ADD_POINT_quad)
  ├─ 选择模式 (mode 1~4)
  ├─ 搜索候选点
  ├─ 质量评估
  ├─ 几何约束检查
  └─ 失败则退化为三角形
  ↓
更新单元数据
  ├─ UpdateQuadCells (四边形)
  └─ UpdateTriCellsHybrid (三角形)
  ↓
可视化 (可选)
  ↓
删除非活跃阵面
  ↓
阵面排序 (Sort_AFT)
  ↓
更新计数器
  ↓
nFronts > 0 ?
  ├─ 是 → 继续循环
  └─ 否 → 结束
```

### 9.2 四边形生成流程（Mode 1）

```
开始 (GenerateQuads_mode1)
  ↓
构建候选点列表 (NodeCandidate)
  ↓
准备相交判断数据
  ↓
计算质量矩阵 Qp(M×N)
  ↓
按质量降序排序
  ↓
遍历候选组合
  ├─ 相交检查 (IsNotCrossAllMode)
  ├─ 左单元检查 (IsLeftCell)
  ├─ 内部点检查 (IsPointInCell)
  ├─ 对角线检查 (IsPointDiagnoal)
  └─ 邻近检查 (IsEdgeClose2Point)
  ↓
找到有效组合？
  ├─ 是 → 设置 node_select
  └─ 否 → 扩大 al2，重试
  ↓
质量验证 (quality >= epsilon?)
  ├─ 是 → 成功
  └─ 否 → 失败，返回 [-1,-1]
```

## 10. 总结

### 10.1 算法优势

1. **高质量网格**：通过严格的质量检查和几何约束，确保生成网格质量
2. **混合策略**：优先四边形，退化三角形，兼顾质量和成功率
3. **鲁棒性强**：多种鲁棒性措施，避免推进中断
4. **自适应步长**：根据尺寸场动态调整网格步长

### 10.2 关键创新点

1. **四种生成模式**：适应不同场景的四边形生成需求
2. **额外单元搜索**：充分利用阵面连接关系，提高生成效率
3. **动态搜索范围**：根据成功率自动调整搜索范围
4. **阵面排序优化**：优先处理短阵面，改善整体网格质量

### 10.3 参考文献

[1] 陈建军，郑建靖，季廷炜，等。前沿推进曲面四边形网格生成算法 [J].计算力学学报，2011,28(05):779-784.

---

**文档版本**：v1.0  
**创建日期**：2026-03-13  
**适用范围**：`backup/AdFront2Hybrid` 目录，`use_ann=0` 情况
