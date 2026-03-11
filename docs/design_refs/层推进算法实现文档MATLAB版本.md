# 层推进算法实现文档

## 1. 算法概述

本文档详细描述了基于 advancing front method (AFM) 的边界层网格层推进算法，包括**多方向推进**和**局部步长因子**两大核心特性。算法实现在 `AdLayers2` 类中，主要用于生成各向异性四边形边界层网格。

---

## 2. 数据结构

### 2.1 核心类 AdLayers2 属性

```matlab
properties
    maxLayers;           % 最大推进层数
    fullLayers;          % 完整推进层数（不进行终止判断）
    stopLayer;           % 凸点停止推进的层数
    stopFlag;            % 凸点停止标志
    stopLayerConcav;     % 凹点停止推进的层数
    stopFlagConcav;      % 凹点停止标志

    firstHeight;         % 第一层高度
    growthRate;          % 增长率
    growthMethod;        % 增长方法（如 "geometric"）
    multi_direction;     % 是否启用多方向推进
    useLocalStepSize;    % 是否使用局部步长因子

    relax_fact = 0.8;    % 方向光滑松弛因子
    smooth_iter = 3;     % 光滑迭代次数

    % 几何数据
    normal_point;        % 节点推进方向 (nWallNodes × 2)
    normal_front;        % 阵面法向
    direction_front;     % 阵面方向向量
    neighbor_front;      % 相邻阵面信息 (cell 数组)

    % 特殊点
    special_points;      % 特殊点标记 (1:凸点，-1:凹点，0:普通点)
    convexPoints;        % 凸点全局编号
    concavPoints;        % 凹点全局编号
    betaConvex;          % 凸点夹角

    % 多方向推进相关
    virtualPoints;       % 虚拟点编号
    nConvexPoints;       % 凸点数量
    nConcavPoints;       % 凹点数量
    localSpCoeff;        % 局部步长系数
    nodePair;            % 真实点与虚拟点的对应关系
    directions;          % 多方向推进的方向矢量
    spNodePair;          % 特殊点推进的节点对
    localSpPair;         % 真实点的步长因子缓存

    % 网格拓扑
    Grid_stack;          % 网格单元栈
    cellNodeTopo;        % 单元节点拓扑
    Wall_stack;          % 物面阵面栈
    Wall_nodes;          % 物面节点
    Coord_AFT;           % 节点坐标
end
```

### 2.2 Wall_stack 阵面数据结构

| 列索引 | 含义 | 说明 |
|--------|------|------|
| 1 | node1 | 阵面起点节点编号 |
| 2 | node2 | 阵面终点节点编号 |
| 3 | leftCell | 左单元编号 |
| 4 | rightCell | 右单元编号 |
| 5 | length | 阵面长度 |
| 6 | neighborFront | 相邻阵面编号 |
| 7 | type | 阵面类型 (3:物面 wall, 2:内部面) |

---

## 3. 算法主流程

```
┌─────────────────────────────────────────────────────────┐
│                   AdvancingLayers()                     │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               │
    for layer = 1 to maxLayers            │
          │                               │
          ▼                               │
    ┌─────────────────────────────────┐   │
    │ 1. ComputeNodeNormal(layer)     │   │
    │    - 计算阵面法向                │   │
    │    - 平均到节点                 │   │
    │    - 标记特殊点 (凹凸点)         │   │
    │    - 计算多方向方向矢量          │   │
    └─────────────────────────────────┘   │
          │                               │
          ▼                               │
    ┌─────────────────────────────────┐   │
    │ 2. AdvancingQuadLayers(layer)   │   │
    │    - 初始化多方向 (layer=1)      │   │
    │    - 光滑推进方向               │   │
    │    - 计算步长缩放系数           │   │
    │    - 逐个阵面推进               │   │
    │    - 相交检查 (ALM)             │   │
    │    - 质量检查                   │   │
    └─────────────────────────────────┘   │
          │                               │
          ▼                               │
    ┌─────────────────────────────────┐   │
    │ 3. DeleteInactiveFront()        │   │
    │    删除非活跃阵面               │   │
    └─────────────────────────────────┘   │
          │                               │
          ▼                               │
    ┌─────────────────────────────────┐   │
    │ 4. ComputeWallNodes()           │   │
    │    更新物面节点                 │   │
    └─────────────────────────────────┘   │
          │                               │
          └───────────────────────────────┘
                          │
          ▼
    ┌─────────────────────────────────┐
    │ PostProcessAfterAdvancingLayers │
    │    - 处理虚拟点                 │
    │    - 恢复边界条件               │
    └─────────────────────────────────┘
```

---

## 4. 核心算法详解

### 4.1 节点法向计算 (ComputeNodeNormal)

**目的**：为每个物面节点计算推进方向。

**步骤**：

```matlab
function ComputeNodeNormal(this, layer)
    1. 初始化 localSpCoeff = ones(nWallNodes, 1)
    2. 初始化 special_points = zeros(nWallNodes, 1)
    
    3. 计算阵面几何属性:
       - direction_front = ComputeFrontDirection()  % 阵面方向向量
       - neighbor_front  = ComputeNeighborFront()   % 相邻阵面
       - normal_front    = ComputeFrontNormal()     % 阵面法向
    
    4. if use_ANN:
          ComputeNodeNormalByANN(layer)  % 使用人工神经网络
          return
    
    5. AverageFrontNormalToNode()  % 将阵面法向平均到节点
    
    6. for iNode = 1 to nWallNodes:
           a. 获取相邻阵面 neighborFront
           b. if length(neighborFront) == 1:
                  continue  % 只有 1 个相邻阵面，不是特殊点
           c. 计算相邻阵面的:
              - 方向向量 va, vb
              - 法向量 n1, n2
              - 叉积 crprod = cross([va,0], [vb,0])
              - 点积 dotprod = n1 * n2'
              - 夹角 theta = acos(dotprod / |n1| / |n2|)
           d. MarkSpecialPoints(iNode, crprod, theta, layer, n1)
    
    7. InitializeStopFlag(layer)
end
```

### 4.2 特殊点标记 (MarkSpecialPoints)

**目的**：识别凸点和凹点，并计算多方向推进的方向矢量。

**判定准则**：

```matlab
function MarkSpecialPoints(this, iNode, crprod, theta, layer, n1)
    THETA1 = π/4;    % 最小角度阈值
    THETA2 = π;      % 最大角度阈值
    
    % 1. 判断凹凸性
    if crprod(3) < 0 && THETA1 < theta < THETA2:
        special_points(iNode) = 1    % 凸点
        thetam = abs(theta)          % 凸角夹角为正
    elseif crprod(3) >= 0 && THETA1 < theta < THETA2:
        special_points(iNode) = -1   % 凹点
        thetam = -abs(theta)         % 凹角夹角为负
    else:
        special_points(iNode) = 0    % 普通点
    
    % 2. 记录凹点
    if special_points(iNode) == -1:
        concavPoints.append(node)
        nConcavPoints++
    
    % 3. 计算局部步长因子
    if NOT multi_direction:
        % 单方向推进：所有点都需要局部步长因子
        localSpCoeff(iNode) = 1 - sign(thetam) * |thetam| / π
    else:
        % 多方向推进：只有凹点需要局部步长因子
        if special_points(iNode) == -1 && useLocalStepSize:
            localSpCoeff(iNode) = 1 - sign(thetam) * |thetam| / π
    
    % 4. layer > 1 时，无需判断凸点和计算多方向
    if layer > 1:
        return
    
    % 5. 记录凸点并计算多方向
    if special_points(iNode) == 1:
        convexPoints.append(node)
        nConvexPoints++
        betaConvex.append(theta)
        
        if multi_direction:
            % 计算多方向数量
            num_directions = round(theta / (1.1 * π/3)) + 1
            
            % 等分夹角，生成多个推进方向
            delta_theta = -theta / (num_directions - 1)
            dividedVectors(:, 1) = n1
            for idir = 2 to num_directions:
                rotationMatrix = [cos(delta_theta*(idir-1)), -sin(delta_theta*(idir-1));
                                  sin(delta_theta*(idir-1)),  cos(delta_theta*(idir-1))]
                dividedVectors(:, idir) = rotationMatrix * n1'
            directions.append(dividedVectors')
        else:
            directions.append(normal_point(iNode, :))
end
```

**多方向数量计算公式**：
$$
\text{num\_directions} = \text{round}\left(\frac{\theta}{1.1 \times \pi/3}\right) + 1
$$

**局部步长因子公式**：
$$
\text{localSpCoeff} = 1 - \text{sign}(\theta_m) \times \frac{|\theta_m|}{\pi}
$$

其中 $\theta_m$ 为凹凸角（凸角为正，凹角为负）。

---

### 4.3 多方向推进初始化 (InitializeMultiDirection)

**目的**：在凸角处创建多个推进方向，通过虚拟点和虚拟阵面实现。

**算法步骤**：

```matlab
function InitializeMultiDirection(this)
    for iConvexPoint = 1 to nConvexPoints:
        node = convexPoints(iConvexPoint)
        direction_single = directions{iConvexPoint}  % 多方向矢量
        num_directions = size(direction_single, 1)
        
        % 1. 设置第一个方向为当前凸点的推进方向
        localIndex = (Wall_nodes == node)
        normal_point(localIndex, :) = direction_single(1, :)
        
        % 2. 获取相邻阵面（在添加虚拟阵面之前）
        neighborFront = neighbor_front{localIndex}
        
        % 3. 为第 2 到第 N 个方向创建虚拟点和虚拟阵面
        nodePair_single = node
        node_last = node
        for i = 2 to num_directions:
            a. 添加虚拟点的推进方向:
               normal_point(end+1, :) = direction_single(i, :)
            
            b. 添加虚拟点编号:
               Wall_nodes(end+1) = size(Coord_AFT, 1) + i - 1
            
            c. 设置虚拟点的局部步长因子:
               localSpCoeff(end+1) = localSpCoeff(localIndex)
            
            d. 添加虚拟阵面 (连接上一个点和当前虚拟点):
               Wall_stack(end+1, :) = [node_last, Wall_nodes(end), -1, 0, 0, 
                                       size(Wall_stack,1)+1, 3]
            
            e. 更新 node_last:
               node_last = Wall_nodes(end)
            
            f. 记录节点对:
               nodePair_single.append(node_last)
        
        % 4. 保存真实点与虚拟点的对应关系
        nodePair.append(nodePair_single)
        
        % 5. 将凸点右侧阵面的左侧点更新为最后一个虚拟点
        Wall_stack(neighborFront(2), 1) = Wall_nodes(end)
        
        % 6. 设置虚拟点坐标（与真实凸点坐标相同）
        Coord_AFT(end+1:end+num_directions-1, :) = ones(num_directions-1, 2) * Coord_AFT(node, :)
    
    % 7. 记录所有虚拟点编号
    for i = 1 to length(nodePair):
        pair = nodePair{i}
        nvirtual = length(pair) - 1  % pair 中第一个点是真实点
        virtualPoints.append(pair(2:end))
    
    % 8. 更新物面节点和阵面数量
    nWallNodes = length(Wall_nodes)
    nWallFronts = size(Wall_stack, 1)
end
```

**多方向推进示意图**：

```
        方向 1
          ↑
          │
          ●─────── 真实凸点
         / \
        /   \
       /     \
   方向 N     方向 2
    (虚拟点)   (虚拟点)
```

---

### 4.4 推进方向光滑 (SmoothAdvancingDirection)

**目的**：对推进方向进行光滑处理，避免网格扭曲。

**算法步骤**：

```matlab
function SmoothAdvancingDirection(this)
    relax_fact = 0.5   % 自身方向权重
    smooth_iter = 5    % 光滑迭代次数
    
    for iter = 1 to smooth_iter:
        for iNode = 1 to nWallNodes:
            node = Wall_nodes(iNode)
            
            % 凹点不参与光滑
            if node in concavPoints:
                continue
            
            % 获取相邻节点
            [neighborNodes, neighborFront] = NeighborWallNodes(neighbor_front, node, iNode, Wall_stack)
            num_neighbors = length(neighborNodes)
            
            % 只对有 2 个相邻物面点的节点进行光滑
            if num_neighbors == 2:
                % 获取邻居点的推进方向
                for ineig = 1 to num_neighbors:
                    localIndex = (neighborNodes(ineig) == Wall_nodes)
                    neig_normal(ineig, :) = normal_point(localIndex, :)
                
                % 加权平均：自身方向 + 邻居方向平均值
                normal_point(iNode, :) = relax_fact * normal_point(iNode, :) + 
                                         (1 - relax_fact) * mean(neig_normal, 1)
            
            % 归一化
            normal_point(iNode, :) = normal_point(iNode, :) / |normal_point(iNode, :)|
end
```

**光滑公式**：
$$
\mathbf{n}^{new} = \alpha \cdot \mathbf{n}^{self} + (1-\alpha) \cdot \frac{1}{N}\sum_{i=1}^{N}\mathbf{n}^{neighbor}_i
$$

其中 $\alpha = 0.5$ 为松弛因子。

---

### 4.5 局部步长因子缩放 (AdvancingStepScaleCoeff)

**目的**：根据相邻阵面法向与推进方向的夹角，对局部步长进行缩放。

**算法步骤**：

```matlab
function AdvancingStepScaleCoeff(this, layer)
    % 1. 对真实点进行步长缩放
    for iNode = 1 to nWallNodes:
        node = Wall_nodes(iNode)
        node = SpNode(node)  % 获取步长因子等价点
        
        % 跳过虚拟点和第一层之后的凸点
        if node in virtualPoints:
            continue
        if layer > 1 && node in convexPoints:
            continue
        
        % 获取相邻阵面法向
        neig_normal = GetNeighborFrontNormal(iNode)
        
        % 获取推进方向
        np = normal_point(iNode, :)
        
        % 计算夹角余弦值
        coeff = neig_normal * np'  % 点积
        coeff(coeff == 0) = []     % 去除 0 值（虚拟阵面）
        coeff = mean(coeff)        % 取平均值
        
        % 缩放局部步长因子
        localSpCoeff(iNode) = localSpCoeff(iNode) / coeff
    
    % 2. 对虚拟点的局部步长进行缩放（必须在真实点之后）
    if layer == 1:
        for i = 1 to length(virtualPoints):
            nodeVirtual = virtualPoints(i)
            nodeReal = ValidPoint(nodeVirtual)
            
            coeff = localSpCoeff(nodeReal == Wall_nodes)
            localSpCoeff(nodeVirtual == Wall_nodes) = coeff
            
            % 缓存真实点的步长因子
            InsertLocalSpPair(nodeReal, coeff)
    
    % 3. 对虚拟点推进生成的点的步长进行缩放
    for iNode = 1 to nWallNodes:
        node1 = Wall_nodes(iNode)
        node2 = SpNode(node1)  % 获取多方向上的第一个点
        
        if node1 != node2:  % 在多方向上
            nodeReal = ValidPoint(node2)  % 获取真实点
            localSpCoeff(iNode) = GetLocalSpPair(nodeReal)
end
```

**步长缩放公式**：
$$
\text{localSpCoeff}^{new} = \frac{\text{localSpCoeff}^{old}}{\cos(\theta)}
$$

其中 $\theta$ 为相邻阵面法向与推进方向的夹角。

---

### 4.6 单层推进 (AdvancingQuadLayers)

**目的**：对当前层的所有物面阵面进行推进，生成四边形单元。

**算法步骤**：

```matlab
function AdvancingQuadLayers(this, layer)
    % 1. 初始化多方向（仅第一层）
    if layer == 1 && multi_direction:
        InitializeMultiDirection()
    
    % 2. 计算相邻阵面（不使用 ANN 时）
    if NOT use_ANN:
        neighbor_front = ComputeNeighborFront(Wall_stack, Wall_nodes)
        SmoothAdvancingDirection()
    
    % 3. 计算步长缩放系数
    AdvancingStepScaleCoeff(layer)
    
    % 4. 初始化推进状态
    nWallNodes = length(Wall_nodes)
    nWallFronts = size(Wall_stack, 1)
    flag_point = zeros(nWallNodes, 1)    % 标记节点是否已外推
    next_layer = zeros(nWallNodes, 1)    % 外推点的索引
    
    % 5. 逐个阵面推进
    for ifront = 1 to nWallFronts:
        % 仅处理物面阵面（type == 3）
        if Wall_stack(ifront, 7) != 3:
            continue
        
        nCells = nCells + 1
        
        % 获取阵面节点
        node1In = Wall_stack(ifront, 1)
        node2In = Wall_stack(ifront, 2)
        node1_local = (node1In == Wall_nodes)
        node2_local = (node2In == Wall_nodes)
        
        % 获取节点法向
        n1 = normal_point(node1_local, :)
        n2 = normal_point(node2_local, :)
        
        % 创建临时变量（用于预检查）
        Coord_AFT_tmp = Coord_AFT
        Wall_stack_tmp = Wall_stack
        flag_point_tmp = flag_point
        spNodePair_tmp = spNodePair
        advancingSp = 0
        
        % 推进第一个点（如果未推进）
        if NOT flag_point(node1_local):
            [Coord_AFT_tmp, Wall_stack_tmp, next_layer, spNodePair_tmp, 
             flag_point_tmp, node3In, advancingSp] = AdvancingOnePoint(layer, 
                node1In, node1_local, n1, Coord_AFT_tmp, Wall_stack_tmp, 
                next_layer, spNodePair_tmp, flag_point_tmp, leftPoint=1)
        else:
            [Wall_stack_tmp, node3In, advancingSp] = UpdateExistedFront(
                node1In, node1_local, Wall_stack_tmp, next_layer)
        
        % 推进第二个点（如果未推进）
        if NOT flag_point(node2_local):
            [Coord_AFT_tmp, Wall_stack_tmp, next_layer, spNodePair_tmp, 
             flag_point_tmp, node4In, advancingSp] = AdvancingOnePoint(layer, 
                node2In, node2_local, n2, Coord_AFT_tmp, Wall_stack_tmp, 
                next_layer, spNodePair_tmp, flag_point_tmp, leftPoint=0)
        else:
            [Wall_stack_tmp, node4In, advancingSp] = UpdateExistedFront(
                node2In, node2_local, Wall_stack_tmp, next_layer)
        
        % 添加新阵面
        Wall_stack_tmp = AddFront2Stack(node3In, node4In, Wall_stack_tmp, 
                                        Coord_AFT_tmp, nCells, type=3)
        
        % 更新当前阵面类型
        Wall_stack_tmp(ifront, 3) = nCells
        Wall_stack_tmp(ifront, 7) = 2  % 变为内部面
        
        % 获取真实点编号
        node1Real = ValidPoint(node1In)
        node2Real = ValidPoint(node2In)
        cell_add = unique([node1Real, node2Real, node4In, node3In], 'stable')
        
        % 6. 相交检查 (ALM)
        searchingSp = max(Wall_stack_tmp(ifront, 5), advancingSp)
        frontCandidate = FaceCandidateALM(Wall_stack_tmp, Coord_AFT_tmp, 
                                          node3In, node4In, ratio=3.0, searchingSp)
        
        % 7. 找到最近的特殊点
        [iconvex, iconcav, dist] = NearestSpecialPoint(node1In, node2In)
        
        % 8. 交叉检查
        flagCross = IsCrossALM(node1Real, node2Real, node3In, node4In, 
                               frontCandidate, Wall_stack_tmp, Coord_AFT_tmp)
        
        if NOT flagCross:
            % 9. 距离检查
            ratio = max(localSpCoeff(node1_local), localSpCoeff(node2_local))
            flagFarAway = IsPointFarFromEdgeALM(Wall_stack_tmp, frontCandidate, 
                                                Coord_AFT_tmp, node3In, node4In, 
                                                advancingSp/ratio)
            
            if flagFarAway:
                % 10. 质量检查
                [aspect_ratio, skewness] = QualityCheckALM(Coord_AFT_tmp, cell_add)
                
                % 11. 终止条件判断
                if iconvex > 0:  % 靠近凸点
                    distFlag = Dist2ConvexPoints(dist, iconvex)
                    
                    flag1 = NOT distFlag && layer <= stopLayer(iconvex)
                    flag2 = distFlag && aspect_ratio > 1.2 && skewness < 0.3
                    
                    cond1 = flag1 || flag2
                    
                    if multi_direction:
                        flag3 = IsSpMember(node1In) || IsSpMember(node2In)
                        flag4 = flag3 && layer <= stopLayer(iconvex)
                        flag5 = NOT flag3 && aspect_ratio > 1.2 && skewness < 0.75
                        cond2 = (flag4 || flag5) && cond1
                    else:
                        cond2 = cond1
                
                elseif iconcav > 0:  % 靠近凹点
                    cond1 = layer <= stopLayerConcav(iconcav) && aspect_ratio > 1.2
                    cond2 = cond1
                else:
                    cond2 = true
                
                cond3 = layer <= fullLayers
                flag7 = skewness < 1
                flag8 = layer <= maxLayers
                
                % 12. 更新网格（如果满足条件）
                if (cond2 || cond3) && flag7 && flag8:
                    Wall_stack = Wall_stack_tmp
                    Coord_AFT = Coord_AFT_tmp
                    flag_point = flag_point_tmp
                    spNodePair = spNodePair_tmp
                    cellNodeTopo.append(cell_add)
                else:
                    % 13. 更新停止标志
                    UpdateStopFlag(layer, iconvex, iconcav)
                    Wall_stack(ifront, 7) = 2  % 排除出物面
                    nCells = nCells - 1
end
```

---

### 4.7 单点外推 (AdvancingOnePoint)

**目的**：根据节点法向和步长外推单个点。

**算法步骤**：

```matlab
function [Coord, Wall_stack, next_layer, spNodePair, flag_point, node3In, advancingSp] = 
         AdvancingOnePoint(this, layer, node1In, node1_local, n1, Coord, Wall_stack, 
                          next_layer, spNodePair, flag_point, leftPoint)
    
    % 1. 计算推进步长
    advancingSp = AdvancingLayerStepSize(firstHeight, growthRate, growthMethod, 
                                         layer, localSpCoeff(node1_local))
    
    % 2. 外推节点
    node3 = ExtrudeOnePoint(node1In, Coord, n1, advancingSp)
    
    % 3. 标记已外推
    flag_point(node1_local) = 1
    
    % 4. 添加新节点坐标
    Coord(end+1, :) = node3
    node3In = size(Coord, 1)
    next_layer(node1_local) = node3In
    
    % 5. 记录特殊点推进的节点对
    if node1In in virtualPoints OR node1In in convexPoints:
        spNodePair.append([node1In, node3In])
    else:
        spNodePair = InsertSpNode(node1In, node3In, spNodePair)
    
    % 6. 添加新阵面（保证法向指向单元外）
    if leftPoint:
        Wall_stack = AddFront2Stack(node1In, node3In, Wall_stack, Coord, nCells)
    else:
        Wall_stack = AddFront2Stack(node3In, node1In, Wall_stack, Coord, nCells)
end
```

**步长计算公式**：
$$
\text{advancingSp} = \text{firstHeight} \times \text{growthRate}^{(\text{layer}-1)} \times \text{localSpCoeff}
$$

**外推公式**：
$$
\mathbf{P}_{new} = \mathbf{P}_{old} + \text{advancingSp} \times \mathbf{n}
$$

---

### 4.8 相交检查 (IsCrossALM)

**目的**：检查新生成的阵面是否与已有阵面相交。

**算法步骤**：

```matlab
function flag = IsCrossALM(node1, node2, node3, node4, frontCandidate, Wall_stack, Coord)
    flag = true
    
    % 检查 node1 是否与候选阵面相交
    flagNotCross1 = IsNotCross(node1, frontCandidate, Wall_stack, Coord, 1, node3, node1)
    if flagNotCross1:
        % 检查 node2 是否与候选阵面相交
        flagNotCross2 = IsNotCross(node2, frontCandidate, Wall_stack, Coord, 1, node4, node1)
        if flagNotCross2:
            % 检查 node3 是否与候选阵面相交
            flagNotCross3 = IsNotCross(node3, frontCandidate, Wall_stack, Coord, 1, node4, node1)
            if flagNotCross3:
                flag = false  % 没有相交
end
```

---

### 4.9 后处理 (PostProcessAfterAdvancingLayers)

**目的**：处理虚拟点，恢复真实边界。

**算法步骤**：

```matlab
function PostProcessAfterAdvancingLayers(this)
    % 1. 处理虚拟阵面
    virtual_Stack = []
    for ifront = 1 to size(Wall_stack, 1):
        Wall_stack(ifront, 7) = 3  % 恢复为物面
        
        node1 = Wall_stack(ifront, 1)
        node2 = Wall_stack(ifront, 2)
        
        flag1 = node1 in virtualPoints
        flag2 = node2 in virtualPoints
        
        if flag1 && flag2:
            % 两个点都是虚拟点，删除该阵面
            virtual_Stack.append(ifront)
        elseif flag1:
            % 第一个点是虚拟点，替换为真实点
            node1 = ValidPoint(node1)
            Wall_stack(ifront, 1) = node1
        elseif flag2:
            % 第二个点是虚拟点，替换为真实点
            node2 = ValidPoint(node2)
            Wall_stack(ifront, 2) = node2
        
        if node1 == node2:
            % 两点重合，删除该阵面
            virtual_Stack.append(ifront)
    
    % 2. 删除虚拟阵面
    Wall_stack(virtual_Stack, :) = []
    
    % 3. 恢复层推进的真实边界
    for iface = 1 to size(Grid_stack, 1):
        if Grid_stack(iface, 3) > 0 && Grid_stack(iface, 4) <= 0:
            Grid_stack(iface, 7) = 3
end
```

---

## 5. 终止条件

### 5.1 凸点终止条件

```matlab
% 1. 离凸点较近的阵面，更新停止层数
flag1 = NOT distFlag && layer <= stopLayer(iconvex)

% 2. 离凸点较远的阵面，满足质量条件
flag2 = distFlag && aspect_ratio > 1.2 && skewness < 0.3

cond1 = flag1 || flag2

% 3. 多方向上的层数不能超过凸点的 stopLayer
flag3 = IsSpMember(node1In) || IsSpMember(node2In)
flag4 = flag3 && layer <= stopLayer(iconvex)

% 4. 非多方向上的，满足质量条件
flag5 = NOT flag3 && aspect_ratio > 1.2 && skewness < 0.75

cond2 = (flag4 || flag5) && cond1
```

### 5.2 凹点终止条件

```matlab
cond1 = layer <= stopLayerConcav(iconcav) && aspect_ratio > 1.2
cond2 = cond1
```

### 5.3 通用终止条件

```matlab
cond3 = layer <= fullLayers      % 完整层数内不终止
flag7 = skewness < 1             % 偏斜度条件
flag8 = layer <= maxLayers       % 最大层数限制

最终条件：(cond2 || cond3) && flag7 && flag8
```

---

## 6. 关键公式汇总

### 6.1 步长计算

**等比增长**：
$$
h_n = h_1 \times r^{(n-1)} \times \text{localSpCoeff}
$$

**总高度**：
$$
H_n = h_1 \times \frac{r^n - 1}{r - 1}
$$

### 6.2 局部步长因子

$$
\text{localSpCoeff} = 1 - \text{sign}(\theta_m) \times \frac{|\theta_m|}{\pi}
$$

- 凸角：$\theta_m > 0$，localSpCoeff < 1
- 凹角：$\theta_m < 0$，localSpCoeff > 1

### 6.3 多方向数量

$$
\text{num\_directions} = \text{round}\left(\frac{\theta}{1.1 \times \pi/3}\right) + 1
$$

### 6.4 方向光滑

$$
\mathbf{n}^{new} = \alpha \cdot \mathbf{n}^{self} + (1-\alpha) \cdot \frac{1}{N}\sum_{i=1}^{N}\mathbf{n}^{neighbor}_i
$$

### 6.5 步长缩放

$$
\text{localSpCoeff}^{new} = \frac{\text{localSpCoeff}^{old}}{\cos(\theta)}
$$

---

## 7. 算法特点

| 特性 | 说明 |
|------|------|
| **多方向推进** | 在凸角处生成多个推进方向，通过虚拟点和虚拟阵面实现 |
| **局部步长因子** | 根据凹凸角大小自动调整步长，凸角减小步长，凹角增大步长 |
| **方向光滑** | 通过迭代加权平均光滑推进方向，避免网格扭曲 |
| **相交检查** | 使用 ALM (Advancing Layer Method) 进行阵面相交检查 |
| **质量检查** | 检查单元长细比 (aspect_ratio) 和偏斜度 (skewness) |
| **自适应终止** | 根据特殊点位置和网格质量自适应终止推进 |

---

## 8. 附录：辅助函数

### 8.1 阵面法向计算
```matlab
function normal_front = ComputeFrontNormal(Wall_stack, Coord_AFT)
    for i = 1 to nWallFronts:
        if Wall_stack(i, 5) < 1e-40:
            normal_front(i, :) = [0, 0]
        else:
            normal_front(i, :) = normal_vector(Wall_stack(i,1), Wall_stack(i,2), 
                                               Coord_AFT(:,1), Coord_AFT(:,2))
```

### 8.2 相邻阵面计算
```matlab
function neighbor_front = ComputeNeighborFront(Wall_stack, Wall_nodes)
    for iNode = 1 to nWallNodes:
        node = Wall_nodes(iNode)
        neighbor = [-1, -1]
        for i = 1 to size(Wall_stack, 1):
            if Wall_stack(i, 7) != 3:
                continue
            if Wall_stack(i, 2) == node:  % 左阵面
                neighbor(1) = i
            elseif Wall_stack(i, 1) == node:  % 右阵面
                neighbor(2) = i
        neighbor(neighbor < 0) = []
        neighbor_front{iNode} = neighbor
```

### 8.3 阵面法向平均到节点
```matlab
function AverageFrontNormalToNode(this)
    for iNode = 1 to nWallNodes:
        [neighborWallNodes, neighborFront] = NeighborWallNodes(neighbor_front, 
                                                                node, iNode, Wall_stack)
        if num_neighbors == 1:
            % 未形成 full layer，方向与上一层相同
            normal_point(iNode, :) = Coord_AFT(node, :) - Coord_AFT(nodeInLastLayer, :)
        elseif num_neighbors == 2:
            % Full layer，法向取左右邻点法向的平均
            normal_point(iNode, :) = mean(normal_front(neighborFront, :), 1)
        normal_point(iNode, :) = normal_point(iNode, :) / |normal_point(iNode, :)|
```

---

**文档版本**: 1.0  
**更新日期**: 2026 年 3 月 10 日
