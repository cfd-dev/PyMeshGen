# 多方向推进算法详细设计文档

## 1. 概述

多方向推进（Multi-Direction Advancing）是边界层网格生成中处理**凸角区域**的关键技术。在凸角处，单一推进方向会导致网格严重扭曲或相交。多方向推进通过在凸角处生成**多个等分的推进方向**，并引入**虚拟点**和**虚拟阵面**的概念，使得每个方向独立推进，最终生成高质量的边界层网格。

---

## 2. 核心概念

### 2.1 凸角与多方向需求

```
           单一方向推进                    多方向推进
               
               ↑                              ↑
               │                              │
          ┌────●────┐                    ┌────●────┐
          │         │                    │ \  │  / │
          │         │                    │  \ │ /  │
          │         │                    │   \|/   │
          └─────────┘                    └─────────┘
          网格扭曲/相交                  网格均匀分布
```

**问题**：凸角处相邻阵面夹角 $\theta > \pi/4$ 时，单一推进方向会导致：
- 网格单元严重扭曲
- 阵面自相交
- 长细比超标

**解决**：将凸角等分为多个方向，每个方向独立推进。

### 2.2 基本术语

| 术语 | 定义 |
|------|------|
| **真实点** | 原始几何边界上的节点 |
| **虚拟点** | 为多方向推进创建的辅助点，坐标与真实凸点重合 |
| **虚拟阵面** | 连接虚拟点之间或虚拟点与真实点的阵面 |
| **方向串线** | 从真实凸点出发，沿某一虚拟方向推进生成的节点序列 |
| **nodePair** | 记录真实点与其所有虚拟点的对应关系 |
| **spNodePair** | 记录特殊点推进过程中生成的节点对 |

---

## 3. 数据结构设计

### 3.1 多方向相关核心属性

```matlab
classdef AdLayers2
    properties
        % 多方向开关
        multi_direction;           % 是否启用多方向推进
        
        % 特殊点信息
        convexPoints;              % 凸点全局编号数组 [nConvexPoints × 1]
        nConvexPoints;             % 凸点数量
        betaConvex;                % 凸点夹角数组 [nConvexPoints × 1]
        
        % 方向信息
        directions;                % cell 数组，每个元素存储一个凸点的多方向矢量
                                   % directions{i} = [num_directions × 2] 矩阵
        
        % 虚拟点信息
        virtualPoints;             % 所有虚拟点的全局编号数组
        nodePair;                  % cell 数组，记录真实点与虚拟点的对应关系
                                   % nodePair{i} = [真实点，虚拟点 1, 虚拟点 2, ...]
        
        % 推进过程记录
        spNodePair;                % 特殊点推进的节点对 cell 数组
                                   % spNodePair{i} = [起点，推进点 1, 推进点 2, ...]
        
        % 局部步长
        localSpCoeff;              % 局部步长系数数组 [nWallNodes × 1]
        localSpPair;               % 真实点步长因子缓存 cell 数组
                                   % localSpPair{i} = [节点编号，系数]
    end
end
```

### 3.2 nodePair 数据结构

```
nodePair 是一个 cell 数组，每个元素对应一个凸点：

nodePair{1} = [P_real, P_virt1, P_virt2, P_virt3]
                       │         │         │
                       └─────────┴─────────┘
                          凸点 P 的 3 个虚拟点

nodePair{2} = [Q_real, Q_virt1]
                       │
                       └──── Q 点的 1 个虚拟点

物理意义：
- nodePair{i}(1)  : 第 i 个凸点的真实点编号
- nodePair{i}(2:end) : 第 i 个凸点的所有虚拟点编号
```

**示意图**：

```
凸角 θ = 120°
                    
        方向 1 (真实点)
            ↑
            │
    虚拟点 2 ●
           / 
          / θ/3
         /   
        ● 虚拟点 1
       /
      /
     ● 真实凸点 P
    
nodePair{i} = [P, 虚拟点 1, 虚拟点 2]
```

### 3.3 directions 数据结构

```matlab
directions 是 cell 数组：
directions{1} = [
    0.0,  1.0;    % 方向 1（原始法向）
    0.5,  0.866;  % 方向 2（旋转 60°）
   -0.5,  0.866;  % 方向 3（旋转 120°）
];

directions{2} = [
    1.0,  0.0;    % 方向 1
    0.0,  1.0;    % 方向 2
];
```

### 3.4 spNodePair 数据结构

```matlab
% 记录从特殊点（凸点/虚拟点）推进生成的节点
spNodePair{1} = [虚拟点 A, 推进点 A1, 推进点 A2, ...]
spNodePair{2} = [虚拟点 B, 推进点 B1, 推进点 B2, ...]

% 用途：
% 1. 追踪多方向串线上的所有节点
% 2. 获取节点的局部步长因子
% 3. 判断节点是否在多方向推进路径上
```

---

## 4. 多方向数量计算

### 4.1 计算公式

$$
\text{num\_directions} = \text{round}\left(\frac{\theta}{1.1 \times \pi/3}\right) + 1
$$

其中：
- $\theta$：凸角夹角（相邻阵面法向夹角）
- $1.1 \times \pi/3 \approx 66°$：每个方向覆盖的角度
- $+1$：包含原始方向

### 4.2 角度与方向数量对照表

| 凸角 θ (度) | num_directions | 方向间隔 (度) |
|------------|----------------|--------------|
| 45° ~ 66°  | 2              | 45°          |
| 66° ~ 132° | 2              | θ            |
| 132° ~ 198°| 3              | θ/2          |
| 198° ~ 264°| 4              | θ/3          |
| 264° ~ 330°| 5              | θ/4          |
| 330° ~ 360°| 6              | θ/5          |

### 4.3 计算代码

```matlab
function MarkSpecialPoints(this, iNode, crprod, theta, layer, n1)
    % ... 凹凸点判断 ...
    
    if special_points(iNode) == 1 && multi_direction
        % 计算多方向数量
        num_directions = round(theta / (1.1 * pi/3)) + 1;
        
        % 生成等分方向矢量
        dividedVectors = zeros(2, num_directions);
        delta_theta = -theta / (num_directions - 1);  % 等分夹角
        
        dividedVectors(:, 1) = n1;  % 第一个方向为原始法向
        
        for idir = 2:num_directions
            % 旋转矩阵
            rotationMatrix = [cos(delta_theta * (idir - 1)), -sin(delta_theta * (idir - 1));
                              sin(delta_theta * (idir - 1)),  cos(delta_theta * (idir - 1))];
            dividedVectors(:, idir) = rotationMatrix * n1';
        end
        
        directions{end+1} = dividedVectors';  % 存储为 [num_directions × 2]
    end
end
```

### 4.4 方向矢量生成原理

**输入**：
- 原始法向 $\mathbf{n}_1 = (n_x, n_y)$
- 凸角 $\theta$
- 方向数量 $N$

**输出**：
- $N$ 个等分方向矢量 $\mathbf{d}_1, \mathbf{d}_2, ..., \mathbf{d}_N$

**算法**：

```
1. 计算方向间隔角：Δθ = -θ / (N - 1)
   （负号表示顺时针旋转）

2. 第一个方向：d₁ = n₁

3. 后续方向：dᵢ = R(Δθ × (i-1)) × n₁
   
   其中 R(α) 为旋转矩阵：
   R(α) = [cos(α), -sin(α);
           sin(α),  cos(α)]
```

**示例**（θ = 120°, N = 3）：

```
原始法向 n₁ = (0, 1)  ↑

方向 1: d₁ = (0, 1)           ↑  (0°)
方向 2: d₂ = R(-60°) × n₁     ↗  (-60°)
方向 3: d₃ = R(-120°) × n₁    ↘  (-120°)

结果：
        d₁
        ↑
       /
      / d₂
     ●────→ d₃
```

---

## 5. 多方向初始化算法

### 5.1 初始化时机

- **触发条件**：`layer == 1 && multi_direction == true`
- **调用位置**：`AdvancingQuadLayers()` 函数开始处
- **执行次数**：仅执行一次

### 5.2 初始化流程

```
┌─────────────────────────────────────────────────────────────┐
│              InitializeMultiDirection()                     │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               │
    for 每个凸点 iConvexPoint             │
          │                               │
          ▼                               │
    ┌─────────────────────────────────┐   │
    │ 1. 获取凸点的多方向矢量          │   │
    │    direction_single =           │   │
    │    directions{iConvexPoint}     │   │
    └─────────────────────────────────┘   │
          │                               │
          ▼                               │
    ┌─────────────────────────────────┐   │
    │ 2. 设置第一个方向为真实点方向    │   │
    │    normal_point(localIndex) =   │   │
    │    direction_single(1, :)       │   │
    └─────────────────────────────────┘   │
          │                               │
          ▼                               │
    ┌─────────────────────────────────┐   │
    │ 3. 获取相邻阵面（添加前）        │   │
    │    neighborFront =              │   │
    │    neighbor_front{localIndex}   │   │
    └─────────────────────────────────┘   │
          │                               │
          ▼                               │
    ┌─────────────────────────────────┐   │
    │ 4. 为第 2~N 个方向创建虚拟点和阵面 │
    │    for i = 2 to num_directions  │   │
    │      a. 添加推进方向            │   │
    │      b. 添加虚拟点编号          │   │
    │      c. 复制局部步长因子        │   │
    │      d. 添加虚拟阵面            │   │
    │      e. 记录 nodePair           │   │
    │    end                          │   │
    └─────────────────────────────────┘   │
          │                               │
          ▼                               │
    ┌─────────────────────────────────┐   │
    │ 5. 保存 nodePair                 │   │
    └─────────────────────────────────┘   │
          │                               │
          ▼                               │
    ┌─────────────────────────────────┐   │
    │ 6. 更新右侧阵面连接              │   │
    │    Wall_stack(neighborFront(2), │   │
    │              1) = last_virt_pt  │   │
    └─────────────────────────────────┘   │
          │                               │
          ▼                               │
    ┌─────────────────────────────────┐   │
    │ 7. 设置虚拟点坐标                │   │
    │    Coord_AFT(virt_pts) =        │   │
    │    Coord_AFT(real_pt)           │   │
    └─────────────────────────────────┘   │
          │                               │
          └───────────────────────────────┘
```

### 5.3 详细代码实现

```matlab
function InitializeMultiDirection(this)
    %% 遍历每个凸点
    for iConvexPoint = 1:this.nConvexPoints
        node = this.convexPoints(iConvexPoint);
        direction_single = this.directions{iConvexPoint};
        num_directions = size(direction_single, 1);
        
        %% 1. 设置真实点的推进方向为第一个方向
        localIndex = (this.Wall_nodes == node);
        this.normal_point(localIndex, :) = direction_single(1, :);
        
        %% 2. 获取当前凸点的相邻阵面（在添加虚拟阵面之前）
        neighborFront = this.neighbor_front{localIndex};
        
        %% 3. 创建虚拟点和虚拟阵面
        nodePair_single = node;    % 记录真实点及其虚拟点
        node_last = node;          % 上一个点（初始为真实点）
        
        for i = 2:num_directions
            % a. 添加虚拟点的推进方向（放在数组末尾）
            this.normal_point(end+1, :) = direction_single(i, :);
            
            % b. 添加虚拟点编号（全局编号为新坐标点）
            this.Wall_nodes(end+1) = size(this.Coord_AFT, 1) + i - 1;
            
            % c. 复制真实点的局部步长因子
            this.localSpCoeff(end+1) = this.localSpCoeff(localIndex);
            
            % d. 添加虚拟阵面（连接上一个点和当前虚拟点）
            %    阵面格式：[node1, node2, leftCell, rightCell, length, neighbor, type]
            this.Wall_stack(end+1, :) = [...
                node_last, ...                          % 起点
                this.Wall_nodes(end), ...               % 终点（新虚拟点）
                -1, 0, 0, ...                           % 无单元，长度待计算
                size(this.Wall_stack, 1) + 1, ...       % 相邻阵面编号
                3];                                     % 物面类型
            
            % e. 更新上一个点
            node_last = this.Wall_nodes(end);
            
            % f. 记录到 nodePair
            nodePair_single(end+1) = node_last;
        end
        
        %% 4. 保存 nodePair（真实点与虚拟点的对应关系）
        this.nodePair{end+1} = nodePair_single;
        
        %% 5. 将凸点右侧阵面的左侧点更新为最后一个虚拟点
        %    这样右侧阵面就连接到多方向的最后一个方向上
        this.Wall_stack(neighborFront(2), 1) = this.Wall_nodes(end);
        
        %% 6. 设置虚拟点坐标（与真实凸点坐标相同）
        %    新增虚拟点数 = num_directions - 1
        this.Coord_AFT(end+1:end+num_directions-1, :) = ...
            ones(num_directions-1, 2) .* this.Coord_AFT(node, :);
    end
    
    %% 7. 收集所有虚拟点编号
    npairs = size(this.nodePair, 2);  % 应等于 nConvexPoints
    for i = 1:npairs
        pair = this.nodePair{i};
        nvirtual = length(pair) - 1;  % pair(1) 是真实点
        this.virtualPoints(end+1:end+nvirtual) = pair(2:end);
    end
    
    %% 8. 更新物面节点和阵面数量
    this.nWallNodes = length(this.Wall_nodes);
    this.nWallFronts = size(this.Wall_stack, 1);
    
    %% 9. 初始化新增阵面的法向
    this.normal_front(end+1:this.nWallFronts, :) = 0;
end
```

### 5.4 初始化前后数据结构变化

**初始化前**（单个凸点 P）：

```
Wall_nodes = [A, B, P, C, D]     % 5 个物面点
Wall_stack = [
    [A, B, ...],                 % 阵面 1
    [B, P, ...],                 % 阵面 2（P 的左侧）
    [P, C, ...],                 % 阵面 3（P 的右侧）
    [C, D, ...]                  % 阵面 4
]
normal_point = [
    n_A; n_B; n_P; n_C; n_D     % 5 个方向
]
```

**初始化后**（凸角 120°，3 个方向）：

```
Wall_nodes = [A, B, P, C, D, V1, V2]   % 新增 2 个虚拟点
                    │  └─ 虚拟点 1
                    └──── 虚拟点 2

Wall_stack = [
    [A, B, ...],                     % 阵面 1
    [B, P, ...],                     % 阵面 2（P 的左侧）
    [P, V1, ...],                    % 虚拟阵面 1（新增）
    [V1, V2, ...],                   % 虚拟阵面 2（新增）
    [V2, C, ...],                    % 阵面 3 修改：P→V2
    [C, D, ...]                      % 阵面 4
]

normal_point = [
    n_A; n_B; n_P; n_C; n_D;        % 原始 5 个方向
    d_2; d_3                         % 新增 2 个虚拟方向
]

nodePair{1} = [P, V1, V2]            % 记录对应关系
virtualPoints = [V1, V2]             % 虚拟点列表
```

---

## 6. 多方向推进过程

### 6.1 推进状态追踪

**spNodePair 的作用**：

```matlab
% 在 AdvancingOnePoint 中记录
if node1In in virtualPoints || node1In in convexPoints
    spNodePair{end+1} = [node1In, node3In];  % 从特殊点推出新点
else
    spNodePair = InsertSpNode(node1In, node3In, spNodePair);  % 插入到已有串线
end

% 示例：
spNodePair{1} = [V1, P1_1, P1_2, P1_3]   % 从虚拟点 V1 推进的串线
spNodePair{2} = [V2, P2_1, P2_2, P2_3]   % 从虚拟点 V2 推进的串线
spNodePair{3} = [P,  R_1,  R_2,  R_3]    % 从真实点 P 推进的串线
```

### 6.2 InsertSpNode 函数

```matlab
function spNodePair = InsertSpNode(this, nodeIn1, nodeIn2, spNodePair)
    npairs = size(spNodePair, 2);
    for i = 1:npairs
        pair = spNodePair{i};
        if ismember(nodeIn1, pair)    % 找到包含 nodeIn1 的串线
            pair(end+1) = nodeIn2;    % 将 nodeIn2 添加到串线末尾
            spNodePair{i} = pair;
            return;
        end
    end
end
```

**功能**：将新推进的点 `nodeIn2` 添加到包含 `nodeIn1` 的串线中。

### 6.3 推进过程示意图

```
第 0 层（物面）:
        V2
        │
        V1
        │
    B───P───C
        │
       (V2 连接到 C)

第 1 层推进后:
        V2_1
        │
        V1_1
        │
    B_1─P_1─C_1
        │
       (V2_1 连接到 C_1)

第 2 层推进后:
        V2_2
        │
        V1_2
        │
    B_2─P_2─C_2

spNodePair 记录:
{1} = [V1, V1_1, V1_2]    % 虚拟点 V1 的推进串线
{2} = [V2, V2_1, V2_2]    % 虚拟点 V2 的推进串线
{3} = [P, P_1, P_2]       % 真实点 P 的推进串线
```

---

## 7. 局部步长因子处理

### 7.1 步长因子的传递

**问题**：虚拟点和多方向串线上的点需要正确的局部步长因子。

**解决方案**：

```matlab
function AdvancingStepScaleCoeff(this, layer)
    %% 阶段 1：计算真实点的步长因子
    for iNode = 1:this.nWallNodes
        node = this.Wall_nodes(iNode);
        node = this.SpNode(node);  % 获取等价点
        
        % 跳过虚拟点和第一层后的凸点
        if ismember(node, this.virtualPoints)
            continue;
        end
        if layer > 1 && ismember(node, this.convexPoints)
            continue;
        end
        
        % 计算步长缩放
        neig_normal = this.GetNeighborFrontNormal(iNode);
        np = this.normal_point(iNode, :);
        coeff = mean(neig_normal * np');
        this.localSpCoeff(iNode) = this.localSpCoeff(iNode) / coeff;
    end
    
    %% 阶段 2：虚拟点步长因子复制（仅 layer=1）
    if layer == 1
        for i = 1:length(this.virtualPoints)
            nodeVirtual = this.virtualPoints(i);
            nodeReal = this.ValidPoint(nodeVirtual);
            
            % 从真实点复制
            coeff = this.localSpCoeff(nodeReal == this.Wall_nodes);
            this.localSpCoeff(nodeVirtual == this.Wall_nodes) = coeff;
            
            % 缓存到 localSpPair
            this.InsertLocalSpPair(nodeReal, coeff);
        end
    end
    
    %% 阶段 3：多方向串线上的点步长因子设置
    for iNode = 1:this.nWallNodes
        node1 = this.Wall_nodes(iNode);
        node2 = this.SpNode(node1);  % 获取多方向第一个点
        
        if node1 ~= node2  % 在多方向串线上
            nodeReal = this.ValidPoint(node2);  % 获取真实点
            this.localSpCoeff(iNode) = this.GetLocalSpPair(nodeReal);
        end
    end
end
```

### 7.2 SpNode 函数

```matlab
function nodeOut = SpNode(this, nodeIn)
    nodeOut = nodeIn;
    npairs = size(this.spNodePair, 2);
    for i = 1:npairs
        pair = this.spNodePair{i};
        if ismember(nodeIn, pair)
            nodeOut = pair(1);  % 返回串线的起点（真实点或虚拟点）
        end
    end
end
```

**功能**：判断节点是否在多方向串线上，如果是则返回串线起点。

### 7.3 localSpPair 缓存机制

```matlab
% 存储结构
localSpPair{1} = [nodeReal_1, coeff_1]
localSpPair{2} = [nodeReal_2, coeff_2]
...

% 插入
function InsertLocalSpPair(this, nodeIn, coeff)
    npairs = size(this.localSpPair, 2);
    flag = true;
    for i = 1:npairs
        pair = this.localSpPair{i};
        if ismember(nodeIn, pair)
            flag = false;  % 已存在，不重复插入
        end
    end
    if flag
        this.localSpPair{end+1} = [nodeIn, coeff];
    end
end

% 查询
function coeff = GetLocalSpPair(this, nodeIn)
    coeff = 1.0;  % 默认值
    npairs = size(this.localSpPair, 2);
    for i = 1:npairs
        pair = this.localSpPair{i};
        if ismember(nodeIn, pair)
            coeff = pair(2);
        end
    end
end
```

---

## 8. 虚拟点处理

### 8.1 ValidPoint 函数

```matlab
function nodeOut = ValidPoint(this, nodeIn)
    if ~ismember(nodeIn, this.virtualPoints)
        nodeOut = nodeIn;  % 不是虚拟点，直接返回
        return;
    end
    
    % 在 nodePair 中查找对应的真实点
    npairs = size(this.nodePair, 2);
    for i = 1:npairs
        pair = this.nodePair{i};
        if ismember(nodeIn, pair)
            nodeOut = pair(1);  % 返回真实点编号
        end
    end
end
```

**功能**：将虚拟点编号转换为对应的真实点编号。

### 8.2 后处理中的虚拟点消除

```matlab
function PostProcessAfterAdvancingLayers(this)
    virtual_Stack = [];
    nfronts = size(this.Wall_stack, 1);
    
    for ifr = 1:nfronts
        % 恢复为物面
        this.Wall_stack(ifr, 7) = 3;
        
        node1 = this.Wall_stack(ifr, 1);
        node2 = this.Wall_stack(ifr, 2);
        
        flag1 = ismember(node1, this.virtualPoints);
        flag2 = ismember(node2, this.virtualPoints);
        
        if flag1 && flag2
            % 两个点都是虚拟点，标记删除
            virtual_Stack(end+1) = ifr;
        elseif flag1
            % 第一个点是虚拟点，替换为真实点
            node1 = this.ValidPoint(node1);
            this.Wall_stack(ifr, 1) = node1;
        elseif flag2
            % 第二个点是虚拟点，替换为真实点
            node2 = this.ValidPoint(node2);
            this.Wall_stack(ifr, 2) = node2;
        end
        
        if node1 == node2
            % 两点重合，标记删除
            virtual_Stack(end+1) = ifr;
        end
    end
    
    % 删除虚拟阵面
    this.Wall_stack(virtual_Stack, :) = [];
    
    % 恢复边界条件
    % ...
end
```

### 8.3 虚拟点消除示例

**消除前**：

```
Wall_stack 包含：
[1] [B, P, ...]        % 真实阵面
[2] [P, V1, ...]       % 虚拟阵面（待删除）
[3] [V1, V2, ...]      % 虚拟阵面（待删除）
[4] [V2, C, ...]       % 需修改为 [P, C, ...]
```

**消除后**：

```
Wall_stack 包含：
[1] [B, P, ...]        % 真实阵面
[2] [P, C, ...]        % 修改后的阵面
```

---

## 9. 多方向终止条件

### 9.1 凸点停止层数记录

```matlab
function UpdateStopFlag(this, layer, iconvex, iconcav)
    if iconvex > 0 && ~this.stopFlag(iconvex)
        % 第一次停止时记录层数
        this.stopLayer(iconvex) = layer;
        this.stopFlag(iconvex) = true;
    elseif iconcav > 0 && ~this.stopFlagConcav(iconcav)
        this.stopLayerConcav(iconcav) = layer;
        this.stopFlagConcav(iconcav) = true;
    end
end
```

### 9.2 多方向终止判断

```matlab
if iconvex > 0
    distFlag = this.Dist2ConvexPoints(dist, iconvex);
    
    % 条件 1：离凸点较近的阵面
    flag1 = ~distFlag && layer <= this.stopLayer(iconvex);
    
    % 条件 2：离凸点较远的阵面，满足质量条件
    flag2 = distFlag && aspect_ratio > 1.2 && skewness < 0.3;
    
    cond1 = flag1 || flag2;
    
    if this.multi_direction
        % 条件 3：多方向上的层数不能超过 stopLayer
        flag3 = this.IsSpMember(node1In) || this.IsSpMember(node2In);
        flag4 = flag3 && layer <= this.stopLayer(iconvex);
        
        % 条件 4：非多方向上的，满足质量条件
        flag5 = ~flag3 && aspect_ratio > 1.2 && skewness < 0.75;
        
        cond2 = (flag4 || flag5) && cond1;
    else
        cond2 = cond1;
    end
end
```

### 9.3 IsSpMember 函数

```matlab
function flag = IsSpMember(this, nodeIn)
    % 判断节点是否在多方向串线上
    npairs = size(this.spNodePair, 2);
    flag = false;
    for i = 1:npairs
        pair = this.spNodePair{i};
        if ismember(nodeIn, pair)
            flag = true;
        end
    end
end
```

---

## 10. 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        多方向推进数据流                              │
└─────────────────────────────────────────────────────────────────────┘

输入：凸点信息 (convexPoints, betaConvex)
      方向矢量 (directions)
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  InitializeMultiDirection (layer=1)                                 │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 对每个凸点：                                                   │  │
│  │   1. 设置真实点方向 = directions{i}(1)                        │  │
│  │   2. 创建虚拟点 (num_directions-1 个)                         │  │
│  │   3. 设置虚拟点方向 = directions{i}(2:end)                    │  │
│  │   4. 创建虚拟阵面连接虚拟点                                   │  │
│  │   5. 记录 nodePair{真实点，虚拟点 1, 虚拟点 2, ...}            │  │
│  │   6. 收集 virtualPoints 列表                                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  输出：Wall_nodes (含虚拟点)                                        │
│        Wall_stack (含虚拟阵面)                                      │
│        normal_point (含虚拟方向)                                    │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SmoothAdvancingDirection                                           │
│  对推进方向进行光滑（凹点除外）                                      │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AdvancingStepScaleCoeff                                            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 阶段 1: 计算真实点的步长因子                                    │  │
│  │ 阶段 2: 虚拟点从真实点复制步长因子 (layer=1)                    │  │
│  │         并缓存到 localSpPair                                   │  │
│  │ 阶段 3: 多方向串线上的点从缓存获取步长因子                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AdvancingOnePoint (逐点推进)                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 1. 计算步长 = firstHeight × growthRate^(layer-1) × localSpCoeff│  │
│  │ 2. 外推节点：node3 = node1 + Sp × normal                      │  │
│  │ 3. 记录 spNodePair: [起点，推进点 1, 推进点 2, ...]            │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AdvancingQuadLayers (单层推进)                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 对每个阵面：                                                   │  │
│  │   1. 推进两个端点                                             │  │
│  │   2. 形成新四边形单元                                         │  │
│  │   3. 相交检查 (IsCrossALM)                                    │  │
│  │   4. 距离检查 (IsPointFarFromEdgeALM)                         │  │
│  │   5. 质量检查 (QualityCheckALM)                               │  │
│  │   6. 终止条件判断                                             │  │
│  │      - 多方向串线上的点：layer <= stopLayer                   │  │
│  │      - 非多方向点：aspect_ratio > 1.2 && skewness < 0.75      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PostProcessAfterAdvancingLayers                                    │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 1. 遍历所有阵面                                               │  │
│  │ 2. 虚拟点→真实点转换 (ValidPoint)                              │  │
│  │ 3. 删除两端都是虚拟点的阵面                                   │  │
│  │ 4. 删除节点重合的阵面                                         │  │
│  │ 5. 恢复边界条件                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  输出：不含虚拟点的最终网格                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 11. 关键函数接口

### 11.1 公共方法

| 函数名 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `InitializeMultiDirection` | 初始化多方向 | - | 更新 Wall_nodes, Wall_stack, virtualPoints 等 |
| `ValidPoint(nodeIn)` | 虚拟点→真实点转换 | nodeIn: 节点编号 | nodeOut: 真实点编号 |
| `SpNode(nodeIn)` | 获取多方向串线起点 | nodeIn: 节点编号 | nodeOut: 串线起点 |
| `IsSpMember(nodeIn)` | 判断是否在多方向串线上 | nodeIn: 节点编号 | flag: 布尔值 |
| `InsertSpNode(n1, n2, sp)` | 插入节点到串线 | n1, n2: 节点，sp:spNodePair | 更新后的 spNodePair |
| `InsertLocalSpPair(node, coeff)` | 缓存步长因子 | node: 节点，coeff: 系数 | - |
| `GetLocalSpPair(node)` | 获取缓存的步长因子 | node: 节点 | coeff: 步长系数 |

### 11.2 私有方法

| 函数名 | 功能 |
|--------|------|
| `MarkSpecialPoints` | 标记特殊点，计算多方向数量，生成方向矢量 |
| `AdvancingStepScaleCoeff` | 计算并缩放局部步长因子 |
| `SmoothAdvancingDirection` | 光滑推进方向 |
| `UpdateStopFlag` | 更新凸点/凹点停止层数 |
| `Dist2ConvexPoints` | 判断是否远离凸点 |

---

## 12. 示例：90°凸角多方向推进

### 12.1 初始状态

```
几何：
    
    B────P────C
    
凸角 θ = 90°
```

### 12.2 多方向计算

```matlab
num_directions = round(90° / 66°) + 1 = 2

directions{1} = [
    0, 1;      % 方向 1：原始法向
   -0.707, 0.707  % 方向 2：旋转 -45°
];
```

### 12.3 初始化后

```
        V1 (虚拟点，方向 2)
        │
    B───P────C
        │
        │
    (V1 连接到 C)

数据结构:
Wall_nodes = [B, P, C, V1]
nodePair{1} = [P, V1]
virtualPoints = [V1]
directions{1} = [(0,1), (-0.707,0.707)]
```

### 12.4 第 1 层推进后

```
        V1_1
        │
    B_1─P_1─C_1

spNodePair:
{1} = [P, P_1]       % 真实点 P 的推进
{2} = [V1, V1_1]     % 虚拟点 V1 的推进
```

### 12.5 第 2 层推进后

```
        V1_2
        │
    B_2─P_2─C_2
```

### 12.6 后处理后

```
删除虚拟阵面 [P, V1], [V1, V1_1], [V1_1, V1_2]...

最终网格:
    B_2─P_2─C_2
    │   │   │
    B_1─P_1─C_1
    │   │   │
    B───P───C
```

---

## 13. 常见问题与解决方案

### 13.1 问题：虚拟点坐标重复

**现象**：多个虚拟点坐标与真实凸点相同。

**解决**：这是设计行为。虚拟点仅用于推进方向控制，后处理时会被消除。

### 13.2 问题：spNodePair 查找效率低

**现象**：每次推进都要遍历 spNodePair。

**解决**：可使用哈希表或映射结构加速查找：
```matlab
% 优化方案：使用 containers.Map
spNodeMap = containers.Map();
spNodeMap(nodeIn) = nodeOut;
```

### 13.3 问题：多方向数量过多

**现象**：凸角接近 360°时，方向数量过多。

**解决**：设置上限：
```matlab
num_directions = min(num_directions, 6);  % 最多 6 个方向
```

---

## 14. 总结

多方向推进算法的核心设计思想：

1. **空间换质量**：通过创建虚拟点和虚拟阵面，为每个凸角方向创建独立的推进路径
2. **数据关联**：使用 nodePair、spNodePair、localSpPair 等数据结构维护点与点之间的关系
3. **延迟消除**：推进过程中保留虚拟点，后处理时统一消除
4. **步长传递**：虚拟点和多方向串线上的点从真实点继承步长因子
5. **独立终止**：多方向串线上的点有独立的终止条件判断

---

**文档版本**: 1.0  
**更新日期**: 2026 年 3 月 10 日
