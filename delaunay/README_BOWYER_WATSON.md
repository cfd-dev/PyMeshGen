# Bowyer-Watson Delaunay 网格生成器使用说明

## 概述

Bowyer-Watson 算法是一种经典的 Delaunay 三角剖分方法，能够生成高质量的三角形网格。本项目已将 Bowyer-Watson 算法集成到 PyMeshGen 框架中，作为现有网格生成方法的补充。

## 主要特性

1. **离散边界输入**：以离散边界网格作为输入，自动提取边界点和边界约束
2. **尺寸场控制**：集成 QuadtreeSizing 模块，支持自适应网格尺寸控制
3. **质量保证**：使用 Laplacian 平滑优化网格质量
4. **高质量三角形**：Delaunay 三角剖分保证最小角最大化，避免狭长三角形

## 使用方法

### 方法一：通过配置文件使用（推荐）

在配置文件中设置 `mesh_type = 4` 即可启用 Bowyer-Watson 算法：

```json
{
  "mesh_type": 4,
  "parts": [
    {
      "part_name": "your_part_name",
      "input_file": "your_input_file.mesh"
    }
  ]
}
```

然后正常运行 PyMeshGen：

```python
from core import generate_mesh
from data_structure.parameters import Parameters

parameters = Parameters("FROM_CASE_JSON", json_file="your_config.json")
result = generate_mesh(parameters)
```

### 方法二：直接调用 API

```python
import numpy as np
from delaunay.bowyer_watson import create_bowyer_watson_mesh
from meshsize.meshsize import QuadtreeSizing

# 1. 准备边界点数据
boundary_points = np.array([
    [0.0, 0.0],
    [0.5, 0.0],
    [1.0, 0.0],
    [1.0, 0.5],
    [1.0, 1.0],
    [0.5, 1.0],
    [0.0, 1.0],
    [0.0, 0.5],
])

# 2. 创建尺寸场（可选）
sizing_system = QuadtreeSizing(
    initial_front=front_list,  # Front 对象列表
    max_size=0.2,
    resolution=0.1,
    decay=1.2,
    visual_obj=visual_obj,
)

# 3. 生成网格
points, simplices, boundary_mask = create_bowyer_watson_mesh(
    boundary_points=boundary_points,
    sizing_system=sizing_system,  # 或使用 max_edge_length=0.15
    target_triangle_count=None,  # 可选：指定目标三角形数量
    smoothing_iterations=3,
    seed=42,  # 可选：随机种子
)

# 返回值说明：
# - points: 点坐标数组，形状为 (N, 2)
# - simplices: 三角形索引数组，形状为 (M, 3)
# - boundary_mask: 边界点掩码，形状为 (N,)，True 表示边界点
```

## 参数说明

### BowyerWatsonMeshGenerator 类

```python
generator = BowyerWatsonMeshGenerator(
    boundary_points=boundary_points,      # 边界点坐标数组 (N, 2)
    boundary_edges=boundary_edges,        # 边界边列表 [(idx1, idx2), ...]（可选）
    sizing_system=sizing_system,          # QuadtreeSizing 尺寸场对象（可选）
    max_edge_length=0.15,                 # 全局最大边长（可选）
    smoothing_iterations=3,               # Laplacian 平滑迭代次数
    seed=42,                              # 随机种子（可选）
)
```

**参数优先级**：
1. `sizing_system`（尺寸场）- 最高优先级
2. `max_edge_length`（全局尺寸）
3. 无限制（仅基于质量细分）

### generate_mesh 方法

```python
points, simplices, boundary_mask = generator.generate_mesh(
    target_triangle_count=None,  # 目标三角形数量（可选）
)
```

## 算法流程

Bowyer-Watson 算法的完整流程：

1. **初始三角剖分**：
   - 创建超级三角形包含所有边界点
   - 逐点插入边界点，构建初始 Delaunay 三角剖分

2. **迭代插入内部点**：
   - 寻找质量最差或尺寸过大的三角形
   - 在其外接圆圆心处插入新点
   - 重新三角剖分
   - 重复直到满足终止条件

3. **Laplacian 平滑**：
   - 迭代调整内部点位置
   - 边界点保持固定
   - 改善网格整体质量

4. **重新剖分**：
   - 平滑后重新进行 Delaunay 三角剖分
   - 保证最终的 Delaunay 性质

## 网格质量控制

### 三角形质量度量

使用纵横比（aspect ratio）作为质量指标：

```
quality = 2 * r_inscribed / r_circumscribed
```

- 值域：[0, 1]
- 1.0：完美等边三角形
- > 0.5：良好质量
- < 0.3：质量较差

### 终止条件

迭代插入内部点在满足以下任一条件时停止：

1. 达到目标三角形数量（如果指定）
2. 所有三角形的边长都小于目标尺寸
3. 所有三角形的质量都高于阈值
4. 达到最大节点数限制（100,000）
5. 达到最大迭代次数限制（50,000）

## 测试验证

运行测试脚本验证算法正确性：

```bash
python test_bowyer_watson.py
```

测试包括：
- 正方形边界网格生成
- 尺寸场集成测试
- 网格质量评估

## 与其他算法的比较

| 特性 | Bowyer-Watson | Adfront2（阵面推进） |
|------|---------------|---------------------|
| 网格质量 | 高（Delaunay 保证） | 中等 |
| 边界贴合 | 优秀 | 优秀 |
| 尺寸控制 | 精确（尺寸场） | 精确（尺寸场） |
| 速度 | 中等 | 较快 |
| 适用场景 | 高质量要求 | 常规网格 |

## 注意事项

1. **边界点顺序**：边界点应按逆时针或顺时针顺序排列
2. **尺寸场参数**：建议使用 QuadtreeSizing 获得最佳尺寸控制效果
3. **平滑迭代**：3-5 次 Laplacian 平滑通常足够，过多可能导致网格失真
4. **随机种子**：设置 seed 可保证结果可重复

## 示例输出

典型正方形边界测试输出：

```
边界点数量: 40
[INFO] 开始 Bowyer-Watson 网格生成...

生成结果:
  - 总节点数: 125
  - 边界节点: 40
  - 内部节点: 85
  - 三角形数: 204

网格质量统计:
  - 平均质量: 0.9574
  - 最小质量: 0.6387
  - 最大质量: 0.9997
  - 平均最小角: 51.29°
```

## 故障排除

### 问题：网格生成速度慢

**解决方案**：
- 减小边界点密度
- 增大 max_edge_length 或调整尺寸场参数
- 减少 smoothing_iterations

### 问题：网格质量不理想

**解决方案**：
- 增加 smoothing_iterations（5-10 次）
- 检查边界点是否有重叠或异常
- 调整尺寸场的 resolution 和 decay 参数

### 问题：内存不足

**解决方案**：
- 减少边界点数量
- 设置更宽松的尺寸约束
- 检查是否有异常小的边界边导致过度细分

## 技术细节

### Bowyer-Watson 算法核心

1. **外接圆测试**：点在三角形外接圆内则需要重新剖分
2. **边界边识别**：坏三角形集合中不共享的边构成多边形边界
3. **超级三角形**：确保所有点都在其内部，最后移除

### 与现有框架集成

- 输入：兼容 Adfront2 的 Front 对象列表
- 尺寸场：复用 QuadtreeSizing 模块
- 输出：转换为 Unstructured_Grid 对象
- 优化：可继续使用 edge_swap、laplacian_smooth 等后处理

## 参考资料

- Bowyer, A. (1981). "Computing Dirichlet tessellations". Computer Journal.
- Watson, D. F. (1981). "Computing the n-dimensional Delaunay tessellation". Computer Journal.
- 现有参考文件：`delaunay/mesh_generator.py`
