# Bowyer-Watson 算法 JSON 配置使用说明

## 概述

Bowyer-Watson Delaunay 网格生成算法已集成到 PyMeshGen 中，只需在 JSON 配置文件中设置 `mesh_type = 4` 即可启用。

## 快速开始

### 1. 修改配置文件

在任何现有的 JSON 配置文件中，添加或修改 `mesh_type` 字段为 `4`：

```json
{
    "debug_level": 0,
    "input_file": "./config/input/naca0012-tri-coarse.cas",
    "output_file": "./out/naca0012_bw.vtk",
    "mesh_type": 4,
    "viz_enabled": false,
    "parts": [
        {
            "part_name": "farfield",
            "max_size": 2.0,
            "PRISM_SWITCH": "off",
            "first_height": 0.1,
            "max_layers": 5,
            "full_layers": 5,
            "multi_direction": false
        },
        {
            "part_name": "wall",
            "max_size": 2.0,
            "PRISM_SWITCH": "wall",
            "first_height": 0.001,
            "max_layers": 30,
            "full_layers": 5,
            "multi_direction": false
        }
    ]
}
```

### 2. 运行网格生成

使用修改后的配置文件运行 PyMeshGen：

```python
from PyMeshGen import PyMeshGen
from data_structure.parameters import Parameters

parameters = Parameters("FROM_CASE_JSON", "your_config.json")
PyMeshGen(parameters)
```

## 配置选项说明

### mesh_type

- `1`: 三角形网格（Adfront2 阵面推进）
- `2`: 直角三角形网格
- `3`: 三角形/四边形混合网格
- **`4`: Bowyer-Watson Delaunay 三角网格** ⭐

### 边界层配置

Bowyer-Watson 模式**支持边界层网格生成**。根据 `parts` 中的 `PRISM_SWITCH` 设置：

#### 无边界层模式

将所有 part 的 `PRISM_SWITCH` 设置为 `"off"`：

```json
"parts": [
    {
        "part_name": "farfield",
        "PRISM_SWITCH": "off"
    }
]
```

**结果**：纯 Bowyer-Watson 三角形网格

#### 带边界层模式

保留 wall 的 `PRISM_SWITCH` 设置为 `"wall"`：

```json
"parts": [
    {
        "part_name": "farfield",
        "PRISM_SWITCH": "off"
    },
    {
        "part_name": "wall",
        "PRISM_SWITCH": "wall",
        "first_height": 0.001,
        "max_layers": 30
    }
]
```

**结果**：Bowyer-Watson 三角形内层网格 + 边界层网格

## 单元测试

Bowyer-Watson 算法的 JSON 配置测试位于：

```
unittests/test_bowyer_watson.py
```

### 运行测试

```bash
# 运行所有 Bowyer-Watson 测试
python unittests/test_bowyer_watson.py

# 运行 JSON 配置测试
python -m unittest unittests.test_bowyer_watson.TestBowyerWatsonJSONConfig -v

# 运行特定算例测试（无边界层）
python -m unittest unittests.test_bowyer_watson.TestBowyerWatsonJSONConfig.test_naca0012_bowyer_watson -v

# 运行特定算例测试（带边界层）
python -m unittest unittests.test_bowyer_watson.TestBowyerWatsonJSONConfig.test_naca0012_bowyer_watson_with_boundary_layer -v
```

### 测试用例清单

| 测试编号 | 测试名称 | 算例 | 边界层 | 状态 |
|---------|---------|------|--------|------|
| 14 | `test_naca0012_bowyer_watson` | NACA0012 | ❌ 无 | ✅ |
| 17 | `test_naca0012_bowyer_watson_with_boundary_layer` | NACA0012 | ✅ 有 | ✅ |
| 15 | `test_anw_bowyer_watson` | ANW | ❌ 无 | ✅ |
| 18 | `test_anw_bowyer_watson_with_boundary_layer` | ANW | ✅ 有 | ✅ |
| 16 | `test_rae2822_bowyer_watson` | RAE2822 | ❌ 无 | ✅ |
| 19 | `test_rae2822_bowyer_watson_with_boundary_layer` | RAE2822 | ✅ 有 | ✅ |

## 示例：创建 Bowyer-Watson 配置文件

### 示例 1：NACA0012 纯三角网格（无边界层）

```json
{
    "debug_level": 0,
    "input_file": "./unittests/test_files/2d_cases/naca0012.cas",
    "output_file": "./out/naca0012_bw.vtk",
    "mesh_type": 4,
    "viz_enabled": false,
    "parts": [
        {
            "part_name": "farfield",
            "max_size": 2.0,
            "PRISM_SWITCH": "off"
        },
        {
            "part_name": "wall",
            "max_size": 2.0,
            "PRISM_SWITCH": "off"
        }
    ]
}
```

### 示例 2：NACA0012 带边界层

```json
{
    "debug_level": 0,
    "input_file": "./unittests/test_files/2d_cases/naca0012.cas",
    "output_file": "./out/naca0012_bw_bl.vtk",
    "mesh_type": 4,
    "viz_enabled": false,
    "parts": [
        {
            "part_name": "farfield",
            "max_size": 2.0,
            "PRISM_SWITCH": "off"
        },
        {
            "part_name": "wall",
            "max_size": 2.0,
            "PRISM_SWITCH": "wall",
            "first_height": 0.001,
            "max_layers": 30,
            "full_layers": 5,
            "multi_direction": false
        }
    ]
}
```

## 算法流程

当 `mesh_type = 4` 时，网格生成流程：

1. **输入解析**：从 CAS 文件读取边界网格
2. **尺寸场计算**：使用 QuadtreeSizing 计算自适应尺寸场
3. **边界层生成**（可选）：如果配置了 `PRISM_SWITCH = "wall"`，生成边界层
4. **Bowyer-Watson 三角剖分**：
   - 从边界阵面提取边界点
   - 初始 Delaunay 三角剖分
   - 迭代插入内部点（基于尺寸场和质量）
   - Laplacian 平滑优化
5. **网格优化**：edge_swap、edge_collapse
6. **网格合并**：合并边界层和内层网格
7. **输出**：保存 VTK 文件

## 优势

与传统 Adfront2 阵面推进算法相比，Bowyer-Watson 算法的优势：

1. **Delaunay 性质**：保证最小角最大化，避免狭长三角形
2. **高质量网格**：三角形质量更均匀
3. **尺寸场控制**：精确的自适应尺寸控制
4. **可重复性**：支持随机种子，结果可重现
5. **边界层兼容**：可与边界层网格无缝结合

## 注意事项

1. **mesh_type=4 仅生成三角形**：即使配置了 `triangle_to_quad_method`，也不会转换为四边形
2. **边界层可选**：根据配置决定是否生成边界层
3. **尺寸场推荐**：建议使用 QuadtreeSizing 获得最佳效果
4. **平滑迭代**：默认 3 次 Laplacian 平滑，可通过代码调整

## 故障排除

### 问题：网格生成失败

**可能原因**：
- 边界点有重叠或异常
- 尺寸场参数不合理

**解决方案**：
- 检查输入边界网格质量
- 调整 `max_size`、`resolution`、`decay` 参数

### 问题：生成速度慢

**解决方案**：
- 增大 `max_size` 参数
- 减少边界点密度
- 减少平滑迭代次数

### 问题：网格质量不理想

**解决方案**：
- 增加 Laplacian 平滑迭代次数（修改 `bowyer_watson.py` 中的 `smoothing_iterations`）
- 调整尺寸场参数
- 检查边界点分布

## 参考资料

- 算法实现：`delaunay/bowyer_watson.py`
- 核心集成：`core.py`（mesh_type == 4 分支）
- 单元测试：`unittests/test_bowyer_watson.py`
- 使用说明：`delaunay/README_BOWYER_WATSON.md`
