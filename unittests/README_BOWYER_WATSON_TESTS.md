# Bowyer-Watson 单元测试说明

## 概述

当前测试套件聚焦 mesh_type=4 的公共入口回归，覆盖：

- `core.py` 与 Delaunay 公共入口的集成
- 基于 JSON 配置的 Bowyer-Watson 回归
- 基于 JSON 配置的 Triangle 后端回归
- 边界恢复、孔洞清理、拓扑洁净检查

## 测试文件位置

```
unittests/test_bowyer_watson.py
```

## 运行测试

### 运行所有测试

```bash
cd C:\Users\HighOrderMesh\.vscode\PyMeshGen
python unittests/test_bowyer_watson.py
```

### 运行详细输出

```bash
python unittests/test_bowyer_watson.py -v
```

### 运行特定测试类

```bash
# 只运行集成测试
python -m unittest unittests.test_bowyer_watson.TestBowyerWatsonIntegration

# 只运行 JSON 配置回归
python -m unittest unittests.test_bowyer_watson.TestBowyerWatsonJSONConfig
```

### 运行单个测试

```bash
# 运行 ANW Bowyer-Watson 回归
python -m unittest unittests.test_bowyer_watson.TestBowyerWatsonJSONConfig.test_anw_bowyer_watson

# 运行 NACA0012 Triangle 后端回归
python -m unittest unittests.test_bowyer_watson.TestBowyerWatsonJSONConfig.test_naca0012_triangle_backend
```

## 测试用例清单

### 1. 核心集成测试 (TestBowyerWatsonIntegration)

| 测试编号 | 测试名称 | 描述 | 验证内容 | 状态 |
|---------|---------|------|---------|------|
| 测试 13 | `test_core_integration` | 与 core.py 的集成 | 模块可正确导入 | ✅ 通过 |

### 2. JSON 配置回归 (TestBowyerWatsonJSONConfig)

| 场景 | 代表测试 | 说明 |
| --- | --- | --- |
| Bowyer-Watson 无边界层 | `test_anw_bowyer_watson`、`test_naca0012_bowyer_watson`、`test_quad_quad_bowyer_watson` | 验证当前主 Bowyer-Watson 后端 |
| Bowyer-Watson 带边界层 | `test_anw_bowyer_watson_with_boundary_layer` 等 | 验证边界层场景下的整体流程 |
| Triangle 后端 | `test_anw_triangle_backend`、`test_naca0012_triangle_backend` | 验证可替代 Delaunay 后端 |
| 统一验收 | 所有 JSON 用例 | 统一执行边界恢复、孔洞清理、拓扑洁净检查 |

## 测试结果示例

典型的测试输出：

```text
test_core_integration (__main__.TestBowyerWatsonIntegration.test_core_integration) ... ok
test_anw_bowyer_watson (__main__.TestBowyerWatsonJSONConfig.test_anw_bowyer_watson) ... ok
test_naca0012_triangle_backend (__main__.TestBowyerWatsonJSONConfig.test_naca0012_triangle_backend) ... ok

----------------------------------------------------------------------
Ran N tests in XX.XXXs

OK
```

## 测试覆盖率

测试套件覆盖以下方面：

- ✅ **实际算例**: ANW、NACA0012、RAE2822、quad_quad、cylinder
- ✅ **后端切换**: Bowyer-Watson 与 Triangle
- ✅ **边界层场景**: 入口行为与后端选择
- ✅ **结果验收**: 边界恢复、孔洞清理、拓扑洁净
- ✅ **核心集成**: 与 core.py 的 mesh_type=4 分支集成

## 添加新测试

### 添加新的 JSON 配置回归

```python
def test_new_case(self):
    """测试新的 mesh_type=4 配置算例"""
    self._test_bowyer_watson_with_config(
        "your_case.json",
        "test_your_case_bw.vtk",
        enable_boundary_layer=False,
        test_name="your_case Bowyer-Watson",
    )
```

### 添加新的根目录配置回归

```python
def test_new_root_case(self):
    """测试新的 config/ 根目录算例"""
    self._test_bowyer_watson_with_config_from_root(
        "your_case.json",
        "test_your_case_bw.vtk",
        enable_boundary_layer=False,
        test_name="your_case Bowyer-Watson",
        check_boundary_recovery=True,
    )
```

## 已知限制

1. **运行时间**: 某些复杂 Bowyer-Watson case（如 NACA 类）运行时间仍可能接近测试阈值。

2. **性能测试**: 当前测试以功能回归为主，未单独维护性能基准。

3. **可视化验证**: 测试仅验证数值结果，未包含可视化质量检查。可以添加 VTK 输出进行视觉验证。

## 故障排除

### 测试失败：导入错误

确保项目根目录和子目录已正确添加到 `sys.path`：

```python
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

### 测试失败：输入文件不存在

确保配置中引用的输入文件位于正确的位置：

```
config/input/quad_quad.cas
unittests/test_files/2d_cases/anw.cas
unittests/test_files/2d_cases/naca0012-tri-coarse.cas
```

### 测试运行缓慢

Bowyer-Watson 算法对于密集边界可能需要较长时间。可以：

- 减少边界点数量
- 增大 `max_edge_length`
- 减少 `smoothing_iterations`
- 使用固定的随机种子以获得可重复的结果

## 持续集成

建议将 Bowyer-Watson 测试添加到 CI/CD 流程中：

```bash
# 在 CI 脚本中添加
python unittests/test_bowyer_watson.py -v
```

这将确保每次代码更改后，Bowyer-Watson 算法的正确性都得到验证。
