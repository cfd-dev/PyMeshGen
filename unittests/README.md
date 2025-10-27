# PyMeshGen 单元测试

本目录包含 PyMeshGen 项目的所有单元测试。

## 文件结构

- `unittests/` - 包含所有单元测试文件
  - `test_*.py` - 单元测试文件
  - `run_tests.py` - 运行所有测试的脚本
  - `test_files/` - 测试所需的文件
- `backup_tests/` - 包含不适合改造为单元测试的原始测试文件

## 如何运行测试

### 运行所有测试

```bash
cd unittests
python run_tests.py
```

### 运行特定测试

```bash
cd unittests
python run_tests.py <test_name>
```

例如，运行CAS文件导入测试：

```bash
python run_tests.py import_cas_file_fix
```

### 直接运行单个测试文件

```bash
cd unittests
python -m unittest test_import_cas_file_fix.py
```

## 测试分类

### 单元测试

这些测试使用unittest框架编写，不需要GUI交互：

- `test_import_cas_file_fix.py` - CAS文件导入功能测试
- `test_import_mesh_fix.py` - 网格导入功能测试
- `test_cas_import.py` - CAS文件解析测试
- `test_cas_parts.py` - CAS文件部件测试
- `test_config_load.py` - 配置加载测试
- `test_environment.py` - 环境测试
- `test_fileIO.py` - 文件I/O测试
- `test_fix.py` - 修复功能测试
- `test_fixes.py` - 修复功能测试（另一个版本）
- `test_front_init.py` - 前端初始化测试
- `test_geo_cal.py` - 几何计算测试
- `test_gui_config.py` - GUI配置测试
- `test_gui_functions.py` - GUI功能测试
- `test_gui_message.py` - GUI消息测试
- `test_mesh_generation.py` - 网格生成测试
- `test_mesh_quality.py` - 网格质量测试
- `test_properties_panel.py` - 属性面板测试
- `test_vtk_import.py` - VTK文件导入测试

### 集成测试

这些测试需要GUI交互，不适合作为单元测试，已移动到backup_tests文件夹：

- `test_cas_import_display.py` - CAS文件导入和显示测试
- `test_vtk_complete.py` - VTK完整功能测试
- `test_vtk_display.py` - VTK显示测试
- `test_vtk_display_gui.py` - VTK GUI显示测试
- `test_vtk_display_comprehensive.py` - VTK综合显示测试
- `test_vtk_embedding_fix.py` - VTK嵌入修复测试
- `test_vtk_embedding_main.py` - VTK嵌入主测试
- `test_vtk_simple.py` - VTK简单测试

## 测试数据

测试所需的数据文件存储在`test_files/`目录中：

- `2d_cad/` - 2D CAD文件
- `2d_cases/` - 2D案例文件
- `naca0012-hybrid.cas` - NACA0012混合网格文件
- `naca0012.vtk` - NACA0012 VTK文件
- `training_mesh.stl` - 训练网格STL文件

## 添加新测试

1. 创建新的测试文件，命名为`test_*.py`
2. 使用unittest框架编写测试
3. 将测试文件放在unittests目录中
4. 如果需要测试数据，将其放在test_files目录中

## 注意事项

- 单元测试应该独立运行，不需要用户交互
- 测试应该覆盖关键功能和边界情况
- 避免在测试中使用硬编码路径，使用相对路径或临时文件
- 测试应该快速执行，避免长时间运行的操作