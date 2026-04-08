# PyMeshGen 单元测试

本目录包含 PyMeshGen 项目的所有单元测试。

## 文件结构

- `unittests/` - 包含所有单元测试文件
  - `test_*.py` - 单元测试文件
  - `run_tests.py` - 运行所有测试的脚本（支持全量和单个测试）
  - `run_quick_tests.py` - 快速测试脚本（跳过耗时的网格生成测试）
  - `test_files/` - 测试所需的文件

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
python run_tests.py cas_file_io
```

### 快速测试（跳过耗时测试）

```bash
cd unittests
python run_quick_tests.py
```

这将运行文件 I/O、几何计算、网格质量等快速测试，跳过耗时的网格生成测试。

### 直接运行单个测试文件

```bash
cd unittests
python -m unittest test_cas_file_io.py
```

## 测试分类

### 单元测试

这些测试使用unittest框架编写，不需要GUI交互：

- `test_vtk_file_io.py` - VTK文件导入功能测试
- `test_cas_file_io.py` - CAS文件导入功能测试
- `test_cas_parts_info.py` - CAS文件部件信息测试
- `test_cgns_reader.py` - CGNS文件读取测试
- `test_core_functionality.py` - 核心功能测试
- `test_edge_binding.py` - 边绑定测试
- `test_geometry_calculations.py` - 几何计算测试
- `test_geometry_import.py` - 几何导入功能测试
- `test_geometry_import_simple.py` - 几何导入简化测试
- `test_gui_functionality.py` - GUI功能测试
- `test_hybrid_smooth.py` - 混合网格平滑测试
- `test_merge_triangles.py` - 三角形合并测试
- `test_mesh_generation.py` - 网格生成测试
- `test_mesh_quality.py` - 网格质量测试
- `test_mesh_to_plt.py` - 网格转PLT测试
- `test_meshio_cgns.py` - MeshIO CGNS测试
- `test_meshio_import.py` - MeshIO导入测试
- `test_optimize_functions.py` - 优化函数测试
- `test_topology_methods.py` - 拓扑方法测试

### 集成测试

这些测试需要GUI交互，不适合作为单元测试：

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

- `2d_cad/` - 2D CAD文件（IGES、PW格式）
- `2d_cases/` - 2D案例配置文件和输出
- `sfmesh/` - 表面网格文件
- `test_outputs/` - 测试输出文件
- `naca0012-hybrid.cas` - NACA0012混合网格CAS文件
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