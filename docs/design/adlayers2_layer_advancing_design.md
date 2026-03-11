# adlayers2 层推进（ALM）详细设计

## 1. 文档目标与范围

本文档描述 `adfront2/adlayers2.py` 的当前层推进实现设计，并覆盖其与 `adfront2/multi_direction.py` 的多方向推进协同机制。  
重点面向开发与维护，解释：

- 边界层推进的主流程与阶段顺序
- 关键数据结构与状态变量
- 单层推进中的几何计算、步长计算、拓扑更新与早停策略
- 多方向推进（凸角分裂、虚拟点/虚拟阵面、步长缩放、后处理）
- 调试日志（INFO/DEBUG/VERBOSE）在流程中的观测点


## 2. 模块定位与依赖关系

`Adlayers2` 是二维边界层推进核心模块，位于整体网格流程中“边界层生成”阶段。其结果是：

- 生成边界层单元（主要为四边形，特殊情况下可退化三角形）
- 输出更新后的全局边界阵面（供后续各向同性推进使用）
- 构建 `Unstructured_Grid`

核心依赖：

- `data_structure.front2d.Front`：阵面拓扑
- `data_structure.basic_elements`：`NodeElementALM` / `Quadrilateral` / `Triangle`
- `data_structure.rtree_space`：R-Tree 空间索引与候选查询
- `meshsize` 尺寸场：用于网格尺寸约束
- `adfront2.multi_direction.MultiDirectionManager`：多方向推进管理
- `utils.message`：分级日志输出


## 3. 核心数据模型与状态

`Adlayers2` 在运行期维护以下关键状态：

- **几何与拓扑容器**
  - `current_part.front_list`：当前部件阵面集合（`wall` / `prism-cap` / `interior`）
  - `front_node_list`：当前层参与几何计算的节点集合
  - `cell_container`：已生成边界层单元对象
  - `node_coords`：全局节点坐标
  - `all_boundary_fronts`：全局边界阵面集合

- **推进控制参数**
  - `ilayer`：当前层号（从 0 计）
  - `max_layers` / `full_layers`
  - `first_height` / `growth_rate` / `growth_method`
  - `multi_direction`：当前部件是否启用多方向
  - `num_prism_cap`：下一层仍需推进的流向阵面数量

- **质量与稳定性参数**
  - `quality_threshold`：单元偏斜度阈值
  - `al`：候选搜索半径因子
  - `relax_factor` / `smooth_iterions`：方向光滑参数

- **空间查询结构**
  - `front_dict` + `space_index`：全局阵面索引（R-Tree）


## 4. 总体流程（按部件、按层）

入口：`Adlayers2.generate_elements()`

1. 遍历 `part_params`，仅处理 `PRISM_SWITCH == "wall"` 的部件。
2. 设置当前部件参数 `set_current_part()`。
3. 若启用多方向，构造 `MultiDirectionManager(self)`。
4. 在 `ilayer < max_layers` 且 `num_prism_cap > 0` 时循环推进单层。
5. 若启用多方向，结束后执行 `multi_direction_manager.post_process()` 清理虚拟拓扑。
6. 构建 `Unstructured_Grid` 并返回。

单层推进由 `_process_single_layer()` 固定顺序驱动：

1. `prepare_geometry_info()`  
2. `log_multi_direction_debug_summary()`（首层多方向统计）  
3. `apply_multi_direction_workflow()`（多方向初始化→方向光滑→步长缩放）  
4. `calculate_marching_distance()`  
5. `advancing_fronts()`（逐阵面推进与拓扑更新）  
6. `log_first_layer_cell_summary()`（首层新增单元统计）  
7. `show_progress()`


## 5. 几何准备阶段设计

### 5.1 阵面几何重建：`compute_front_geometry()`

执行顺序：

1. `build_front_rtree()`：基于**全局部件阵面**建立 R-Tree
2. `reconstruct_node2front()`：重建节点-阵面关系并统一为 `NodeElementALM`
3. `reconstruct_node2node()`：重建节点邻接（环状相邻）
4. `compute_corner_attributes()`：计算角度、凹凸属性、局部步长因子、多方向数量

### 5.2 节点推进方向

`prepare_geometry_info()` 内执行：

- `compute_point_normals()`：根据相邻阵面法向计算初始推进方向
- `laplacian_smooth_normals()`：做基于邻接关系的方向光滑（含距离和角度权重）

说明：当节点 `marching_direction` 为多方向列表时，该阶段遵循“优先处理当前有效方向（第一个方向）”策略。


## 6. 多方向推进设计（重点）

多方向能力由 `MultiDirectionManager` 管理，核心目标是改善凸角区域层推进质量与拓扑稳定性。

### 6.1 首层初始化：`initialize_multi_direction()`

仅在 `ilayer == 0` 且 `multi_direction=True` 时执行：

1. 收集凸点（`convex_flag=True` 且 `num_multi_direction>1`）。
2. 每个凸点执行 `_create_virtual_points_for_convex_node()`：
   - 保留真实点第一个方向作为主方向
   - 为其余方向创建虚拟点（坐标初始与真实点重合）
   - 在真实点与虚拟点链之间创建虚拟 `prism-cap` 阵面
   - 将“右侧原阵面起点”重接到最后一个虚拟点
   - 同步 `node2front` / `node2node`，保证后续光滑与推进拓扑一致

### 6.2 多方向流程调度：`run_layer_workflow()`

每层执行：

1. 首层初始化（仅首层）
2. `smooth_advancing_direction()`：多方向专用方向光滑
3. `calculate_step_scale_coeff()`：步长缩放计算

### 6.3 步长缩放：`calculate_step_scale_coeff()`

分三阶段：

1. **真实点计算**：根据推进方向与相邻 wall/prism-cap 法向夹角投影修正 `local_step_factor`
2. **首层虚拟点继承**：虚拟点从对应真实点复制步长因子
3. **串线继承**：多方向串线上的点从串线起点（真实点）继承缓存步长因子

### 6.4 虚拟阵面推进：`Adlayers2._advance_virtual_front()`

对零长度阵面（典型虚拟阵面）进行特殊处理：

- 推进后若新点重合则跳过
- 通过“真实点 + 新点”去重构造单元：
  - 4 点 -> `Quadrilateral`
  - 3 点 -> `Triangle`
- 同时创建新的 `prism-cap` 与两条 `interior` 阵面，纳入统一拓扑更新

### 6.5 后处理：`post_process()`

推进结束后清理虚拟拓扑：

- 虚拟-虚拟阵面直接删除
- 虚拟-真实阵面将虚拟点替换回真实点
- 删除重合节点形成的无效阵面


## 7. 单层推进核心算法

### 7.1 步长计算：`calculate_marching_distance()`

- 基础步长（几何增长）：
  - `current_step_size = first_height * growth_rate ** ilayer`
- 多方向模式：
  - `marching_distance = current_step_size * local_step_factor`
- 非多方向模式：
  - 在双 wall 邻接条件下，按方向投影进行修正，避免法向夹角导致厚度失真

### 7.2 逐阵面推进：`advancing_fronts()`

每个阵面处理顺序：

1. `interior` 阵面直接保留
2. 已早停阵面保留（计入 `num_old_prism_cap`）
3. 零长度阵面走 `_advance_virtual_front()`
4. 层差约束 `_check_neighbor_layer_difference()`（BFS 经 interior 链路找邻接 prism-cap）
5. `create_new_cell_and_front()` 创建候选单元与 3 条新阵面
6. `geometric_checker()` 质量/尺寸/长宽比判定
7. `proximity_checker()` 邻近碰撞判定
8. 通过后提交：
   - 新节点落盘（含 `corresponding_node` 回写）
   - 新单元加入 `cell_container`
   - 新阵面通过 `_finalize_new_fronts()` 与 `update_front_list_globally()` 合并

层末更新：

- `num_prism_cap = len(new_prism_cap_list) - num_old_prism_cap`
- `current_part.front_list = new_prism_cap_list + new_interior_list`


## 8. 早停与质量控制策略

### 8.1 几何质量早停：`geometric_checker()`

- 偏斜度低于阈值：早停
- 超过完整层后：
  - 单元尺寸 `> 1.3 * isotropic_size`：早停
  - 长宽比 `< 1.1`：早停

### 8.2 邻近早停：`proximity_checker()`

核心过滤：

- 与新阵面共点/共坐标的候选跳过
- 处于反推进方向的候选跳过
- 若线段距离小于 `safe_distance=0.5*min(marching_distance)`，触发早停

### 8.3 层差早停：`_check_neighbor_layer_difference()`

通过 BFS 沿 interior 链路搜索可达 `prism-cap`，若相邻层差 `> 2` 则早停当前阵面，防止局部层数跳变。


## 9. 拓扑一致性与不变量

实现依赖以下不变量维持稳定：

1. 节点索引全局唯一，新增点从 `num_nodes` 递增分配。
2. `corresponding_node` 仅在推进提交后回写，避免半提交状态污染。
3. `prism-cap` 阵面不去重，`interior` 阵面按 hash 去重并跨 part 清理重复。
4. 每层推进后，`current_part.front_list` 必须由“新 prism-cap + interior”重构。
5. 多方向首层后，虚拟点拓扑必须在 `post_process()` 完成收敛（真实拓扑输出）。


## 10. 日志与可观测性设计

日志分级：

- `INFO`：层推进进度、步长、节点/单元数量
- `DEBUG`：首层多方向统计、层差检查、虚拟阵面处理、首层单元数量
- `VERBOSE`：凸角角度详情、四边形样本坐标、细粒度流程日志

关键观测点：

- 首层多方向节点分布与角度明细（`log_multi_direction_debug_summary`）
- 首层新增单元类型统计与样本坐标（`log_first_layer_cell_summary`）
- 邻近/层差早停原因


## 11. 性能特征与复杂度要点

- 主计算热点在逐阵面推进 + 候选查询。
- R-Tree 降低了全局邻近检查代价（由全表扫描转为局部候选）。
- 多方向会增加首层节点/阵面数量，主要开销来自虚拟点链与额外拓扑维护。


## 12. 已知约束与后续可演进点

1. `safe_distance` 当前为经验值（`0.5 * marching_distance`），可按工况自适应。
2. `size_factor=1.3` 与长宽比阈值为固定值，可参数化到配置层。
3. `proximity_checker_with_cartesian_index()` 目前保留备用，默认走 R-Tree 路径。
4. 多方向数量策略与旋转角分配可进一步做曲率/边界类型感知。


## 13. 代码导航（建议阅读顺序）

1. `Adlayers2.generate_elements`
2. `Adlayers2._process_single_layer`
3. `Adlayers2.compute_front_geometry` + `compute_corner_attributes`
4. `MultiDirectionManager.run_layer_workflow`
5. `Adlayers2.calculate_marching_distance`
6. `Adlayers2.advancing_fronts`
7. `MultiDirectionManager.post_process`

