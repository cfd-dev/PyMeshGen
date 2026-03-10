"""
多方向推进算法实现模块

实现凸角处的多方向推进功能，包括：
1. 多方向初始化
2. 虚拟点和虚拟阵面创建
3. 推进方向光滑
4. 局部步长因子计算
5. 多方向串线追踪
6. 后处理虚拟点消除
"""

import numpy as np
from math import pi
from utils.message import info, debug, verbose, warning


class MultiDirectionManager:
    """多方向推进管理器
    
    负责管理多方向推进的所有相关数据和操作
    """
    
    def __init__(self, adlayers_instance):
        self.adlayers = adlayers_instance
        
        # 多方向相关数据结构
        self.convex_points = []  # 凸点列表
        self.concav_points = []  # 凹点列表
        self.virtual_points = []  # 虚拟点列表
        self.node_pair = {}  # 真实点与虚拟点的对应关系 {real_idx: [real_node, virt_node1, virt_node2, ...]}
        self.sp_node_pair = {}  # 多方向串线追踪 {start_idx: [start_node, pushed_node1, pushed_node2, ...]}
        self.local_sp_pair = {}  # 局部步长因子缓存 {node_idx: factor}
        
        # 停止推进标志
        self.stop_flag_convex = {}  # 凸点停止标志 {idx: flag}
        self.stop_layer_convex = {}  # 凸点停止层数 {idx: layer}
        self.stop_flag_concav = {}  # 凹点停止标志 {idx: flag}
        self.stop_layer_concav = {}  # 凹点停止层数 {idx: layer}
    
    def initialize_multi_direction(self, front_node_list, ilayer):
        """初始化多方向推进
        
        在第一层推进时，为每个凸点创建多个推进方向和虚拟点
        
        Args:
            front_node_list: 当前阵面节点列表
            ilayer: 当前推进层数
        """
        if ilayer != 0 or not self.adlayers.multi_direction:
            return
        
        verbose("开始初始化多方向推进...")
        
        # 收集所有凸点
        self.convex_points = []
        for node in front_node_list:
            if node.convex_flag and node.num_multi_direction > 1:
                self.convex_points.append(node)
        
        if not self.convex_points:
            verbose("未找到需要多方向推进的凸点")
            return
        
        verbose(f"找到 {len(self.convex_points)} 个凸点需要多方向推进")
        
        # 为每个凸点创建虚拟点和虚拟阵面
        for convex_node in self.convex_points:
            debug(f"[多方向初始化] 处理凸点 {convex_node.idx}, 方向数量={convex_node.num_multi_direction}")
            self._create_virtual_points_for_convex_node(convex_node)
        
        verbose(f"多方向推进初始化完成，创建了 {len(self.virtual_points)} 个虚拟点")
    
    def _create_virtual_points_for_convex_node(self, convex_node):
        """为单个凸点创建虚拟点
        
        Args:
            convex_node: 凸点节点对象
        """
        num_directions = convex_node.num_multi_direction
        multi_directions = convex_node.marching_direction
        
        debug(f"[虚拟点创建] 凸点{convex_node.idx}, num_directions={num_directions}, marching_direction类型={type(multi_directions)}, 长度={len(multi_directions) if isinstance(multi_directions, list) else 'N/A'}")
        
        if not isinstance(multi_directions, list) or len(multi_directions) < 2:
            warning(f"[虚拟点创建] 凸点{convex_node.idx}的marching_direction不是列表或长度<2，跳过虚拟点创建")
            return
        
        # 记录真实点
        real_idx = convex_node.idx
        self.node_pair[real_idx] = [convex_node]
        
        # 设置第一个方向为真实点的推进方向
        convex_node.marching_direction = multi_directions[0]
        convex_node.direction_idx = 0
        
        # 创建虚拟点
        base_coords = np.array(convex_node.coords)
        
        # 获取凸点右侧的相邻阵面（在添加虚拟阵面之前）
        neighbor_fronts = convex_node.node2front[:]
        right_front = None
        right_neighbor = None
        for front in neighbor_fronts:
            if front.node_ids[0] == convex_node.idx:
                right_front = front
                right_neighbor = front.node_elems[1]
                break
        
        # 创建虚拟点（不创建虚拟阵面）
        for i in range(1, num_directions):
            # 创建虚拟点
            virtual_node = self._create_virtual_point(
                coords=base_coords.copy(),
                direction=multi_directions[i],
                real_point_idx=real_idx,
                local_step_factor=convex_node.local_step_factor,
            )
            
            # 将虚拟点添加到推进节点列表
            self.adlayers.front_node_list.append(virtual_node)
            debug(f"[虚拟点] 创建虚拟点: idx={virtual_node.idx}, 真实点={real_idx}, 方向={i}")
            
            # 记录虚拟点
            convex_node.virtual_points.append(virtual_node)
            self.virtual_points.append(virtual_node)
            self.node_pair[real_idx].append(virtual_node)
        
        # 创建虚拟阵面：连接真实凸点和所有虚拟点
        # 第一个虚拟阵面：真实凸点 -> 第一个虚拟点
        if len(convex_node.virtual_points) > 0:
            virtual_front1 = self._create_virtual_front(convex_node, convex_node.virtual_points[0])
            debug(f"[虚拟阵面] 创建虚拟阵面: {convex_node.idx} -> {convex_node.virtual_points[0].idx}")
        
        # 后续虚拟阵面：前一个虚拟点 -> 后一个虚拟点
        for j in range(len(convex_node.virtual_points) - 1):
            virtual_front = self._create_virtual_front(
                convex_node.virtual_points[j], 
                convex_node.virtual_points[j+1]
            )
            debug(f"[虚拟阵面] 创建虚拟阵面: {convex_node.virtual_points[j].idx} -> {convex_node.virtual_points[j+1].idx}")
        
        # 将凸点右侧阵面的起点更新为最后一个虚拟点
        if right_front is not None:
            last_virtual = convex_node.virtual_points[-1]
            old_real = right_front.node_elems[0]
            right_front.node_elems[0] = last_virtual
            right_front.node_ids[0] = last_virtual.idx

            # 同步节点-阵面关联，确保后续方向光滑和步长计算使用最新拓扑
            self._remove_front_link(old_real, right_front)
            self._add_front_link(last_virtual, right_front)

            # 同步节点邻接关系：真实凸点不再与右邻点直连，末虚拟点接管该连接
            if right_neighbor is not None:
                self._remove_neighbor_link(old_real, right_neighbor)
                self._add_neighbor_link(last_virtual, right_neighbor)

            debug(f"[阵面修改] 阵面{right_front.node_ids}的起点更新为虚拟点{last_virtual.idx}")
        
        # 初始化停止标志
        self.stop_flag_convex[real_idx] = False
        self.stop_layer_convex[real_idx] = 1000
    
    def _create_virtual_point(self, coords, direction, real_point_idx, local_step_factor):
        """创建虚拟点
        
        Args:
            coords: 坐标 (与真实凸点相同)
            direction: 推进方向
            real_point_idx: 对应的真实点编号
            local_step_factor: 局部步长因子
            
        Returns:
            virtual_node: 虚拟点对象
        """
        from data_structure.basic_elements import NodeElementALM
        
        coords_list = coords.tolist() if isinstance(coords, np.ndarray) else list(coords)

        # 创建新节点，使用全局唯一索引
        virtual_node = NodeElementALM(
            coords=coords_list,
            idx=self.adlayers.num_nodes,
            part_name=self.adlayers.current_part.part_name,
            bc_type="wall",
        )
        self.adlayers.node_coords.append(coords_list)
        self.adlayers.num_nodes += 1
        
        # 设置虚拟点属性
        virtual_node.is_virtual_point = True
        virtual_node.real_point_idx = real_point_idx
        virtual_node.marching_direction = direction
        virtual_node.local_step_factor = local_step_factor
        virtual_node.sp_line_start = virtual_node  # 虚拟点自身是串线起点

        # 虚拟点拓扑关系在创建虚拟阵面/重接右侧阵面时逐步构建，避免继承旧邻接导致方向光滑失真
        virtual_node.node2front = []
        virtual_node.node2node = []
        
        return virtual_node
    
    def _create_virtual_front(self, node1, node2):
        """创建虚拟阵面
        
        Args:
            node1: 起点
            node2: 终点
        """
        from data_structure.front2d import Front
        
        # 创建虚拟阵面
        virtual_front = Front(
            node1,
            node2,
            -1,
            "prism-cap",
            self.adlayers.current_part.part_name,
        )
        
        # 将虚拟阵面添加到推进列表
        self.adlayers.current_part.front_list.append(virtual_front)
        
        # 将虚拟阵面添加到节点的 node2front 中
        self._add_front_link(node1, virtual_front)
        self._add_front_link(node2, virtual_front)
        
        # 更新节点的 node2node 关系
        self._add_neighbor_link(node1, node2)
        
        debug(f"[虚拟阵面] 创建虚拟阵面: {node1.idx} -> {node2.idx}, bc_type=prism-cap")
        
        return virtual_front
    
    def _get_node_by_idx(self, idx):
        """根据编号查找节点
        
        Args:
            idx: 节点编号
            
        Returns:
            node: 节点对象，未找到返回 None
        """
        for node in self.adlayers.front_node_list:
            if node.idx == idx:
                return node
        for node in self.virtual_points:
            if node.idx == idx:
                return node
        return None

    @staticmethod
    def _add_neighbor_link(node1, node2):
        if node1 is None or node2 is None or node1.idx == node2.idx:
            return
        if all(nei.idx != node2.idx for nei in node1.node2node):
            node1.node2node.append(node2)
        if all(nei.idx != node1.idx for nei in node2.node2node):
            node2.node2node.append(node1)

    @staticmethod
    def _remove_neighbor_link(node1, node2):
        if node1 is None or node2 is None:
            return
        node1.node2node = [nei for nei in node1.node2node if nei.idx != node2.idx]
        node2.node2node = [nei for nei in node2.node2node if nei.idx != node1.idx]

    @staticmethod
    def _add_front_link(node, front):
        if node is None or front is None:
            return
        if all(existing is not front for existing in node.node2front):
            node.node2front.append(front)

    @staticmethod
    def _remove_front_link(node, front):
        if node is None or front is None:
            return
        node.node2front = [existing for existing in node.node2front if existing is not front]
    
    def get_real_point(self, node):
        """获取虚拟点对应的真实点
        
        Args:
            node: 节点对象
            
        Returns:
            real_node: 真实点对象
        """
        if not node.is_virtual_point:
            return node
        
        real_idx = node.real_point_idx
        return self._get_node_by_idx(real_idx)
    
    def is_sp_member(self, node):
        """判断节点是否在多方向串线上
        
        Args:
            node: 节点对象
            
        Returns:
            flag: 是否在多方向串线上
        """
        return node.sp_line_start is not None
    
    def get_sp_line_start(self, node):
        """获取节点所在多方向串线的起点
        
        Args:
            node: 节点对象
            
        Returns:
            start_node: 串线起点
        """
        if node.sp_line_start:
            return node.sp_line_start
        return node
    
    def insert_sp_node(self, start_node, new_node):
        """将新推进的节点插入到多方向串线中
        
        Args:
            start_node: 串线起点
            new_node: 新推进的节点
        """
        start_idx = start_node.idx
        
        if start_idx not in self.sp_node_pair:
            self.sp_node_pair[start_idx] = [start_node]
        
        self.sp_node_pair[start_idx].append(new_node)
        
        # 设置新节点的串线起点
        new_node.sp_line_start = start_node

    def _extract_direction(self, direction_value):
        """提取当前有效推进方向向量（多方向列表默认取第一个）"""
        try:
            arr = np.asarray(direction_value, dtype=float)
        except (TypeError, ValueError):
            return None

        if arr.ndim == 0:
            return None
        if arr.ndim >= 2:
            if arr.shape[0] == 0:
                return None
            arr = arr[0]

        arr = np.asarray(arr, dtype=float).reshape(-1)
        if arr.size < 2:
            return None
        return self._normalize(arr[:2])

    def _pick_wall_fronts(self, node):
        wall_fronts = [
            f
            for f in node.node2front
            if self.adlayers.is_wall_front(f) and getattr(f, "length", 1.0) > 1e-10
        ]
        if len(wall_fronts) < 2:
            return None
        return wall_fronts[0], wall_fronts[1]

    def run_layer_workflow(
        self, front_node_list, ilayer, smooth_iterations=3, relax_factor=0.5
    ):
        """按设计文档执行单层多方向流程

        流程：初始化（首层）→ 方向光滑 → 步长缩放
        """
        if not self.adlayers.multi_direction:
            return

        if ilayer == 0:
            self.initialize_multi_direction(front_node_list, ilayer)
            self._log_convex_direction_snapshot("初始化后")

        self.smooth_advancing_direction(
            front_node_list,
            smooth_iterations=smooth_iterations,
            relax_factor=relax_factor,
        )
        if ilayer == 0:
            self._log_convex_direction_snapshot("光滑后")
        self.calculate_step_scale_coeff(front_node_list, ilayer)

    def _insert_local_sp_pair(self, node_idx, coeff):
        """缓存真实点步长因子，已存在则覆盖为最新值"""
        self.local_sp_pair[node_idx] = coeff

    def _get_local_sp_pair(self, node_idx):
        return self.local_sp_pair.get(node_idx, 1.0)
    
    def smooth_advancing_direction(self, front_node_list, smooth_iterations=3, relax_factor=0.5):
        """光滑推进方向
        
        Args:
            front_node_list: 阵面节点列表
            smooth_iterations: 光滑迭代次数
            relax_factor: 松弛因子
        """
        if not self.adlayers.multi_direction:
            return
        
        verbose(f"开始光滑推进方向 (迭代{smooth_iterations}次)...")
        
        for iteration in range(smooth_iterations):
            updated_directions = {}

            for node in front_node_list:
                # 凹点不参与光滑
                if node.concav_flag:
                    continue

                # 获取相邻节点
                neighbors = node.node2node
                if len(neighbors) < 2:
                    continue
                
                # 计算邻居点的平均方向
                neighbor_dirs = []
                for neighbor in neighbors:
                    if neighbor.concav_flag:
                        continue
                    neighbor_dir = self._extract_direction(neighbor.marching_direction)
                    if neighbor_dir is not None:
                        neighbor_dirs.append(neighbor_dir)
                
                if not neighbor_dirs:
                    continue
                
                avg_neighbor_dir = np.mean(neighbor_dirs, axis=0)
                avg_neighbor_dir = self._normalize(avg_neighbor_dir)
                
                # 加权平均
                current_dir = self._extract_direction(node.marching_direction)
                if current_dir is None:
                    continue
                # relax_factor=0 表示不光滑，=1 表示完全采用邻居平均方向
                new_dir = (1 - relax_factor) * current_dir + relax_factor * avg_neighbor_dir
                new_dir = self._normalize(new_dir)

                updated_directions[node.idx] = new_dir

            # 统一更新，避免本轮迭代内的更新顺序影响其他节点
            for node in front_node_list:
                if node.idx not in updated_directions:
                    continue

                new_dir = updated_directions[node.idx]
                if isinstance(node.marching_direction, list):
                    if len(node.marching_direction) == 0:
                        node.marching_direction = [tuple(new_dir)]
                    else:
                        node.marching_direction[0] = tuple(new_dir)
                else:
                    node.marching_direction = tuple(new_dir)
        
        verbose("推进方向光滑完成")

    def _log_convex_direction_snapshot(self, stage):
        """输出凸点及虚拟点推进方向，便于多方向调试"""
        if not self.convex_points:
            return

        for convex_node in self.convex_points:
            chain_nodes = [convex_node] + list(getattr(convex_node, "virtual_points", []))
            direction_text = []
            for node in chain_nodes:
                vec = self._extract_direction(node.marching_direction)
                if vec is None:
                    continue
                direction_text.append(
                    f"id={node.idx},dir=({vec[0]:.4f},{vec[1]:.4f}),nei={len(node.node2node)}"
                )
            if direction_text:
                debug(f"[多方向方向][{stage}] " + " | ".join(direction_text))
    
    def calculate_step_scale_coeff(self, front_node_list, ilayer):
        """计算步长缩放系数
        
        根据相邻阵面法向与推进方向的夹角，对局部步长进行缩放
        
        Args:
            front_node_list: 阵面节点列表
            ilayer: 当前推进层数
        """
        verbose("计算步长缩放系数...")

        # 清空缓存，避免跨层残留
        self.local_sp_pair = {}

        # 阶段 1：计算真实点步长因子（跳过虚拟点；首层后跳过真实凸点）
        for node in front_node_list:
            if node.is_virtual_point:
                continue

            if ilayer > 0 and node.convex_flag and not node.is_virtual_point:
                continue

            picked = self._pick_wall_fronts(node)
            if picked is None:
                continue

            front1, front2 = picked
            np_dir = self._extract_direction(node.marching_direction)
            if np_dir is None:
                continue

            coeff1 = np.dot(np.array(front1.normal), np_dir)
            coeff2 = np.dot(np.array(front2.normal), np_dir)
            coeff = (abs(coeff1) + abs(coeff2)) / 2.0
            if coeff > 1e-10:
                node.local_step_factor = node.local_step_factor / coeff

            if node.convex_flag:
                self._insert_local_sp_pair(node.idx, node.local_step_factor)

        # 阶段 2：首层虚拟点步长因子从真实点复制，并缓存真实点系数
        if ilayer == 0:
            for node in front_node_list:
                if not node.is_virtual_point:
                    continue

                real_node = self.get_real_point(node)
                if real_node is None:
                    continue

                node.local_step_factor = real_node.local_step_factor
                self._insert_local_sp_pair(real_node.idx, real_node.local_step_factor)

        # 阶段 3：多方向串线上的点从缓存继承真实点步长
        for node in front_node_list:
            if not node.sp_line_start:
                continue

            start_node = self.get_sp_line_start(node)
            if start_node.idx == node.idx:
                continue

            real_node = self.get_real_point(start_node)
            if real_node:
                node.local_step_factor = self._get_local_sp_pair(real_node.idx)

        verbose("步长缩放系数计算完成")
    
    def update_stop_flag(self, node, ilayer, iconvex=False, iconcav=False):
        """更新停止推进标志
        
        Args:
            node: 节点对象
            ilayer: 当前推进层数
            iconvex: 是否为凸点
            iconcav: 是否为凹点
        """
        if iconvex and node.idx not in self.stop_flag_convex:
            self.stop_flag_convex[node.idx] = False
            self.stop_layer_convex[node.idx] = 1000
        
        if iconcav and node.idx not in self.stop_flag_concav:
            self.stop_flag_concav[node.idx] = False
            self.stop_layer_concav[node.idx] = 1000
        
        if iconvex and not self.stop_flag_convex.get(node.idx, False):
            self.stop_layer_convex[node.idx] = ilayer + 1
            self.stop_flag_convex[node.idx] = True
        
        if iconcav and not self.stop_flag_concav.get(node.idx, False):
            self.stop_layer_concav[node.idx] = ilayer + 1
            self.stop_flag_concav[node.idx] = True
    
    def check_termination(self, node, ilayer, aspect_ratio, skewness):
        """检查是否满足终止条件
        
        Args:
            node: 节点对象
            ilayer: 当前推进层数
            aspect_ratio: 单元长细比
            skewness: 单元偏斜度
            
        Returns:
            should_stop: 是否应该停止推进
        """
        full_layers = self.adlayers.full_layers
        max_layers = self.adlayers.max_layers
        
        # 完整层数内不终止
        if ilayer < full_layers:
            return False
        
        # 超过最大层数
        if ilayer >= max_layers:
            return True
        
        # 偏斜度条件
        if skewness >= 1.0:
            return True
        
        # 凸点终止条件
        if node.convex_flag:
            real_idx = node.idx
            if real_idx in self.stop_flag_convex and self.stop_flag_convex[real_idx]:
                stop_layer = self.stop_layer_convex[real_idx]
                
                # 多方向串线上的点
                if node.sp_line_start:
                    if ilayer >= stop_layer:
                        return True
                else:
                    # 非多方向点，检查质量
                    if aspect_ratio < 1.2 or skewness >= 0.75:
                        return True
        
        # 凹点终止条件
        if node.concav_flag:
            if node.idx in self.stop_flag_concav and self.stop_flag_concav[node.idx]:
                stop_layer = self.stop_layer_concav[node.idx]
                if ilayer >= stop_layer:
                    return True
        
        return False
    
    def post_process(self):
        """后处理：消除虚拟点
        
        将虚拟阵面转换为真实阵面，删除无效的虚拟阵面
        """
        verbose("开始后处理：消除虚拟点...")
        
        # 标记需要删除的虚拟阵面
        fronts_to_remove = []
        
        for part in self.adlayers.part_params:
            for front in part.front_list:
                node1, node2 = front.node_elems
                
                # 检查是否为虚拟点
                is_virt1 = getattr(node1, 'is_virtual_point', False)
                is_virt2 = getattr(node2, 'is_virtual_point', False)
                
                if is_virt1 and is_virt2:
                    # 两个点都是虚拟点，标记删除
                    fronts_to_remove.append(front)
                elif is_virt1:
                    # 第一个点是虚拟点，替换为真实点
                    real_node = self.get_real_point(node1)
                    if real_node:
                        front.node_elems[0] = real_node
                        front.node_ids[0] = real_node.idx
                elif is_virt2:
                    # 第二个点是虚拟点，替换为真实点
                    real_node = self.get_real_point(node2)
                    if real_node:
                        front.node_elems[1] = real_node
                        front.node_ids[1] = real_node.idx
                
                # 检查两点是否重合
                if node1.idx == node2.idx:
                    fronts_to_remove.append(front)
        
        # 删除虚拟阵面
        for part in self.adlayers.part_params:
            part.front_list = [f for f in part.front_list if f not in fronts_to_remove]
        
        verbose(f"删除了 {len(fronts_to_remove)} 个虚拟阵面")
        verbose("虚拟点消除完成")
    
    @staticmethod
    def _normalize(vec):
        """归一化向量
        
        Args:
            vec: 向量
            
        Returns:
            normalized: 归一化后的向量
        """
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            return vec / norm
        return vec


def compute_multi_directions(node, angle_degrees, normal1):
    """计算多方向矢量
    
    Args:
        node: 节点对象
        angle_degrees: 凸角角度
        normal1: 第一个阵面法向
        
    Returns:
        directions: 多方向矢量列表
    """
    # 计算多方向数量
    num_directions = int(np.radians(angle_degrees) / (1.1 * pi / 3)) + 1
    num_directions = max(2, min(num_directions, 6))  # 限制在 2-6 之间
    
    if num_directions == 1:
        return [tuple(normal1)]
    
    # 计算方向间隔角
    delta_theta = -np.radians(angle_degrees) / (num_directions - 1)
    
    # 生成等分方向
    directions = []
    initial_vector = np.array(normal1)
    
    for i in range(num_directions):
        angle = i * delta_theta
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        rotated_vector = np.dot(rotation_matrix, initial_vector)
        directions.append(tuple(rotated_vector))
    
    return directions
