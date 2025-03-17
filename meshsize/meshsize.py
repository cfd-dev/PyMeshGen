from math import log
import math

class QuadTreeNode:
    def __init__(self, bounds):
        self.bounds = bounds
        self.children = None 
        self.father = None
        self.sp = None 
        
        self.level = 0
        self.subflag = 0

    def draw_node(self):
        pass
        return

    def draw_quadtree(self):
        pass
        return

class QuadtreeSizing:
    def __init__(self, initial_front=None, max_size=1.0, resolution=0.1, decay=1.2):
        
        self.max_size = max_size  # 最大网格尺寸
        self.resolution = resolution  # 分辨率  
        self.decay = decay  # 网格尺寸场的decay
        self.initial_front = initial_front  # 阵面列表

        self.bg_bounds = []  # 背景网格边界
        self.global_spacing = None  # 全局网格尺寸
        self.depth = None  # 叉树深度
        self.bg_dim = []  # 背景网格维度
        self.quad_tree = None  # 四叉树     

        self.generate_bg_mesh()

    def generate_bg_mesh(self):
        # 初始化计算域的BoundingBox及全局尺寸及叉树深度
        self.compute_global_parameters()

        # 初始叉树网格
        self.initial_quadtree()

        # 细分网格
        self.refine_quadtree()

        # 加密区扩散，避免level相差>=2
        self.level_refinement()

        # 根据decay参数计算网格尺度场的decay
        self.compute_spacing_decay()

        # 网格尺度场过渡
        self.spacing_transition()

        self.grid_summary()

        return

    def compute_global_parameters(self):
        print(f'计算域的BoundingBox...')
        all_points = []
        min_edge_len = float('inf')
        max_edge_len = 0
        for front in self.initial_front:
            all_points.extend(front.nodes_coords)
            min_edge_len = min(min_edge_len, front.length)
            max_edge_len = max(max_edge_len, front.length)

        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]

        # 计算初始边界范围（扩展10%防止边界情况）
        pad = 0.1
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_range = x_max - x_min
        y_range = y_max - y_min

        self.bg_bounds = (x_min - x_range*pad,
                y_min - y_range*pad,
                x_max + x_range*pad,
                y_max + y_range*pad)
        
        print(f'计算最大尺度及树深度...')
        # 计算全局尺寸及叉树深度
        if self.max_size > min_edge_len and self.max_size < max_edge_len:
            self.global_spacing = self.max_size
        else:
            self.global_spacing = max_edge_len
            
        self.depth = round(1.4427*log(self.global_spacing / min_edge_len) + 0.5); 

    def initial_quadtree(self):
        root_bounds = (*self.bg_bounds[:2],  # min_x, min_y
                      self.bg_bounds[0] + self.global_spacing,  # max_x
                      self.bg_bounds[1] + self.global_spacing)  # max_y
        self.quad_tree = QuadTreeNode(root_bounds)
        self.quad_tree.sp = self.global_spacing

    def refine_quadtree(self):
        """实现基于表面节点尺寸的四叉树细分"""        
        def _should_refine(node, surface_size):
            """细分条件判断"""
            if node.level >= self.depth:
                return False
            if node.sp <= surface_size:
                return False
            diff = (node.sp - surface_size) / node.sp
            return diff > self.resolution

        def _locate_quadtree(point, node):
            """定位点在四叉树中的叶节点"""
            min_x, min_y, max_x, max_y = node.bounds
            if not (min_x <= point[0] <= max_x and min_y <= point[1] <= max_y):
                # raise ValueError(f"Point {point} 超出叉树节点边界 {node.bounds}")
                kkk = 1

            while node.children:
                x_center = (node.bounds[0] + node.bounds[2])/2
                y_center = (node.bounds[1] + node.bounds[3])/2
                
                # 判断点所在象限
                quadrant = 0
                if point[0] > x_center: quadrant += 1
                if point[1] > y_center: quadrant += 2
                node = node.children[quadrant]
            return node       

        # 遍历所有表面节点
        for front in self.initial_front:
            face_center = front.front_center
            target_size = front.length

            current = self.quad_tree
            current = _locate_quadtree(face_center, current)
            # 执行递归细分
            while _should_refine(current, target_size):
                if not current.children:
                    # 细分当前节点
                    current.subflag = 1
                    current.children = [
                            QuadTreeNode(divide_bounds(current.bounds)) 
                            for _ in range(4)
                        ]
                    for child in current.children:
                            child.level = current.level + 1
                            child.father = current
                            child.sp = current.sp / 2
                    
                # 更新当前节点到子节点
                current = _locate_quadtree(face_center, current)
                
    def level_refinement(self):
        """平衡树层级差异"""
        from collections import deque

        def find_neighbors(node):
            """查找当前叶节点四个方向的相邻叶节点"""
            neighbors = []
            # 获取当前节点边界
            min_x, min_y, max_x, max_y = node.bounds
            cell_size = max_x - min_x

            # 四个方向的搜索向量 (dx_min, dx_max, dy_min, dy_max)
            directions = [
                (-cell_size, 0, 0, 0),        # 西
                (cell_size, 0, 0, 0),         # 东 
                (0, 0, -cell_size, 0),        # 南
                (0, 0, cell_size, 0)          # 北
            ]

            # 广度优先搜索每个方向的邻居
            for dx_min, dx_max, dy_min, dy_max in directions:
                search_bounds = (
                    min_x + dx_min + dx_max,
                    min_y + dy_min + dy_max,
                    max_x + dx_min + dx_max,
                    max_y + dy_min + dy_max
                )
                
                # 从根节点开始搜索
                queue = deque([self.quad_tree])
                while queue:
                    current = queue.popleft()
                    if current == node:
                        continue
                    
                    # 检查边界重叠
                    c_min_x, c_min_y, c_max_x, c_max_y = current.bounds
                    if (search_bounds[0] < c_max_x and 
                        search_bounds[2] > c_min_x and 
                        search_bounds[1] < c_max_y and 
                        search_bounds[3] > c_min_y):
                        
                        if current.children:
                            queue.extend(current.children)
                        else:
                            neighbors.append(current)
            
            return neighbors

        def force_refine(node):
            """强制细分节点"""
            if not node.children and node.level < self.depth:
                node.subflag = 1
                node.children = [QuadTreeNode(divide_bounds(node.bounds)) for _ in range(4)]
                for child in node.children:
                    child.level = node.level + 1
                    child.father = node
                    child.sp = node.sp / 2

        # 主平衡逻辑
        queue = deque([self.quad_tree])
        visited = set()

        while queue:
            node = queue.popleft()
            if id(node) in visited:
                continue
            visited.add(id(node))

            if node.children:
                queue.extend(node.children)
                continue

            # 获取相邻叶节点
            neighbors = find_neighbors(node)
            
            for neighbor in neighbors:
                # 层级差>=2时需要细分
                while node.level - neighbor.level >= 2:
                    force_refine(neighbor)
                    # 细分后需要处理新生成的子节点
                    if neighbor.children:
                        queue.extend(neighbor.children)
                    neighbor = neighbor.father  # 向上追溯直到满足条件

    def traverse_tree(self, node):
        """递归枚举四叉树节点"""
        if not node.children:
            yield node
        else:
            for child in node.children:
                yield from traverse_tree(child)

    def compute_spacing_decay(self):
        """计算考虑表面节点影响的网格尺寸"""
        def source_spacing(base_size, dist, expand):
            """基于指数衰减的尺寸计算"""
            pwr = 0.5 * dist * (expand - 1.0) / base_size
            pwr = min(pwr, 50.0)  # 限制最大指数
            return base_size * math.exp(pwr)

        if self.decay < 1.0:
            return 

        # 遍历四叉树所有节点
        for node in self.traverse_tree(self.quad_tree):
            min_sp = float('inf')
            x_center = (node.bounds[0] + node.bounds[2])/2
            y_center = (node.bounds[1] + node.bounds[3])/2
             # 遍历所有表面节点
            for front in self.initial_front:
                face_center = front.front_center
                target_size = front.length

                # 计算子节点中心到表面节点的距离
                dx = x_center - face_center[0]
                dy = y_center - face_center[1]
                dist = math.sqrt(dx**2 + dy**2)
                            
                # 计算该表面节点影响的尺寸
                sp = source_spacing(target_size, dist, self.decay)
                min_sp = min(min_sp, sp)

            # 限制最小尺寸不超过全局尺寸
            node.sp = min(min_sp, self.global_spacing)

    def spacing_transition(self):
        """基于相邻节点尺寸的平滑过渡"""
        from collections import deque
        
        # 四叉树节点相邻关系映射表 [子节点索引][方向] -> 相邻节点索引
        bg_mapSides = [
            [-1, 1, -1, 2],  # 子节点0 (NW) 的西、东、南、北相邻
            [0, -1, -1, 3],  # 子节点1 (NE)
            [-1, 3, 0, -1],  # 子节点2 (SW)
            [2, -1, 1, -1]   # 子节点3 (SE)
        ]
        
        def update_sizes(node, neighbors):
            """递归更新节点尺寸"""
            changed = 0
            if node.children:
                # 处理有子节点的情况
                for child_idx in range(4):
                    child = node.children[child_idx]
                    child_neighbors = []
                    
                    # 获取子节点的四个方向相邻节点
                    for dir in range(4):  # 方向: 0=西 1=东 2=南 3=北
                        map_idx = bg_mapSides[child_idx][dir]
                        
                        if map_idx == -1:  # 需要从父节点的邻居获取
                            neighbor_node = neighbors[dir]
                            if neighbor_node and neighbor_node.children:
                                # 取邻居节点的对应子节点
                                oppo_dir = dir ^ 1  # 取反方向
                                child_neighbors.append(neighbor_node.children[oppo_dir])
                            else:
                                child_neighbors.append(None)
                        else:
                            child_neighbors.append(node.children[map_idx])
                    
                    # 递归更新子节点
                    changed += update_sizes(child, child_neighbors)
                return changed
            else:
                # 处理叶节点尺寸更新
                for dir in range(4):
                    neighbor = neighbors[dir]
                    if not neighbor or neighbor.children:
                        continue
                    
                    # 层级相同时直接比较
                    if neighbor.level == node.level:
                        if neighbor.sp < node.sp:
                            node.sp = 0.5 * (node.sp + neighbor.sp)
                            changed += 1
                    # 相邻节点更粗糙时（层级小1）
                    elif neighbor.level == node.level - 1:
                        if neighbor.sp < node.sp:
                            node.sp = 0.5 * (node.sp + neighbor.sp)
                            changed += 1
                return changed
        
        # 主循环直到没有尺寸变化
        while True:
            total_changed = 0
            # 广度优先遍历所有节点
            queue = deque([(self.quad_tree, [None]*4)])  # (节点, [西,东,南,北邻居])
            
            while queue:
                node, neighbors = queue.popleft()
                
                # 更新当前节点尺寸
                total_changed += update_sizes(node, neighbors)
                
                if node.children:
                    # 为子节点准备邻居信息
                    for child_idx, child in enumerate(node.children):
                        child_neighbors = []
                        for dir in range(4):
                            map_idx = bg_mapSides[child_idx][dir]
                            if map_idx != -1:
                                child_neighbors.append(node.children[map_idx])
                            else:
                                child_neighbors.append(neighbors[dir])
                        queue.append((child, child_neighbors))
            
            if total_changed == 0:
                break

    def grid_summary(self):
        """输出网格统计信息"""
        node_count = 0
        max_level = 0
        for node in self.traverse_tree(self.quad_tree):
            node_count += 1
            max_level = max(max_level, node.level)
            
        print(f"网格统计:")
        print(f"- 总节点数: {node_count}")
        print(f"- 最大深度: {max_level}")
        print(f"- 最小尺寸: {self.global_spacing / (2 ** max_level):.4f}")



