"""GNN 方向预测器

使用训练好的 GNN 模型预测层推进方向矢量，
替代传统的阵面法向平均 + Laplacian 光滑流程。
"""

import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.message import info, warning


class GNNDirectionPredictor:
    """使用训练好的 GNN 模型预测推进方向"""

    def __init__(self, model, model_config, norm_params, device):
        self.model = model
        self.model_config = model_config
        self.norm_params = norm_params
        self.device = device
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, model_path):
        """从模型检查点加载预测器

        复用 tester.py 中的 load_model_from_checkpoint 函数。

        Args:
            model_path: 模型文件路径 (.pth)，支持绝对路径或相对于项目根目录的相对路径

        Returns:
            GNNDirectionPredictor 实例
        """
        # 延迟导入，避免循环依赖和顶层依赖问题
        from neural.GNN_ALM.nn_utils.tester import load_model_from_checkpoint

        model_path = Path(model_path)
        # 相对路径相对于项目根目录解析
        if not model_path.is_absolute():
            project_root = Path(__file__).parent.parent
            model_path = project_root / model_path

        # 复用 tester.py 中的加载函数
        model, device, _ = load_model_from_checkpoint(str(model_path))

        # 直接从 checkpoint 文件中提取 model_config 和 norm_params
        # （与 test_cas_file.py 保持一致）
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
        model_config = checkpoint.get('model_config', {})
        norm_params = checkpoint.get('norm_params') or model_config.get('norm_params')

        if norm_params is not None:
            info(f"[GNN] 从模型中提取到归一化参数:")
            info(f"     X范围: [{norm_params['x_min']:.4f}, {norm_params['x_max']:.4f}], range={norm_params['x_range']:.4f}")
            info(f"     Y范围: [{norm_params['y_min']:.4f}, {norm_params['y_max']:.4f}], range={norm_params['y_range']:.4f}")
        else:
            warning("[GNN] 模型中未保存归一化参数，将使用原始坐标")

        info(f"[GNN] 已加载模型: {model_path.name}")
        return cls(model, model_config, norm_params, device)

    def _build_graph(self, node_list):
        """从阵面节点列表构建图数据

        与 test_cas_file.py 保持一致：先归一化坐标，再构建边特征。
        这样所有特征（长度、相对长度、方向、曲率）都在归一化空间中计算，
        与训练数据的分布完全一致。

        Args:
            node_list: NodeElementALM 对象列表

        Returns:
            PyG Data 对象，构建失败返回 None
        """
        n = len(node_list)
        if n < 2:
            return None

        # 建立节点索引映射
        node_index_map = {id(node): i for i, node in enumerate(node_list)}

        # 节点坐标（原始物理坐标）
        raw_coords = [np.array(node.coords[:2], dtype=np.float64) for node in node_list]

        # === 先归一化坐标（与 test_cas_file.py 的 extract_ground_truth 一致）===
        if self.norm_params:
            from neural.GNN_ALM.nn_utils.data_utils import normalize_xy

            coords = [
                np.array(normalize_xy(c[0], c[1], self.norm_params))
                for c in raw_coords
            ]
        else:
            coords = raw_coords

        node_features = np.array([[c[0], c[1]] for c in coords], dtype=np.float64)

        # 从 node2node 追踪环路，确保边沿顺时针方向
        # 构建局部邻接表（node_list 索引 → 邻居索引列表）
        adj = [[] for _ in range(n)]
        for i, node in enumerate(node_list):
            for neighbor in node.node2node:
                j = node_index_map.get(id(neighbor))
                if j is not None and j != i:
                    adj[i].append(j)

        # 追踪环路并确保顺时针方向
        edge_set = set()
        visited = [False] * n
        for start in range(n):
            if visited[start]:
                continue
            # 沿邻接表追踪环路
            cycle = [start]
            visited[start] = True
            prev = -1
            curr = start
            while True:
                neighbors = [nb for nb in adj[curr] if nb != prev]
                if not neighbors:
                    break
                next_node = neighbors[0]
                if next_node == start:
                    cycle.append(next_node)  # 闭合
                    break
                if visited[next_node]:
                    break
                visited[next_node] = True
                cycle.append(next_node)
                prev, curr = curr, next_node

            if len(cycle) < 3:
                continue

            # 用 shoelace 公式计算有向面积，判断方向
            signed_area = 0.0
            for k in range(len(cycle) - 1):
                c0, c1 = coords[cycle[k]], coords[cycle[k + 1]]
                signed_area += c0[0] * c1[1] - c1[0] * c0[1]

            # 有向面积 > 0 表示逆时针，需要反转为顺时针
            if signed_area > 0:
                # 去掉闭合节点后反转，再补回闭合
                cycle = cycle[:-1]
                cycle.reverse()
                cycle.append(cycle[0])

            # 构建有向边（沿环路方向）
            for k in range(len(cycle) - 1):
                edge_set.add((cycle[k], cycle[k + 1]))

        if not edge_set:
            return None

        # 构建边列表
        edge_list = list(edge_set)
        edge_index = [[a, b] for a, b in edge_list]

        # 计算边向量、长度和方向（基于归一化后的坐标）
        edge_vecs = []
        edge_lengths = []
        for a, b in edge_list:
            vec = coords[b] - coords[a]
            length = float(np.linalg.norm(vec))
            edge_vecs.append(vec)
            edge_lengths.append(length)

        avg_length = sum(edge_lengths) / len(edge_lengths) if edge_lengths else 1.0

        edge_dirs = []
        for vec, length in zip(edge_vecs, edge_lengths):
            if length > 1e-12:
                edge_dirs.append(vec / length)
            else:
                edge_dirs.append(np.zeros(2))

        # 构建节点出边索引（用于计算边的曲率）
        node_out_edges = [[] for _ in range(n)]
        for e_idx, (a, b) in enumerate(edge_list):
            node_out_edges[a].append(e_idx)

        # 计算每条边的曲率（与训练数据一致：基于目标节点出边的转向角）
        edge_curvatures = []
        for e_idx, (a, b) in enumerate(edge_list):
            dir_e = edge_dirs[e_idx]
            out_edges = node_out_edges[b]  # 目标节点 b 的出边

            max_turn = 0.0
            for e2 in out_edges:
                dir_e2 = edge_dirs[e2]
                # 计算转向角（有向）
                cross = dir_e[0] * dir_e2[1] - dir_e[1] * dir_e2[0]
                dot = dir_e[0] * dir_e2[0] + dir_e[1] * dir_e2[1]
                turn = abs(np.arctan2(cross, dot))
                max_turn = max(max_turn, turn)

            edge_curvatures.append(max_turn)

        # 组装边特征: [长度, 相对长度, dx, dy, 曲率]
        edge_features = []
        for e_idx, length in enumerate(edge_lengths):
            rel_len = length / (avg_length + 1e-8)
            dx, dy = edge_dirs[e_idx]
            curvature = edge_curvatures[e_idx]
            edge_features.append([length, rel_len, dx, dy, curvature])

        # 创建 PyG Data（坐标已归一化，无需再调用 normalize_data）
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
        )

        return data

    def _predict_with_data(self, node_list):
        """内部方法：执行推理并返回 (prediction, data) 供复用

        Args:
            node_list: NodeElementALM 对象列表

        Returns:
            (prediction_tensor, data) 元组，失败时返回 (None, None)
        """
        from neural.GNN_ALM.nn_utils.tester import test_single_sample

        n = len(node_list)
        if n == 0:
            return None, None

        data = self._build_graph(node_list)
        if data is None:
            warning("[GNN] 图构建失败")
            return None, None

        data.y = torch.zeros((n, 2), dtype=torch.float)
        prediction, _ = test_single_sample(self.model, data, self.device)
        return prediction, data

    def predict(self, node_list):
        """预测节点推进方向

        复用 tester.py 中的 test_single_sample 逻辑进行推理。

        Args:
            node_list: NodeElementALM 对象列表

        Returns:
            预测方向列表 [(dx, dy), ...]，与 node_list 一一对应
        """
        n = len(node_list)
        prediction, _ = self._predict_with_data(node_list)
        if prediction is None:
            return [(0.0, 0.0)] * n

        pred_np = prediction.cpu().numpy()
        return [(float(pred_np[i, 0]), float(pred_np[i, 1])) for i in range(n)]

    def predict_and_assign(self, node_list, prediction=None, data=None):
        """预测并写回推进方向

        对多方向节点（marching_direction 为 list），保持已有方向扇不变。
        支持传入预计算的 prediction 以避免重复推理。

        Args:
            node_list: NodeElementALM 对象列表
            prediction: 预计算的预测结果（可选，为 None 时调用 _predict_with_data）
            data: 预构建的图数据（可选，仅供 visualize_predictions 复用）

        Returns:
            (prediction_tensor, data) 元组，供可视化复用
        """
        if prediction is None:
            prediction, data = self._predict_with_data(node_list)

        if prediction is None:
            return None, None

        pred_np = prediction.cpu().numpy()
        directions = [(float(pred_np[i, 0]), float(pred_np[i, 1])) for i in range(len(node_list))]

        assigned = 0
        skipped = 0
        for node, direction in zip(node_list, directions):
            # 保持多方向列表不变（凸角方向扇由 compute_corner_attributes 设置）
            if isinstance(node.marching_direction, list):
                skipped += 1
                continue
            node.marching_direction = direction
            assigned += 1

        info(
            f"[GNN] 方向预测完成: {assigned} 个节点已赋值, "
            f"{skipped} 个多方向节点保持原方向扇"
        )

        return prediction, data

    def visualize_predictions(self, node_list, output_path,
                              title="GNN Predicted Marching Directions",
                              prediction=None, data=None):
        """可视化预测的推进方向

        复用 tester.py 中的 visualize_prediction 函数进行可视化。
        支持传入预计算的 prediction 和 data 以避免重复推理。

        Args:
            node_list: NodeElementALM 对象列表
            output_path: 输出图片路径
            title: 图片标题
            prediction: 预计算的预测结果（可选，为 None 时重新推理）
            data: 预构建的图数据（可选，为 None 时重新构建）
        """
        # 延迟导入，避免顶层依赖问题
        from neural.GNN_ALM.nn_utils.tester import visualize_prediction, compute_metrics, test_single_sample

        if not node_list:
            warning("[GNN] 无可视化数据：节点列表为空")
            return

        # 如果未传入预计算结果，则重新构建图和推理
        if data is None or prediction is None:
            data = self._build_graph(node_list)
            if data is None:
                warning("[GNN] 图构建失败，无法可视化")
                return
            n = len(node_list)
            data.y = torch.zeros((n, 2), dtype=torch.float)
            prediction, ground_truth = test_single_sample(self.model, data, self.device)
        else:
            ground_truth = data.y if data.y is not None else torch.zeros_like(prediction)

        # 计算评估指标
        metrics = compute_metrics(prediction, ground_truth)

        # 确保输出目录存在
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 复用 tester.py 中的 visualize_prediction 函数
        visualize_prediction(
            data,
            prediction,
            ground_truth,
            title=title,
            save_path=str(output_path),
            metrics=metrics
        )

        info(f"[GNN] 方向预测可视化已保存: {output_path}")


def find_trained_model(model_dir=None, keyword=None):
    """查找已训练的模型文件

    Args:
        model_dir: 模型目录，默认为 GNN_ALM/naca/models
        keyword: 文件名过滤关键字

    Returns:
        模型文件路径，未找到返回 None
    """
    if model_dir is None:
        model_dir = (
            Path(__file__).parent.parent
            / "neural"
            / "GNN_ALM"
            / "naca"
            / "models"
        )
    else:
        model_dir = Path(model_dir)

    if not model_dir.exists():
        return None

    pth_files = list(model_dir.glob("saved_model_*.pth"))
    if not pth_files:
        return None

    if keyword:
        pth_files = [f for f in pth_files if keyword in f.stem]
        if not pth_files:
            return None

    return sorted(pth_files)[-1]
