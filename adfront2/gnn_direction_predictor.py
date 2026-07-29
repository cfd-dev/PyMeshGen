"""GNN 方向预测器

使用训练好的 GNN 模型预测层推进方向矢量，
替代传统的阵面法向平均 + Laplacian 光滑流程。
"""

import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data

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

        Args:
            model_path: 模型文件路径 (.pth)，支持绝对路径或相对于项目根目录的相对路径

        Returns:
            GNNDirectionPredictor 实例
        """
        from neural.GNN_ALM.nn_utils.model_checkpoint import load_model
        from neural.GNN_ALM.nn_utils.trainer import create_model

        model_path = Path(model_path)
        # 相对路径相对于项目根目录解析
        if not model_path.is_absolute():
            project_root = Path(__file__).parent.parent
            model_path = project_root / model_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        result = load_model(model_path, device)
        if not result:
            raise FileNotFoundError(f"无法加载模型: {model_path}")

        model_config = result.get("model_config")
        if not model_config:
            raise ValueError(f"模型文件缺少 model_config: {model_path}")

        model = create_model(model_config, device)
        model.load_state_dict(result["model_state"])
        model.eval()

        norm_params = model_config.get("norm_params")

        info(f"[GNN] 已加载模型: {model_path.name}")
        return cls(model, model_config, norm_params, device)

    def _build_graph(self, node_list):
        """从阵面节点列表构建图数据

        Args:
            node_list: NodeElementALM 对象列表

        Returns:
            PyG Data 对象，构建失败返回 None
        """
        n = len(node_list)
        if n < 2:
            return None

        node_index_map = {id(node): i for i, node in enumerate(node_list)}

        # 从 node2node 收集无向边
        edge_set = set()
        for i, node in enumerate(node_list):
            for neighbor in node.node2node:
                j = node_index_map.get(id(neighbor))
                if j is not None and j != i:
                    edge_set.add((min(i, j), max(i, j)))

        if not edge_set:
            return None

        # 构建双向边列表和边特征
        edge_index = []
        edge_features = []

        for a, b in edge_set:
            edge_index.append([a, b])
            edge_index.append([b, a])

        coords = [np.array(node.coords[:2], dtype=np.float64) for node in node_list]

        # 边特征: [长度, 相对长度, dx, dy, 曲率]
        edge_lengths = []
        edge_raw = []
        for a, b in edge_set:
            vec = coords[b] - coords[a]
            length = float(np.linalg.norm(vec))
            edge_lengths.append(length)
            direction = vec / length if length > 1e-12 else np.zeros(2)
            edge_raw.append((length, direction))

        avg_length = sum(edge_lengths) / len(edge_lengths) if edge_lengths else 1.0

        # 计算每个节点的曲率（相邻边的最大转向角）
        node_curvatures = {}
        for i, node in enumerate(node_list):
            neighbor_indices = []
            for neighbor in node.node2node:
                j = node_index_map.get(id(neighbor))
                if j is not None and j != i:
                    neighbor_indices.append(j)

            if len(neighbor_indices) < 2:
                node_curvatures[i] = 0.0
                continue

            max_turn = 0.0
            for ni in range(len(neighbor_indices)):
                for nj in range(ni + 1, len(neighbor_indices)):
                    j1, j2 = neighbor_indices[ni], neighbor_indices[nj]
                    v1 = coords[j1] - coords[i]
                    v2 = coords[j2] - coords[i]
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    if n1 > 1e-12 and n2 > 1e-12:
                        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                        turn = abs(np.arccos(cos_angle))
                        max_turn = max(max_turn, turn)
            node_curvatures[i] = max_turn

        # 组装边特征（双向边各一条）
        edge_idx = 0
        for a, b in edge_set:
            length, direction = edge_raw[edge_idx]
            rel_len = length / (avg_length + 1e-8)
            curv = max(node_curvatures.get(a, 0.0), node_curvatures.get(b, 0.0))
            feat = [length, rel_len, direction[0], direction[1], curv]
            edge_features.append(feat)
            edge_features.append(feat)
            edge_idx += 1

        # 节点坐标
        node_features = np.array([[c[0], c[1]] for c in coords], dtype=np.float64)

        # 归一化（与训练时一致）
        if self.norm_params:
            coord_scale = self.norm_params.get("data_range", 1.0) or 1.0
            node_features[:, 0] = (
                node_features[:, 0] - self.norm_params["x_min"]
            ) / coord_scale
            node_features[:, 1] = (
                node_features[:, 1] - self.norm_params["y_min"]
            ) / coord_scale
            for ef in edge_features:
                ef[0] /= coord_scale

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
        )

        return data

    def predict(self, node_list):
        """预测节点推进方向

        Args:
            node_list: NodeElementALM 对象列表

        Returns:
            预测方向列表 [(dx, dy), ...]，与 node_list 一一对应
        """
        n = len(node_list)
        if n == 0:
            return []

        data = self._build_graph(node_list)
        if data is None:
            warning("[GNN] 图构建失败，返回零向量")
            return [(0.0, 0.0)] * n

        data = data.to(self.device)

        with torch.no_grad():
            pred = self.model(data)
            pred_np = pred.cpu().numpy()

        return [(float(pred_np[i, 0]), float(pred_np[i, 1])) for i in range(n)]

    def predict_and_assign(self, node_list):
        """预测并写回推进方向

        对多方向节点（marching_direction 为 list），保持已有方向扇不变。

        Args:
            node_list: NodeElementALM 对象列表
        """
        directions = self.predict(node_list)

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
