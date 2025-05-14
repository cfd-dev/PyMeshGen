MODEL_CONFIG = {
    "model_name": "GAT",
    "hidden_channels": 64,
    "num_gcn_layers": 4,
    "residual_switch": True,
    "dropout": 0.3,
    "normalization": "LayerNorm",
}

TRAINING_CONFIG = {
    "train_ratio": 1.0,  # 训练集比例
    "batch_size": 3,  # 批量大小
    "total_epochs": 50000,  # 总训练轮次
    "log_interval": 50,
    "learning_rate": 0.01,  # 学习率
    "validation_interval": 200,  # 验证间隔
    "lr_stepsize": 1000,  # 学习率调整步长
    "lr_gamma": 0.9,  # 学习率调整因子
}
