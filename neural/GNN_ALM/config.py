MODEL_CONFIG = {
    "model_type": "GCN",
    "num_gcn_layers": 8,
    "hidden_channels": 64,
    "residual_switch": True,
    "dropout": 0.0,
    "normalization": "LayerNorm",
}

TRAINING_CONFIG = {
    "train_ratio": 0.8,  # 训练集比例
    "batch_size": 3,  # 批量大小
    "total_epochs": 200000,  # 总训练轮次
    "log_interval": 50,
    "learning_rate": 0.01,  # 学习率
    "validation_interval": 200,  # 验证间隔
    "lr_stepsize": 2000,  # 学习率调整步长
    "lr_gamma": 0.9,  # 学习率调整因子
    "visualize_training": True,  # 是否绘制训练历史
}
