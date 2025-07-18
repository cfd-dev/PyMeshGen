MODEL_CONFIG = {
    "model_type": "GCN",
    "num_gcn_layers": 5,
    "hidden_channels": 32,
    "residual_switch": False,
    "dropout": 0.1,
    "normalization": "GroupNorm",
}

TRAINING_CONFIG = {
    "train_ratio": 0.8,  # 训练集比例
    "batch_size": 32,  # 批量大小
    "total_epochs": 100000,  # 总训练轮次
    "log_interval": 50,
    "learning_rate": 0.01,  # 学习率
    "validation_interval": 200,  # 验证间隔
    "lr_stepsize": 500,  # 学习率调整步长
    "lr_gamma": 0.95,  # 学习率调整因子
    "visualize_training": True,  # 是否绘制训练历史 
}
