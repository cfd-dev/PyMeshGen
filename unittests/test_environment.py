import torch

print(f"PyTorch 版本: {torch.__version__}")  # 应显示 GPU 版本（如 2.0.0+cu117）
print(f"CUDA 是否可用: {torch.cuda.is_available()}")  # 应输出 True
print(f"CUDA 版本: {torch.version.cuda}")  # 应输出 11.7
