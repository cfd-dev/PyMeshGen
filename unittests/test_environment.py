import unittest
import torch


class TestCUDAEnvironment(unittest.TestCase):
    """CUDA环境检测测试套件"""

    def test_cuda_availability(self):
        """检测CUDA是否可用"""
        self.assertTrue(
            torch.cuda.is_available(), "CUDA不可用，请检查显卡驱动和PyTorch安装"
        )
        print(f"CUDA可用性验证通过: {torch.cuda.is_available()}")

    def test_device_properties(self):
        """检测GPU设备属性"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            print(f"当前设备: {torch.cuda.get_device_name(device)}")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"PyTorch版本: {torch.__version__}")

            # 验证显存容量
            total_mem = torch.cuda.get_device_properties(device).total_memory
            print(
                f"显存容量: {total_mem / 1024/1024/1024:.2f} GB"
            )  # 字节转GB并保留两位小数
            self.assertGreater(total_mem, 0, "显存容量检测异常")

    def test_gpu_count(self):
        """检测可用GPU数量"""
        gpu_count = torch.cuda.device_count()
        print(f"检测到GPU数量: {gpu_count}")
        self.assertGreater(gpu_count, 0, "未检测到可用GPU设备")


if __name__ == "__main__":
    unittest.main()
