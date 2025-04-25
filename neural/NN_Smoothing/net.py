import torch
import torch.nn.functional as F  # 激励函数都在这


class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)  # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden2, n_output)  # 输出层线性输出

    def forward(self, x):  # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.sigmoid(self.hidden1(x))  # 激励函数(隐藏层的线性值)
        x = F.sigmoid(self.hidden2(x))  # 激励函数(隐藏层的线性值)
        x = self.predict(x)  # 输出值
        return x


if __name__ == "__main__":
    print("This is net file!")
