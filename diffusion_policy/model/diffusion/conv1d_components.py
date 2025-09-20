# 这段代码定义了一个一维卷积块 Conv1dBlock，包含卷积、归一化和激活函数。它可以用于处理一维数据，如时间序列或音频信号。
# Downsample1d 用于一维下采样（通过一维卷积）,用 stride=2 的卷积实现下采样。
# Upsample1d 用于一维上采样（通过转置卷积）,用 stride=2 的转置卷积实现上采样。
# Conv1dBlock 是一个卷积+归一化+激活的模块。

import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange
# Rearrange 表明 代码是从其他维度（如二维）的模块修改而来，但在一维场景下不需要调整维度。确保这些注释不会影响代码逻辑即可


class Downsample1d(nn.Module):  # 父类是 nn.Module, 它们是 PyTorch 神经网络模块的子类
    def __init__(self, dim):
        super().__init__()      # 调用 nn.Module 的初始化方法
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

        # self.block = nn.Sequential(
        #     nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
        #     # Rearrange('batch channels horizon -> batch channels 1 horizon'),
        #     nn.GroupNorm(n_groups, out_channels),
        #     # Rearrange('batch channels 1 horizon -> batch channels horizon'),
        #     nn.Mish(),
        # )

    def forward(self, x):
        return self.block(x)


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    print(cb)                       # 打印模型结构Conv1dBlock(256, 128, kernel_size=3)

    x = torch.zeros((1, 256, 16))   # 输入形状 (batch=1, channels=256, length=16)
    print("x", x.shape)             # torch.Size([1, 256, 16])

    o = cb(x)                       # 输出形状应为 (1, 128, 16)
    print("o", o.shape)             # torch.Size([1, 128, 16])
    print(o)                        # 输出的张量

# 确保测试函数被调用
if __name__ == "__main__":
    test()
"""
Conv1dBlock(
  (block): Sequential(
    (0): Conv1d(256, 128, kernel_size=(3,), stride=(1,), padding=(1,))
    (1): GroupNorm(8, 128, eps=1e-05, affine=True)
    (2): Mish()
  )
)
x torch.Size([1, 256, 16])
o torch.Size([1, 128, 16])
tensor([[[-0.2539, -0.2539, -0.2539,  ..., -0.2539, -0.2539, -0.2539],
         [-0.3012, -0.3012, -0.3012,  ..., -0.3012, -0.3012, -0.3012],
         [ 0.9573,  0.9573,  0.9573,  ...,  0.9573,  0.9573,  0.9573],
         ...,
         [ 0.1268,  0.1268,  0.1268,  ...,  0.1268,  0.1268,  0.1268],
         [ 0.8750,  0.8750,  0.8750,  ...,  0.8750,  0.8750,  0.8750],
         [ 0.5703,  0.5703,  0.5703,  ...,  0.5703,  0.5703,  0.5703]]],
       grad_fn=<MishBackward0>)
"""
