# unet or transformer 都是用的这个进行 positional embedding
# 不同位置和维度的编码值交替变化
# 这种​​正交频率设计​​使模型可通过注意力机制选择性地组合不同频率的编码，从而解析出XY轴几何关系。例如：
# 低频维度（热图顶部）编码全局位置（如物体在图像左侧/右侧）
# 高频维度（热图底部）编码局部位置（如边缘对齐）
# ​​UNet的扩展性​​：通过显式添加位置编码，UNet可学习类似的位置感知能力，尤其在需要精确定位的任务中表现更优。
# ​​ViT的几何解码能力​​：通过正交频率编码和自注意力机制，ViT天然具备解析XY轴信息的能力，可视化结果验证了其空间推理的有效性。
import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        # 频率计算，当前值为10000，实际上，减小值为1000适合图像的任务，局部相关性更强，增大则更适合长序列任务
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 相位计算
        emb = x[:, None] * emb[None, :]
        # 正弦余弦拼接
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
