"""
网络架构：
    一个面向一维时序数据的条件​​扩散模型​
    核心是基于UNet结构改进的条件卷积网络，通过引入​​时间步编码​​、​​全局条件​​（如观察特征）和​​局部条件​​（如历史动作）来实现可控生成。
    适用于时间序列预测任务（如机器人动作生成）。

​​条件注入机制​​：
    ​​全局条件​​（如扩散时间步、语言指令）：通过拼接扩散时间步特征和全局条件，输入到所有残差块。通过逐层跳跃连接保持高频信息，避免生成结果过度模糊。
    ​​局部条件​​（如历史动作）：通过独立的编码器处理后，在下采样和上采样的特定层注入。
​​UNet结构​​：
    ​​下采样​​：逐步压缩特征维度，保存跳跃连接。
    ​​上采样​​：逐步恢复维度，拼接跳跃连接。
    ​​中间层​​：进一步处理深层特征。
​​扩散时间步编码​​：
    使用Sinusoidal位置编码 + MLP层，增强模型对扩散过程当前去噪阶段的时序感知。
​​FiLM调制​​：
    可选是否通过条件预测scale和bias，增强模型对条件的响应能力。


"""
# 时间步编码（Sinusoidal） SinusoidalPosEmb(dsed) → 行1	                            ConditionalUnet1D.__init__()
# 时间步与全局条件融合	    torch.cat([global_feature, global_cond]) → 行5	         ConditionalUnet1D.forward()
# FiLM参数生成	          self.cond_encoder(cond) → 行7 + scale/bias切片 → 行8-9	ConditionalResidualBlock1D.forward()
# FiLM调制应用	          out = scale * out + bias → 行10	                       ConditionalResidualBlock1D.forward()
# 残差块条件注入	       resnet(x, global_feature) → 行6	                        ConditionalUnet1D.forward()


from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import ( Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

# 条件残差块
# ​​输入输出​​：接受形状为 [B, C, T] 的输入和条件 [B, cond_dim]
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        # 定义两个卷积块Conv1dBlock
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        # 每个Conv1dBlock都是1D卷积块，包含卷积、归一化和激活函数：
        # self.block = nn.Sequential(
        #     nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
        #     nn.GroupNorm(n_groups, out_channels),
        #     nn.Mish(),
        # )
        # forward方法是直接调用self.block(x)


        # FiLM调制参数：生成scale和bias
        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        # ​​FiLM条件调制​​：通过条件生成scale和bias（cond_predict_scale=True）或直接相加特征（cond_predict_scale=False）
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2            # 同时预测scale和bias
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        
        # 条件编码器（将条件向量映射到通道数）
        self.cond_encoder = nn.Sequential(
            nn.Mish(),                                  # 激活函数
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),          # 调整维度为 [B, T, 1]
        )

        # 残差连接（处理输入输出通道不一致的情况）稳定训练，保持梯度流动 make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    # 前向传播​
    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        # 第一个卷积块处理输入
        out = self.blocks[0](x)
        
        # 编码条件
        embed = self.cond_encoder(cond)                     # 行7：条件编码生成参数
        
        # FiLM调制的核心计算（缩放和偏移）——FiLM调制或直接相加
        if self.cond_predict_scale:
            # 通过条件编码器生成调制参数：缩放(scale)和偏移(shift)
            # ​​输入​:embed 的原始形状为 (B, 2*C, 1)（假设 cond_encoder 的输出维度是 2*C）。
            # ​​输出​​：通过 reshape 将 2*C 拆分为两个独立的通道维度 (2, C)，最终形状为 (B, 2, C, 1)
            # 原始输出：[0.8, 0.1, -0.3, 0.5, 1.2, -0.7]  # 形状 (B, 6, 1)
            # reshape后：[[0.8, -0.3, 1.2], [0.1, 0.5, -0.7]]  # 形状 (B, 2, 3, 1)
            embed = embed.reshape( embed.shape[0], 2, self.out_channels, 1 )
            # 从 embed 张量的第二个维度（索引为 0）提取数据，生成 ​​缩放系数（scale）斯瓦夫萨大色想；；；
            # scale[0,0,0] = embed[0,0,0,0]  # 样本0，通道0的缩放值
            # scale[1,2,0] = embed[1,0,2,0]  # 样本1，通道2的缩放值
            scale = embed[:,0,...]      # [B, C, 1]     # 行8：切片取缩放系数
            bias = embed[:,1,...]       # [B, C, 1]     # 行9：切片取偏移系数
            out = scale * out + bias    # (B, C, T)     # 行10：按通道调制
        else:
            out = out + embed           # 直接相加（如 B=2, C=3, T=5）
        
        # 第二个卷积块处理调制后的输出
        out = self.blocks[1](out)
        
        # 残差连接
        out = out + self.residual_conv(x)
        return out

# 主网络 ConditionalUnet1D 模块​ 
# 典型的UNet下采样-跳跃连接-上采样架构（共4层，默认下采样尺度down_dims=[256,512,1024]）
# ​​条件集成​​：
#     ​​时间步条件​​：通过扩散步骤的正弦编码引入时间依赖性。
#     ​​全局条件​​（如场景观察特征）：与时间步编码拼接。
#     局部条件​​（如历史动作轨迹）：通过local_cond_encoder编码并注入到特定层。
class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        # 定义各层维度（输入层 + 下采样层）
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]


        # 扩散时间步编码器（Sinusoidal位置编码）是一个线性层，将离散时间步 t 映射为特征向量（如 t=100 → [0.3, -0.5, ...]）
        # 该时间步编码器，输入为扩散时间步的正弦位置编码，输入维度为 diffusion_step_embed_dim
        # 该时间步编码器，将离散的时间步 t 映射为一个连续的特征向量，输出维度为 diffusion_step_embed_dim
        # 时间步特征向量将与全局条件拼接后，插入到UNet的各层中，例如通过​​相加​​或​​拼接​​到网络中间的特征图
        dsed = diffusion_step_embed_dim # 256维位置编码
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),     # Step1：生成正弦位置编码
            nn.Linear(dsed, dsed * 4),  # Step2：扩展维度
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),  # Step3：压缩回目标维度，最终得到扩散时间步特征
        )

        # 全局条件拼接扩散时间步特征
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        # 定义下采样和上采样的层对
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # 局部条件编码器（若存在）
        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # 下采样编码器 down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # 上采样编码器 up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])
        
        # 中间层模块
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        # 下采样模块（包含多层条件残差块）
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity() # 下采样操作
            ]))

        # 上采样模块
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,               # 拼接跳跃连接
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()    # 上采样操作
            ]))

        # 最终输出层
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),                         # 1x1卷积恢复输入维度
        )

        # 将所有模块组合在一起，注册模块
        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, 
            global_cond=None, 
            **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # 1. ​​输入预处理​，调整输入维度：[B, T, H] -> [B, H, T]
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 2. ​​时间步编码与全局条件集成​or处理局部条件 local_cond
        # 1. 处理时间步编码 time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        # 生成扩散时间步特征（[B, dsed]），当 global_cond 不存在时​​：global_feature 保持为时间步编码结果，无需任何操作
        global_feature = self.diffusion_step_encoder(timesteps) # Step4：时间步编码调用 [B, dsed]

        if global_cond is not None: # 仅在 global_cond 存在时
            # 拼接全局条件（如场景特征、观测特征）——​​将扩散时间步编码（diffusion_step_encoder 的输出）与全局条件（global_cond）拼接
            global_feature = torch.cat([global_feature, global_cond], axis=-1)# Step5：将时间步特征与其他全局条件（如观测特征）融合 [B, dsed + global_cond_dim]
            # 如果diffusion_step_encoder 的输出维度为 dsed（如 256），global_cond 的维度为 global_cond_dim（如 128）
            # 拼接后的 global_feature 维度为 dsed + global_cond_dim（如 384）。如果后续模块（如 ConditionalResidualBlock1D）的 cond_dim 参数设置为此值，则维度匹配
        
        # encode local features
        # 编码局部条件（如历史动作）
        h_local = list()
        if local_cond is not None:
            # 处理局部条件（如历史动作轨迹）
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            # local_cond_encoder构成：
            # 下采样编码器 down encoder ConditionalResidualBlock1D
            # 上采样编码器 up encoder ConditionalResidualBlock1D
            resnet, resnet2 = self.local_cond_encoder   # 局部条件编码器
            x = resnet(local_cond, global_feature)
            h_local.append(x)                           # 保存到列表供后续注入
            x = resnet2(local_cond, global_feature)
            h_local.append(x)                           # 保存到列表供后续注入
        
        # 下采样过程：特征分辨率逐步降低，通道数逐步增加
        x = sample
        h = []      # 保存跳跃连接
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)                               # Step6：将global_feature输入残差块
            if idx == 0 and len(h_local) > 0:   # 第一层注入局部条件
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)                         # 保存特征用于跳跃连接
            x = downsample(x)                   # 下采样（Conv1d stride=2）

        # 中间层处理：维持最低分辨率下的特征处理
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # 上采样过程：利用跳跃连接恢复空间细节，通过注入局部条件（h_local[1]）到最终上采样层
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)  # 拼接跳跃连接维度（通道维度）
            x = resnet(x, global_feature)                               # Step6：将global_feature输入残差块
            # 在循环条件判断中，原本应该使用len(self.up_modules)-1（最后一个索引）来触发局部条件的注入，
            # 但是为了兼容已经发布的模型（这些模型是在没有上采样部分注入局部条件的情况下训练的），
            # 所以保留了一个永远不会触发的条件（即不注入）。这是一种为了兼容性的折中做法。
            # The correct condition should be:
            # dp3考虑了这一点
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:    # 最后一层注入局部条件
            # 局部条件未正确注入上采样层​​，这会损害模型的条件生成能力
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)             # 上采样（ConvTranspose1d）   

        # 最终卷积层
        # 1x1卷积恢复输入维度
        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')# 恢复维度 [B, H, T]
        return x

