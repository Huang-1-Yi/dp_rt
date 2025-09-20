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

        # 扩散时间步编码器
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # 确保全局条件维度正确
        self.global_cond_dim = global_cond_dim
        
        # 条件编码器
        if global_cond_dim is not None:
            # 创建投影层，将全局条件维度映射到扩散步嵌入维度
            self.global_cond_proj = nn.Linear(global_cond_dim, diffusion_step_embed_dim)
            
            # 条件编码器使用扩散步嵌入维度作为输入
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
                nn.Mish()
            )
        
        # 局部条件编码器（若存在）
        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            self.local_cond_encoder = nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=diffusion_step_embed_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=diffusion_step_embed_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])
        
        # 中间层模块
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=diffusion_step_embed_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=diffusion_step_embed_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        # 下采样模块
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            is_last = ind >= (len(all_dims) - 2)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=diffusion_step_embed_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=diffusion_step_embed_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        # 上采样模块
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(list(zip(all_dims[1:], all_dims[:-1])))):
            is_last = ind >= (len(all_dims) - 2)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=diffusion_step_embed_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=diffusion_step_embed_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        # 将所有模块组合在一起
        self.down_modules = down_modules
        self.up_modules = up_modules

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        # 1. 输入预处理
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 2. 时间步编码
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        timestep_embed = self.diffusion_step_encoder(timesteps)

        # 3. 全局条件处理
        if global_cond is not None:
            # 确保全局条件维度正确
            if global_cond.shape[-1] != self.global_cond_dim:
                # 使用投影层修正维度
                global_cond = self.global_cond_proj(global_cond)
            
            # 通过条件编码器
            global_feature = self.cond_encoder(global_cond)
            
            # 将时间步长嵌入与条件特征相加
            aug_timestep_embed = timestep_embed + global_feature
        else:
            aug_timestep_embed = timestep_embed
        
        # 4. 局部条件处理
        h_local = []
        if local_cond is not None and self.local_cond_encoder is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, aug_timestep_embed)
            h_local.append(x)
            x = resnet2(local_cond, aug_timestep_embed)
            h_local.append(x)
        
        # 5. 下采样过程
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, aug_timestep_embed)
            if idx == 0 and h_local:
                x = x + h_local[0]
            x = resnet2(x, aug_timestep_embed)
            h.append(x)
            x = downsample(x)
        
        # 6. 中间层处理
        for mid_module in self.mid_modules:
            x = mid_module(x, aug_timestep_embed)
        
        # 7. 上采样过程
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, aug_timestep_embed)
            if idx == len(self.up_modules) - 1 and len(h_local) > 1:
                x = x + h_local[1]
            x = resnet2(x, aug_timestep_embed)
            x = upsample(x)
        
        # 8. 最终输出
        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x
