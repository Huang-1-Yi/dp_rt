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

# 共享投影 + 头
"""
先把 global_feature = [time_emb; global_cond] 过一个共享 cond_base: Linear(dcond → r)(r=128 推荐）得到 cond_r。
再准备三个一次性 head: head_256: Linear(r → C256), head_512, head_1024 (若开启 scale 就是 → 2*C)
在 forward 的开头算一次这三组向量，缓存到 dict, 后续 12 个块直接拿来用 (不再在块内各算各的 MLP)
"""
# 需要把残差块改为“直接吃预先算好的 cond 张量”，权重组织略变；微调很快收敛。


from typing import Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import ( Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class CrossAttention(nn.Module):
    def __init__(self, in_dim, cond_dim, out_dim):
        super().__init__()
        self.query_proj = nn.Linear(in_dim, out_dim)
        self.key_proj = nn.Linear(cond_dim, out_dim)
        self.value_proj = nn.Linear(cond_dim, out_dim)

    def forward(self, x, cond):
        # x: [batch_size, t_act, in_dim]
        # cond: [batch_size, t_obs, cond_dim]

        # Project x and cond to query, key, and value
        query = self.query_proj(x)  # [batch_size, horizon, out_dim]
        key = self.key_proj(cond)   # [batch_size, horizon, out_dim]
        value = self.value_proj(cond)  # [batch_size, horizon, out_dim]


        # Compute attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, horizon, horizon]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention
        attn_output = torch.matmul(attn_weights, value)  # [batch_size, horizon, out_dim]
        
        return attn_output

# 条件残差块
# ​​输入输出​​：接受形状为 [B, C, T] 的输入和条件 [B, cond_dim]
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,

            condition_type='film',
            expects_preproj: bool = False  # <<< 新增：是否期望外部预投影的 cond
        ):
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

        self.condition_type = condition_type
        self.expects_preproj = expects_preproj
        self.out_channels = out_channels

        # FiLM调制参数：生成scale和bias
        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        # ​​FiLM条件调制​​：通过条件生成scale和bias（cond_predict_scale=True）或直接相加特征（cond_predict_scale=False）
        # 构建 cond 编码器（若 expects_preproj=True 且不是 cross-attn，则不再建 Linear）
        if condition_type in {'film', 'add', 'mlp_film'}:
            if self.expects_preproj:
                self.cond_encoder = nn.Identity()  # 直接吃 [B,C] 或 [B,2C]
            else:
                if condition_type == 'film':
                    cond_channels = out_channels * 2
                    self.cond_encoder = nn.Sequential(
                        nn.Mish(), nn.Linear(cond_dim, cond_channels),
                        Rearrange('batch t -> batch t 1'),
                    )
                elif condition_type == 'add':
                    self.cond_encoder = nn.Sequential(
                        nn.Mish(), nn.Linear(cond_dim, out_channels),
                        Rearrange('batch t -> batch t 1'),
                    )
                elif condition_type == 'mlp_film':
                    cond_channels = out_channels * 2
                    self.cond_encoder = nn.Sequential(
                        nn.Mish(), nn.Linear(cond_dim, cond_dim),
                        nn.Mish(), nn.Linear(cond_dim, cond_channels),
                        Rearrange('batch t -> batch t 1'),
                    )
        elif condition_type in {'cross_attention_add', 'cross_attention_film'}:
            # cross-attn 仍需内部编码器
            if condition_type == 'cross_attention_add':
                self.cond_encoder = CrossAttention(in_channels, cond_dim, out_channels)
            else:  # cross_attention_film
                self.cond_encoder = CrossAttention(in_channels, cond_dim, out_channels * 2)
        else:
            raise NotImplementedError(f"condition_type {condition_type} not implemented")

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

        if cond is not None:
            if self.condition_type in {'film', 'add', 'mlp_film'}:
                if self.expects_preproj:
                    # 预投影：cond 已是 [B, C] 或 [B, 2C]
                    if self.condition_type in self._two_param_types:
                        # [B, 2C] -> [B,2,C,1]
                        embed = cond.view(cond.shape[0], 2, self.out_channels, 1)
                        scale, bias = embed[:, 0, ...], embed[:, 1, ...]
                        out = scale * out + bias
                    else:
                        embed = cond.unsqueeze(-1)  # [B,C,1]
                        out = out + embed
                else:
                    # 旧路径：块内线性+reshape
                    embed = self.cond_encoder(cond)
                    if self.condition_type in self._two_param_types:
                        embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                        scale, bias = embed[:, 0, ...], embed[:, 1, ...]
                        out = scale * out + bias
                    else:
                        out = out + embed
            elif self.condition_type == 'cross_attention_add':
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)  # [B,T,C]
                out = out + embed.permute(0, 2, 1)
            elif self.condition_type == 'cross_attention_film':
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)  # [B,T,2C]
                embed = embed.permute(0, 2, 1).reshape(embed.shape[0], 2, self.out_channels, -1)
                scale, bias = embed[:, 0, ...], embed[:, 1, ...]
                out = scale * out + bias
            else:
                raise NotImplementedError(f"condition_type {self.condition_type} not implemented")
        
        out = self.blocks[1](out)
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
        condition_type='film',
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        ):
        super().__init__()

        self.condition_type = condition_type 
        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition

        # ----------------- 共享 cond 预投影（新增） -----------------
        # r: 共享瓶颈维度
        r = 128
        self.cond_bottleneck_dim = r
        # film / cross_attention_film / mlp_film 需要 (scale,bias)=2*C，其它仅 C
        two_param_types = {'film', 'cross_attention_film', 'mlp_film'}
        m = 2 if condition_type in two_param_types else 1

        # cond_base: [dsed + global_cond_dim] -> r
        self.cond_base = nn.Sequential(
            nn.Linear(diffusion_step_embed_dim + (global_cond_dim or 0), r),
            nn.SiLU(),
        )
        # 需要的 head 通道集合（覆盖下/中/上所有 block 的 out_channels）
        # 对该 UNet1D，所有 block 的 out_channels 都来自 down_dims；
        # up 路径的两个块 out_channels = 对应尺度的 dim_in（即 down_dims[:-1]），也包含在集合里。
        head_channels = sorted(set(down_dims))
        self._cond_head_channels = head_channels
        self.cond_heads = nn.ModuleDict({
            str(C): nn.Linear(r, C * m) for C in head_channels
        })
        expects_preproj = condition_type in {'film', 'add', 'mlp_film'}
        # -----------------------------------------------------------

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
                    condition_type=condition_type,
                    expects_preproj=expects_preproj),
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type,
                    expects_preproj=expects_preproj)
            ])

        # 中间层模块
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type,
                expects_preproj=expects_preproj
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type,
                expects_preproj=expects_preproj
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
                    condition_type=condition_type,
                    expects_preproj=expects_preproj),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type,
                    expects_preproj=expects_preproj),
                Downsample1d(dim_out) if not is_last else nn.Identity() # 下采样操作
            ]))

        # 上采样模块
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type,
                    expects_preproj=expects_preproj),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type,
                    expects_preproj=expects_preproj),
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

    def _build_global_feature(self, timestep_embed, global_cond, B, device, dtype):
        """返回 [B, dsed+G]（非 cross-attn）或 [B,T,dsed+G]（cross-attn）"""
        if self.condition_type in {'cross_attention_add', 'cross_attention_film'}:
            assert global_cond is not None and global_cond.dim() == 3, \
                "cross-attn 需 [B,T,G] 的 global_cond"
            timestep_seq = timestep_embed.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
            return torch.cat([timestep_seq, global_cond], dim=-1)
        else:
            # 非 cross-attn：若 global_cond 缺省，用 0 向量补齐到维度
            if global_cond is None:
                if self.global_cond_dim > 0:
                    pad = torch.zeros(B, self.global_cond_dim, device=device, dtype=dtype)
                    return torch.cat([timestep_embed, pad], dim=-1)
                return timestep_embed
            else:
                assert global_cond.dim() == 2, "非 cross-attn 条件应为 [B,G]"
                return torch.cat([timestep_embed, global_cond], dim=-1)

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):

        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 时间步
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.dim() == 0:
            timesteps = timesteps[None].to(sample.device)
        B = sample.shape[0]
        timesteps = timesteps.expand(B)
        timestep_embed = self.diffusion_step_encoder(timesteps)  # [B, dsed]

        # 构造 global_feature
        global_feature = self._build_global_feature(
            timestep_embed, global_cond, B, sample.device, timestep_embed.dtype
        )

        # cond_cache（仅 film/add/mlp_film 使用）
        use_preproj = self.condition_type in {'film', 'add', 'mlp_film'}
        cond_cache = None
        if use_preproj:
            # global_feature: [B, dsed+G]
            assert global_feature.dim() == 2, "预投影路径需 [B, dsed+G]"
            cond_r = self.cond_base(global_feature)              # [B, r]
            cond_cache = {C: self.cond_heads[str(C)](cond_r)     # [B, C] 或 [B, 2C]
                          for C in self._cond_head_channels}

        # local 条件
        h_local = list()
        if local_cond is not None and self.local_cond_encoder is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            if use_preproj:
                x = resnet(local_cond, cond_cache[resnet.out_channels])
                h_local.append(x)
                x = resnet2(local_cond, cond_cache[resnet2.out_channels])
                h_local.append(x)
            else:
                x = resnet(local_cond, global_feature)
                h_local.append(x)
                x = resnet2(local_cond, global_feature)
                h_local.append(x)

        # 下采样
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            if self.use_down_condition:
                if use_preproj:
                    x = resnet(x, cond_cache[resnet.out_channels])
                    if idx == 0 and len(h_local) > 0:
                        x = x + h_local[0]
                    x = resnet2(x, cond_cache[resnet2.out_channels])
                else:
                    x = resnet(x, global_feature)
                    if idx == 0 and len(h_local) > 0:
                        x = x + h_local[0]
                    x = resnet2(x, global_feature)
            else:
                x = resnet(x, None)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, None)
            h.append(x)
            x = downsample(x)

        # 中间层
        for mid_module in self.mid_modules:
            if self.use_mid_condition:
                if use_preproj:
                    x = mid_module(x, cond_cache[mid_module.out_channels])
                else:
                    x = mid_module(x, global_feature)
            else:
                x = mid_module(x, None)

        # 上采样
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            if self.use_up_condition:
                if use_preproj:
                    x = resnet(x, cond_cache[resnet.out_channels])
                    if idx == len(self.up_modules) - 1 and len(h_local) > 0:
                        x = x + h_local[1]
                    x = resnet2(x, cond_cache[resnet2.out_channels])
                else:
                    x = resnet(x, global_feature)
                    if idx == len(self.up_modules) - 1 and len(h_local) > 0:
                        x = x + h_local[1]
                    x = resnet2(x, global_feature)
            else:
                x = resnet(x, None)
                if idx == len(self.up_modules) - 1 and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, None)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x

