from typing import Dict, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, 'CropRandomizer'):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.vision.rot_randomizer import RotRandomizer


class SlidingWindowDiffusionPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            rot_aug=False,
            # 滑动窗口相关参数
            window_loss_weights="linear",  # 窗口内不同位置的损失权重
            parallel_value=16,  # 并行扩展的批次大小
            **kwargs):
        super().__init__()

        # 解析shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # 获取原始robomimic配置
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # 初始化全局状态
        ObsUtils.initialize_obs_utils_with_config(config)

        # 加载模型
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # 创建扩散模型
        obs_feature_dim = obs_encoder.output_shape()[0]
        print(f"Obs encoder output shape: {obs_encoder.output_shape()}")
        print(f"Obs feature dim: {obs_feature_dim}")

        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        
        # 初始化noise_scheduler
        self.noise_scheduler = noise_scheduler

        # 预计算alpha_bar
        self.num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        beta_schedule = self.noise_scheduler.config.beta_schedule
        if beta_schedule == "squaredcos_cap_v2":
            s = 0.008
            t = torch.arange(0, self.num_train_timesteps + 1, dtype=torch.float32)
            alpha_bar = torch.cos((t / self.num_train_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            self.alpha_bar = alpha_bar
        else:
            beta_start = self.noise_scheduler.config.beta_start
            beta_end = self.noise_scheduler.config.beta_end
            betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=torch.float32)
            alphas = 1.0 - betas
            alpha_bar = torch.cumprod(alphas, dim=0)
            alpha_bar = torch.cat([torch.tensor([1.0]), alpha_bar])
            self.register_buffer('alpha_bar', alpha_bar)

        self.normalizer = LinearNormalizer()
        self.rot_randomizer = RotRandomizer() if rot_aug else None
        self.rot_aug = rot_aug

        self.parallel_value = parallel_value    # 并行扩展的批次大小
        self.horizon = horizon                  # 滑动窗口参数
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.rot_aug = rot_aug
        self.kwargs = kwargs

        
        # 设置窗口内损失权重
        if window_loss_weights == "linear":
            self.window_weights = torch.linspace(1.0, 0.1, horizon)
        elif window_loss_weights == "constant":
            self.window_weights = torch.ones(horizon)
        else:
            raise ValueError(f"Unknown window loss weights: {window_loss_weights}")
            
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # 推理状态
        self._inference_buffer = None
        self._inference_t_buffer = None
        self._inference_global_cond = None

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    def _prepare_action_blocks(self, actions: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        将动作序列分割成块，并进行必要的填充
        
        参数:
            actions: 形状为 [B, T, D] 的动作序列
            
        返回:
            padded_actions: 填充后的动作序列 [B, padded_T, D]
            num_blocks: 块的数量
        """
        B, T, D = actions.shape
        
        # 打印原始动作序列形状
        print(f"Original actions shape: {actions.shape}")

        # 计算需要填充的长度
        remainder = T % self.n_action_steps
        if remainder != 0:
            pad_length = self.n_action_steps - remainder
            # 使用最后一个动作值进行填充
            last_action = actions[:, -1:, :]
            padding = last_action.repeat(1, pad_length, 1)
            actions = torch.cat([actions, padding], dim=1)
            T = actions.shape[1]  # 更新序列长度
        
        # 计算块的数量
        num_blocks = T // self.n_action_steps
        
        # 打印填充后的动作形状
        print(f"Padded actions shape: {actions.shape}")
        print(f"Number of blocks: {num_blocks}")

        """
        部分输出如下：
            Original actions shape: torch.Size([1, 100, 10])
            Padded actions shape: torch.Size([1, 104, 10])
            Number of blocks: 13
            Original actions shape: torch.Size([1, 98, 10])
            Padded actions shape: torch.Size([1, 104, 10])
            Number of blocks: 13
            Original actions shape: torch.Size([1, 110, 10])
            Padded actions shape: torch.Size([1, 112, 10])
            Number of blocks: 14
            Original actions shape: torch.Size([1, 109, 10])
            Padded actions shape: torch.Size([1, 112, 10])
            Number of blocks: 14
            Original actions shape: torch.Size([1, 101, 10])
            Padded actions shape: torch.Size([1, 104, 10])
            Number of blocks: 13
        这些输出表明，动作序列被正确地填充到可以被n_action_steps整除的长度，
        并且块的数量也被正确计算出来。
        """

        return actions, num_blocks
    
    def _apply_noise_to_blocks(self, actions: torch.Tensor, window_start: int, noise_levels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对动作序列的指定窗口应用不同程度的噪声
        
        参数:
            actions: 形状为 [B, T, D] 的动作序列
            window_start: 窗口起始位置
            noise_levels: 形状为 [horizon] 的噪声强度
            
        返回:
            noisy_actions: 加噪后的动作序列
            true_noise: 真实添加的噪声
        """
        B, T, D = actions.shape
        H = self.horizon
        S = self.n_action_steps
        
        # 提取当前窗口的动作 [B, H*S, D]
        window_actions = actions[:, window_start*S:(window_start+H)*S, :]
        
        # 确保 noise_levels 与 self.alpha_bar 在同一设备上
        noise_levels = noise_levels.to(self.alpha_bar.device)
        
        # 获取对应的alpha_bar值
        alpha_bar_t = self.alpha_bar[noise_levels]  # [H]
        
        # 为每个块重复alpha_bar值
        # expanded_noise_levels 使每个位置一个噪声强度[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        alpha_bar_t = alpha_bar_t.repeat_interleave(S).view(1, H*S, 1)  # [1, H*S, 1] 

        # 确保 alpha_bar_t 与 actions 在同一设备上
        if alpha_bar_t.device != actions.device:
            alpha_bar_t = alpha_bar_t.to(actions.device)
        
        # 生成随机噪声 - 每个时间步都有不同的随机值
        true_noise = torch.randn_like(window_actions)  # [B, H*S, D]
        
        # 应用噪声 - 使用扩展后的alpha_bar值
        noisy_actions = torch.sqrt(alpha_bar_t) * window_actions + torch.sqrt(1 - alpha_bar_t) * true_noise
        
        return noisy_actions, true_noise
    
    def _prepare_global_condition(self, obs_dict: Dict[str, torch.Tensor], 
                                 batch_size: int) -> torch.Tensor:
        """
        准备全局条件信息
        
        参数:
            obs_dict: 观测字典
            batch_size: 批次大小
            
        返回:
            global_cond: 形状为 [batch_size * parallel, global_cond_dim] 的全局条件
        """
        # 编码观测
        this_obs = dict_apply(obs_dict, lambda x: x.reshape(-1, *x.shape[2:]))
        obs_features = self.obs_encoder(this_obs)
        
        # 打印观测特征形状以调试
        print(f"Obs features shape after encoder: {obs_features.shape}")    # Obs features shape after encoder: torch.Size([2, 137])

        # 计算实际的特征总数
        num_features = obs_features.shape[1]  # 假设形状为 [B*T, D]
        
        # 重塑回 [B, n_obs_steps * actual_feature_dim]
        obs_features = obs_features.reshape(batch_size, self.n_obs_steps * num_features)
        
        # 打印重塑后的形状
        print(f"Obs features shape after reshape: {obs_features.shape}")    # Obs features shape after reshape: torch.Size([1, 274])
        
        # 扩展批次维度以匹配并行处理
        global_cond = obs_features.repeat(self.parallel_value, 1)
        return global_cond
    
    def compute_loss(self, batch):
        """
        基于滑动窗口的因果迭代去噪训练
        
        参数:
            batch: 包含'obs'和'action'的字典，视作一整条episode
            
        返回:
            loss: 加权平均损失
        """
        # 1. 标准化输入
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        if self.rot_aug and self.rot_randomizer is not None:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
        
        # 2. 批次检查
        B, T, Da = nactions.shape
        if B != 1:
            raise ValueError("批次设置错误：本算法强制使用单批次样本进行训练")
        
        # 3. 将动作分割成块并进行填充
        padded_actions, num_blocks = self._prepare_action_blocks(nactions)
        
        # 4. 滑动窗口处理
        device = nactions.device    # 获取设备，损失设置为张量，确保返回的是张量而非浮点数
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        window_count = 0
        
        # 滑动窗口: 每次滑动一个块
        # 确保窗口起始位置 + n_obs_steps 不超过序列长度
        max_window_start = num_blocks - self.horizon + 1 - (self.n_obs_steps - 1)
        for window_start in range(0, max_window_start):
            
            # 5. 对窗口内的块应用不同程度的噪声
            # 位置0: 无噪声 (t=0)
            # 位置1: 强度1 (t=1)
            # ...
            # 位置6: 强度6 (t=6)
            noise_levels = torch.arange(self.horizon, device=self.alpha_bar.device)
            noisy_actions, true_noise = self._apply_noise_to_blocks(padded_actions, window_start, noise_levels)
      
            # 位置0保持无噪声 (覆盖加噪结果)
            # 注意：这里我们需要单独处理第一个块
            start_idx = window_start * self.n_action_steps
            noisy_actions[:, :self.n_action_steps] = padded_actions[:, start_idx:start_idx+self.n_action_steps, :]
            
            # 6. 准备全局条件 (使用第一个块的观测)
            window_obs = {k: v[:, window_start:window_start+self.n_obs_steps] for k, v in nobs.items()}
            # 打印窗口观测形状以调试
            print(f"Window obs shape: {next(iter(window_obs.values())).shape}")     #  Window obs shape: torch.Size([1, 2, 3, 84, 84])
            global_cond = self._prepare_global_condition(window_obs, B)

            # print(f"Before Expanded noisy shape: {noisy_actions.shape}")  # Before Expanded noisy shape: torch.Size([1, 128, 10])

            # 7. 扩展批次维度以进行并行处理
            # [1, H*S, Da] -> [parallel, H*S, Da]
            expanded_noisy = noisy_actions.repeat(self.parallel_value, 1, 1)
            
            # # 打印扩展后的数据形状
            # print(f"Expanded noisy shape: {expanded_noisy.shape}")    # Expanded noisy shape: torch.Size([16, 128, 10])
            # print(f"Expected shape: [{self.parallel_value}, {self.horizon*self.n_action_steps}, {Da}]")   # Expected shape: [16, 128, 10]
        
            # 8. UNet前向传播
            # 使用虚拟时间步 (这里使用中等噪声强度)
            timesteps = torch.randint(
                self.num_train_timesteps // 2, self.num_train_timesteps,
                (self.parallel_value,), device=expanded_noisy.device
            ).long()
            
            # 由于数据已经是3维的，不需要额外重塑
            # 直接使用 expanded_noisy 作为模型输入
            print(f"Model input shape: {expanded_noisy.shape}") # Model input shape: torch.Size([16, 128, 10])
            print(f"Expected shape: [{self.parallel_value}, {self.horizon*self.n_action_steps}, {Da}]") # Expected shape: [16, 128, 10]
            
            # 调用模型预测噪声
            pred_noise = self.model(expanded_noisy, timesteps, global_cond=global_cond)
            
            # 9. 计算加权损失
            # 只计算目标块 (第1到horizon-1块) 的损失
            # 位置0的损失权重为0 (忽略)
            weights = self.window_weights.to(pred_noise.device)
            weights = weights.repeat_interleave(self.n_action_steps).view(1, -1, 1)  # [1, H*S, 1]
            weights[:self.n_action_steps] = 0  # 位置0权重为0
            
            # 计算每个位置的MSE损失
            loss_per_element = F.mse_loss(pred_noise, 
                                        true_noise.repeat(self.parallel_value, 1, 1), 
                                        reduction='none')
            
            # 应用位置权重
            weighted_loss = loss_per_element * weights  # 现在repeat_interleave后都是3维，可以正确广播
            
            # 对所有并行样本和位置求平均
            loss = weighted_loss.mean() # 确保 loss 是一个张量，而不是标量值
            total_loss = total_loss + loss
            window_count += 1
        
        # 计算平均损失
        if window_count > 0:
            total_loss = total_loss / window_count  # 确保在计算平均损失时，使用的是张量操作    
        
        return total_loss

    
    def _initialize_inference_buffer(self, obs_dict: Dict[str, torch.Tensor], 
                                    action_history: torch.Tensor = None):
        """
        初始化推理缓冲区
        
        参数:
            obs_dict: 当前观测字典
            action_history: 历史动作序列 [B, T, D]
        """
        B = next(iter(obs_dict.values())).shape[0]
        device = self.device
        dtype = self.dtype
        
        # 编码当前观测作为全局条件
        nobs = self.normalizer.normalize(obs_dict)
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, -1)  # [B, To * Do]
        self._inference_global_cond = nobs_features
        
        # 初始化缓冲区
        if action_history is not None:
            # 使用历史动作初始化缓冲区
            nactions = self.normalizer['action'].normalize(action_history)
            # 使用与训练相同的方法准备动作块
            padded_actions, num_blocks = self._prepare_action_blocks(nactions)
            
            # 取最后horizon个块
            if num_blocks >= self.horizon:
                # 提取最后horizon个块的动作
                start_idx = (num_blocks - self.horizon) * self.n_action_steps
                buffer_actions = padded_actions[:, start_idx:start_idx+self.horizon*self.n_action_steps, :]
            else:
                # 填充到horizon长度
                pad_length = self.horizon * self.n_action_steps - padded_actions.shape[1]
                last_action = padded_actions[:, -1:, :]
                padding = last_action.repeat(1, pad_length, 1)
                buffer_actions = torch.cat([padded_actions, padding], dim=1)
        else:
            # 没有历史动作，使用零初始化
            buffer_actions = torch.zeros(B, self.horizon * self.n_action_steps, self.action_dim,
                                    device=device, dtype=dtype)
        
        # 应用不同程度的噪声
        # 位置0: 无噪声 (t=0)
        # 位置1: 强度1 (t=1)
        # ...
        # 位置H-1: 强度H-1 (t=H-1)
        noise_levels = torch.arange(self.horizon, device=device)
        
        # 使用与训练相同的方法应用噪声
        # 注意：这里window_start=0，因为我们处理的是整个缓冲区
        noisy_buffer, _ = self._apply_noise_to_blocks(buffer_actions, 0, noise_levels)
        
        # 位置0保持无噪声
        start_idx = 0 * self.n_action_steps
        noisy_buffer[:, :self.n_action_steps] = buffer_actions[:, start_idx:start_idx+self.n_action_steps, :]
        
        # 将缓冲区重塑为块形式以便后续处理
        self._inference_buffer = noisy_buffer.reshape(B, self.horizon, self.n_action_steps, self.action_dim)
        self._inference_t_buffer = noise_levels.repeat(B, 1)  # [B, horizon]
    
    def _denoise_step(self, pred_noise: torch.Tensor) -> torch.Tensor:
        """
        执行一步去噪
        
        参数:
            pred_noise: 预测的噪声 [B, H*S, D]
            
        返回:
            denoised_buffer: 去噪后的缓冲区 [B, H, S, D]
        """
        B, H, S, D = self._inference_buffer.shape
        device = self._inference_buffer.device
        
        # 将缓冲区转换为3D格式
        buffer_3d = self._inference_buffer.reshape(B, H*S, D)
        
        # 将预测的噪声重塑为4D格式 [B, H, S, D]
        pred_noise_4d = pred_noise.reshape(B, H, S, D)
        
        # 对每个位置应用去噪公式
        denoised_buffer_3d = torch.zeros_like(buffer_3d)
        
        for i in range(H):
            t = self._inference_t_buffer[0, i].item()  # 当前噪声强度
            start_idx = i * S
            end_idx = (i+1) * S
            
            if t == 0:
                # 位置i: 无噪声，保持不变
                denoised_buffer_3d[:, start_idx:end_idx] = buffer_3d[:, start_idx:end_idx]
            else:
                # 应用去噪公式
                alpha_bar_t = self.alpha_bar[t].to(device)
                
                # 确保形状匹配
                current_block = buffer_3d[:, start_idx:end_idx]  # [B, S, D]
                current_noise = pred_noise_4d[:, i]  # [B, S, D]
                
                denoised_block = (
                    current_block - 
                    torch.sqrt(1 - alpha_bar_t) * current_noise
                ) / torch.sqrt(alpha_bar_t)
                
                denoised_buffer_3d[:, start_idx:end_idx] = denoised_block
                
                # 更新噪声强度
                self._inference_t_buffer[:, i] = t - 1
        
        # 将去噪后的缓冲区转换回4D格式
        return denoised_buffer_3d.reshape(B, H, S, D)
    
    def _slide_window(self):
        """
        滑动窗口并添加新块
        """
        B, H, S, D = self._inference_buffer.shape
        
        # 将缓冲区转换为3D格式
        buffer_3d = self._inference_buffer.reshape(B, H*S, D)
        
        # 滑动窗口: 移除第一个块，其余块前移
        new_buffer_3d = torch.zeros_like(buffer_3d)
        
        # 移除第一个块 (S个时间步)
        new_buffer_3d[:, :-S] = buffer_3d[:, S:]
        
        # 添加新块: 使用最后一个块的值加强度H-1的噪声
        last_block = new_buffer_3d[:, -S:]  # 最后一个块
        noise_level = H - 1  # 最高强度
        
        # 生成随机噪声
        noise = torch.randn_like(last_block)
        
        # 应用噪声
        alpha_bar_t = self.alpha_bar[noise_level].to(self._inference_buffer.device)
        new_block = torch.sqrt(alpha_bar_t) * last_block + torch.sqrt(1 - alpha_bar_t) * noise
        
        new_buffer_3d[:, -S:] = new_block
        
        # 更新缓冲区
        self._inference_buffer = new_buffer_3d.reshape(B, H, S, D)
        
        # 更新噪声强度缓冲区
        new_t_buffer = torch.zeros_like(self._inference_t_buffer)
        new_t_buffer[:, :-1] = self._inference_t_buffer[:, 1:]
        new_t_buffer[:, -1] = noise_level
        self._inference_t_buffer = new_t_buffer
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        基于滑动窗口的迭代去噪推理
        
        参数:
            obs_dict: 包含当前和过去观测的字典
            
        返回:
            包含预测动作的字典
        """
        # 如果是第一次调用，初始化缓冲区
        if self._inference_buffer is None:
            self._initialize_inference_buffer(obs_dict)
        
        B = next(iter(obs_dict.values())).shape[0]
        device = self.device
        dtype = self.dtype
        
        # 1. 更新全局条件
        nobs = self.normalizer.normalize(obs_dict)
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, -1)  # [B, To * Do]
        self._inference_global_cond = nobs_features
        
        # 2. 迭代去噪
        for i in range(self.num_inference_steps):
            # 将缓冲区转换为3D格式
            buffer_3d = self._inference_buffer.reshape(B, -1, self.action_dim)
            
            # 使用当前时间步
            timesteps = torch.full((B,), 
                                self.num_train_timesteps - i - 1, 
                                device=device, dtype=torch.long)
            
            # 预测噪声
            pred_noise = self.model(buffer_3d, timesteps, global_cond=self._inference_global_cond)
            
            # 执行去噪
            self._inference_buffer = self._denoise_step(pred_noise)
        
        # 3. 输出位置1的动作 (去噪后变为干净值)
        output_actions = self._inference_buffer[:, 1]  # [B, n_action_steps, D]
        
        # 4. 滑动窗口
        self._slide_window()
        
        # 5. 去标准化并返回
        actions = self.normalizer['action'].unnormalize(output_actions)
        
        return {
            'action': actions,
            'action_pred': self.normalizer['action'].unnormalize(self._inference_buffer)
        }
    
    def reset(self):
        """重置推理状态"""
        self._inference_buffer = None
        self._inference_t_buffer = None
        self._inference_global_cond = None
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    # https://yuanbao.tencent.com/chat/naQivTmsDa/065256f4-7061-4c6b-bf5e-1608b743af30