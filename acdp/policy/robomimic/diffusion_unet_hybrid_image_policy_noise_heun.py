from typing import Dict
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


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,# 设置扩散步骤的嵌入维度，并传递给模型的初始化
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            rot_aug=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
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

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
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

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        # 代码11 - 简单的条件输入处理 输入包括动作和观察的联合特征
        # 代码11 总是将动作和观察数据联合处理，因此输入维度为动作维度加上观察特征维度
        input_dim = action_dim + obs_feature_dim 
        global_cond_dim = None
        if obs_as_global_cond:
            # obs_as_global_cond如果为 True，仅使用观察数据的特征作为条件输入；
            # obs_as_global_cond如果为 False，将观察特征和动作特征结合作为输入
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

        
        # 创建 diffusion model 后，初始化 noise_scheduler
        self.noise_scheduler = noise_scheduler

        # 预计算 alpha_bar
        self.num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        beta_schedule = self.noise_scheduler.config.beta_schedule
        if beta_schedule == "squaredcos_cap_v2":
            s = 0.008
            t = torch.arange(0, self.num_train_timesteps + 1, dtype=torch.float32)
            alpha_bar = torch.cos((t / self.num_train_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
        else:
            # 线性调度作为备选
            beta_start = self.noise_scheduler.config.beta_start
            beta_end = self.noise_scheduler.config.beta_end
            betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=torch.float32)
            alphas = 1.0 - betas
            alpha_bar = torch.cumprod(alphas, dim=0)
            alpha_bar = torch.cat([torch.tensor([1.0]), alpha_bar])
        self.register_buffer('alpha_bar', alpha_bar)


        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = RotRandomizer()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.rot_aug = rot_aug
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
        ):
        """
        Euler采样器

        参数:
            net: 封装的扩散模型
            latents: PyTorch张量，在时间`sigma_max`的输入样本
            class_labels: PyTorch张量，条件采样的条件或引导采样
            condition: PyTorch张量，用于LDM和Stable Diffusion模型的调节条件
            unconditional_condition: PyTorch张量，用于LDM和Stable Diffusion模型的无条件调节
            num_steps: `int`，带`num_steps-1`间隔的总时间步数
            sigma_min: `float`，采样结束时的sigma值
            sigma_max: `float`，采样开始时的sigma值
            schedule_type: `str`，时间调度类型。支持三种类型:
                - 'polynomial': 多项式时间调度（EDM推荐）
                - 'logsnr': 均匀logSNR时间调度（小分辨率数据集DPM-Solver推荐）
                - 'time_uniform': 均匀时间调度（高分辨率数据集DPM-Solver推荐）
                - 'discrete': LDM使用的时间调度（使用LDM和Stable Diffusion代码库的预训练扩散模型时推荐）
            schedule_rho: `float`，时间步指数。当schedule_type为['polynomial', 'time_uniform']时需要指定
            afs: `bool`，是否在采样开始时使用解析第一步（AFS）
            denoise_to_zero: `bool`，是否在采样结束时从`sigma_min`去噪到`0`
            return_inters: `bool`，是否保存中间结果（整个采样轨迹），从初始噪声状态到最终生成结果的所有中间采样状态
            return_denoised: `bool`，是否保存中间去噪结果
            return_eps: `bool`，是否保存中间分数函数（梯度）
            step_idx: `int`，指定训练中采样步骤的索引
            train: `bool`，是否在训练循环中？
        返回:
            PyTorch张量。如果return_inters=True则返回生成样本或采样轨迹的批次
        """
        model = self.model
        num_steps = self.num_inference_steps
        device=condition_data.device

        # # ===== 替换原有固定值 =====
        # # 获取调度器的累积alpha乘积
        # alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
        
        # # 计算噪声水平σ (sigma = sqrt((1 - α_bar)/α_bar))
        # sigmas = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        
        # # 设置sigma范围
        # sigma_min = sigmas[-1].item()  # 最大噪声对应训练结束时的σ
        # sigma_max = sigmas[0].item()   # 最小噪声对应训练开始时的σ
        
        # # 避免数值问题
        # sigma_min = max(sigma_min, 1e-3)
        # sigma_max = max(sigma_max, sigma_min + 1e-3)

        # # 0.自定义欧拉采样器,设置核心参数，时间表类型schedule_type='polynomial'
        sigma_min=0.0001          # 最小噪声水平
        sigma_max=2.6             # 最大噪声水平
        schedule_rho=3              # 调度指数=3 适配 squaredcos_cap_v2调度器的特性
        # sigma_min=0.002
        # sigma_max=80
        # schedule_rho=7
        afs=False                    # 启用解析第一步
        prediction_type = self.noise_scheduler.config.prediction_type

        # 1. 替代Diffusers的set_timesteps实现时间步生成（基于连续sigma值）
        # if schedule_type == 'polynomial':
        step_indices = torch.arange(num_steps, device=device)
        t_steps = (sigma_max ** (1 / schedule_rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / schedule_rho) - sigma_max ** (1 / schedule_rho))) ** schedule_rho

        # 2. 初始化噪声轨迹
        a = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )

        # 3. 应用条件约束
        a_next = a * t_steps[0]
        a_next[condition_mask] = condition_data[condition_mask]

        # 3. 主采样循环
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):   # 0, ..., N-1
            a_cur = a_next
            if afs and i == 0:      # afs技巧：首次步使用分析解（可选），提高采样质量
                d_cur = a_cur / ((1 + t_cur**2).sqrt())
            else:                   # 其它时候直接调用模型获取去噪结果
                # 调用模型预测去噪结果
                model_output = model(a_cur, t_cur, local_cond=local_cond, global_cond=global_cond)
                # 根据prediction_type转换模型输出为去噪数据
                if prediction_type == 'epsilon':  # 模型预测噪声
                    a_denoised = a_cur - t_cur * model_output
                elif prediction_type == 'sample':  # 模型预测去噪后的数据
                    a_denoised = model_output
                elif prediction_type == 'v_prediction':  # 模型预测速度
                    # 速度与噪声的关系: v = -σ·ε
                    a_denoised = a_cur + t_cur * model_output
                else:
                    raise ValueError(f"Unsupported prediction type {prediction_type}")
                # 确保条件部分不变
                a_denoised[condition_mask] = condition_data[condition_mask]      
                # 计算当前梯度
                d_cur = (a_cur - a_denoised) / t_cur      
            # 欧拉法更新轨迹
            a_next = a_cur + (t_next - t_cur) * d_cur
        
        return a_denoised

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)



        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # 自定义加噪过程替换 noise_scheduler.add_noise
        alpha_bar_t = self.alpha_bar[timesteps]  # 形状 [bsz]
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1)  # 扩展为 [bsz, 1, 1]
        noisy_trajectory = torch.sqrt(alpha_bar_t) * trajectory + torch.sqrt(1 - alpha_bar_t) * noise
        

        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
