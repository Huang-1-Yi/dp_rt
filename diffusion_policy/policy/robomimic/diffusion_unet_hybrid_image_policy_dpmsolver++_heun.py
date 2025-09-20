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
            eps_scaler=0.99,  # 添加这行
            # parameters passed to step
            **kwargs):
        super().__init__()

        
        print("测试时的eps_scaler==", eps_scaler,"\n")
        kwargs['eps_scaler'] = eps_scaler   # 将 eps_scaler 添加到 kwargs 中
        self.schedule_type = kwargs['schedule_type']
        print(f"Using schedule type: {self.schedule_type}")

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
        self.noise_scheduler = noise_scheduler
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
            # dpm-solver parameters
            class_labels=None, 
            condition=None, 
            unconditional_condition=None,
            num_steps=None, 
            sigma_min=0.002, 
            sigma_max=80, 
            schedule_rho=7, 
            afs=False, 
            denoise_to_zero=False, 
            return_inters=False, 
            inner_steps=2,  # New parameter for the number of inner steps
            r=0.5,
            eps_scaler=1.0,  # 添加 Epsilon Scaling 参数
            # keyword arguments to scheduler.step
            **kwargs
            ):
        """
        功能：基于给定的条件数据和条件掩码，通过扩散模型进行采样。
        参数：
            condition_data、condition_mask: 条件数据和掩码。
            local_cond、global_cond: 本地条件和全局条件（根据 obs_as_global_cond 来决定）。
            generator: 随机数生成器，用于采样。
        关键步骤：
            初始化扩散轨迹。
            设置调度器的时间步。
            在每个时间步上，应用条件数据和掩码进行采样。
            调用扩散模型的输出，并通过调度器一步步反向扩散生成目标数据。
        """
        model = self.model
        schedule_type = self.schedule_type
        x_next = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        get_schedule = self.get_schedule

        # 在 conditional_sample 里，调用 get_schedule 之前
        if num_steps is None:
            num_steps = self.num_inference_steps

        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=x_next.device, schedule_type=schedule_type, schedule_rho=schedule_rho)
        x_next = x_next * t_steps[0]
        x_next[condition_mask] = condition_data[condition_mask]
        inters = [x_next.unsqueeze(0)]

        x_list = []
        x_list.append(x_next)

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        
            x_cur = x_next
            # Compute the inner step size
            t_s = get_schedule(inner_steps, t_next, t_cur, device=x_cur.device, schedule_type=schedule_type, schedule_rho=7)
            for i, (t_c, t_n) in enumerate(zip(t_s[:-1],t_s[1:])):
                # Euler step.
                use_afs = (afs and i == 0)
                if use_afs:
                    d_cur = x_cur / ((1 + t_c**2).sqrt())
                else:
                    denoised = model(x_cur, t_c, local_cond=local_cond, global_cond=global_cond)
                    
                    # 添加 Epsilon Scaling
                    if eps_scaler != 1.0:
                        pred_eps = (x_cur - denoised) / t_c
                        pred_eps = pred_eps / eps_scaler
                        denoised = x_cur - pred_eps * t_c
                    
                    denoised[condition_mask] = condition_data[condition_mask]
                    d_cur = (x_cur - denoised) / t_c
                x_next = x_cur + (t_n - t_c) * d_cur

                # Apply 2nd order correction.
                denoised = model(x_next, t_n, local_cond=local_cond, global_cond=global_cond)

                # 添加 Epsilon Scaling
                if eps_scaler != 1.0:
                    pred_eps = (x_next - denoised) / t_n
                    pred_eps = pred_eps / eps_scaler
                    denoised = x_next - pred_eps * t_n
                
                denoised[condition_mask] = condition_data[condition_mask]
                
                d_prime = (x_next - denoised) / t_n
                x_cur = x_cur + (t_n - t_c) * (0.5 * d_cur + 0.5 * d_prime)
            
            x_next = x_cur
            x_list.append(x_next)

            if return_inters:
                inters.append(x_cur.unsqueeze(0))

        if denoise_to_zero:
            x_next = model(x_next, t_next, local_cond=local_cond, global_cond=global_cond)

            # 添加 Epsilon Scaling
            if eps_scaler != 1.0:
                pred_eps = (x_next - denoised) / t_next
                pred_eps = pred_eps / eps_scaler
                denoised = x_next - pred_eps * t_next

            x_next[condition_mask] = condition_data[condition_mask]   
            if return_inters:
                inters.append(x_next.unsqueeze(0))

        if return_inters:
            return torch.cat(inters, dim=0).to(x_next.device)
        # return x_next, x_list     

        return x_next


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        功能：根据当前观察数据预测动作。
        参数：
            obs_dict: 包含观察数据的字典，必须包括键 "obs"。
        关键步骤：
            1.对观察数据进行归一化处理。
            2.构建输入数据（观察和动作数据）。
            3.根据是否使用全局条件 (obs_as_global_cond)，分别处理本地条件和全局条件。
            4.调用 conditional_sample 方法进行条件采样，生成轨迹。
            5.生成预测的动作，并将其反归一化（恢复到原始尺度）。
            6.返回预测的动作。
        """
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
            # eps_scaler=self.kwargs.get('eps_scaler', 1.0),  # 添加这行
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
        """
        功能：设置归一化器（LinearNormalizer）。
        参数：
            normalizer: 传入的归一化器。
        关键步骤：
            加载归一化器的状态字典。
        """
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        功能：计算模型的损失函数。
        参数：
            batch: 输入的批次数据，包含观察数据和动作数据。
        关键步骤：
            1.对输入数据进行归一化处理。
            2.根据是否使用旋转增强 (rot_aug)，对数据进行旋转增强。
            3.处理观察数据，通过全局条件或其他方式构建条件数据。
            4.使用 mask_generator 生成损失掩码。
            5.计算噪声并添加到图像中，模拟扩散过程。
            6.根据预测类型（epsilon 或 sample）计算目标（噪声或轨迹）。
            7.使用均方误差（MSE）损失函数计算损失。
            8.计算和返回最终的平均损失。
        """
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
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
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

    @staticmethod
    # def get_schedule(num_steps, sigma_min, sigma_max, device=None, schedule_type='polynomial', schedule_rho=7, net=None):
    def get_schedule(num_steps, sigma_min, sigma_max, *, device=None, schedule_type='polynomial', schedule_rho=7, net=None):
        """
        获取采样的时间调度。

        参数：
            num_steps: 一个 `int` 类型。时间步数的总数，间隔为 `num_steps-1`。
            sigma_min: 一个 `float` 类型。采样结束时的 sigma 值。
            sigma_max: 一个 `float` 类型。采样开始时的 sigma 值。
            device: 一个 torch 设备。
            schedule_type: 一个 `str` 类型。时间调度的类型。支持以下几种类型：
                - 'polynomial': 多项式时间调度。（推荐用于 EDM）
                - 'logsnr': 均匀 logSNR 时间调度。（推荐用于小分辨率数据集的 DPM-Solver）
                - 'time_uniform': 均匀时间调度。（推荐用于高分辨率数据集的 DPM-Solver）
                - 'discrete': LDM 使用的时间调度。（推荐用于 LDM 和 Stable Diffusion 代码库中的预训练扩散模型）
            schedule_rho: 一个 `float` 类型。时间步指数。
            net: 一个预训练的扩散模型。当 schedule_type == 'discrete' 时需要。
        返回值：
            一个形状为 [num_steps] 的 PyTorch 张量。

        实现逻辑分析：
        根据不同的时间调度类型，计算时间步的值。多项式调度通过指数公式计算，logsnr 调度通过对数公式计算，time_uniform 调度通过 beta 参数计算，discrete 调度依赖于预训练模型的 sigma 函数。
        """
        import numpy as np
        if schedule_type == 'polynomial':
            step_indices = torch.arange(num_steps, device=device)
            t_steps = (sigma_max ** (1 / schedule_rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / schedule_rho) - sigma_max ** (1 / schedule_rho))) ** schedule_rho
        elif schedule_type == 'logsnr':
            logsnr_max = -1 * torch.log(torch.tensor(sigma_min))
            logsnr_min = -1 * torch.log(torch.tensor(sigma_max))
            t_steps = torch.linspace(logsnr_min.item(), logsnr_max.item(), steps=num_steps, device=device)
            t_steps = (-t_steps).exp()
        elif schedule_type == 'time_uniform':
            epsilon_s = 1e-3
            vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
            vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
            step_indices = torch.arange(num_steps, device=device)
            vp_beta_d = 2 * (np.log(torch.tensor(sigma_min).cpu() ** 2 + 1) / epsilon_s - np.log(torch.tensor(sigma_max).cpu() ** 2 + 1)) / (epsilon_s - 1)
            vp_beta_min = np.log(torch.tensor(sigma_max).cpu() ** 2 + 1) - 0.5 * vp_beta_d
            t_steps_temp = (1 + step_indices / (num_steps - 1) * (epsilon_s ** (1 / schedule_rho) - 1)) ** schedule_rho
            t_steps = vp_sigma(vp_beta_d.clone().detach().cpu(), vp_beta_min.clone().detach().cpu())(t_steps_temp.clone().detach().cpu())
        elif schedule_type == 'discrete':
            assert net is not None
            t_steps_min = net.sigma_inv(torch.tensor(sigma_min, device=device))
            t_steps_max = net.sigma_inv(torch.tensor(sigma_max, device=device))
            step_indices = torch.arange(num_steps, device=device)
            t_steps_temp = (t_steps_max + step_indices / (num_steps - 1) * (t_steps_min ** (1 / schedule_rho) - t_steps_max)) ** schedule_rho
            t_steps = net.sigma(t_steps_temp)
        else:
            raise ValueError("Got wrong schedule type {}".format(schedule_type))
        return t_steps.to(device)

