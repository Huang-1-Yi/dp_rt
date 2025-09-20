"""
代码对比分析
1. 类名和模型结构
    代码1定义了 DiffusionEquiUNetCNNEncPolicy 类,主要使用 EquivariantObsEnc 进行观察编码,并通过 EquiDiffusionUNet 执行扩散过程。
    代码2定义了 DiffusionUnetHybridImagePolicy 类,模型部分使用 ConditionalUnet1D 进行扩散过程,同时在图像编码部分使用了来自 robomimic 的模块,特别是 PolicyAlgo 和 obs_encoder。
2. 扩散模型
    代码1使用了 EquiDiffusionUNet,这是一个处理具有等变性的扩散模型,适用于复杂的视觉任务,重点是通过全局和局部条件进行生成。
    代码2使用了 ConditionalUnet1D,它与 EquiDiffusionUNet 类似,但也专注于通过全局和局部条件进行生成,但在实现上有所不同,尤其是与 obs_encoder 结合的部分。
3. 观察数据处理
    代码1通过 EquivariantObsEnc 处理输入的图像数据,并结合 LowdimMaskGenerator 为扩散过程生成掩码。
    代码2通过 obs_encoder(一个基于 robomimic 的编码器)处理图像数据,且在其中添加了对旋转增强(RotRandomizer)和裁剪随机化(CropRandomizer)的支持。对于 obs_encoder,还支持通过 GroupNorm 替换 BatchNorm。
4. 动作预测
    代码1的 predict_action 方法先通过 normalizer 对观测数据进行标准化,然后使用 conditional_sample 生成动作预测,最终根据 n_action_steps 获取最终的动作。
    代码2的 predict_action 方法的流程与代码1类似,也是通过标准化的输入生成条件采样,并对预测的动作进行去标准化。但它在处理全局和局部条件时有所不同,尤其是在如何使用 obs_as_global_cond 进行条件控制。
5. 损失计算
    代码1在 compute_loss 方法中,使用 MSE损失来计算噪声预测与目标的差异,并使用 LowdimMaskGenerator 来生成损失掩码。
    代码2在 compute_loss 方法中,基本逻辑与代码1相同,但它多了 dict_apply 和条件控制的细节,确保在不同的条件下合理传递观测数据。
6. 超参数和配置
    代码1和代码2都通过初始化时的超参数(如 horizon,n_action_steps,n_obs_steps 等)控制模型的行为,但代码2涉及到更多的外部配置,尤其是 robomimic 中的 config,并且支持动态调整图像裁剪大小等设置。
总结
    相似性:两个代码都基于扩散模型进行图像生成,并且通过条件采样进行动作预测。它们都处理了观察数据的标准化、生成的掩码以及损失函数计算。
    差异性:代码1使用了一个专门设计的 EquivariantObsEnc 和 EquiDiffusionUNet 进行处理,而代码2则集成了 robomimic 框架,使用了更为复杂的图像处理模块(如旋转增强、裁剪随机化等),并且在配置和模型细节上有更多的灵活性。
"""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, 'CropRandomizer'):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
from diffusion_policy.model.equi.equi_obs_encoder import EquivariantObsEnc
from diffusion_policy.model.equi.equi_conditional_unet1d import EquiDiffusionUNet
from diffusion_policy.model.vision.rot_randomizer import RotRandomizer


class DiffusionEquiUNetCNNEncPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # image
            crop_shape=(76, 76),
            # arch
            N=8,
            enc_n_hidden=64,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            rot_aug=False,
            # parameters passed to step
            **kwargs):
        """
        
        """
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        
        self.enc = EquivariantObsEnc(
            obs_shape=obs_shape_meta['agentview_image']['shape'], 
            crop_shape=crop_shape, 
            n_hidden=enc_n_hidden, 
            N=N)
        
        obs_feature_dim = enc_n_hidden
        global_cond_dim = obs_feature_dim * n_obs_steps
        
        self.diff = EquiDiffusionUNet(
            act_emb_dim=64,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            N=N,
        )
        
        print("Enc params: %e" % sum(p.numel() for p in self.enc.parameters()))
        print("Diff params: %e" % sum(p.numel() for p in self.diff.parameters()))

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = RotRandomizer()

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.crop_shape = crop_shape
        self.obs_feature_dim = obs_feature_dim
        self.rot_aug = rot_aug

        self.kwargs = kwargs

        self.noise_scheduler = noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            weight_decay: float, 
            learning_rate: float, 
            betas: Tuple[float, float],
            eps: float
        ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(), weight_decay=weight_decay, lr=learning_rate, betas=betas, eps=eps
        )
        return optimizer
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.diff
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        nobs_features = self.enc(nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=None,
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
        """
        观察数据（nobs）被直接通过一个编码器（self.enc）进行特征提取，然后将结果用于生成global_cond，即全局条件。这里假设观察数据只是提供全局条件，用于影响预测过程中的全局信息。
        特征提取后的global_cond是一个批次大小的二维张量，即(batch_size, -1)，用作预测时的全局条件。
        """
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
        batch_size = nactions.shape[0]


        trajectory = nactions
        cond_data = trajectory
        nobs_features = self.enc(nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)



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
        pred = self.diff(noisy_trajectory, timesteps, 
            local_cond=None, global_cond=global_cond)

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
    
