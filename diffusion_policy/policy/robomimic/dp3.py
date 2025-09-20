"""
1.总结
代码2通过点云编码器和交叉注意力机制，将扩散策略从图像域扩展到3D空间操作任务，核心改进在于几何感知与时序条件交互。若任务需处理深度信息或复杂空间关系（如机械臂抓取），
代码2是更优选择；若依赖RGB图像特征（如场景理解），代码1仍具优势。
2. 关键改进点详解
2.1 点云支持与编码器升级
    DP3Encoder：
        使用PointNet结构处理点云，支持坐标（+可选颜色）输入。
        替换代码1的CNN，适应3D空间感知需求。
            # 代码2的点云处理逻辑（移除RGB，保留坐标）
            if not self.use_pc_color:
                nobs['point_cloud'] = nobs['point_cloud'][..., :3]
    点云特性：
        更适合抓取、避障等依赖几何信息的任务。

2.2 条件注入机制增强
    Cross-Attention条件：
        通过condition_type参数支持交叉注意力，替代简单的特征拼接。
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
        优势：时序特征与动作生成动态交互，提升长序列建模能力。

2.3 扩散预测目标扩展
    v_prediction支持：
        新增对速度预测目标的适配，扩展模型灵活性。
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], ...
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t  # 速度预测目标
        优点：可能加速收敛，适合动态变化剧烈的任务。

2.4 计算效率优化
    去除冗余模块：
        移除代码1中的RotRandomizer和CropRandomizer，简化预处理流程。
        点云无需图像式增强，减少计算开销。
"""




from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
# from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.dp3_conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.pointnet_extractor import DP3Encoder


class DP3(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,

            condition_type="film",# 通过condition_type参数支持交叉注意力，替代简单的特征拼接
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta: 解析形状元数据
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        # 解析观察形状元数据
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # 初始化观察特征提取器（DP3Encoder）
        obs_encoder = DP3Encoder(   observation_space=obs_dict,
                                    img_crop_shape=crop_shape,
                                    out_channel=encoder_output_dim,
                                    pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                    use_pc_color=use_pc_color,
                                    pointnet_type=pointnet_type,
                                )

        # create diffusion model 创建扩散模型
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")


        # 创建 ConditionalUnet1D 模型
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )
        
        # 将观察特征提取器和模型赋值
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        # 创建副本和掩码生成器
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        # 归一化器和其他参数初始化
        self.normalizer         = LinearNormalizer()
        self.horizon            = horizon
        self.obs_feature_dim    = obs_feature_dim
        self.action_dim         = action_dim
        self.n_action_steps     = n_action_steps
        self.n_obs_steps        = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs             = kwargs

        # 如果没有提供推理步骤数，使用噪声调度器中的训练时间步数
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        # print_params(self)
        
    # ========= inference 推理部分 ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        # 随机大小相同的随机噪声
        # 实际就是：初始化轨迹为标准正态分布
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values 设置时间步
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning 应用条件化：取特征图部分，部分预测内容所在掩码：直接删除了无效的图像预测内容
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output 模型输出
            model_output = model(   sample=trajectory,
                                    timestep=t, 
                                    local_cond=local_cond,
                                    global_cond=global_cond)
            
            # 3. compute previous image:计算前一动作 a_t -> a_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
        # finally make sure conditioning is enforced
        # 确保掩码被执行：得到最新的动作+观测特征图
        trajectory[condition_mask] = condition_data[condition_mask]   

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # 清理不需要的观察数据
        if 'robot0_eye_in_hand_image' in obs_dict:
            del obs_dict['robot0_eye_in_hand_image']
        if 'agentview_image' in obs_dict:
            del obs_dict['agentview_image']
        # normalize input # 归一化输入数据
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate坐标
        # 不使用颜色时，只保留点云坐标
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        # 获取批次和时间步的形状
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
        # 处理条件化
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature 通过全局特征条件化
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence 作为序列处理
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action 为动作创建空数据
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # 使用补全进行条件化
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do 
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling 执行条件化采样
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)

        # unnormalize prediction 反归一化预测的动作
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action 获取动作
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction 返回结果
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # 清理不需要的观察数据
        if 'robot0_eye_in_hand_image' in batch['obs']:
            del batch['obs']['robot0_eye_in_hand_image']
        if 'agentview_image' in batch['obs']:
            del batch['obs']['agentview_image']
        # normalize input 归一化输入数据
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action']) # 计算损失专用的，获取真实数据
        # 不使用颜色时，只保留点云坐标
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        # 处理不同的观察传递方式（预测步骤是得到b to da do）
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
       
        
        if self.obs_as_global_cond:
            # 通过全局特征条件化
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence 作为序列处理
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # 通过补全进行条件化，而且不用cond_data[:,:To,Da:]这样掩码
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()



        # 前面部分得到的trajectory是真实且完全的，相比于直接调用预测函数，进行后面开始前向扩散，开始预测残差

        # generate impainting mask 生成补全掩码
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images 随机生成噪声，并采样时间步
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]   # trajectory 张量的第一个维度的大小(batch_size, time_steps, feature_dim)，通常对应于批次大小
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # 添加噪声到图像中（前向扩散过程）
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask 计算损失掩码
        loss_mask = ~condition_mask

        # apply conditioning 应用条件
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual 预测噪声残差
        pred = self.model(  sample=noisy_trajectory, 
                            timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)

        # 计算目标噪声，即训练噪声预测网络，而不是如预测部分计算前一动作 a_t -> a_t-1
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # 根据“加噪与预测结果”计算损失
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        

        loss_dict = {
                'bc_loss': loss.item(),
            }

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")
        
        return loss, loss_dict