"""
架构设计
    观测编码器：使用robomimic的CNN（可能含CropRandomizer）提取图像特征，支持替换BatchNorm为GroupNorm。
    扩散模型：ConditionalUnet1D处理时间序列动作，支持通过全局条件（观测特征）或局部条件（观测与动作拼接）注入信息。
    条件机制：通过obs_as_global_cond控制是否将观测特征作为全局条件，影响模型输入维度。

关键组件
    调度器：DDPMScheduler管理扩散过程的时间步。
    掩码生成器：LowdimMaskGenerator处理条件掩码，用于训练时数据修补。
    数据增强：RotRandomizer支持旋转增强，CropRandomizer处理图像裁剪。

训练流程
    损失计算：预测噪声与真实噪声的MSE损失，通过掩码忽略条件部分。
    归一化：LinearNormalizer统一处理观测和动作的归一化。

推理流程
    采样过程：通过conditional_sample逐步去噪生成动作序列，最终提取有效动作步。
"""


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


from diffusion_policy.model.equi.equi_obs_encoder import EquivariantObsEnc
from diffusion_policy.model.equi.equi_conditional_unet1d import EquiDiffusionUNet


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
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
            diffusion_step_embed_dim=256,# 设置扩散步骤的嵌入维度，并传递给模型的初始化
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            rot_aug=False,
            
            model_type='dp',  # 'dp' or 'eqdp'

            # dp encoder
            obs_as_global_cond=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,

            # eqdp encoder
            N=8,
            enc_n_hidden=64,

            # parameters passed to step
            **kwargs):
        """
        功能：初始化 DiffusionUnetHybridImagePolicy 类，设置模型的所有超参数，加载配置，初始化各类组件。
        参数：
            shape_meta: 包含动作和观察数据的形状信息。
            noise_scheduler: 用于控制扩散过程的调度器(DDPM调度器)。
            horizon: 轨迹的长度。
            n_action_steps: 每次动作的时间步长。
            n_obs_steps: 观察数据的时间步长。
            其他相关的配置参数，如 crop_shape、diffusion_step_embed_dim 等。
        关键步骤：
            解析形状元数据 (shape_meta)。
            配置观察数据的类型和模态（RGB、低维、深度等)。
            加载 Robomimic 配置，设定观察数据的随机化方式（如裁剪）。
            初始化 obs_encoder（用于处理观察数据）和 model（扩散模型）。
            初始化 noise_scheduler 和 mask_generator 等其他相关组件。
        """
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']


        if model_type=='dp':
            # dp encoder
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

            # 初始化与观察数据处理相关的全局工具类，确保在整个系统中共享一致的配置
            ObsUtils.initialize_obs_utils_with_config(config)

            # 加载模型，PolicyAlgo 是策略算法的类，包含了训练模型的各类组件。该工厂函数使用了前面加载的配置，生成了一个适合当前任务和数据集的策略模型
            policy: PolicyAlgo = algo_factory(
                    algo_name=config.algo_name,
                    config=config,
                    obs_key_shapes=obs_key_shapes,
                    ac_dim=action_dim,
                    device='cpu',
                )

            # 观察数据编码器
            obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
            # 替换批归一化（BatchNorm）为组归一化（GroupNorm）
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
            # 替换裁剪随机化模块（CropRandomizer）
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
            # 将动作和观察数据联合处理，因此输入维度为动作维度加上观察特征维度
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
            self.obs_as_global_cond = obs_as_global_cond


        elif model_type=='eqdp':

            obs_encoder = EquivariantObsEnc(
                obs_shape=obs_shape_meta['agentview_image']['shape'], 
                crop_shape=crop_shape, 
                n_hidden=enc_n_hidden, 
                N=N)
            obs_feature_dim = enc_n_hidden
            global_cond_dim = obs_feature_dim * n_obs_steps
            model = EquiDiffusionUNet(
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
            self.crop_shape = crop_shape    # 好像无效
            self.obs_as_global_cond = True


        self.obs_encoder = obs_encoder
        self.model =model
        print("model_type = ", model_type,"(dp or eqdp)")
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))


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
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.rot_aug = rot_aug
        
        self.kwargs = kwargs

        self.noise_scheduler = noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
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
        接收一个批量数据batch作为输入。这个批量数据通常包含了观察数据、动作数据等
        """
        # normalize input                                               确保输入数据在一定的范围内，避免某些特征值过大或过小影响训练过程
        assert 'valid_mask' not in batch                                # 确保输入的batch数据中没有valid_mask字段。如果有这个字段，程序会抛出一个错误，保证后续代码不会出现不预期的行为
        nobs = self.normalizer.normalize(batch['obs'])                  # batch['obs']是观察数据，使用self.normalizer.normalize进行标准化
        nactions = self.normalizer['action'].normalize(batch['action']) # batch['action']是动作数据，使用self.normalizer['action'].normalize进行标准化
        if self.rot_aug:                                                # 如果启用了旋转增强（self.rot_aug为True），
            nobs, nactions = self.rot_randomizer(nobs, nactions)        # 通过self.rot_randomizer对观察数据和动作数据进行随机旋转增强。这种数据增强技术可以帮助模型学习到对旋转的鲁棒性，增强模型的泛化能力
        batch_size = nactions.shape[0]                                  # 批量中的样本数量，即动作数据的第一维大小
        horizon = nactions.shape[1]                                     # 方法2获取轨迹的长度，通常指每个样本中动作的时间步数或长度
        
        # handle different ways of passing observation
        local_cond = None                                               # 方法2中的本地条件,局部的上下文信息
        global_cond = None                                              # 方法2中的全局条件,全局的上下文信息
        trajectory = nactions                                           # trajectory是动作数据nactions，它将作为预测过程中的目标轨迹
        cond_data = trajectory                                          # cond_data是条件数据，最初与trajectory相同。后面可能会根据不同的条件进行修改

        # 将eqdp中的self.enc(nobs)作为特征提取模块
        # 只使用观察数据的前self.n_obs_steps个时间步（self.n_obs_steps是一个超参数，表示我们关心的观察历史长度）
        # 将整个观察数据（包括所有时间步）合并为一个大批次，并重塑形状。-1仍然是合并批次和时间步维度，*x.shape[2:]保留数据的其他维度
        if self.obs_as_global_cond:
            """
            如果obs_as_global_cond为True,将观察数据nobs作为全局条件global_cond处理
            即，通过self.enc对观察数据进行特征提取，并reshape为global_cond。
            # 如果观察数据nobs包含多步时序信息，这一行将前self.n_obs_steps步观察数据提取出来，并将其重塑为一个二维张量（将时间步展开为一个批次）.dict_apply用于应用这个操作到nobs字典的每个项。
            # 使用obs_encoder观察数据this_nobs进行特征提取,eqdp的obs_encoder是一个等变网络，能够在处理旋转和平移等空间变换时具有不变性。
            # 将提取的观察数据特征nobs_features重塑为global_cond，这是全局条件数据，用于后续的噪声预测。
            """
            # reshape B, T, ... to B*T      ？？？
            this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)  
            # reshape back to B, Do         ？？？
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            """
            如果obs_as_global_cond为False，观察数据与动作数据nactions被拼接，形成局部和全局的条件数据cond_data，并用于后续的预测
            # 将所有的观察数据nobs重塑为二维张量，展平时间步，使得每个观察样本的时间步都被“拉平”以供后续处理。
            # 使用obs_encoder来提取特征。eqdp的特征提取方式与全局条件部分相似，只是这里的特征提取是为了提取局部条件。
            # 将提取的特征nobs_features重塑为(batch_size, horizon, -1)的形状，表示每个时间步的观察特征
            # 将动作数据nactions和观察数据特征nobs_features在最后一维上拼接，形成cond_data。这将成为后续噪声预测过程的条件数据这里将cond_data通过.detach()从计算图中分离，以确保它不会在反向传播过程中计算梯度。
            # trajectory作为目标，且不需要计算梯度
            """
            # reshape B, T, ... to B*T      ？？？
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do      ？？？
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()




        # generate impainting mask
        """
        生成遮罩（Mask）：
            condition_mask生成的目的是用于确定哪些部分的轨迹数据应该计算损失。
            ~condition_mask意味着对不满足条件的地方计算损失，具体的条件通过后续的噪声加注来决定。
        """
        condition_mask = self.mask_generator(trajectory.shape)

        """
        噪声处理：
            噪声noise会被加入到轨迹中，以模拟噪声处理过程。timesteps随机选择每个轨迹对应的时间步长。
            然后，noisy_trajectory将噪声加入到原始轨迹中。
        生成与轨迹trajectory形状相同的随机噪声。这些噪声将添加到轨迹数据中，模拟噪声的扩散过程。
        """ 
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]               # 获取批量大小bsz，即轨迹的第一维大小
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()                                # 随机选择每个轨迹的时间步长timesteps。这些时间步将在噪声扩散过程中使用，以决定在每个时间点加多少噪声
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        """
        使用self.noise_scheduler.add_noise将噪声添加到轨迹数据trajectory中，生成带噪声的轨迹noisy_trajectory。
        这个过程模拟了前向扩散过程
        """
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        # 创建loss_mask，它是condition_mask的取反，表示哪些位置不应该进行条件掩蔽。最终会应用到损失计算中
        loss_mask = ~condition_mask

        # apply conditioning
        # 将noisy_trajectory中符合condition_mask条件的位置替换为cond_data中的数据。这样做的目的是保持条件数据的准确性，同时添加噪声
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        # 使用模型预测noisy_trajectory的噪声残差。模型根据noisy_trajectory和对应的时间步timesteps，以及局部条件local_cond和全局条件global_cond进行预测
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        # 根据self.noise_scheduler.config.prediction_type选择预测的目标类型。
        # 如果是epsilon，目标是噪声；如果是sample，目标是轨迹数据
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise      # dp预测的是噪声，而不是直接的轨迹
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        """
        根据预测类型（epsilon或sample），目标target可以是噪声或轨迹。这里使用MSE损失来计算预测与目标之间的均方误差（MSE）。
        损失计算：最终的损失loss会应用loss_mask对损失进行加权，从而只计算需要计算损失的位置的数据点的损失，
        使用reduce对损失进行归约，将损失从多维数据减少到批次大小维度上的平均损失
        对损失取均值，最终得到整个批次的损失值。
        """        
        loss = F.mse_loss(pred, target, reduction='none')# reduction='none'表示不对损失进行聚合
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
