if __name__ == "__main__":  # 主程序入口
    import sys              # 导入sys模块
    import os               # 导入os模块
    import pathlib          # 导入pathlib模块

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent) # 获取项目根目录
    sys.path.append(ROOT_DIR)                                   # 将项目根目录添加到sys.path
    os.chdir(ROOT_DIR)                                          # 切换到项目根目录

import os                                                   # 导入os模块
import hydra                                                # 导入hydra模块
import torch                                                # 导入torch模块
from omegaconf import OmegaConf                             # 导入OmegaConf类
import pathlib                                              # 导入pathlib模块
from torch.utils.data import DataLoader                     # 导入DataLoader类
import copy                                                 # 导入copy模块
import random                                               # 导入random模块
import wandb                                                # 导入wandb模块
import tqdm                                                 # 导入tqdm模块
import numpy as np                                          # 导入numpy模块
import shutil                                               # 导入shutil模块
from diffusion_policy.workspace.base_workspace                              import BaseWorkspace
from diffusion_policy.policy.real_franka_diffusion_unet_image_policy        import DiffusionUnetImagePolicy
# from diffusion_policy.policy.real_franka_diffusion_unet_image_policy_0723   import DiffusionUnetImagePolicyTimm
from diffusion_policy.policy.diffusion_unet_timm_policy                     import DiffusionUnetTimmPolicy      # 导入DiffusionUnetTimmPolicy类
from diffusion_policy.dataset.base_dataset                                  import BaseImageDataset             # 导入BaseImageDataset类
from diffusion_policy.env_runner.base_image_runner                          import BaseImageRunner              # 导入BaseImageRunner类
from diffusion_policy.common.checkpoint_util                                import TopKCheckpointManager        # 导入TopKCheckpointManager类
from diffusion_policy.common.json_logger                                    import JsonLogger                   # 导入JsonLogger类
from diffusion_policy.common.pytorch_util                                   import dict_apply, optimizer_to     # 导入dict_apply和optimizer_to函数
from diffusion_policy.model.diffusion.ema_model                             import EMAModel                     # 导入EMAModel类
from diffusion_policy.model.common.lr_scheduler                             import get_scheduler                # 导入get_scheduler函数

# 加速用
from accelerate import Accelerator
import pickle

# 注册OmegaConf自定义解析器，用于解析配置中的eval表达式
OmegaConf.register_new_resolver("eval", eval, replace=True)

import json

# 在类定义前添加
def log_action_mse(step_log, category, pred_action, gt_action):
    """记录动作MSE分解误差"""
    B, T, _ = pred_action.shape
    action_dim = pred_action.shape[-1]
    
    # 整体MSE
    step_log[f'{category}_action_mse'] = torch.nn.functional.mse_loss(pred_action, gt_action).item()
    
    # 位置误差 (前3维)
    if action_dim >= 3:
        step_log[f'{category}_action_mse_pos'] = torch.nn.functional.mse_loss(
            pred_action[..., :3], gt_action[..., :3]).item()
    
    # 旋转误差 (第3-6维)
    if action_dim >= 6:
        step_log[f'{category}_action_mse_rot'] = torch.nn.functional.mse_loss(
            pred_action[..., 3:6], gt_action[..., 3:6]).item()
    
    # 夹爪误差 (第6维)
    if action_dim >= 7:
        step_log[f'{category}_action_mse_gripper'] = torch.nn.functional.mse_loss(
            pred_action[..., 6], gt_action[..., 6]).item()
    
    # 记录动作维度信息
    step_log[f'{category}_action_dim'] = action_dim


class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    # 定义需要包含在checkpoint中的键
    include_keys = ['global_step', 'epoch']

    # 加速用
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        """
        1. 初始化和配置管理
            ​Hydra配置框架：使用Hydra管理配置，从指定路径加载YAML配置，支持动态解析（如eval解析器）。
            ​工作空间设置：初始化时设置随机种子确保可复现性，实例化扩散模型（及EMA模型）、优化器，并初始化训练状态变量。
        """
        super().__init__(cfg, output_dir=output_dir)

         # 设置随机种子（保证可重复性）
        seed = cfg.training.seed    # 获取配置中的随机种子
        torch.manual_seed(seed)     # 设置torch的随机种子
        np.random.seed(seed)        # 设置numpy的随机种子
        random.seed(seed)           # 设置random的随机种子

        print(f"\n=== 初始化配置 ===")  # 初始化调试打印
        print(f"训练种子: {seed}")
        print(f"使用EMA: {cfg.training.use_ema}")
        print(f"冻结编码器: {cfg.training.freeze_encoder}")
        print(f"恢复训练: {cfg.training.resume}")
        print(f"优化器配置: {OmegaConf.to_container(cfg.optimizer)}")

        # DiffusionUnetImagePolicy DiffusionUnetTimmPolicy
        # 实例化策略模型（通过Hydra配置TimmObsEncoder）
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        # 关键修改：确保模型在CPU上初始化
        self.model.cpu()

        print(f"\n=== 模型架构 ===")
        print(f"模型类型: {type(self.model).__name__}")
        print(f"观测编码器: {type(self.model.obs_encoder).__name__}")
        print(f"主干网络: {type(self.model.model).__name__}")
        
        # DiffusionUnetImagePolicy DiffusionUnetTimmPolicy
        # 初始化EMA模型（如果启用）
        self.ema_model: DiffusionUnetImagePolicy = None          # 初始化EMA模型为None

        if cfg.training.use_ema:                                # 如果使用EMA
            self.ema_model = copy.deepcopy(self.model)          # 深拷贝模型
            self.ema_model.cpu()  # 同样确保在CPU上

        # # 实例化优化器
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())
        # ========= 优化1：分层学习率 =========
        def build_optimizer(cfg, params):
            """兼容Hydra配置的优化器构建"""
            import copy
            
            # 处理不同配置场景
            if hasattr(cfg.optimizer, '_target_'):
                # Hydra风格配置
                optimizer_cfg = copy.deepcopy(OmegaConf.to_container(cfg.optimizer, resolve=True))
                target_cls = optimizer_cfg.pop('_target_', None)
                
                # 动态加载优化器类
                if target_cls:
                    from hydra.utils import get_class
                    optimizer_class = get_class(target_cls)
                    return optimizer_class(params, **optimizer_cfg)
            
            # 回退到AdamW（兼容老代码）
            return torch.optim.AdamW(params, **cfg.optimizer)

        # ========= 修复点1：安全访问pretrained配置 =========
        try:
            # 尝试获取预训练标志
            pretrained_flag = cfg.policy.obs_encoder.pretrained
            print(f"找到配置: policy.obs_encoder.pretrained = {pretrained_flag}")
        except:
            # 如果配置中不存在该字段
            pretrained_flag = False
            print(f"警告: 配置中未找到 policy.obs_encoder.pretrained，使用默认值 False")

        # 训练代码部分
        base_lr = cfg.optimizer.lr

        # 1. 主干网络参数组（始终存在）
        param_groups = [{'params': self.model.model.parameters()}]

        # 2. 编码器参数组（智能添加）
        obs_encoder_params = [
            p for p in self.model.obs_encoder.parameters() 
            if p.requires_grad
        ]

        # 计算编码器学习率（兼容代码A和B的逻辑）
        obs_lr = base_lr
        if pretrained_flag:
            obs_lr *= 0.1
            print('==> 降低预训练编码器学习率')
            print(f"分层学习率: 主干网络LR={base_lr}, 编码器LR={obs_lr}")

        # 仅当有可训练参数时添加组
        if len(obs_encoder_params) > 0:
            param_groups.append({
                'params': obs_encoder_params, 
                'lr': obs_lr
            })
        print(f'编码器可训练参数数量: {len(obs_encoder_params)}')# 日志输出（兼容代码A）
        self.optimizer = build_optimizer(cfg, param_groups)# 实例化优化器（兼容两种模式）


        print(f"\n=== 优化器配置 ===")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数: {total_params}, 可训练参数: {trainable_params} ({trainable_params/total_params:.1%})")
        for i, group in enumerate(param_groups):
            print(f"参数组 #{i}: {len(group['params'])}个参数, LR={group.get('lr', base_lr)}")


        self.global_step = 0  # 初始化全局训练步数
        self.epoch = 0        # 初始化当前epoch数

        if not cfg.training.resume:
            print("\n=== 恢复训练配置 ===")
            print(f"排除键: {self.exclude_keys}")
            self.exclude_keys = ['optimizer']
        self.non_blocking = cfg.training.get('non_blocking_transfer', True)
        self.horizon = cfg.task.dataset.horizon if hasattr(cfg.task.dataset, 'horizon') else 8

        # ========= 恢复训练逻辑 =========
        if cfg.training.resume:
            # 优先使用配置中指定的检查点路径
            ckpt_path = cfg.training.get('resume_ckpt_path', None)
            if ckpt_path is None:
                # 如果未指定，则使用默认的最新检查点
                ckpt_path = self.get_checkpoint_path()
            else:
                ckpt_path = pathlib.Path(ckpt_path)
            
            if ckpt_path.is_file():
                print(f"从检查点恢复训练: {ckpt_path}")
                
                # 加载检查点
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                
                # 加载模型状态
                model_state = checkpoint['model_state_dict']
                if hasattr(self.model, 'module'):  # 处理分布式包装
                    self.model.module.load_state_dict(model_state)
                else:
                    self.model.load_state_dict(model_state)
                
                # 加载优化器状态
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # 加载训练状态
                self.global_step = checkpoint.get('global_step', 0)
                self.epoch = checkpoint.get('epoch', 0)
                
                # 保存学习率调度器状态（稍后在run方法中恢复）
                if 'lr_scheduler_state_dict' in checkpoint:
                    self.lr_scheduler_state = checkpoint['lr_scheduler_state_dict']
                    print("已保存学习率调度器状态，将在训练开始时恢复")
                else:
                    print("警告: 检查点中未找到学习率调度器状态")
                # 这里只记录，稍后在run方法中恢复
                
                # 加载EMA模型状态
                if cfg.training.use_ema and 'ema_state_dict' in checkpoint:
                    ema_state = checkpoint['ema_state_dict']
                    if hasattr(self.ema_model, 'module'):  # 处理分布式包装
                        self.ema_model.module.load_state_dict(ema_state)
                    else:
                        self.ema_model.load_state_dict(ema_state)
                
                print(f"恢复成功! 从epoch {self.epoch} 步数 {self.global_step} 继续训练")
            else:
                print(f"警告: 检查点文件不存在 {ckpt_path}")
        # ========= 结束恢复逻辑 =========




    # 执行训练过程，包括数据集配置、模型训练、验证、采样、日志记录和检查点保存
    def run(self):
        """
        2. 训练流程
            ​数据加载：
                使用BaseImageDataset加载训练和验证数据集，通过DataLoader进行批处理。
                数据归一化器（Normalizer）被应用到模型，确保输入数据标准化。
            ​学习率调度：
                使用get_scheduler创建学习率调度器，支持热身（warmup）和训练过程中的动态调整。
            EMA模型管理：
                若启用EMA，使用深拷贝的模型副本，并通过EMAModel类进行参数平滑更新，提升模型鲁棒性。
        """
        print("\n=== 训练启动 ===")
        print(f"输出目录: {self.output_dir}")
        print(f"设备: {self.cfg.training.device}")
        print(f"梯度累积步数: {self.cfg.training.gradient_accumulate_every}")
        print(f"总训练轮数: {self.cfg.training.num_epochs}")
        
        cfg = copy.deepcopy(self.cfg)                                   # 深拷贝配置以避免修改原始配置



        # 确保日志目录存在
        tensorboard_logdir = os.path.join(self.output_dir, 'tensorboard_runs')
        os.makedirs(tensorboard_logdir, exist_ok=True)
        
        # 创建兼容不同版本的Accelerator
        try:
            # 尝试使用带project_dir的初始化方式
            accelerator = Accelerator(
                log_with='tensorboard',
                project_dir=self.output_dir,
                mixed_precision=cfg.training.get('mixed_precision', 'bf16')  # 添加这一行
            )
            
            # 新版本只需要传递额外的参数
            init_kwargs = {
                "tensorboard": {
                    "flush_secs": 60
                },
                "log_dir": tensorboard_logdir
            }
        except TypeError as e:
            # 对于不支持project_dir的旧版本
            accelerator = Accelerator(log_with='tensorboard')
            # 旧版本需要传递log_dir参数
            init_kwargs = {"log_dir": tensorboard_logdir}
            
        # 初始化追踪器
        accelerator.init_trackers(
            project_name="diffusion_policy",
            init_kwargs=init_kwargs
        )

        # 实例化数据集
        dataset: BaseImageDataset                                       # 声明数据集类型
        dataset = hydra.utils.instantiate(cfg.task.dataset)             # 实例化数据集
        
        print("\n=== 数据集信息 ===")
        print(f"数据集类型: {type(dataset).__name__}")
        print(f"训练样本数: {len(dataset)}")
        print(f"观测步数: {dataset.n_obs_steps}")
        print(f"预测步数: {dataset.horizon}")
        
        assert isinstance(dataset, BaseImageDataset)                    # 确认数据集类型
        train_dataloader = DataLoader(dataset, **cfg.dataloader)        # 创建训练数据加载器
        val_dataset = dataset.get_validation_dataset()                  # 创建验证数据集加载器
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))
        print(f"数据集配置: n_obs_steps={dataset.n_obs_steps}")
        print(f"数据集配置: n_obs_steps={dataset.horizon}")

        # 只在主进程计算并保存归一化器
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()               # 获取数据归一化器
            pickle.dump(normalizer, open(normalizer_path, 'wb'))
            print(f"\n主进程保存归一化器到: {normalizer_path}")
        
        accelerator.wait_for_everyone()                         # 所有进程等待主进程完成
        normalizer = pickle.load(open(normalizer_path, 'rb'))   # 所有进程加载归一化器

        self.model.set_normalizer(normalizer)                   # 设置模型归一化器
        print(f"所有进程加载归一化器完成")
        if cfg.training.use_ema:                                # 如果使用EMA
            self.ema_model.set_normalizer(normalizer)           # 设置EMA模型归一化器


        # 配置初始化学习率调度器
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,# 学习率预热步数
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch假设每个训练周期都会执行一次LRScheduler（学习率调度器）的更新      pytorch assumes stepping LRScheduler every epoch
            # 然而，huggingface diffusers在每个批次都会对其进行处理                 however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1       # 恢复训练时使用
        )
        # 如果恢复训练，恢复学习率调度器状态
        if cfg.training.resume and hasattr(self, 'lr_scheduler_state'):
            try:
                lr_scheduler.load_state_dict(self.lr_scheduler_state)
                print("成功恢复学习率调度器状态")
                del self.lr_scheduler_state  # 清理临时存储
            except Exception as e:
                print(f"恢复学习率调度器状态失败: {str(e)}")
                # 回退到默认行为
                if self.global_step > 0:
                    # 手动设置调度器步数
                    for _ in range(self.global_step):
                        lr_scheduler.step()
        
        print(f"\n学习率调度器: {type(lr_scheduler).__name__}")
        # 添加学习率预热结束日志
        if self.global_step == cfg.training.lr_warmup_steps:
            current_lr = lr_scheduler.get_last_lr()[0]
            print(f"学习率预热结束! 当前学习率: {current_lr:.2e}")


        # 实例化环境运行器（用于策略评估）
        env_runner: BaseImageRunner                         # 声明环境运行器类型
        env_runner = hydra.utils.instantiate(               # 实例化环境运行器
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)      # 确认环境运行器类型


        # 初始化TopK检查点管理器,根据验证指标保存表现最好的K个模型，避免过拟合
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        """
        3. 训练循环
            设备转移:将模型、EMA模型和优化器移至指定设备（如GPU）。
            ​梯度累积:通过gradient_accumulate_every配置累积多个批次的梯度后再更新参数，模拟更大批次训练。
            ​损失计算与反向传播：
            调用model.compute_loss计算损失，反向传播并执行梯度裁剪。
            每隔一定步数更新优化器及学习率，并更新EMA模型参数。
            ​日志记录：
            使用WandB记录训练损失、学习率、验证指标等。
            JsonLogger将日志写入本地文件，便于后续分析。
        """
        
        # ========= 优化3：分布式准备 =========
        components = [self.model, self.optimizer, train_dataloader, val_dataloader]
        if lr_scheduler is not None:
            components.append(lr_scheduler)

        if accelerator:
            # 确保模型在CPU上，等待Accelerator处理
            self.model.cpu()
            if self.ema_model is not None:
                self.ema_model.cpu()
            
            prepared_components = accelerator.prepare(*components)
            # 解包
            self.model = prepared_components[0]
            self.optimizer = prepared_components[1]
            train_dataloader = prepared_components[2]
            val_dataloader = prepared_components[3]
            if lr_scheduler is not None:
                lr_scheduler = prepared_components[4]
            # 在初始化 EMA 管理器时明确指定设备, 确保EMA模型在正确设备上
            if cfg.training.use_ema and self.ema_model is not None:
                self.ema_model.to(accelerator.device)
        else:
            # 单设备设置设备转移（GPU/CPU）
            device = torch.device(cfg.training.device)
            self.model.to(device)
            if self.ema_model is not None:
                self.ema_model.to(device)
            optimizer_to(self.optimizer, device)    # 将优化器状态转移到设备

        # 设备一致性检查
        if accelerator:
            model_device = next(self.model.parameters()).device
            print(f"模型设备: {model_device}")
            print(f"加速器设备: {accelerator.device}")
            
            if model_device != accelerator.device:
                print(f"警告: 模型设备 ({model_device}) 与加速器设备 ({accelerator.device}) 不一致")
                # 自动修复 - 将模型转移到加速器设备
                self.model.to(accelerator.device)

        # 初始化EMA模型管理器,在分布式准备后手动转移 EMA 模型
        ema: EMAModel = None                                # 初始化EMA为None
        if cfg.training.use_ema:                            # 如果使用EMA
            # 确保EMA模型在正确设备上
            if accelerator:
                device = accelerator.device
            else:
                device = torch.device(cfg.training.device)
            
            # 将EMA模型转移到正确设备
            if self.ema_model is not None:
                self.ema_model.to(device)  # 这里应该是 self.ema_model，不是 self.model

            # 初始化EMA管理器，不传递device参数
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model
            )


        ######################### 调试信息 #########################
        print("\n" + "="*50)
        print("模型输入维度调试信息:")
        # 安全获取动作维度
        if hasattr(self.model, 'module'):  # 检查是否是分布式包装的模型
            action_dim = self.model.module.action_dim
        else:
            action_dim = self.model.action_dim
        print(f"模型动作维度 (action_dim): {action_dim}")

        # 打印第一个批次的形状
        # # 模型输入维度调试
        # print("\n=== 模型输入验证 ===")
        # with torch.no_grad():
        #     # 获取动作维度
        #     action_dim = self.model.module.action_dim if hasattr(self.model, 'module') else self.model.action_dim
        #     print(f"模型动作维度: {action_dim}")
        #     first_batch = next(iter(train_dataloader))
        #     print("\n第一个批次形状:")
        #     for key, value in first_batch.items():
        #         if isinstance(value, torch.Tensor):
        #             print(f"  {key}: {value.shape}")
        #         else:
        #             print(f"  {key}: {type(value)}")
            
        #     # 计算观测编码器输出
        #     # 需要安全获取模型

        #     model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        #     obs_features = model_to_use.obs_encoder(first_batch['obs'])
        #     print(f"\n观测编码器输出形状: {obs_features.shape}")
        #     # 测试动作预测
        #     with torch.no_grad():
        #         batch = dict_apply(first_batch, lambda x: x.to(device, non_blocking=True))
        #         gt_action = batch['action']
        #         print(f"真实动作: {gt_action.shape}")
        #         pred_action = self.model.module.predict_action(batch['obs'])['action_pred']
        #         print(f"动作预测输出: {pred_action.shape}") 
        # print("="*50 + "\n")
        ######################### 结束调试信息 #########################

        train_sampling_batch = None # 初始化训练采样批次为None, 用于保存一个训练批次用于后续采样

        # 调试模式设置（减少数据量）
        if cfg.training.debug:                  # 如果处于调试模式
            cfg.training.num_epochs = 2         # 设置训练轮数为2
            cfg.training.max_train_steps = 3    # 设置最大训练步数为3
            cfg.training.max_val_steps = 3      # 设置最大验证步数为3
            cfg.training.rollout_every = 1      # 设置每隔1个epoch进行一次rollout
            cfg.training.checkpoint_every = 1   # 设置每隔1个epoch保存一次检查点
            cfg.training.val_every = 1          # 设置每隔1个epoch进行一次验证
            cfg.training.sample_every = 1       # 设置每隔1个epoch进行一次采样

        # 初始空间检查
        total, used, free = shutil.disk_usage(self.output_dir)
        print(f"初始磁盘空间 - 总: {total//(1024**3)}GB, 已用: {used//(1024**3)}GB, 空闲: {free//(1024**3)}GB")
        MIN_DISK_SPACE = 7 * 1024**3  # 7GB最小要求

        if free < MIN_DISK_SPACE:
            raise RuntimeError(
                f"磁盘空间不足! 需要至少{MIN_DISK_SPACE/(1024**3):.1f}GB空闲空间, "
                f"当前只有{free/(1024**3):.1f}GB"
            )
    
        # 在训练循环开始前，读取所有需要的配置到局部变量
        use_ema = cfg.training.use_ema
        freeze_encoder = cfg.training.freeze_encoder
        gradient_accumulate_every = cfg.training.gradient_accumulate_every
        rollout_every = cfg.training.rollout_every
        sample_every = cfg.training.sample_every
        checkpoint_every = cfg.training.checkpoint_every
        max_train_steps = cfg.training.max_train_steps
        val_every = cfg.training.val_every
        tqdm_interval_sec = cfg.training.tqdm_interval_sec
        debug_mode = cfg.training.debug

        # 训练循环，先初始化JSON日志记录器
        log_path = os.path.join(self.output_dir, 'logs.json.txt')   # 日志文件路径
        # 在训练循环开始前
        metrics_log_path = os.path.join(self.output_dir, 'action_metrics.json.txt')
        # 确保目录存在
        os.makedirs(os.path.dirname(metrics_log_path), exist_ok=True)
        
        with JsonLogger(log_path) as json_logger:                   # 使用JsonLogger记录日志
            # 训练主循环
            for local_epoch_idx in range(cfg.training.num_epochs):  # 遍历训练轮数
                # print(f"\n--- 训练轮次 {self.epoch}/{self.cfg.training.num_epochs} ---")
                # print(f"全局步数: {self.global_step}")

                step_log = dict()                                   # 初始化步日志
                self.model.train()# 设置模型为训练模式

                # EMA模型始终设为评估模式
                if use_ema:
                    self.ema_model.eval()
                
                # 冻结编码器（如果配置）
                if freeze_encoder:                     # 如果冻结编码器
                    self.model.obs_encoder.eval()                   # 设置编码器为评估模式
                    self.model.obs_encoder.requires_grad_(False)    # 不需要计算编码器的梯度

                # ========= train for this epoch ==========
                train_losses = list()                               # 记录本epoch的损失
                accumulation_steps = 0
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval = tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # 将数据转移到指定设备（支持Accelerator），Accelerator会自动处理数据转移
                        if not accelerator:
                            device = torch.device(cfg.training.device)
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=self.non_blocking))

                        # batch = accelerator.prepare(batch)
                        if train_sampling_batch is None:    # 如果训练采样批次为空，设置训练采样批次
                            train_sampling_batch = batch

                        # 计算损失（前向传播）,在计算损失前添加混合精度上下文
                        with accelerator.autocast():
                            raw_loss = self.safe_compute_loss(self.model, batch)
                        # 梯度累积（将损失除以累积步数）
                        loss = raw_loss / gradient_accumulate_every
                        accelerator.backward(loss)  # 使用加速器统一处理反向传播

                        accumulation_steps += 1
                        # 执行优化器步骤（达到累积步数时,self.global_step 在每次迭代中都会增加,使用批次索引判断梯度累积可能会跳过最后部分的梯度下降）
                        if (accumulation_steps) % gradient_accumulate_every == 0: # 如果到达梯度累积步数
                            self.optimizer.step()                   # 更新优化器
                            self.optimizer.zero_grad()              # 梯度归零
                            lr_scheduler.step()                     # 更新学习率调度器
                            accumulation_steps = 0
                        # ========= 优化4：EMA更新改进 =========
                        # 更新EMA模型参数
                        if use_ema and ema is not None:  # 使用EMA管理器对象
                            # 统一使用加速器的unwrap_model方法,移除 DDP 包装,返回原始模型结构,保留所有参数和状态
                            model_to_update = accelerator.unwrap_model(self.model) if accelerator else self.model
                            if model_to_update.device != ema.averaged_model.device: # 设备一致性检查
                                print(f"警告：模型设备 ({model_to_update.device}) 与EMA模型设备 ({ema.averaged_model.device}) 不一致")
                                # 自动修复 - 将EMA模型转移到相同设备
                                ema.averaged_model.to(model_to_update.device)
                            ema.step(model_to_update)  # 通过EMA管理器更新

                        # logging
                        raw_loss_cpu = raw_loss.item()              # 获取原始损失值
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False) # 更新进度条显示损失
                        train_losses.append(raw_loss_cpu)           # 添加损失到训练损失列表
                        step_log = {                                # 记录步日志
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        # 非最后批次时记录日志
                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            json_logger.log(step_log)               # 记录json日志
                            self.global_step += 1                   # 全局步数加1

                        # 提前终止训练（调试模式）
                        if (max_train_steps is not None) and batch_idx >= (max_train_steps - 1):
                            break

                # 在每个epoch结束时，用epoch平均值替换训练损失，计算本epoch平均训练损失
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ========== # 评估当前epoch
                policy = self.model             # 获取模型策略
                if use_ema:        # 如果使用EMA
                    policy = self.ema_model     # 获取EMA模型策略
                policy.eval()                   # 设置模型为评估模式

                # run rollout 运行策略评估（定期执行）
                if (self.epoch % rollout_every) == 0:  # 如果到达rollout间隔
                    runner_log = env_runner.run(policy) # 运行环境并获取日志
                    step_log.update(runner_log)         # 更新步日志,记录评估指标

                # run diffusion sampling on a training batch
                # 训练批次采样评估（定期执行）
                if (self.epoch % sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        # 从训练集中采样轨迹，并评估差异 sample trajectory from training set, and evaluate difference
                        # 获取当前设备
                        device = accelerator.device if accelerator else torch.device(cfg.training.device)
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=self.non_blocking))
                        gt_action = batch['action']
                        
                        pred_action = policy.predict_action(batch['obs'])['action_pred']
                        # 对齐序列长度
                        min_length = min(pred_action.shape[1], gt_action.shape[1])
                        truncated_pred_action = pred_action[:, :min_length, :]
                        truncated_gt_action = gt_action[:, :min_length, :]
                        mse = torch.nn.functional.mse_loss(truncated_pred_action, truncated_gt_action)
                        step_log['train_action_mse_error'] = mse.item()

                        # 额外的动作分解误差分析
                        log_action_mse(step_log, 'val', truncated_pred_action, truncated_gt_action)
                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            gt_action = batch['action']
                            pred_action = policy.predict_action(batch['obs'])['action_pred']

                            min_length = min(pred_action.shape[1], gt_action.shape[1])
                            truncated_pred_action = pred_action[:, :min_length, :]
                            truncated_gt_action = gt_action[:, :min_length, :]

                            mse = torch.nn.functional.mse_loss(truncated_pred_action, truncated_gt_action)

                            log_action_mse(step_log, 'val', truncated_pred_action, truncated_gt_action)
                        del batch
                        del gt_action
                        del pred_action
                        del mse
                    torch.cuda.empty_cache()  # 释放显存  

                # ========= 检查点保存 ==========
                if (self.epoch % checkpoint_every) == 0 and accelerator.is_main_process:
                    # 确保checkpoints目录存在
                    checkpoints_dir = os.path.join(self.output_dir, 'checkpoints')
                    os.makedirs(checkpoints_dir, exist_ok=True)
                    
                    # 保存前先解包模型（如果是分布式训练）
                    model_to_save = accelerator.unwrap_model(self.model) if accelerator else self.model
                    original_model = self.model
                    self.model = model_to_save

                    # 模型保存前设置模型为评估模式
                    self.model.eval()

                    # 保存EMA模型（如果启用）
                    if use_ema and self.ema_model is not None:
                        ema_to_save = accelerator.unwrap_model(self.ema_model) if accelerator else self.ema_model
                        original_ema = self.ema_model
                        self.ema_model = ema_to_save
                        self.ema_model.eval()
                    
                    try:
                        # 1. 保存最新独立模型检查点（最小化）
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint(use_thread=False)  # 确保同步保存
                        
                        # 2. 保存最新的可恢复训练检查点（完整状态）
                        # 获取训练损失值（用于文件名）
                        train_loss_val = step_log.get('train_loss', 0.0)
                        
                        # 创建完整的检查点数据
                        trainable_checkpoint = {
                            'epoch': self.epoch,
                            'global_step': self.global_step,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                            'ema_state_dict': self.ema_model.state_dict() if use_ema and self.ema_model is not None else None
                        }
                        
                        # 创建唯一文件名（基于epoch和损失）
                        checkpoint_filename = f"epoch_{self.epoch:04d}_train_loss={train_loss_val:.4f}.ckpt"
                        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_filename)
                        
                        # 保存检查点
                        torch.save(trainable_checkpoint, checkpoint_path)
                        print(f"可恢复训练检查点已保存至: {checkpoint_path}")
                        

                        # 2. 保存快照（完整模型状态）
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()
                        
                        # 3. 处理TopK检查点（仅当有新的TopK时保存）
                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value
                        
                        # 获取topk检查点路径，比较当前模型的验证指标与已保存检查点的指标
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        
                        if topk_ckpt_path is not None:
                            # 如果是TopK模型，额外保存一次
                            self.save_checkpoint(path=topk_ckpt_path, use_thread=False)
                    
                    finally:
                        # 恢复模型引用
                        self.model = original_model
                        if use_ema and self.ema_model is not None:
                            self.ema_model = original_ema
                
                # ========= eval end for this epoch ==========
                # end of epoch，log of last step is combined with validation and rollout
                # 记录最终日志
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

                # 下面是您添加的新代码
                if accelerator.is_main_process:  # 确保只由主进程执行
                    # 添加以下两行打开文件的代码
                    with open(metrics_log_path, 'a') as metrics_file:
                        metrics_log = {
                            'epoch': self.epoch,
                            'global_step': self.global_step,
                            # 包含动作分解指标
                            'train_action_mse': step_log.get('train_action_mse', None),
                            'train_action_mse_pos': step_log.get('train_action_mse_pos', None),
                            'train_action_mse_rot': step_log.get('train_action_mse_rot', None),
                            'train_action_mse_gripper': step_log.get('train_action_mse_gripper', None),
                            'val_action_mse': step_log.get('val_action_mse', None),
                            'val_action_mse_pos': step_log.get('val_action_mse_pos', None),
                            'val_action_mse_rot': step_log.get('val_action_mse_rot', None),
                            'val_action_mse_gripper': step_log.get('val_action_mse_gripper', None),
                            'rollout_success_rate': step_log.get('rollout/success_rate', None),
                            'rollout_reward': step_log.get('rollout/reward', None)
                        }
                        # 写入专用日志文件
                        json.dump(metrics_log, metrics_file)  # 这里需要json模块
                        metrics_file.write('\n')  # 换行分隔记录

                # 添加梯度统计
                if batch_idx % 50 == 0:
                    total_grad = 0.0
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            total_grad += grad_norm
                    print(f"批次 {batch_idx}: 平均梯度范数: {total_grad/(batch_idx+1):.4f}")
                
                # 添加内存使用监控
                if batch_idx % 100 == 0:
                    print(f"GPU内存使用: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

            # 在训练循环结束后
            if accelerator:
                accelerator.end_training()
            
            # 清理资源
            del train_dataloader, val_dataloader
            torch.cuda.empty_cache()

    def safe_compute_loss(self, model, batch):
        """安全计算损失，处理分布式包装的模型"""
        if hasattr(model, 'module'):
            return model.module.compute_loss(batch)
        else:
            return model.compute_loss(batch)
    
    def safe_predict_action(self, model, obs):
        """安全预测动作，处理分布式包装的模型"""
        if hasattr(model, 'module'):
            return model.module.predict_action(obs)
        else:
            return model.predict_action(obs)
    
    def safe_get_model(self, model):
        """安全获取原始模型"""
        if hasattr(model, 'module'):
            return model.module
        else:
            return model

    def save_model_checkpoint(self, output_dir, epoch, train_loss, model, ema_model=None):
        """保存模型检查点，文件名包含epoch和损失信息"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if hasattr(self, 'lr_scheduler') else None,
        }
        
        if ema_model is not None:
            checkpoint['ema_state_dict'] = ema_model.state_dict()
        
        # 创建带损失信息的文件名
        checkpoint_filename = f"epoch_{epoch:04d}_train_loss={train_loss:.4f}.pt"
        checkpoint_path = os.path.join(output_dir, checkpoint_filename)
        
        # 保存检查点
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存至: {checkpoint_path}")
        
        return checkpoint_path

    def save_last_checkpoint(self, output_dir, model, ema_model=None):
        """保存最新的检查点，用于恢复训练"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if hasattr(self, 'lr_scheduler') else None,
        }
        
        if ema_model is not None:
            checkpoint['ema_state_dict'] = ema_model.state_dict()
        
        # 保存为last.ckpt
        last_ckpt_path = os.path.join(output_dir, 'last.ckpt')
        torch.save(checkpoint, last_ckpt_path)
        print(f"最新检查点已保存至: {last_ckpt_path}")
        
        return last_ckpt_path






# Hydra主函数装饰器（指定配置路径）
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    # 实例化工作空间并运行
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()



# if __name__ == "__main__":
#     import sys
#     import os
#     import pathlib

#     ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
#     sys.path.append(ROOT_DIR)
#     os.chdir(ROOT_DIR)

#     # 快照路径
#     snapshot_path = "path/to/snapshot.pkl"

#     # 加载快照
#     import dill
#     workspace = torch.load(open(snapshot_path, 'rb'), pickle_module=dill)

#     # 继续训练
#     workspace.run()