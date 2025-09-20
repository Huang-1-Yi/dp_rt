"""
在dp_multi_image_obs_encoder的基础上
增加了对低维离散状态（如任务阶段）的one-hot编码支持，同时优化了设备兼容性
"""
# 特征编码流程
# 1. 缩放图像至224x224
# 2. 随机裁剪至192x192
# 3. ResNet18提取视觉特征
# 4. 拼接位姿数据
# 5. 输出合并特征向量 (32, 512 * 2 + 6/7)
from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

'''
# ========== 使用示例 ==========================================
# 配置示例
shape_meta = {
    'obs': {
        'camera_0': {'shape': (3, 240, 320), 'type': 'rgb'},
        'robot_eef_pose': {'shape': (6,), 'type': 'low_dim'}
    }
}

# 使用ResNet18作为共享骨干网络
encoder = MultiImageObsEncoder(
    shape_meta=shape_meta,
    rgb_model=torchvision.models.resnet18(pretrained=False),
    share_rgb_model=True,
    resize_shape=(224, 224),
    crop_shape=(192, 192),
    imagenet_norm=True
)

# 前向计算
obs = {
    'camera_0': torch.randn(32, 3, 240, 320),  # 批量大小32
    'robot_eef_pose': torch.randn(32, 6)
}
features = encoder(obs)  # 输出形状 (32, 512+6)
'''

class MultiImageObsEncoder(ModuleAttrMixin):
    """
    多模态观测编码器，支持处理：
    - 多个RGB摄像头输入 (支持共享/独立视觉骨干网络)
    - 低维状态输入 (如机械臂位姿、夹爪状态)
    
    设计特点：
    1. 灵活支持不同传感器配置 (通过shape_meta定义)
    2. 可配置图像预处理 (缩放、随机裁剪、归一化)
    3. 支持BatchNorm -> GroupNorm替换 (提升小批量训练稳定性)
    4. 支持单模型多摄像头共享 (减少参数量)
    """
    def __init__(self,
            shape_meta: dict,                                                # 观测数据结构定义（类型和形状，必须包含'obs'键）
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],                # RGB处理模型（单个或字典）（共享或独立）
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None, # 图像缩放尺寸（全局或各摄像头独立）
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,   # 随机裁剪尺寸
            random_crop: bool=True,                                          # 是否启用随机裁剪（数据增强）
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,                                      # 是否使用GroupNorm替换BatchNorm
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,                                     # 多个摄像头 是/否 共享同一个RGB模型
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,                                        # 是否使用ImageNet标准化（输入需在[0,1]范围）
            # 你希望最大支持 stage=4 的 one-hot，向量维度为 5
            stage_num_classes: int=5,                                          # 任务阶段分类数（用于one-hot编码）5
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        # ========== 初始化存储结构 =============================================
        rgb_keys = list()                   # 存储所有RGB观测键（如'camera_0'）
        low_dim_keys = list()               # 存储低维观测键（如'robot_eef_pose'）
        key_model_map = nn.ModuleDict()     # 各观测键对应的处理模型
        key_transform_map = nn.ModuleDict() # 各观测键对应的预处理序列
        key_shape_map = dict()              # 各观测键原始形状记录
        self.one_hot_keys = list()  # 记录需要 one-hot 编码的低维键

        # ========== 共享视觉骨干网络配置 ========================================
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module) # 验证变量 rgb_model 的类型是否合法,共享模式下必须是单个模型
            key_model_map['rgb'] = rgb_model        # 使用统一键名'rgb'存储共享模型

        # ========== 遍历shape_meta解析每个观测项 ================================
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])         # 原始数据形状（如(3,240,320)）
            type = attr.get('type', 'low_dim')   # 默认为传感器等低维数据
            key_shape_map[key] = shape

            # === RGB图像处理分支 ===
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None

                # 非共享模式下的模型配置
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]             # 为每个摄像头分配独立模型
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)   # 深拷贝避免参数共享
                
                # 模型替换BatchNorm为GroupNorm（提升小批量训练稳定性）
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16,  # 16通道/组
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # 图像尺寸调整
                # configure resize
                input_shape = shape    # (C,H,W)
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    # 支持全局或各摄像头独立设置缩放尺寸
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)    # 更新输入形状

                # 随机裁剪配置
                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(  # 随机裁剪增强
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        # this_normalizer = torchvision.transforms.CenterCrop(
                        #     size=(h,w)
                        # )
                        # 修复预处理流程顺序（原错误会导致 CenterCrop 被错误赋值为 this_normalizer）。
                        # 确保裁剪操作在归一化前执行，符合视觉任务标准流程。
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )# 正确变量名
                
                # 图像归一化（Imagenet标准）
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],    # ImageNet均值
                        std=[0.229, 0.224, 0.225]      # ImageNet标准差
                    )
                
                # 组合预处理流程：缩放 -> 裁剪 -> 归一化
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            
            # === 低维数据处理分支 ===
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}") # 无需模型，直接拼接
        
        # 按键名字母序排列（确保特征拼接顺序一致）
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        # 存储关键配置
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.stage_num_classes = stage_num_classes
        print(f"[MultiImageObsEncoder] stage_num_classes: {self.stage_num_classes}")
        self.one_hot_keys = ['stage']  # 哪些低维键要 one-hot 编码


    def forward(self, obs_dict):
        """
        输入: 
            obs_dict: 观测字典，键值示例：
                'camera_0': (B, C, H, W) RGB图像
                'robot_eef_pose': (B, D) 低维状态
        
        输出: 
            (B, total_feature_dim) 拼接后的特征向量
        """
        # # 输入维度打印
        # print("\n[MultiImageObsEncoder] Input shapes:")
        # for key, value in obs_dict.items():
        #     print(f"  {key}: {value.shape}")

        batch_size = None
        features = list()

        # ========== 处理RGB输入 ============================================
        if self.share_rgb_model:
            # 共享模式：拼接所有图像后统一处理
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]   # (B,C,H,W)
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                # 应用预处理链
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # 拼接所有图像到批次维度 (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # 通过共享模型 (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # print(f"  Shared features shape1: {feature.shape}")

            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            # print(f"  Shared features shape2: {feature.shape}")
            features.append(feature)
        else:
            # 独立模式：每个摄像头独立处理
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)  # 应用预处理
                
                feature = self.key_model_map[key](img)  # (B,D) rgb观测键对应的独立模型推理
                # print(f"  Feature {key} shape: {feature.shape}")
                features.append(feature)
        
        # ========== 处理低维输入 ============================================
        # 无参数化处理:低维状态不经过任何可学习层（如MLP）
        for key in self.low_dim_keys:
            data = obs_dict[key]     # 直接提取原始张量，如 (B,6)
            if batch_size is None:
                batch_size = data.shape[0]
            # else:
            #     assert batch_size == data.shape[0]
            # # assert data.shape[1:] == self.key_shape_map[key]
            # # 修改后
            # if data.shape[1:] != self.key_shape_map[key]:
            #     raise RuntimeError(
            #         f"观测键 '{key}' 维度不匹配！\n"
            #         f"配置形状: {self.key_shape_map[key]}\n"
            #         f"实际形状: {data.shape[1:]}\n"
            #         f"可能原因: \n"
            #         "  1. shape_meta配置错误\n"
            #         "  2. 数据集包含动态尺寸图像\n"
            #         "  3. 数据加载预处理改变了维度"
            #     )
            # if data.shape[1:] != self.key_shape_map[key]:
            #     raise RuntimeError(f"Shape mismatch for key: {key}. Expected {self.key_shape_map[key]}, got {data.shape[1:]}.")

            if key in self.one_hot_keys:
                index = data.long().squeeze(-1)        # .round() 安全起见，先 round
                # if torch.any(index >= self.stage_num_classes) or torch.any(index < 0):
                #     bad_max = int(index.max())
                #     bad_min = int(index.min())
                #     raise ValueError(
                #         f"[{key}] out‑of‑range value(s) detected! "
                #         f"min={bad_min}, max={bad_max}, "
                #         f"but num_classes={self.stage_num_classes}."
                #     )
                one_hot = torch.nn.functional.one_hot(
                    index, num_classes=self.stage_num_classes
                ).float()
                features.append(one_hot)
            else:
                features.append(data)

            # features.append(data)
        
        # ========== 沿特征维度拼接 (B, total_dim) ============================
        result = torch.cat(features, dim=-1)   # (B, D_image + D_lowdim)即(B, total_feature_dim)
        # print(f"[MultiImageObsEncoder] Output shape: {result.shape}")
        return result
    

    @torch.no_grad()
    def output_shape(self):
        '''动态计算特征总维度，避免手动维护维度参数'''
        # 生成虚拟输入计算输出维度
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape,
                dtype=torch.float32,
                device=next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device("cpu")
            )
            example_obs_dict[key] = this_obs
        # 前向传播获取输出形状
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape

    # # 动态输出维度优化​：首次调用后缓存结果，后续调用时间从 ​​50ms​​ 降至 ​​0.1ms​​
    # @torch.no_grad()
    # def output_shape(self):
    #     if hasattr(self, '_cached_output_shape'):  # 缓存结果
    #         return self._cached_output_shape
    #     # 生成虚拟输入计算输出维度
    #     example_obs_dict = dict()
    #     obs_shape_meta = self.shape_meta['obs']
    #     batch_size = 1
    #     for key, attr in obs_shape_meta.items():
    #         shape = tuple(attr['shape'])
    #         this_obs = torch.zeros(
    #             (batch_size,) + shape, 
    #             dtype=self.dtype,
    #             device=self.device)
    #         example_obs_dict[key] = this_obs
    #     # 前向传播获取输出形状
    #     example_output = self.forward(example_obs_dict)
    #     output_shape = example_output.shape[1:]
    #     self._cached_output_shape = output_shape
    #     return output_shape
