from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

import logging
import torch
import torch.nn as nn
import torchvision.transforms as T
from einops.layers.torch import Rearrange



class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,   # [240, 320]
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,     # [240, 320]
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            # 新增参数
            use_mask: bool = False,         # 是否启用mask处理
            mask_channels: int = 1,         # mask通道数
            fusion_dim: int = 512           # RGB与Mask融合后的维度
        ):
        """
        精简版编码器：只提取mask区域的RGB特征
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        mask_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # 处理共享视觉主干
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        # 解析形状元数据
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape

            if type == 'rgb':
                rgb_keys.append(key)
                # 配置模型 for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # 配置预处理 configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # 配置随机裁剪 configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # 配置归一化 configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform

            elif type == 'mask':
                mask_keys.append(key)
                # 不需要为mask单独配置模型和预处理
            
            elif type == 'low_dim':
                # 低维输入，直接跳过
                pass
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        rgb_keys = sorted(rgb_keys)
        mask_keys = sorted(mask_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.mask_keys = mask_keys
        self.key_shape_map = key_shape_map

    def forward(self, obs_dict):
        batch_size = None
        features = []  # 使用列表存储特征向量
        
        # 处理共享模型情况
        if self.share_rgb_model:
            imgs_list = []  # 存储所有masked RGB图像
            mask_keys = []  # 记录对应的mask键
            
            # 收集所有masked RGB图像
            for mask_key in self.mask_keys:
                rgb_key = mask_key.replace('_mask', '')
                rgb_img = obs_dict[rgb_key]
                mask = obs_dict[mask_key]
                
                # 生成masked RGB
                if mask.max() > 1:
                    mask = (mask > 0).float()
                if mask.shape[1] == 1 and rgb_img.shape[1] > 1:
                    mask = mask.repeat(1, rgb_img.shape[1], 1, 1)
                masked_rgb = rgb_img * mask
                
                # 应用预处理
                preprocessed = self.key_transform_map[rgb_key](masked_rgb)
                imgs_list.append(preprocessed)
                mask_keys.append(mask_key)
                
                if batch_size is None:
                    batch_size = preprocessed.shape[0]
            
            # 如果有需要处理的图像
            if imgs_list:
                # 沿批次维度拼接 (N*B, C, H, W)
                imgs = torch.cat(imgs_list, dim=0)
                
                # 一次性处理所有图像
                features_all = self.key_model_map['rgb'](imgs)
                
                # 重组特征维度 (N, B, D)
                features_all = features_all.reshape(
                    len(imgs_list), batch_size, *features_all.shape[1:]
                )
                
                # 转换为 (B, N, D)
                features_all = torch.moveaxis(features_all, 0, 1)
                
                # 展平每个图像的特征 (B, N*D)
                features_all = features_all.reshape(batch_size, -1)
                features.append(features_all)
        
        # 非共享模型情况
        else:
            for mask_key in self.mask_keys:
                # 方案1：处理每个mask_key对应的masked RGB
                rgb_key = mask_key.replace('_mask', '')
                rgb_img = obs_dict[rgb_key]
                mask = obs_dict[mask_key]
                # 生成masked RGB
                if mask.max() > 1:
                    mask = (mask > 0).float()
                if mask.shape[1] == 1 and rgb_img.shape[1] > 1:
                    mask = mask.repeat(1, rgb_img.shape[1], 1, 1)
                masked_rgb = rgb_img * mask
                # 方案2：原始代码 img = obs_dict[key]

                # 应用预处理
                preprocessed = self.key_transform_map[rgb_key](masked_rgb)
                
                # 应用编码器
                feature = self.key_model_map[rgb_key](preprocessed)
                
                # 展平特征，原始rgb+低维是在，在最终输出前展平
                feature = feature.flatten(start_dim=1)      # 从第1维开始展平（保留批次维度）
                features.append(feature)                    # 将特征向量添加到列表中
                
                if batch_size is None:
                    batch_size = feature.shape[0]
        
        # 拼接所有特征
        if features:
            # 对于共享模型，features只有一个元素
            # 对于非共享模型，features有多个元素需要拼接
            result = torch.cat(features, dim=1)
        else:
            result = torch.zeros((batch_size, 0), device=self.device)
        
        return result


    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        
        # 创建示例输入
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        
        # 计算输出形状
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape



    # def forward(self, obs_dict):
    #     batch_size = None                   # 初始化batch_size变量，稍后用于记录批次大小
    #     features = []                       # 创建一个空列表，用于存储所有masked RGB的特征向量
    #     # features = {}  # 改用字典存储特征

    #     # 处理每个mask_key对应的masked RGB（如'camera_1_mask'）
    #     for mask_key in self.mask_keys:
    #         rgb_key = mask_key.replace('_mask', '')     # 从mask键名推导出对应的RGB键名（如'camera1_mask' -> 'camera1'）
            
    #         # 获取原始RGB和mask
    #         rgb_img = obs_dict[rgb_key]                 # 从输入字典中获取原始RGB图像 [B, C, H, W]
    #         mask = obs_dict[mask_key]                   # 从输入字典中获取对应的mask [B, 1, H, W]
            
    #         # 生成masked RGB (非mask区域置黑)
    #         if mask.max() > 1:                          # 确保mask是二值化的（0或1）：如果最大值超过1，则进行二值化
    #             mask = (mask > 0).float()
    #         if mask.shape[1] == 1 and rgb_img.shape[1] > 1:# 如果mask是单通道而RGB是多通道，复制mask使其通道数与RGB匹配
    #             mask = mask.repeat(1, rgb_img.shape[1], 1, 1)
    #         masked_rgb = rgb_img * mask                 # 生成masked RGB：将非mask区域置为0（黑色）
            
    #         # 应用预处理
    #         preprocessed = self.key_transform_map[rgb_key](masked_rgb)# 使用RGB键对应的预处理流程（resize/crop/normalize）
            
    #         # 选择模型：如果是共享模型则使用共享模型，否则使用该RGB键对应的模型
    #         if self.share_rgb_model:
    #             model = self.key_model_map['rgb']
    #         else:
    #             model = self.key_model_map[rgb_key]
            
    #         # 使用模型提取特征：输入预处理后的masked RGB图像
    #         feature = model(preprocessed)
            
    #         # 展平特征：将多维特征图转换为一维特征向量 [B, D]
    #         feature = feature.flatten(start_dim=1)      # 从第1维开始展平（保留批次维度）
    #         features.append(feature)                    # 将特征向量添加到列表中
            
    #         # 如果是第一次处理，记录batch_size
    #         if batch_size is None:
    #             batch_size = feature.shape[0]
        
    #     # 拼接所有masked RGB特征
    #     if features:                                    # 两个摄像头各提取512维特征 → 1024维特征向量
    #         result = torch.cat(features, dim=1)         # 将所有masked RGB特征沿特征维度拼接 [B, D1+D2+...]
    #     else:
    #         # 如果没有masked RGB特征需要处理，创建空特征向量 [B, 0]，确保函数始终返回有效的张量
    #         result = torch.zeros((batch_size, 0), device=self.device)
        
    #     return result


    #     # 处理每个mask_key对应的masked RGB
    #     if self.share_rgb_model:
    #         imgs = list()
    #         for key in self.rgb_keys:
    #             img = obs_dict[key]
    #             if batch_size is None:
    #                 batch_size = img.shape[0]
    #             else:
    #                 assert batch_size == img.shape[0]

    #             assert img.shape[1:] == self.key_shape_map[key]
    #             img = self.key_transform_map[key](img)
    #             imgs.append(img)
    #         # (N*B,C,H,W)
    #         imgs = torch.cat(imgs, dim=0)
    #         # (N*B,D)
    #         feature = self.key_model_map['rgb'](imgs)
    #         # (N,B,D)
    #         feature = feature.reshape(-1,batch_size,*feature.shape[1:])
    #         # (B,N,D)
    #         feature = torch.moveaxis(feature,0,1)
    #         # (B,N*D)
    #         feature = feature.reshape(batch_size,-1)
    #         features['rgb'] = feature
    #     else:
    #         # run each rgb obs to independent models
    #         for key in self.rgb_keys:
    #             img = obs_dict[key]
    #             if batch_size is None:
    #                 batch_size = img.shape[0]
    #             else:
    #                 assert batch_size == img.shape[0]

    #             img = self.key_transform_map[key](img)
    #             # print(f"key: {key}, img.shape: {img.shape[1:]}, key_shape_map[key]: {self.key_shape_map[key]}") 
    #             # assert img.shape[1:] == self.key_shape_map[key]
    #             feature = self.key_model_map[key](img)
    #             features[key] = feature

    #     # ========== 处理带Mask的摄像头 ==================================
    #     for mask_key in self.mask_keys:
    #         rgb_key = mask_key.replace('_mask', '')
    #         rgb_feat = features[rgb_key]  # 直接通过键获取
            
    #         # 处理Mask (已经是B,1,H,W格式)
    #         mask = self.key_transform_map[mask_key](obs_dict[mask_key])
    #         mask_feat = self.key_model_map[mask_key](mask).flatten(start_dim=1)
            
    #         # 特征融合
    #         fused = self.fusion_layers[mask_key](torch.cat([rgb_feat, mask_feat], dim=-1))
    #         features[mask_key] = fused

    #     # process lowdim input
    #     for key in self.low_dim_keys:
    #         data = obs_dict[key]
    #         if batch_size is None:
    #             batch_size = data.shape[0]
    #         else:
    #             assert batch_size == data.shape[0]
    #         assert data.shape[1:] == self.key_shape_map[key]
    #         features[key] = data
        
    #     # ========== 沿特征维度拼接 (B, total_dim) ============================
    #     result = torch.cat(list(features.values()), dim=-1)   # (B, 1518)
    #     return result