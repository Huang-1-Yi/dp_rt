from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply, dict_apply_reduce, dict_apply_split
import numpy as np
import torch

# 通用范围归一化
def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    """
    功能：创建基于数据范围的标准线性归一化器
    目的：将输入数据线性映射到指定输出范围（默认[-1,1]）
    实现逻辑：
        1. 计算数据范围：input_range = max - min
        2. 处理常值维度：当范围<eps时视为无效维度
        3. 计算变换参数：
           scale = (output_max - output_min) / input_range
           offset = output_min - scale * input_min
        4. 无效维度处理：offset = (output_max+output_min)/2 - input_min
    数学原理：
        output = scale * input + offset
        满足：min(input) → output_min, max(input) → output_max
    使用场景：
        通用数值型数据的标准化处理，适用于传感器读数、关节角度等连续值
    参数说明：
        stat：输入统计字典 {'min':, 'max':, 'mean':, 'std':}
        output_max/min：目标归一化范围（默认[-1,1]）
        range_eps：范围阈值，小于此值视为常值维度
    返回：配置完成的SingleFieldLinearNormalizer实例
    """
    input_max = stat['max']
    input_min = stat['min']
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

# 通用对称归一化
def get_range_symmetric_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    """
    功能：创建具有坐标对称特性的范围归一化器
    目的：针对机器人控制场景，确保前两维（通常是XY平面坐标）的归一化范围对称于原点
    实现逻辑：
        1. 计算前两个维度的最大绝对值：abs_max = max(|max_x|, |min_x|, |max_y|, |min_y|)
        2. 强制设定归一化范围：input_max = [abs_max, abs_max, 原始max_z...]
                            input_min = [-abs_max, -abs_max, 原始min_z...]
        3. 处理常值维度：当input_range<eps时保持范围不变
        4. 计算线性变换参数：scale = (output_max - output_min) / (input_max - input_min)
                          offset = output_min - scale * input_min
    数学原理：
        output = scale * input + offset
        其中XY维度满足：max(output) = -min(output)
    使用场景：
        机器人轨迹控制任务，确保X/Y坐标对称归一化（如桌面型机械臂工作空间）
    参数说明：
        stat：输入统计字典 {'min':, 'max':, 'mean':, 'std':}
        output_max/min：目标归一化范围（默认[-1,1]）
        range_eps：防止除零的阈值
    返回：配置完成的SingleFieldLinearNormalizer实例
    """
    # -1, 1 normalization
    input_max = stat['max']
    input_min = stat['min']
    abs_max = np.max([np.abs(stat['max'][:2]), np.abs(stat['min'][:2])])
    input_max[:2] = abs_max
    input_min[:2] = -abs_max
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

# 体素数据恒等变换
def get_voxel_identity_normalizer():
    """
    功能：创建体素数据的恒等归一化器
    目的：为体素数据（三维网格表示）提供无变换的归一化处理
    实现逻辑：
        scale = [1] (无缩放)
        offset = [0] (无偏移)
        预设统计参数：min=0, max=1, mean=0.5, std=√(1/12)
    数学原理：output = 1 * input + 0
    物理意义：
        体素值通常表示占用概率（0-1范围），此归一化器保持原始数值不变
    使用场景：
        3D场景理解任务中的体素数据处理（如点云体素化表示）
    注意：
        与图像恒等归一化器(get_image_identity_normalizer)数学等价但语义场景不同
    返回：配置完成的SingleFieldLinearNormalizer实例
    """
    scale = np.array([1], dtype=np.float32)
    offset = np.array([0], dtype=np.float32)
    stat = {
        'min': np.array([0], dtype=np.float32),
        'max': np.array([1], dtype=np.float32),
        'mean': np.array([0.5], dtype=np.float32),
        'std': np.array([np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

# 图像范围归一化：像素值从[0,1]范围线性映射到[-1,1]
def get_image_range_normalizer():
    """
    功能：创建图像的标准化归一化器
    目的：将图像像素值从[0,1]范围线性映射到[-1,1]
    实现逻辑：
        scale = [2]（缩放系数）
        offset = [-1]（平移系数）
        预设统计参数：min=0, max=1, mean=0.5, std=√(1/12)
    数学原理：
        output = 2 * input - 1
        映射关系：0 → -1, 0.5 → 0, 1 → 1
    使用场景：
        图像数据预处理，符合深度学习模型常用输入范围
    注意：
        区别于恒等归一化器，此变换会改变原始像素值范围
    返回：配置完成的SingleFieldLinearNormalizer实例
    """
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        'min': np.array([0], dtype=np.float32),
        'max': np.array([1], dtype=np.float32),
        'mean': np.array([0.5], dtype=np.float32),
        'std': np.array([np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

# 图像恒等归一化：不做任何标准化变换
def get_image_identity_normalizer():
    """
    功能：创建图像的恒等归一化器​​
    ​​目的​​：生成一个对图像数据不做任何标准化变换的归一化器（即输入输出相同）。
    ​​实现逻辑​​：设置缩放系数 scale = 1，偏移量 offset = 0
    定义统计信息：
        min=0, max=1：表示像素值范围 [0,1]
        mean=0.5：均匀分布的期望值
        std=√(1/12)：均匀分布的标准差（方差=1/12）
    调用 SingleFieldLinearNormalizer.create_manual()创建归一化器
    ​​效果​​：任何输入图像通过此归一化器后，输出与原始输入完全一致（数学形式：output = 1 * input + 0）。
    ​​使用场景​​：
        当需要保持图像原始数值范围（如[0,1]）时使用，避免标准化操作改变像素分布。
    """
    scale = np.array([1], dtype=np.float32)
    offset = np.array([0], dtype=np.float32)
    stat = {
        'min': np.array([0], dtype=np.float32),
        'max': np.array([1], dtype=np.float32),
        'mean': np.array([0.5], dtype=np.float32),
        'std': np.array([np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

# 统计保持型恒等归一化器，不改变输入数据的归一化器，同时保留原始统计信息
def get_identity_normalizer_from_stat(stat):
    """
    功能：创建统计保持型恒等归一化器
    目的：生成不改变输入数据的归一化器（恒等变换），同时保留原始统计信息
    实现逻辑：
        scale = [1,1,...]（维度匹配统计最小值）
        offset = [0,0,...]（维度匹配统计最小值）
    数学原理：
        output = 1 * input + 0 = input
    使用场景：
        1. 调试阶段禁用归一化
        2. 需要保持原始数值范围的特殊字段（如二进制标志位）
        3. 数据预处理流水线的中性操作
    参数说明：
        stat：输入统计字典，仅用于确定维度信息
    返回：配置完成的SingleFieldLinearNormalizer实例
    """
    scale = np.ones_like(stat['min'])
    offset = np.zeros_like(stat['min'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

# Robomimic绝对动作归一化器，对机器人动作的不同部分（位置/旋转/夹持器）实施定制归一化策略
def robomimic_abs_action_normalizer_from_stat(stat, rotation_transformer):
    """
    功能：创建Robomimic绝对动作的分段归一化器
    目的：对机器人动作的不同部分（位置/旋转/夹持器）实施定制归一化策略
    实现逻辑：
        1. 动作分解：pos(3D), rot(3D), gripper(1D)
        2. 位置处理：标准范围归一化（同get_range_normalizer_from_stat）
        3. 旋转处理：通过rotation_transformer转换为向量后恒等归一化
        4. 夹持器处理：恒等归一化
        5. 合并参数：scale = concat([pos_scale, rot_scale, grip_scale])
    特殊设计：
        旋转参数通过外部转换器（如四元数/欧拉角→旋转6D）处理
    使用场景：
        Robomimic数据集的机器人动作建模，需分离处理位姿和夹持状态
    参数说明：
        stat：动作统计字典
        rotation_transformer：旋转变换器（需实现forward方法）
    返回：配置完成的SingleFieldLinearNormalizer实例
    """
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'rot': x[...,3:6],
            'gripper': x[...,6:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    def get_rot_param_info(stat):
        example = rotation_transformer.forward(stat['mean'])
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info
    
    def get_gripper_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    rot_param, rot_info = get_rot_param_info(result['rot'])
    gripper_param, gripper_info = get_gripper_param_info(result['gripper'])

    param = dict_apply_reduce(
        [pos_param, rot_param, gripper_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, rot_info, gripper_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )

# Robomimic位置敏感动作归一化器，仅对位置分量执行范围归一化，其他动作维度保持原样
def robomimic_abs_action_only_normalizer_from_stat(stat):
    """
    功能：创建Robomimic位置敏感动作归一化器
    目的：仅对位置分量执行范围归一化，其他动作维度保持原样
    实现逻辑：
        1. 动作分解：pos(3D), other(N维)
        2. 位置处理：标准范围归一化（同get_range_normalizer_from_stat）
        3. 其他分量：恒等归一化（scale=1, offset=0）
        4. 合并参数：scale = concat([pos_scale, other_scale])
    特殊设计：
        动作向量被解释为[位置, 姿态/配置]的组合
    使用场景：
        位置精度要求高但姿态精度要求低的机器人任务
    参数说明：
        stat：动作统计字典（需包含min/max/mean/std）
    返回：配置完成的SingleFieldLinearNormalizer实例
    """
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'other': x[...,3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    other_param, other_info = get_other_param_info(result['other'])

    param = dict_apply_reduce(
        [pos_param, other_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, other_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )

# Robomimic动作数据的绝对动作数据对称处理，仅对位置坐标的前两维（XY）进行对称归一化
def robomimic_abs_action_only_symmetric_normalizer_from_stat(stat):
    """
    功能：创建Robomimic动作数据的对称位置归一化器
    目的：针对机器人绝对动作数据，仅对位置坐标的前两维（XY）进行对称归一化
    实现逻辑：
        1. 分解动作：{'pos': [x,y,z], 'other': [旋转, 夹持器...]}
        2. 位置处理：对pos的前两维应用对称范围归一化（同get_range_symmetric_normalizer_from_stat）
        3. 其他部分：保持恒等变换（scale=1, offset=0）
        4. 重建参数：scale = concat([pos_scale, other_scale])
                   offset = concat([pos_offset, other_offset])
    特殊设计：
        仅作用于动作向量的前三维（位置），其他维度（旋转/夹持等）保持不变
        位置数据中仅XY维度对称处理，Z轴保持原始范围
    使用场景：
        Robomimic数据集的双臂机器人控制任务，需要对称平面运动的工作空间
    参数说明：
        stat：Robomimic动作统计字典（需包含min/max/mean/std）
    返回：配置完成的SingleFieldLinearNormalizer实例
    """
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'other': x[...,3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        abs_max = np.max([np.abs(stat['max'][:2]), np.abs(stat['min'][:2])])
        input_max[:2] = abs_max
        input_min[:2] = -abs_max
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    other_param, other_info = get_other_param_info(result['other'])

    param = dict_apply_reduce(
        [pos_param, other_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, other_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )

# 独立处理双臂系统的各机械臂动作分量
def robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat):
    """
    功能：创建双臂机器人的动作归一化器
    目的：独立处理双臂系统的各机械臂动作分量
    实现逻辑：
        1. 动作分解：左右臂的位置(pos0/pos1)和配置(other0/other1)
        2. 位置处理：对各臂位置执行标准范围归一化
        3. 配置处理：各臂其他分量恒等归一化
        4. 参数重建：scale = concat([pos0_scale, other0_scale, pos1_scale, other1_scale])
    数学表示：
        output = [
            scale_pos0 * pos0 + offset_pos0,
            other0,
            scale_pos1 * pos1 + offset_pos1,
            other1
        ]
    使用场景：
        Robomimic数据集的双臂机器人任务（如bimanual manipulation）
    参数说明：
        stat：动作统计字典，维度需为双臂动作总维度
    返回：配置完成的SingleFieldLinearNormalizer实例
    """
    Da = stat['max'].shape[-1]
    Dah = Da // 2
    result = dict_apply_split(
        stat, lambda x: {
            'pos0': x[...,:3],
            'other0': x[...,3:Dah],
            'pos1': x[...,Dah:Dah+3],
            'other1': x[...,Dah+3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos0_param, pos0_info = get_pos_param_info(result['pos0'])
    pos1_param, pos1_info = get_pos_param_info(result['pos1'])
    other0_param, other0_info = get_other_param_info(result['other0'])
    other1_param, other1_info = get_other_param_info(result['other1'])

    param = dict_apply_reduce(
        [pos0_param, other0_param, pos1_param, other1_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos0_info, other0_info, pos1_info, other1_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )



# 从数据数组计算基本统计量
def array_to_stats(arr: np.ndarray):
    """
    功能：从数据数组计算基本统计量
    目的：为归一化器创建提供必要的统计信息
    实现逻辑：
        按列（特征维度）计算：
          min = 最小值
          max = 最大值
          mean = 平均值
          std = 标准差
    数学原理：
        min = min(arr, axis=0)
        std = sqrt(mean((arr - mean)^2))
    使用场景：
        数据集预处理阶段，为归一化器提供输入统计
    参数说明：
        arr：二维数组[样本数, 特征维度]
    返回：统计字典{'min':, 'max':, 'mean':, 'std':}
    """
    stat = {
        'min': np.min(arr, axis=0),
        'max': np.max(arr, axis=0),
        'mean': np.mean(arr, axis=0),
        'std': np.std(arr, axis=0)
    }
    return stat

# 沿特征维度拼接多个归一化器
# 输出 = 各字段归一化结果的拼接
# 动态拼接scale/offset参数
def concatenate_normalizer(normalizers: list):
    """
    功能：沿特征维度拼接多个归一化器​​
    ​​目的​​：将多个单字段归一化器合并为一个能处理多字段数据的统一归一化器。
    ​​实现逻辑​​：
        ​​提取参数​​：
            从每个输入归一化器中提取 scale、offset和统计字典 input_stats
        ​​沿最后一维拼接​​：
            缩放系数：torch.cat([norm1.scale, norm2.scale, ...], dim=-1)
            偏移量：torch.cat([norm1.offset, norm2.offset, ...], dim=-1)
            统计信息：对每个统计量（min/max/mean/std）分别拼接：
                input_stats_dict = {
                    'min': torch.cat([norm1.min, norm2.min], dim=-1),
                    'max': torch.cat([norm1.max, norm2.max], dim=-1),
                    ... 
                }
        ​​创建新归一化器​​：使用拼接后的参数调用 SingleFieldLinearNormalizer.create_manual()
    ​​数学表示​​：若有两个归一化器：output1 = scale1 * input1 + offset1output2 = scale2 * input2 + offset2则合并后：output_combined = concat(scale1, scale2) * concat(input1, input2) + concat(offset1, offset2)
    使用场景​​：
        当处理多模态或多字段数据（如拼接图像特征+关节角度+传感器数据）时，将各字段独立的归一化器合并为统一的处理模块。
    """
    scale = torch.concatenate([normalizer.params_dict['scale'] for normalizer in normalizers], axis=-1)
    offset = torch.concatenate([normalizer.params_dict['offset'] for normalizer in normalizers], axis=-1)
    input_stats_dict = dict_apply_reduce(
        [normalizer.params_dict['input_stats'] for normalizer in normalizers], 
        lambda x: torch.concatenate(x,axis=-1))
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=input_stats_dict
    )