from typing import Dict, Callable, Tuple, List
import numpy as np
import collections
from diffusion_policy.common.cv2_util import get_image_transform,get_mask_transform
from diffusion_policy.common.pose_repr_util import (
    compute_relative_pose, 
    convert_pose_mat_rep
)
from diffusion_policy.common.pose_util import (
    pose_to_mat, mat_to_pose, 
    mat_to_pose10d, pose10d_to_mat)

"""
您提供的 get_real_obs_resolution 函数的目的是从给定的 shape_meta 字典中提取观测(observation)的分辨率。
这个函数假定 shape_meta 字典包含一个 'obs' 键，该键对应的值是一个字典，其中包含关于不同类型观测的信息。
函数遍历这些观测信息，当找到类型为 'rgb' 的观测时，它会提取其高度(ho)和宽度(wo)，并确保所有 'rgb' 类型的观测具有相同的分辨率。
"""
def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
            # print(f'观测obs {key} shape: {this_data_in.shape}')
    return obs_dict_np

def mask_get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    
    # 创建图像转换器的缓存字典
    transform_cache = {}
    
    for key, attr in obs_shape_meta.items():
        data_type = attr.get('type', 'low_dim')
        target_shape = tuple(attr['shape'])
        
        # 获取原始数据并验证基础形状
        raw_data = env_obs[key]
        t = raw_data.shape[0]  # 时间步数
        
        if data_type == 'rgb':
            # 原有RGB处理逻辑保持不变
            hi, wi = raw_data.shape[1:3]
            co, ho, wo = target_shape
            
            # 生成转换器缓存键
            transform_key = ('rgb', wi, hi, wo, ho)
            if transform_key not in transform_cache:
                transform_cache[transform_key] = get_image_transform(
                    input_res=(wi, hi),
                    output_res=(wo, ho),
                    bgr_to_rgb=False
                )
            tf = transform_cache[transform_key]
            
            processed = np.stack([tf(x) for x in raw_data])
            if raw_data.dtype == np.uint8:
                processed = processed.astype(np.float32) / 255
                
            # THWC -> TCHW
            obs_dict_np[key] = np.moveaxis(processed, -1, 1)

        elif data_type == 'mask':
            # 新增Mask处理分支
            # 输入形状验证（允许单通道或没有通道维度）
            if raw_data.ndim == 4 and raw_data.shape[-1] == 1:
                raw_data = raw_data.squeeze(-1)  # 去除冗余通道维度
                
            # 获取目标形状参数
            c_out, ho, wo = target_shape
            assert c_out == 1, f"Mask {key} 的通道数必须为1"
            
            # 尺寸转换逻辑
            hi, wi = raw_data.shape[1:3]
            if (hi != ho) or (wi != wo):
                # 使用独立缓存键（不同插值方法）
                transform_key = ('mask', wi, hi, wo, ho)
                if transform_key not in transform_cache:
                    transform_cache[transform_key] = get_mask_transform(
                        input_res=(wi, hi),
                        output_res=(wo, ho)
                    )
                tf = transform_cache[transform_key]
                processed = np.stack([tf(x) for x in raw_data])
            else:
                processed = raw_data

            # 数据类型处理（保持与参考代码一致）
            processed = processed.astype(np.float32)
            if raw_data.dtype == np.uint8:
                processed = processed / 255.0

            # 通道维度处理（与参考代码逻辑相同）
            if processed.ndim == 3:  # (T,H,W)
                processed = np.expand_dims(processed, axis=-1)  # (T,H,W,1)
                
            # THWC -> TCHW
            obs_dict_np[key] = np.moveaxis(processed, -1, 1)

        elif data_type == 'low_dim':
            # 保持原有低维数据处理
            obs_dict_np[key] = raw_data.astype(np.float32)

        else:
            raise ValueError(f"未知的观测类型: {data_type} (key: {key})")

    return obs_dict_np

# 接收一个字典 env_obs，它包含环境观测数据；返回一个字典 obs_dict_np，它包含处理后的观测数据
# 一个字典 shape_meta，它包含观测数据的形状元数据，以及其他一些参数
def get_real_umi_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        obs_pose_repr: str='abs',
        tx_robot1_robot0: np.ndarray=None,
        episode_start_pose: List[np.ndarray]=None,
        ) -> Dict[str, np.ndarray]:
    # print("env_obs ==", env_obs)
    obs_dict_np = dict()
    # process non-pose
    """
    获取 shape_meta 字典中的 'obs' 部分，然后遍历这些键和属性。
    对于 'rgb' 类型的数据，它会处理图像数据；
    对于 'low_dim' 类型的数据，并且不是末端执行器的数据，它会直接添加到 obs_dict_np 中
    同时，它还会处理多机器人情况，记录每个机器人的键
    """
    obs_shape_meta = shape_meta['obs']
    robot_prefix_map = collections.defaultdict(list)
    # expected_keys = {
    #     'robot0_eef_pos': [0, 0, 0],
    #     'robot0_eef_rot_axis_angle': [0, 0, 0, 1],
    #     'robot0_eef_rot_axis_angle_wrt_start': [0, 0, 0, 1],
    #     'robot0_gripper_width': [0]
    # }
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim' and ('eef' not in key):
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
            # 处理多机器人情况 handle multi-robots
            ks = key.split('_')
            if ks[0].startswith('robot'):
                robot_prefix_map[ks[0]].append(key)

    # 生成相对姿态——将每个机器人末端执行器的姿态转换为矩阵，计算相对于自身末端执行器的相对姿态，添加到 obs_dict_np 中
    for robot_prefix in robot_prefix_map.keys():
        # 转换姿态为矩阵
        pose_mat = pose_to_mat(np.concatenate([
            env_obs[robot_prefix + '_eef_pos'],
            env_obs[robot_prefix + '_eef_rot_axis_angle']
        ], axis=-1))
        # pose_mat = pose_to_mat(np.concatenate([
        #     env_obs.get(robot_prefix + '_eef_pos', np.zeros((1, 3))),
        #     env_obs.get(robot_prefix + '_eef_rot_axis_angle', np.zeros((1, 4)))
        # ], axis=-1))
        # solve reltaive obs
        obs_pose_mat = convert_pose_mat_rep(
            pose_mat, 
            base_pose_mat=pose_mat[-1],
            pose_rep=obs_pose_repr,
            backward=False)

        obs_pose = mat_to_pose10d(obs_pose_mat)
        obs_dict_np[robot_prefix + '_eef_pos'] = obs_pose[...,:3]
        obs_dict_np[robot_prefix + '_eef_rot_axis_angle'] = obs_pose[...,3:]
    
    # 生成相对于其他机器人的姿态——使用一个变换矩阵 tx_robot1_robot0 来转换姿态，然后计算相对于另一个机器人的相对姿态。这个相对姿态也被添加到 obs_dict_np 中
    n_robots = len(robot_prefix_map)
    for robot_id in range(n_robots):
        # 转换姿态为矩阵
        assert f'robot{robot_id}' in robot_prefix_map
        tx_robota_tcpa = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_id}_eef_pos'],
            env_obs[f'robot{robot_id}_eef_rot_axis_angle']
        ], axis=-1))
        for other_robot_id in range(n_robots):
            if robot_id == other_robot_id:
                continue
            tx_robotb_tcpb = pose_to_mat(np.concatenate([
                env_obs[f'robot{other_robot_id}_eef_pos'],
                env_obs[f'robot{other_robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            # tx_robotb_tcpb = pose_to_mat(np.concatenate([
            #     env_obs.get(f'robot{other_robot_id}_eef_pos', np.zeros((1, 3))),
            #     env_obs.get(f'robot{other_robot_id}_eef_rot_axis_angle', np.zeros((1, 4)))
            # ], axis=-1))
            tx_robota_robotb = tx_robot1_robot0
            if robot_id == 0:
                tx_robota_robotb = np.linalg.inv(tx_robot1_robot0)
            tx_robota_tcpb = tx_robota_robotb @ tx_robotb_tcpb

            rel_obs_pose_mat = convert_pose_mat_rep(
                tx_robota_tcpa,
                base_pose_mat=tx_robota_tcpb[-1],
                pose_rep='relative',
                backward=False)
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            obs_dict_np[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]

    # generate relative pose with respect to episode start
    # 计算每个机器人相对于初始姿态的相对姿态。它使用一个起始姿态列表 episode_start_pose 来计算相对于初始姿态的相对姿态。这个相对姿态也被添加到 obs_dict_np 中。
    if episode_start_pose is not None:
        for robot_id in range(n_robots):        
            # convert pose to mat
            # 获取机器人的末端执行器位置和旋转角，并转换为姿态矩阵
            pose_mat = pose_to_mat(np.concatenate([
                env_obs[f'robot{robot_id}_eef_pos'],
                env_obs[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            # pose_mat = pose_to_mat(np.concatenate([
            #     env_obs.get(f'robot{robot_id}_eef_pos', np.zeros((1, 3))),
            #     env_obs.get(f'robot{robot_id}_eef_rot_axis_angle', np.zeros((1, 4)))
            # ], axis=-1))
            # get start pose
            # 获取每个机器人的初始姿态
            start_pose = episode_start_pose[robot_id]
            # 将初始姿态转换为姿态矩阵
            start_pose_mat = pose_to_mat(start_pose)
            # 计算相对于初始姿态的相对姿态矩阵
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)
            # 将相对姿态矩阵转换为 10 维姿态表示
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # obs_dict_np[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            # 存储相对于初始姿态的末端执行器的相对旋转轴角
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]
            # print("obs_dict_np==",obs_dict_np)
    # 函数返回处理后的观测数据字典 obs_dict_np
    return obs_dict_np

def get_real_umi_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs'
    ):

    n_robots = int(action.shape[-1] // 10)
    env_action = list()
    for robot_idx in range(n_robots):
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_idx}_eef_pos'][-1],
            env_obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1]
        ], axis=-1))

        start = robot_idx * 10
        action_pose10d = action[..., start:start+9]
        action_grip = action[..., start+9:start+10]
        action_pose_mat = pose10d_to_mat(action_pose10d)

        # solve relative action
        action_mat = convert_pose_mat_rep(
            action_pose_mat, 
            base_pose_mat=pose_mat,
            pose_rep=action_pose_repr,
            backward=True)

        # convert action to pose
        action_pose = mat_to_pose(action_mat)
        env_action.append(action_pose)
        env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    return env_action
