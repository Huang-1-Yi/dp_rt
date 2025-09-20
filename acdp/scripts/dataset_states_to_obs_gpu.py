"""
​​环境缓存机制 (--use_env_cache)​​
    使用LRU缓存避免重复初始化环境
    通过@lru_cache装饰器实现
    显著减少环境初始化开销
​​GPU加速支持 (--use_gpu)​​
    自动检测GPU可用性
    使用PyTorch进行图像数据处理
    多进程自动分配不同GPU设备
​​警告抑制 (--suppress_warnings)​​
    减少robosuite/robomimic的警告输出
    使用Python logging模块控制日志级别
​​进度显示增强​​
    添加批次处理进度信息
    更清晰的处理状态反馈
"""

import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
import logging  # 添加日志控制

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# ==== 新增：环境缓存机制 ====
from functools import lru_cache

@lru_cache(maxsize=32)
def create_cached_env(env_meta_json, camera_names, height, width, shaped):
    """
    带缓存的环境创建函数，避免重复初始化
    """
    env_meta = json.loads(env_meta_json)
    return EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=camera_names,
        camera_height=height,
        camera_width=width,
        reward_shaping=shaped
    )

# ==== 新增：GPU加速支持 ====
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

def process_obs_with_gpu(obs_dict, device):
    """
    使用GPU加速处理图像观测数据
    """
    for key in obs_dict:
        if "image" in key or "depth" in key:
            # 转换为PyTorch tensor并发送到GPU
            obs_dict[key] = torch.tensor(obs_dict[key]).to(device)
    return obs_dict

def extract_trajectory(
    env_meta,
    args, 
    initial_state, 
    states, 
    actions,
):
    done_mode = args.done_mode
    
    # ==== 修改：使用缓存环境或原始方式 ====
    if args.use_env_cache:
        # 使用缓存环境
        env = create_cached_env(
            json.dumps(env_meta),  # 序列化为可哈希类型
            tuple(args.camera_names), 
            args.camera_height,
            args.camera_width,
            args.shaped
        )
    else:
        # 原始环境创建方式
        if env_meta['env_name'].startswith('PickPlace'):
            camera_names=['birdview', 'agentview', 'robot0_eye_in_hand']
        else:
            camera_names=['birdview', 'agentview', 'sideview', 'robot0_eye_in_hand']
        
        env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            camera_names=camera_names, 
            camera_height=args.camera_height, 
            camera_width=args.camera_width, 
            reward_shaping=args.shaped,
        )
    
    assert states.shape[0] == actions.shape[0]

    # ==== 新增：GPU设备设置 ====
    gpu_device = None
    if args.use_gpu and GPU_AVAILABLE:
        # 为每个进程分配不同的GPU（如果可用）
        try:
            gpu_id = multiprocessing.current_process()._identity[0] % torch.cuda.device_count()
            gpu_device = torch.device(f"cuda:{gpu_id}")
        except:
            gpu_device = torch.device("cuda:0")

    # 加载初始状态
    env.reset()
    obs = env.reset_to(initial_state)
    
    # ==== 新增：GPU处理初始观测 ====
    if args.use_gpu and GPU_AVAILABLE and gpu_device:
        obs = process_obs_with_gpu(obs, gpu_device)

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]
    
    # 迭代处理轨迹
    for t in range(1, traj_len + 1):
        # 获取下一个观测
        if t == traj_len:
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            next_obs = env.reset_to({"states": states[t]})
        
        # ==== 新增：GPU处理下一个观测 ====
        if args.use_gpu and GPU_AVAILABLE and gpu_device:
            next_obs = process_obs_with_gpu(next_obs, gpu_device)

        # 推断奖励信号
        r = env.get_reward()

        # 推断完成信号
        done = False
        if done_mode in (1, 2):
            done = done or (t == traj_len)
        if done_mode in (0, 2):
            done = done or env.is_success()["task"]
        done = int(done)

        # 收集转换
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # 更新下一个迭代
        obs = deepcopy(next_obs)

    # 转换数据结构
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # 列表转numpy数组
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                # ==== 修改：处理GPU tensor ====
                if args.use_gpu and GPU_AVAILABLE and isinstance(traj[k][kp], torch.Tensor):
                    traj[k][kp] = traj[k][kp].cpu().numpy()
                else:
                    traj[k][kp] = np.array(traj[k][kp])
        else:
            if args.use_gpu and GPU_AVAILABLE and isinstance(traj[k], torch.Tensor):
                traj[k] = traj[k].cpu().numpy()
            else:
                traj[k] = np.array(traj[k])

    return traj

def worker(x):
    env_meta, args, initial_state, states, actions = x
    traj = extract_trajectory(
        env_meta=env_meta,
        args=args,
        initial_state=initial_state, 
        states=states, 
        actions=actions,
    )
    return traj

def dataset_states_to_obs(args):
    # ==== 新增：抑制日志输出 ====
    if args.suppress_warnings:
        logging.getLogger("robosuite").setLevel(logging.ERROR)
        logging.getLogger("robomimic").setLevel(logging.ERROR)
        os.environ["ROBOMIMIC_SUPPRESS_WARNINGS"] = "1"

    num_workers = args.num_workers
    
    # 创建数据处理环境
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.input)
    
    # ==== 修改：使用缓存环境或原始方式 ====
    if args.use_env_cache:
        # 使用缓存环境（主进程也需要初始化）
        _ = create_cached_env(
            json.dumps(env_meta),
            tuple(args.camera_names), 
            args.camera_height,
            args.camera_width,
            args.shaped
        )
        env = None  # 不需要实际使用
    else:
        # 原始环境创建方式
        if env_meta['env_name'].startswith('PickPlace'):
            camera_names=['birdview', 'agentview', 'robot0_eye_in_hand']
        else:
            camera_names=['birdview', 'agentview', 'sideview', 'robot0_eye_in_hand']
        
        env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            camera_names=camera_names, 
            camera_height=args.camera_height, 
            camera_width=args.camera_width, 
            reward_shaping=args.shaped,
        )

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env_meta, indent=4))
    print("")

    # 确定是否是robosuite环境
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # 获取所有演示片段
    f = h5py.File(args.input, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # 可能减少演示数量
    if args.n is not None:
        demos = demos[:args.n]

    # 输出文件
    output_path = args.output
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print(f"input file: {args.input}")
    print(f"output file: {output_path}")

    total_samples = 0
    for i in range(0, len(demos), num_workers):
        end = min(i + num_workers, len(demos))
        initial_state_list = []
        states_list = []
        actions_list = []
        for j in range(i, end):
            ep = demos[j]
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            if is_robosuite_env:
                initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            actions = f["data/{}/actions".format(ep)][()]

            initial_state_list.append(initial_state)
            states_list.append(states)
            actions_list.append(actions)
        
        # ==== 新增：进度显示 ====
        print(f"Processing trajectories {i} to {end-1} of {len(demos)}...")
            
        with multiprocessing.Pool(num_workers) as pool:
            worker_args = [
                [env_meta, args, initial_state_list[j], states_list[j], actions_list[j]] 
                for j in range(len(initial_state_list))
            ]
            trajs = pool.map(worker, worker_args)

        for j, ind in enumerate(range(i, end)):
            ep = demos[ind]
            traj = trajs[j]
            if args.copy_rewards:
                traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            if args.copy_dones:
                traj["dones"] = f["data/{}/dones".format(ep)][()]

            ep_data_grp = data_grp.create_group(ep)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            
            for k in traj["obs"]:
                if args.compress:
                    ep_data_grp.create_dataset(f"obs/{k}", data=np.array(traj["obs"][k]), compression="gzip")
                else:
                    ep_data_grp.create_dataset(f"obs/{k}", data=np.array(traj["obs"][k]))
                
                if not args.exclude_next_obs:
                    if args.compress:
                        ep_data_grp.create_dataset(f"next_obs/{k}", data=np.array(traj["next_obs"][k]), compression="gzip")
                    else:
                        ep_data_grp.create_dataset(f"next_obs/{k}", data=np.array(traj["next_obs"][k]))

            if is_robosuite_env:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
            total_samples += traj["actions"].shape[0]
            print(f"ep {ind}: wrote {ep_data_grp.attrs['num_samples']} transitions to group {ep}")
        
        del trajs

    if "mask" in f:
        f.copy("mask", f_out)

    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env_meta, indent=4)
    print(f"Wrote {len(demos)} trajectories to {output_path}")

    f.close()
    f_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=84,
        help="(optional) height of image observations",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
        help="(optional) width of image observations",
    )
    parser.add_argument(
        "--done_mode",
        type=int,
        default=2,
        help="how to write done signal (0, 1, or 2)",
    )
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file",
    )
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file",
    )
    parser.add_argument(
        "--exclude-next-obs", 
        type=bool,
        default=True,
        help="(optional) exclude next obs in dataset",
    )
    parser.add_argument(
        "--compress", 
        type=bool,
        default=True,
        help="(optional) compress observations with gzip",
    )
    
    # ==== 新增：优化选项 ====
    parser.add_argument(
        "--use_env_cache",
        action='store_true',
        help="(optional) use environment caching to reduce initialization overhead",
    )
    parser.add_argument(
        "--use_gpu",
        action='store_true',
        help="(optional) use GPU acceleration for image processing",
    )
    parser.add_argument(
        "--suppress_warnings",
        action='store_true',
        help="(optional) suppress robosuite/robomimic warning messages",
    )

    args = parser.parse_args()
    
    # ==== 新增：GPU可用性检查 ====
    if args.use_gpu:
        if not GPU_AVAILABLE:
            print("WARNING: --use_gpu requested but GPU not available. Disabling GPU acceleration.")
            args.use_gpu = False
        else:
            print(f"Using GPU acceleration with {torch.cuda.device_count()} devices")
    
    dataset_states_to_obs(args)