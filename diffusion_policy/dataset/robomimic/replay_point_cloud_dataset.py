"""
这段代码定义了一个自定义的数据集类 RobomimicReplayPointCloudDataset，它继承自 BaseImageDataset 类，目的是将数据从 HDF5 文件转换为 Zarr 格式，并提供数据的样本生成、归一化等功能。它主要用于处理包含点云数据和图像数据的训练数据集。
数据处理与转换：RobomimicReplayPointCloudDataset 类负责从 HDF5 数据文件中加载原始数据，进行必要的转换（如旋转转换、数据标准化等），并将其存储为 Zarr 格式。它支持缓存机制，能够避免重复的计算。
并行化处理：使用 ThreadPoolExecutor 和 multiprocessing 来并行化数据的加载和转换，确保高效处理大规模数据。
数据集划分与采样：通过 SequenceSampler 和 get_val_mask 方法，数据集被划分为训练集和验证集，并支持按序列批量采样
其中，数据存储和并行处理
    使用 zarr 存储：数据被存储为 Zarr 格式，这是一种非常高效的多维数据存储格式，适用于大规模数据集。
    并行处理：在 _convert_point_cloud_to_replay 中，使用 concurrent.futures.ThreadPoolExecutor 来并行处理图像和点云数据，最大化 I/O 性能，减少处理时间。
"""

from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_voxel_identity_normalizer,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)

from matplotlib import pyplot as plt
from plyfile import PlyData, PlyElement
from tqdm import tqdm


register_codecs()

class RobomimicReplayPointCloudDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            n_demo=100,
            save_dir="/home/hy/equidiff/test/test_output",
        ):
        """
        初始化数据集类的一些关键参数：
            参数：
                shape_meta：形状元数据，用来描述数据的维度和类型。
                dataset_path：数据集的路径。
                horizon, pad_before, pad_after：数据的处理范围和前后填充。
                n_obs_steps：观察步数，控制从数据中选择多少步的观察。
                abs_action：是否使用绝对动作（例如，位置和旋转数据被合并为单一动作）。
                use_cache：是否使用缓存，以避免每次都重新处理数据。
                seed, val_ratio, n_demo：用于控制随机性、验证集比例以及使用的演示数量。
            关键逻辑：
                缓存机制：如果 use_cache=True，则使用 zarr 格式的缓存。程序首先检查缓存是否存在，如果不存在，程序将处理数据并保存到缓存中，若缓存已存在，直接从缓存加载数据。
                数据转换：通过 _convert_point_cloud_to_replay 函数，将点云数据和其他数据转换为 ReplayBuffer 格式，并将其存储在 replay_buffer 中。
                分配数据类型：根据 shape_meta 中的元数据，将数据分为三类：rgb_keys（RGB图像数据），pc_keys（点云数据）和 lowdim_keys（低维数据）。
        """
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir) and self.save_dir is not None:
            os.makedirs(self.save_dir)  # Ensure the output directory exists

        self.n_demo = n_demo
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + f'.{n_demo}.' + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_point_cloud_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer,
                            n_demo=n_demo)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_point_cloud_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer,
                n_demo=n_demo)

        rgb_keys = list()
        pc_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            if type == 'point_cloud':
                pc_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + pc_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.pc_keys = pc_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

    def get_validation_dataset(self):
        """
        返回一个新的数据集实例，用于验证集，区别在于它的 sampler 会使用验证集的掩码 train_mask 的反值（即 ~self.train_mask），确保使用不同的数据进行验证。
        """
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs) -> LinearNormalizer:
        """
        该方法根据存储的数据计算并返回一个 LinearNormalizer 对象，该对象用于对数据进行标准化。主要步骤：
            从 replay_buffer 中提取所需数据：动作（action）、机器人末端执行器位置（robot0_eef_pos）、机器人末端执行器四元数（robot0_eef_quat）等。
            使用 LinearNormalizer 对这些数据进行标准化。
            SingleFieldLinearNormalizer.create_identity() 被注释掉了，说明原本可能考虑对点云数据不做归一化（即保持原数据），但这一行被注释掉了，表明可能会在某些情况下使用自定义的标准化方法。
        """
        data = {
            'action': self.replay_buffer['action'],
            'robot0_eef_pos': self.replay_buffer['robot0_eef_pos'][...,:],
            'robot0_eef_quat': self.replay_buffer['robot0_eef_quat'][...,:],
            'robot0_gripper_qpos': self.replay_buffer['robot0_gripper_qpos'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """
        从 replay_buffer 中提取所有的动作数据，并返回一个 torch.Tensor 类型的对象
        """
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        """
        __len__：返回数据集的长度，等于 self.sampler 的长度，self.sampler 是用于从 replay_buffer 中抽取样本的工具。
        """
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本，索引 idx 对应的样本数据，包括图像、点云和低维数据。每个数据样本被转换为 torch.Tensor 类型，并返回一个字典形式的数据。
        """
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1).astype(np.float32) / 255.
            
            # # save image
            # if self.save_dir is not None:
            #     # 检查图像的通道数是否是 1（灰度图像）、3（RGB图像）或 4（RGBA图像）。这三种格式是 matplotlib 支持的格式
            #     # Check if image is 3D with the last dimension being the channel dimension
            #     if obs_dict[key].ndim == 3 and obs_dict[key].shape[-1] in [1, 3, 4]:
            #         # If it's grayscale (1 channel), we can make it 3 channels by repeating the grayscale channel
            #         # 沿着最后一个维度（即颜色通道维度）将单通道的灰度图像重复 3 次，创建一个具有 3 个相同通道的 RGB 图像
            #         if obs_dict[key].shape[-1] == 1:
            #             obs_dict[key] = np.repeat(obs_dict[key], 3, axis=-1)  # Convert grayscale to RGB
            #     else:
            #         # 如果图像的通道数既不是 1、3，也不是 4，则抛出一个错误，提示图像的形状不符合预
            #         raise ValueError(f"Unexpected image shape: {obs_dict[key].shape}")
            #     image_path = os.path.join(self.save_dir, f"image_{key}_{idx}.png")
            #     plt.imsave(image_path, obs_dict[key][0])  # Save only the first frame of the image (T=0)
            
            
            # save image
            
            # if idx ==0 and self.save_dir is not None:
            #     if key =="agentview_image":
            #         image_data = obs_dict[key]
            #         # 如果是两张拼接的图像 (2, 3, 84, 84)
            #         if image_data.ndim == 4 and image_data.shape[0] == 2 and image_data.shape[1] == 3:
            #             # 分离成两张图像
            #             for i in range(2):
            #                 image = image_data[i]  # 取每一张图像
            #                 image_path = os.path.join(self.save_dir, f"{key}_image_{idx}_{i}.png")
            #                 plt.imsave(image_path, np.moveaxis(image, 0, -1))  # 转换为 [H, W, C] 格式保存
            #                 print(f"Saved image {i} for index {idx} to {image_path}")
            #         else:
            #             # 如果只有一张图像
            #             image_path = os.path.join(self.save_dir, f"{key}_image_{idx}.png")
            #             plt.imsave(image_path, np.moveaxis(image_data[0], 0, -1))  # 直接保存
            #             print(f"Saved image for index {idx} to {image_path}")
            # # T,C,H,W
            del data[key]
        for key in self.pc_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)

            # # save point cloud
            # if idx ==0 and self.save_dir is not None:
            #     pc_path = os.path.join(self.save_dir, f"point_cloud_{key}_{idx}.ply")
            #     self.save_point_cloud_to_ply(obs_dict[key][0], pc_path)  # Save only the first frame of the point cloud (T=0)
            
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    """
    转换原始的动作数据，特别是当使用绝对动作时，分离位置、旋转和抓取信息，并根据所选的旋转表示法（如 rotation_6d）进行旋转转换。
    """
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions


def _convert_point_cloud_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None, n_demo=100):
    """
    将原始的点云和动作数据转换为 ReplayBuffer 格式，并存储在 Zarr 文件中。此过程涉及读取 HDF5 文件，提取图像、点云和动作数据，并将它们压缩存储。使用 ThreadPoolExecutor 并行处理图像和点云数据，以提高性能
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    pc_keys = list()
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'point_cloud':
            pc_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        n_demo = min(n_demo, len(demos))
        for i in range(n_demo):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(n_demo):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                print(f"this_data.shape: {this_data.shape}")
                print(f"Expected shape: {(n_steps,)} + {tuple(shape_meta['action']['shape'])}")
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        def pc_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
            
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(n_demo):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))
        
        with tqdm(total=n_steps*len(pc_keys), desc="Loading point cloud data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in pc_keys:
                    data_key = key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    n, c = shape
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, n, c),
                        chunks=(1, n, c),
                        dtype=np.float32
                    )
                    for episode_idx in range(n_demo):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(pc_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

def normalizer_from_stat(stat):
    """
    根据统计数据（如最大值和最小值）创建一个标准化器，用于规范化数据，使其具有指定的范围
    """
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def save_point_cloud_to_ply(self, point_cloud, file_path):
    """
    将点云数据保存为 PLY 文件。
    """
    # 假设点云是 N x 3 数组 (每个点有 x, y, z 坐标)
    vertices = np.array([(x, y, z) for x, y, z in point_cloud], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # 创建 PLY 元素
    vertex_element = PlyElement.describe(vertices, 'vertex')

    # 保存为 PLY 文件
    ply_data = PlyData([vertex_element])
    ply_data.write(file_path)
    print(f"Point cloud saved to {file_path}")