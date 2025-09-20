import copy
import json
import hashlib
import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil
import cv2
from omegaconf import OmegaConf

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.real_data_conversion import real_data_to_replay_buffer
from umi.common.pose_util import pose_to_mat, mat_to_pose10d

register_codecs()

class UmiDataset(BaseDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        repeat_frame_prob: float=0.0,
        seed: int=42,
        val_ratio: float=0.0,
        max_duration: Optional[float]=None,
        use_ratio: float = 1.0,
        dataset_idx: Optional[str] = None,
        use_cache: bool=False  # 新增：启用缓存机制
    ):
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')
        
        # === 新增：数据转换功能 ===
        if use_cache:
            # 生成缓存文件名
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(cache_dir, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # 缓存不存在，创建新缓存
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = self._get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore()
                        )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = self._get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore()
            )
        # === 结束新增 ===
        
        self.num_robot = 0
        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

            if key.endswith('eef_pos'):
                self.num_robot += 1

            # solve obs_horizon
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            # solve latency_steps
            latency_steps = shape_meta['obs'][key]['latency_steps']
            key_latency_steps[key] = latency_steps

            # solve down_sample_steps
            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # solve action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']
        if dataset_idx is None:
            if val_ratio == 0.0:
                val_mask = np.zeros(replay_buffer.n_episodes, dtype=bool)
            else:
                val_mask = get_val_mask(
                    n_episodes=replay_buffer.n_episodes, 
                    val_ratio=val_ratio,
                    seed=seed
                )
        
            train_mask = ~val_mask
        assert use_ratio <= 1.0

        if dataset_idx is not None:
            dataset_name_len = len(dataset_path.split('/')[-1])
            with open(os.path.join(dataset_path[:-dataset_name_len], 'count.txt'), 'r') as f:
                counts = f.readlines()
            counts = [int(count[:-1]) if count[-1] == '\n' else int(count) for count in counts]
            counts = [0] + list(np.cumsum(counts))
            dataset_idx = dataset_idx.split(',')
            train_mask, val_mask = np.zeros(replay_buffer.n_episodes, dtype=bool), np.zeros(replay_buffer.n_episodes, dtype=bool)
            dataset_idx_list = []
            for idx in dataset_idx:
                if '-' not in idx:
                    dataset_start_idx, dataset_end_idx = int(idx), int(idx)
                else:
                    dataset_start_idx, dataset_end_idx = int(idx.split('-')[0]), int(idx.split('-')[1])
                train_mask[counts[dataset_start_idx - 1] : counts[dataset_end_idx]] = True
                dataset_idx_list += list(range(dataset_start_idx, dataset_end_idx + 1))
            true_indices = np.where(train_mask == True)[0]
            rng = np.random.default_rng(seed=seed)
            val_num = round(len(true_indices) * val_ratio)
            val_idx = rng.choice(true_indices, val_num, replace=False)
            train_mask[val_idx] = False
            val_mask[val_idx] = True
            print('dataset index: ', dataset_idx_list)
            if use_ratio < 1.0:
                print('use train ratio:', use_ratio)
                for i in range(len(counts) - 1):
                    if i + 1 in dataset_idx_list:
                        now_mask = train_mask[counts[i] : counts[i+1]]
                        now_mask_true = np.where(now_mask == True)[0]
                        sample_num = max(round(len(now_mask_true) * use_ratio), 1)
                        sample_index = rng.choice(now_mask_true, sample_num, replace=False)
                        train_mask[counts[i] : counts[i+1]] = False
                        train_mask[counts[i] + sample_index] = True

        elif dataset_idx is None and use_ratio < 1.0:
            print('use train ratio:', use_ratio)
            dataset_name_len = len(dataset_path.split('/')[-1])
            with open(os.path.join(dataset_path[:-dataset_name_len], 'count.txt'), 'r') as f:
                counts = f.readlines()
            counts = [int(count[:-1]) if count[-1] == '\n' else int(count) for count in counts]
            counts = [0] + list(np.cumsum(counts))
            for i in range(len(counts) - 1):
                now_mask = train_mask[counts[i] : counts[i+1]]
                now_mask_true = np.where(now_mask == True)[0]
                rng = np.random.default_rng(seed=seed)
                sample_num = max(1, round(len(now_mask_true) * use_ratio))
                sample_index = rng.choice(now_mask_true, sample_num, replace=False)
                train_mask[counts[i] : counts[i+1]] = False
                train_mask[counts[i] + sample_index] = True

        print('use total episode number:', train_mask.shape[0])
        print('use training episode number:', (train_mask == True).sum())

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key:
                self.sampler_lowdim_keys.append(key)
    
        for key in replay_buffer.keys():
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                self.sampler_lowdim_keys.append(key)
                query_key = key.split('_')[0] + '_eef_pos'
                key_horizon[key] = shape_meta['obs'][query_key]['horizon']
                key_latency_steps[key] = shape_meta['obs'][query_key]['latency_steps']
                key_down_sample_steps[key] = shape_meta['obs'][query_key]['down_sample_steps']

        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration
        )
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

    # === 新增：数据转换函数 ===
    def _get_replay_buffer(self, dataset_path, shape_meta, store):
        # 解析 shape_meta
        rgb_keys = list()
        lowdim_keys = list()
        out_resolutions = dict()
        lowdim_shapes = dict()
        obs_shape_meta = shape_meta['obs']
        
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            shape = tuple(attr.get('shape'))
            if type == 'rgb':
                rgb_keys.append(key)
                c, h, w = shape
                out_resolutions[key] = (w, h)
            elif type == 'low_dim':
                lowdim_keys.append(key)
                lowdim_shapes[key] = tuple(shape)
                if 'pose' in key:
                    assert tuple(shape) in [(2,), (6,), (7,)]
        
        action_shape = tuple(shape_meta['action']['shape'])
        assert action_shape in [(2,), (6,), (7,)]
        
        # 加载数据
        cv2.setNumThreads(1)
        with threadpool_limits(1):
            replay_buffer = real_data_to_replay_buffer(
                dataset_path=dataset_path,
                out_store=store,
                out_resolutions=out_resolutions,
                lowdim_keys=lowdim_keys + ['action'],
                image_keys=rgb_keys
            )
        
        return replay_buffer
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration
        )
        val_set.val_mask = ~self.val_mask
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, D)

        # action
        assert data_cache['action'].shape[-1] % self.num_robot == 0
        dim_a = data_cache['action'].shape[-1] // self.num_robot
        action_normalizers = list()
        for i in range(self.num_robot):
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a: i * dim_a + 3])))              # pos
            action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a + 3: (i + 1) * dim_a - 1]))) # rot
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., (i + 1) * dim_a - 1: (i + 1) * dim_a])))  # gripper

        normalizer['action'] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])

            if key.endswith('pos') or 'pos_wrt' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos_abs'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('rot_axis_angle') or 'rot_axis_angle_wrt' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('gripper_width'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.sampler_lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]
        
        # generate relative pose between two ees
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            for other_robot_id in range(self.num_robot):
                if robot_id == other_robot_id:
                    continue
                if not f'robot{robot_id}_eef_pos_wrt{other_robot_id}' in self.lowdim_keys:
                    continue
                other_pose_mat = pose_to_mat(np.concatenate([
                    obs_dict[f'robot{other_robot_id}_eef_pos'],
                    obs_dict[f'robot{other_robot_id}_eef_rot_axis_angle']
                ], axis=-1))
                rel_obs_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=other_pose_mat[-1],
                    pose_rep='relative',
                    backward=False)
                rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
                obs_dict[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]
                
        # generate relative pose with respect to episode start
        for robot_id in range(self.num_robot):
            # HACK: add noise to episode start pose
            if (f'robot{other_robot_id}_eef_pos_wrt_start' not in self.shape_meta['obs']) and \
                (f'robot{other_robot_id}_eef_rot_axis_angle_wrt_start' not in self.shape_meta['obs']):
                continue
            
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            
            # get start pose
            start_pose = obs_dict[f'robot{robot_id}_demo_start_pose'][0]
            # HACK: add noise to episode start pose
            start_pose += np.random.normal(scale=[0.05,0.05,0.05,0.05,0.05,0.05],size=start_pose.shape)
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)
            
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # HACK: add noise to episode start pose
            # obs_dict[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

        del_keys = list()
        for key in obs_dict:
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                del_keys.append(key)
        for key in del_keys:
            del obs_dict[key]

        actions = list()
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            action_mat = pose_to_mat(data['action'][...,7 * robot_id: 7 * robot_id + 6])
            
            # solve relative obs
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
            action_pose_mat = convert_pose_mat_rep(
                action_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
        
            # convert pose to pos + rot6d representation
            obs_pose = mat_to_pose10d(obs_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)
        
            action_gripper = data['action'][..., 7 * robot_id + 6: 7 * robot_id + 7]
            actions.append(np.concatenate([action_pose, action_gripper], axis=-1))

            # generate data
            obs_dict[f'robot{robot_id}_eef_pos'] = obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle'] = obs_pose[:,3:]
            
        data['action'] = np.concatenate(actions, axis=-1)
        
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data