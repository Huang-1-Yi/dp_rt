from typing import Sequence, Tuple, Dict, Optional, Union
import os
import pathlib
import numpy as np
import av
import zarr
import numcodecs
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer, get_optimal_chunks
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.real_world.video_recorder import read_video
from diffusion_policy.codecs.imagecodecs_numcodecs import (
    register_codecs,
    Jpeg2k
)
register_codecs()      

                                                                                                                                                         
def real_data_to_replay_buffer(
        dataset_path: str, 
        out_store: Optional[zarr.ABSStore]=None, 
        out_resolutions: Union[None, tuple, Dict[str,tuple]]=None, # (width, height)
        lowdim_keys: Optional[Sequence[str]]=None,
        image_keys: Optional[Sequence[str]]=None,
        mask_keys: Optional[Sequence[str]] = None,  # 新增mask_keys参数
        lowdim_compressor: Optional[numcodecs.abc.Codec]=None,
        image_compressor: Optional[numcodecs.abc.Codec]=None,
        mask_compressor: Optional[numcodecs.abc.Codec]=None,
        n_decoding_threads: int=multiprocessing.cpu_count(),
        n_encoding_threads: int=multiprocessing.cpu_count(),
        max_inflight_tasks: int=multiprocessing.cpu_count()*5,
        verify_read: bool=True
        ) -> ReplayBuffer:
    """
    It is recommended to use before calling this function
    to avoid CPU oversubscription
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)

    out_resolution:
        if None:
            use video resolution
        if (width, height) e.g. (1280, 720)
        if dict:
            camera_0: (1280, 720)
    image_keys: ['camera_0', 'camera_1']
    """
    if out_store is None:
        out_store = zarr.MemoryStore()
    # if n_decoding_threads <= 0:
    #     n_decoding_threads = multiprocessing.cpu_count()
    # if n_encoding_threads <= 0:
    #     n_encoding_threads = multiprocessing.cpu_count()
    # if image_compressor is None:
    #     image_compressor = Jpeg2k(level=50)
    # if mask_compressor is None:
    #     # 为mask数据默认使用无损PNG压缩
    #     mask_compressor = numcodecs.PNG()

    print(f"222222222222222222222真实数据转换函数调用数据: {dataset_path}")

    # 验证输入路径 ------------------------------------------------------
    input_path = pathlib.Path(os.path.expanduser(dataset_path))
    if not input_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"路径不是目录: {input_path}")
    
    print(f"33333333333333正在加载数据集: {input_path}")
    
    # 直接使用传入路径作为Zarr存储路径
    in_replay_buffer = ReplayBuffer.create_from_path(
        str(input_path.absolute()), mode='r')
    
    # 合并所有需要复制的键 ----------------------------------------------
    all_keys = []
    key_sources = {
        'lowdim': lowdim_keys,
        'image': image_keys,
        'mask': mask_keys
    }
    for key_type, keys in key_sources.items():
        if keys is not None:
            # 验证键是否存在于源数据
            missing = [k for k in keys if k not in in_replay_buffer]
            if len(missing) > 0:
                raise KeyError(f"源数据中缺失以下{key_type}键: {missing}")
            all_keys.extend(keys)


    # save lowdim data to single chunk
    # 配置数据类型特定的压缩器
    compressor_map = {}
    for key in all_keys:
        if image_keys and (key in image_keys):
            compressor_map[key] = image_compressor
        elif mask_keys and (key in mask_keys):
            compressor_map[key] = mask_compressor
        else:
            compressor_map[key] = lowdim_compressor

    # 配置chunk策略
    chunks_map = {}
    for key in all_keys:
        arr = in_replay_buffer[key]
        # 保持原始chunk结构，但对图像/mask优化
        if (image_keys and key in image_keys) or (mask_keys and key in mask_keys):
            # 图像/Mask数据
            if key in (image_keys or []):
                chunks = (1,) + arr.shape[1:]  # (1, H, W, C)
            elif key in (mask_keys or []):
                chunks = (1,) + arr.shape[1:]  # (1, H, W, 1)
        else:
            chunks = arr.chunks
        chunks_map[key] = chunks

    print(f"正在复制数据键: {all_keys}")
    out_replay_buffer = ReplayBuffer.copy_from_store(
        src_store=in_replay_buffer.root.store,
        store=out_store,
        keys=all_keys,
        chunks=chunks_map,
        compressors=compressor_map
        )
    
    # 复制元数据
    print("正在复制元数据...")
    for meta_key in ['episode_ends', 'timestamp']:
        if meta_key in in_replay_buffer.root:
            out_replay_buffer.root[meta_key] = in_replay_buffer.root[meta_key][:]

    # 最终验证 ---------------------------------------------------------
    print("最终验证:")
    print("输入数据集步数:", in_replay_buffer.n_steps)
    print("输出数据集步数:", out_replay_buffer.n_steps)
    assert out_replay_buffer.n_steps == in_replay_buffer.n_steps
    
    return out_replay_buffer