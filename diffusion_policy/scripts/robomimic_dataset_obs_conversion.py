"""
将 Robomimic 数据集中的观察数据(obs)从时间步分离的结构转换为按传感器类型聚合的结构

原始格式 (转换前)
在 Robomimic 的 HDF5 文件中，每个时间步的观察数据是分散存储的：

/demo_0/
    ├── actions
    ├── rewards
    ├── dones
    └── obs/
        ├── time_step_0/
        │   ├── camera_image (128,128,3)
        │   ├── robot_state (7,)
        │   └── ...
        ├── time_step_1/
        │   ├── camera_image (128,128,3)
        │   ├── robot_state (7,)
        │   └── ...
        └── ...
转换后格式
代码会将数据重组为​​按传感器类型聚合的时间序列​​：

/demo_0/
    ├── actions
    ├── rewards
    ├── dones
    └── obs/
        ├── camera_image (T, 128,128,3)  # 整个episode的图像序列
        ├── robot_state (T, 7)           # 整个episode的状态序列
        └── ...                          # 其他传感器数据序列
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import multiprocessing
import os
import shutil
import click
import pathlib
import h5py
from tqdm import tqdm
import numpy as np
import collections
import pickle
from diffusion_policy.common.robomimic_util import RobomimicObsConverter

# 强制使用 spawn方式创建子进程，避免 fork 方式在 Unix 上的潜在问题
multiprocessing.set_start_method('spawn', force=True)

def worker(x):
    path, idx = x
    converter = RobomimicObsConverter(path)
    obss = converter.convert_idx(idx)
    return obss

@click.command()
@click.option('-i', '--input', required=True, help='input hdf5 path')
@click.option('-o', '--output', required=True, help='output hdf5 path. Parent directory must exist')
@click.option('-n', '--num_workers', default=None, type=int)
def main(input, output, num_workers):
    # process inputs
    # 1. 路径验证和准备
    input = pathlib.Path(input).expanduser()
    output = pathlib.Path(output).expanduser()
    assert input.is_file()
    assert output.parent.is_dir()
    assert not output.is_dir()
    
    # 3. 创建转换器实例
    converter = RobomimicObsConverter(input)

    # save output
    # 2. 复制原始文件作为处理基础
    print('Copying hdf5')
    shutil.copy(str(input), str(output))

    # 4. 分批次并行处理
    idx = 0
    while idx < len(converter):
        # 4.1 计算当前批次范围
        end = min(idx + num_workers, len(converter))
        
        # 4.2 并行处理当前批次
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(worker, [(input, i) for i in range(idx, end)])
        
        # 4.3 写入处理结果到输出文件
        print(f'Writing {idx} to {end}')
        with h5py.File(output, 'r+') as out_file:
            for i in tqdm(range(idx, end), desc="Writing to output"):
                obss = results[i - idx]  # 获取对应结果
                demo = out_file[f'data/demo_{i}']
                del demo['obs']  # 删除原始obs
                
                # 写入新的obs数据
                for k in obss:
                    demo.create_dataset(
                        f"obs/{k}", 
                        data=np.array(obss[k]), 
                        compression="gzip"
                    )
        
        # 4.4 更新索引和清理
        idx = end
        del results


if __name__ == "__main__":
    main()
