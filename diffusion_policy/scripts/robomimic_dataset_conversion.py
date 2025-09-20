"""
Robomimic 动作格式转换工具

功能：
1. 将 HDF5 数据集中的相对动作(增量动作)转换为绝对动作(世界坐标系动作)
2. 可选地评估转换质量并生成可视化报告
3. 支持多进程并行处理提高效率

使用场景：
- 将增量动作数据集转换为绝对动作数据集
- 评估增量动作到绝对动作的转换质量
- 为需要绝对动作的算法准备数据集

命令行参数：
-i/--input: 输入 HDF5 文件路径（必需）
-o/--output: 输出 HDF5 文件路径（必需）
-e/--eval_dir: 评估结果输出目录（可选）
-n/--num_workers: 并行工作进程数（可选）

处理流程：
1. 验证输入输出路径
2. 并行转换所有演示(demo)的动作
3. 复制原始文件作为输出基础
4. 用转换后的动作替换原始动作
5. 生成评估报告（如果启用）
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
import collections
import pickle
from diffusion_policy.common.robomimic_util import RobomimicAbsoluteActionConverter

def worker(x):
    path, idx, do_eval = x
    converter = RobomimicAbsoluteActionConverter(path)
    if do_eval:
        abs_actions, info = converter.convert_and_eval_idx(idx)
    else:
        abs_actions = converter.convert_idx(idx)
        info = dict()
    return abs_actions, info

@click.command()
@click.option('-i', '--input', required=True, help='input hdf5 path')
@click.option('-o', '--output', required=True, help='output hdf5 path. Parent directory must exist')
@click.option('-e', '--eval_dir', default=None, help='directory to output evaluation metrics')
@click.option('-n', '--num_workers', default=None, type=int)
def main(input, output, eval_dir, num_workers):
    # process inputs
    input = pathlib.Path(input).expanduser()
    assert input.is_file()
    output = pathlib.Path(output).expanduser()
    assert output.parent.is_dir()
    assert not output.is_dir()

    do_eval = False
    if eval_dir is not None:
        eval_dir = pathlib.Path(eval_dir).expanduser()
        assert eval_dir.parent.exists()
        do_eval = True
    
    converter = RobomimicAbsoluteActionConverter(input)

    # run
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(worker, [(input, i, do_eval) for i in range(len(converter))])
    
    # save output
    print('Copying hdf5')
    shutil.copy(str(input), str(output))

    # modify action
    with h5py.File(output, 'r+') as out_file:
        for i in tqdm(range(len(converter)), desc="Writing to output"):
            abs_actions, info = results[i]
            demo = out_file[f'data/demo_{i}']
            demo['actions'][:] = abs_actions
    
    # save eval
    if do_eval:
        eval_dir.mkdir(parents=False, exist_ok=True)

        print("Writing error_stats.pkl")
        infos = [info for _, info in results]
        pickle.dump(infos, eval_dir.joinpath('error_stats.pkl').open('wb'))

        print("Generating visualization")
        metrics = ['pos', 'rot']
        metrics_dicts = dict()
        for m in metrics:
            metrics_dicts[m] = collections.defaultdict(list)

        for i in range(len(infos)):
            info = infos[i]
            for k, v in info.items():
                for m in metrics:
                    metrics_dicts[m][k].append(v[m])

        from matplotlib import pyplot as plt
        plt.switch_backend('PDF')

        fig, ax = plt.subplots(1, len(metrics))
        for i in range(len(metrics)):
            axis = ax[i]
            data = metrics_dicts[metrics[i]]
            for key, value in data.items():
                axis.plot(value, label=key)
            axis.legend()
            axis.set_title(metrics[i])
        fig.set_size_inches(10,4)
        fig.savefig(str(eval_dir.joinpath('error_stats.pdf')))
        fig.savefig(str(eval_dir.joinpath('error_stats.png')))


if __name__ == "__main__":
    main()
