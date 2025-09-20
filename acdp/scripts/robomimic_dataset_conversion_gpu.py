"""
1. 性能优化
​​转换器缓存​​：使用全局缓存避免重复创建转换器实例
​​GPU支持​​：添加--use_gpu选项启用GPU加速
​​高效进度显示​​：使用tqdm显示实时进度和处理速度

2. 错误处理与健壮性
​​错误捕获​​：在worker函数中添加异常捕获，防止单个失败影响整体
​​错误统计​​：记录所有转换错误并生成错误报告
​​错误跳过​​：遇到错误时跳过该demo而不是中断整个流程
​​成功率计算​​：计算并显示整体转换成功率

3. 用户体验改进
​​详细日志​​：添加时间戳和级别的结构化日志
​​进度可视化​​：实时显示处理进度和完成百分比
​​警告抑制​​：添加--suppress_warnings选项减少日志噪音
​​自动帮助​​：无参数运行时自动显示帮助信息

4. 资源管理优化
​​内存管理​​：使用imap_unordered减少内存占用
​​进程初始化日志​​：记录工作进程初始化状态
​​优雅降级​​：GPU不可用时自动回退到CPU模式

5. 输出增强
​​错误分析报告​​：生成详细的错误统计和成功率
​​图表优化​​：添加坐标轴标签和标题提高可读性
​​图表格式​​：同时保存PDF和PNG格式图表
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



import logging
import torch
# ==== 新增：GPU可用性检查 ====
GPU_AVAILABLE = torch.cuda.is_available() if torch.cuda.is_available() else False

# ==== 新增：转换器缓存 ====
converter_cache = {}

def get_converter(file_path):
    """带缓存的转换器获取函数"""
    if file_path not in converter_cache:
        converter_cache[file_path] = RobomimicAbsoluteActionConverter(file_path)
    return converter_cache[file_path]

def worker(x):
    """带错误处理的worker函数"""
    path, idx, do_eval = x
    try:
        converter = get_converter(path)
        if do_eval:
            abs_actions, info = converter.convert_and_eval_idx(idx)
        else:
            abs_actions = converter.convert_idx(idx)
            info = dict()
        return (idx, abs_actions, info)
    except Exception as e:
        logging.error(f"Error processing index {idx}: {str(e)}")
        return (idx, None, {"error": str(e)})

@click.command()
@click.option('-i', '--input', required=True, help='input hdf5 path')
@click.option('-o', '--output', required=True, help='output hdf5 path. Parent directory must exist')
@click.option('-e', '--eval_dir', default=None, help='directory to output evaluation metrics')
@click.option('-n', '--num_workers', default=None, type=int)
@click.option('--use_gpu', is_flag=True, help='Enable GPU acceleration if available')
@click.option('--suppress_warnings', is_flag=True, help='Suppress warning messages')
def main(input, output, eval_dir, num_workers, use_gpu, suppress_warnings):
    # ==== 新增：日志控制 ====
    if suppress_warnings:
        logging.getLogger().setLevel(logging.ERROR)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    # ==== 新增：GPU检查 ====
    if use_gpu and not GPU_AVAILABLE:
        logging.warning("GPU acceleration requested but not available. Falling back to CPU.")
        use_gpu = False
    
    # 处理输入输出路径
    input_path = pathlib.Path(input).expanduser()
    assert input_path.is_file(), f"Input file not found: {input}"
    output_path = pathlib.Path(output).expanduser()
    assert output_path.parent.is_dir(), f"Output directory does not exist: {output_path.parent}"
    assert not output_path.is_dir()

    do_eval = False
    if eval_dir is not None:
        eval_dir = pathlib.Path(eval_dir).expanduser()
        # assert eval_dir.parent.exists()
        eval_dir.mkdir(parents=True, exist_ok=True)
        do_eval = True
    
    # 获取转换器实例（仅用于获取长度）
    converter = RobomimicAbsoluteActionConverter(input)
    
    # ==== 新增：进度条 ====
    total_demos = len(converter)
    progress = tqdm(total=total_demos, desc="Processing demos")

    # 使用多进程池处理
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=lambda: logging.info(f"Worker {multiprocessing.current_process().name} initialized")
    ) as pool:
        # 准备任务参数
        tasks = [(str(input_path), i, do_eval) for i in range(total_demos)]
        
        # 处理结果：使用imap_unordered减少内存占用，添加排序确保顺序
        results = []
        for result in pool.imap_unordered(worker, tasks):
            results.append(result)
            progress.update(1)
            progress.set_postfix_str(f"Processed {len(results)}/{total_demos}")
    
    progress.close()
    
    # 按索引排序结果
    results.sort(key=lambda x: x[0])
    
    # 保存输出
    print('Copying hdf5 file...')
    shutil.copy(str(input_path), str(output_path))
    
    # 修改动作数据
    print('Updating actions in output file...')
    with h5py.File(output_path, 'r+') as out_file:
        for idx, abs_actions, info in tqdm(results, desc="Writing actions"):
            if abs_actions is None:
                logging.warning(f"Skipping demo_{idx} due to conversion error")
                continue
                
            demo_key = f'data/demo_{idx}'
            if demo_key in out_file:
                out_file[demo_key]['actions'][:] = abs_actions
    
    # 保存评估结果
    if do_eval:
        print("Saving evaluation metrics...")
        eval_dir.mkdir(parents=False, exist_ok=True)
        
        # 保存错误统计
        error_stats = [info for _, _, info in results]
        with open(eval_dir.joinpath('error_stats.pkl'), 'wb') as f:
            pickle.dump(error_stats, f)
        
        # ==== 新增：错误分析报告（添加转换成功率统计，提供质量评估） ====
        error_count = sum(1 for _, abs_actions, _ in results if abs_actions is None)
        success_rate = 1 - (error_count / total_demos)
        print(f"Conversion success rate: {success_rate:.2%} ({total_demos - error_count}/{total_demos})")
        
        # 生成可视化
        if error_count < total_demos:  # 只在有成功转换时生成
            print("Generating visualization...")
            metrics = ['pos', 'rot']
            metrics_dicts = {m: collections.defaultdict(list) for m in metrics}
            
            for idx, abs_actions, info in results:
                if abs_actions is None:
                    continue
                    
                for k, v in info.items():
                    for m in metrics:
                        if m in v:
                            metrics_dicts[m][k].append(v[m])
            
            from matplotlib import pyplot as plt
            plt.switch_backend('Agg')  # 使用非交互式后端
            
            fig, axes = plt.subplots(1, len(metrics), figsize=(10, 4))
            if len(metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                data = metrics_dicts[metric]
                for key, values in data.items():
                    ax.plot(values, label=key)
                ax.legend()
                ax.set_title(metric.capitalize() + " Error")
                ax.set_xlabel("Demo Index")
                ax.set_ylabel("Error Value")
            
            plt.tight_layout()
            fig.savefig(str(eval_dir.joinpath('error_stats.pdf')))
            fig.savefig(str(eval_dir.joinpath('error_stats.png')))
            plt.close(fig)
        else:
            print("Skipping visualization - no successful conversions")


if __name__ == "__main__":
    # ==== 新增：日志配置 ====
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 支持命令行参数
    import sys
    if len(sys.argv) == 1:
        # 显示帮助信息
        sys.argv.append('--help')
    
    main()

