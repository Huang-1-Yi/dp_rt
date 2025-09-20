# 成功实现数据集合并、裁剪、修改

import zarr
import numpy as np
from pathlib import Path
import json
import os
import cv2
import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import shutil
from tqdm import tqdm, trange
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from diffusion_policy.common.pose_util import pose_to_mat, mat_to_pose10d

class ZarrInspector:
    def __init__(self, zarr_path):
        self.zarr_path = Path(zarr_path)
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr路径不存在: {self.zarr_path}")
        
        self.store = zarr.open(str(self.zarr_path), mode='r')
        self.tree = {}
        self.physical_size = self._calculate_directory_size(self.zarr_path)
        self._init_cache()
    
    def _init_cache(self):
        """初始化缓存"""
        self._episode_ends_cache = None
        self._min_length_cache = None
    
    @lru_cache(maxsize=32)
    def _calculate_directory_size(self, path):
        """递归计算目录大小 - 使用缓存优化"""
        total = 0
        path = Path(path)
        if not path.exists():
            return total
        
        for entry in os.scandir(str(path)):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += self._calculate_directory_size(entry.path)
        return total
    
    def get_physical_size(self):
        """获取Zarr目录的实际磁盘占用"""
        return self.physical_size / (1024**3)  # 转换为GB
    
    def detailed_storage_report(self):
        """生成详细存储报告 - 优化性能"""
        report = []
        total_theoretical = 0
        total_physical = 0
        
        # 遍历Zarr结构
        def traverse_nodes(group, base_path=""):
            nonlocal total_theoretical, total_physical
            for name, node in group.items():
                full_path = f"{base_path}/{name}" if base_path else name
                node_path = self.zarr_path / full_path
                
                if isinstance(node, zarr.Array):
                    # 计算理论大小
                    if node.shape:
                        theo_size = np.prod(node.shape) * np.dtype(node.dtype).itemsize
                        dim0 = node.shape[0]
                    else:
                        theo_size = 0
                        dim0 = 0
                    
                    # 实际大小
                    phys_size = self._calculate_directory_size(node_path)
                    
                    # 累加总数
                    total_theoretical += theo_size
                    total_physical += phys_size
                    
                    # 添加节点
                    node_info = {
                        "path": full_path,
                        "type": "Array",
                        "dim0": dim0,
                        "theoretical_size": theo_size,
                        "physical_size": phys_size,
                        "dtype": str(node.dtype),
                        "shape": node.shape
                    }
                    report.append(node_info)
                
                elif isinstance(node, zarr.Group):
                    group_phys_size = self._calculate_directory_size(node_path)
                    node_info = {
                        "path": full_path,
                        "type": "Group",
                        "subkeys_count": len(node.keys()),
                        "physical_size": group_phys_size
                    }
                    report.append(node_info)
                    
                    # 递归遍历子节点
                    traverse_nodes(node, full_path)
        
        traverse_nodes(self.store)
        
        return report, total_theoretical, total_physical
    
    def print_detailed_storage_report(self):
        """打印详细存储报告 - 优化输出格式"""
        report, total_theoretical, total_physical = self.detailed_storage_report()
        
        print(f"\n{'='*80}")
        print(f"精确检查Zarr文件: {self.zarr_path}")
        print(f"物理磁盘占用: {self.get_physical_size():.2f} GB")
        print(f"{'='*80}")
        
        # 打印报告头
        print("\n详细存储报告:")
        print(f"{'路径':<50} | {'类型':<8} | {'长度':<8} | {'理论大小(MB)':>12} | {'实际大小(MB)':>12}")
        print("-"*100)
        
        # 打印每一行
        for node in report:
            theo_mb = node["theoretical_size"] / (1024**2) if "theoretical_size" in node else 0
            phys_mb = node["physical_size"] / (1024**2)
            
            if node["type"] == "Array":
                print(f"{node['path']:<50} | {'Array':<8} | {node['dim0']:<8} | {theo_mb:>12.1f} | {phys_mb:>12.1f}")
            else:
                print(f"{node['path']:<50} | {'Group':<8} | {node['subkeys_count']:<8} | {'-':>12} | {phys_mb:>12.1f}")
        
        # 打印统计信息
        total_theo_gb = total_theoretical / (1024**3)
        total_phys_gb = self.physical_size / (1024**3)
        
        print(f"\n总结:")
        print(f"理论数据总量: {total_theo_gb:.2f} GB")
        print(f"实际磁盘占用: {total_phys_gb:.2f} GB")
        
        if total_theoretical > 0:
            compression_rate = (1 - total_physical/total_theoretical) * 100
            print(f"压缩/稀疏率: {compression_rate:.1f}%")
        
        print(f"{'='*80}")

    def build_tree_structure(self):
        """构建Zarr结构的树形表示 - 优化性能"""
        def traverse_group(group, path=""):
            node = {"type": "group", "path": path, "children": []}
            for name, item in group.items():
                item_path = f"{path}/{name}" if path else name
                
                if isinstance(item, zarr.Array):
                    array_node = {
                        "type": "array",
                        "path": item_path,
                        "shape": item.shape,
                        "dtype": str(item.dtype),
                        "chunks": item.chunks,
                        "dim0_length": item.shape[0] if item.shape else 0
                    }
                    node["children"].append(array_node)
                elif isinstance(item, zarr.Group):
                    group_node = traverse_group(item, item_path)
                    node["children"].append(group_node)
            return node
        
        self.tree = traverse_group(self.store)
        return self.tree
    
    def print_tree(self):
        """打印树形结构到控制台 - 优化输出格式"""
        if not self.tree:
            print("未构建树形结构，请先调用build_tree_structure()")
            return
        
        def print_node(node, indent=0):
            prefix = "│   " * (indent) + "├── "
            
            if node["type"] == "group":
                print(f"{prefix}[Group] {node['path'] or '/'} ({len(node['children'])}项)")
                for child in node["children"]:
                    print_node(child, indent + 1)
            else:  # array
                shape_str = "x".join(map(str, node["shape"]))
                print(f"{prefix}[Array] {node['path']} (dim0={node['dim0_length']}, {shape_str})")
        
        print(f"\n{'-'*80}")
        print(f"Zarr树形结构: {self.zarr_path}")
        print(f"{'-'*80}")
        print_node(self.tree)
        print(f"{'-'*80}\n")
    
    def inspect_key(self, key_path):
        """检查特定键的详细信息 - 优化错误处理"""
        result = {}
        try:
            # 分割键路径为层级
            path_parts = key_path.strip('/').split('/')
            
            # 遍历路径
            current_node = self.store
            for part in path_parts:
                if part not in current_node:
                    result = {"error": f"键路径'{key_path}'中'{part}'不存在"}
                    logger.error(result["error"])
                    return result
                current_node = current_node[part]
            
            # 处理不同类型
            if isinstance(current_node, zarr.Array):
                result = {
                    "type": "array",
                    "path": key_path,
                    "shape": current_node.shape,
                    "dtype": str(current_node.dtype),
                    "chunks": current_node.chunks,
                    "dim0_length": current_node.shape[0] if current_node.shape else 0,
                    "compressor": str(current_node.compressor) if current_node.compressor else None,
                    "physical_size": self._calculate_directory_size(self.zarr_path / key_path)
                }
            elif isinstance(current_node, zarr.Group):
                result = {
                    "type": "group",
                    "path": key_path,
                    "subkeys": list(current_node.keys()),
                    "subkey_count": len(current_node),
                    "physical_size": self._calculate_directory_size(self.zarr_path / key_path)
                }
            else:
                result = {"error": f"未知节点类型: {type(current_node)}"}
                logger.error(result["error"])
                return result
                
        except Exception as e:
            result = {"error": f"检查键时出错: {str(e)}"}
            logger.exception(result["error"])
            return result
            
        # 成功获取信息时打印结果
        print("键详细信息:")
        print(f"  路径: {result['path']}")
        print(f"  类型: {result['type']}")
        
        if result["type"] == "array":
            print(f"  形状: {result['shape']} (元素总数: {np.prod(result['shape'])})")
            print(f"  数据类型: {result['dtype']}")
            print(f"  分块: {result['chunks']}")
            print(f"  第一维度长度: {result['dim0_length']}")
            print(f"  压缩器: {result.get('compressor', '无')}")
            print(f"  物理大小: {result['physical_size'] / (1024**2):.1f} MB")
        else:  # group
            print(f"  子键数量: {result['subkey_count']}")
            print(f"  物理大小: {result['physical_size'] / (1024**2):.1f} MB")
            print("  子键列表:")
            for i, sk in enumerate(result['subkeys'][:10]):  # 显示前10个
                print(f"    {i+1}. {sk}")
            if len(result['subkeys']) > 10:
                print(f"    ... 共 {len(result['subkeys'])} 个子键")
        
        return result
    
    def get_episode_ends(self):
        """获取episode_ends值 - 使用缓存优化"""
        if self._episode_ends_cache is not None:
            return self._episode_ends_cache
            
        key_path = "meta/episode_ends"
        result = self.inspect_key(key_path)
        
        if "error" in result:
            logger.error(f"获取episode_ends失败: {result['error']}")
            return None
            
        try:
            episode_ends = self.store["meta"]["episode_ends"][:]
            self._episode_ends_cache = episode_ends
            return episode_ends
        except Exception as e:
            logger.exception(f"加载episode_ends时出错: {str(e)}")
            return None
    
    def inspect_episode_ends(self, print_flag=True):
        """查看episode_ends键的值 - 优化性能"""
        episode_ends = self.get_episode_ends()
        if episode_ends is None:
            if print_flag:
                print("无法获取有效的episode_ends数据")
            return None
        
        # 计算实际帧数（基于数据数组）
        actual_frames = self.get_min_length()
        
        if print_flag:
            print(f"episode_ends值 (总数: {len(episode_ends)}):")
            # 打印部分值
            max_display = 30  # 最多显示30个值
            if len(episode_ends) <= max_display:
                for i, value in enumerate(episode_ends):
                    print(f"  [{i}]: {value}")
            else:
                # 显示前15个
                for i in range(15):
                    print(f"  [{i}]: {episode_ends[i]}")
                # 中间省略提示
                print(f"  ... 省略中间 {len(episode_ends) - max_display} 个值 ...")
                # 显示后15个
                for i in range(len(episode_ends)-15, len(episode_ends)):
                    print(f"  [{i}]: {episode_ends[i]}")
            
            if len(episode_ends) > 0:
                # 计算每个episode的起始和长度
                starts = [0]
                ends = [episode_ends[0] - 1]
                for i in range(1, len(episode_ends)):
                    starts.append(episode_ends[i-1])
                    ends.append(episode_ends[i] - 1)
                
                # 计算每个episode的长度
                episode_lengths = []
                for i in range(len(episode_ends)):
                    end_idx = ends[i]
                    if end_idx >= actual_frames:
                        end_idx = actual_frames - 1
                    length = end_idx - starts[i] + 1
                    episode_lengths.append(length)
                
                # 打印统计信息
                print(f"\n实际帧数: {actual_frames} (基于数据数组)")
                print("\nEpisode统计:")
                print(f"  总episode数: {len(episode_ends)}")
                print(f"  平均长度: {np.mean(episode_lengths):.0f} 帧")
                print(f"  最小长度: {min(episode_lengths)} 帧")
                print(f"  最大长度: {max(episode_lengths)} 帧")
                
                # 计算累计帧数 (使用各个episode的真实长度求和)
                total_frames = sum(episode_lengths)
                print(f"  累计帧数: {total_frames}")
                
                # 打印与数据集长度的关系
                if actual_frames == total_frames:
                    print("  状态: 累计帧数与数据集长度一致")
                elif actual_frames > total_frames:
                    print(f"  警告: 数据集长度({actual_frames}) > 累计帧数({total_frames})")
                else:
                    print(f"  警告: 数据集长度({actual_frames}) < 累计帧数({total_frames})")

                # 检测异常值并打印（仅当print_flag为True时）
                outlier_threshold = np.mean(episode_lengths) * 5
                outliers = [i for i, length in enumerate(episode_lengths) if length > outlier_threshold]
                if outliers:
                    print(f"\n警告: 检测到可能异常的episode:")
                    for i in outliers:
                        print(f"  episode {i}: {starts[i]} -> {ends[i]} (长度: {episode_lengths[i]})")
            
        return episode_ends
    
    def get_min_length(self):
        """获取所有数组中的最小长度 - 使用缓存优化"""
        if self._min_length_cache is not None:
            return self._min_length_cache
            
        min_length = float('inf')
        for key, array in self.store['data'].items():
            if isinstance(array, zarr.Array) and array.shape:
                length = array.shape[0]
                if length < min_length:
                    min_length = length
        
        self._min_length_cache = min_length if min_length != float('inf') else 0
        return self._min_length_cache

    def save_tree_json(self, file_path):
        """将树结构保存为JSON文件 - 优化性能"""
        if not self.tree:
            self.build_tree_structure()
        
        with open(file_path, 'w') as f:
            json.dump(self.tree, f, indent=2)
        logger.info(f"树结构已保存到 {file_path}")

    def export_episode_stages(self, output_dir="episode_stages"):
        """提取每个episode的stage数据并保存为文件 - 优化性能"""
        try:
            # 获取episode_ends值
            episode_ends = self.get_episode_ends()
            if episode_ends is None:
                logger.error("无法获取有效的episode_ends数据")
                return False
            
            # 尝试获取stage数组
            try:
                stage_array = self.store["data"]["stage"][:]
            except KeyError:
                logger.error("未找到stage数据路径 'data/stage'")
                return False
            
            # 计算episode边界
            starts = [0]
            ends = [episode_ends[0] - 1]
            for i in range(1, len(episode_ends)):
                starts.append(episode_ends[i-1])
                ends.append(episode_ends[i] - 1)
            
            logger.info(f"导出episode stages数据...")
            logger.info(f"检测到 {len(starts)} 个episode")
            logger.info(f"总帧数: {len(stage_array)}")
            
            # 确保输出目录存在
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 使用多线程处理
            def process_episode(i):
                start_idx = starts[i]
                end_idx = ends[i]
                
                # 验证索引
                if start_idx < 0 or end_idx < 0:
                    return f"episode_{i}: 无效索引 (start={start_idx}, end={end_idx})"
                    
                # 验证索引范围
                if end_idx >= len(stage_array):
                    end_idx = len(stage_array) - 1
                    if start_idx > end_idx:
                        return f"episode_{i}: 无法调整索引 (start={start_idx}, end={end_idx})"
                    
                # 检查episode长度
                episode_length = end_idx - start_idx + 1
                if episode_length <= 0:
                    return f"episode_{i}: 无效长度 {episode_length} (start={start_idx}, end={end_idx})"
                
                # 提取当前episode的stage数据
                episode_data = stage_array[start_idx:end_idx+1]
                
                # 保存为文本文件
                txt_filename = output_path / f"episode_{i:03d}_stage.txt"
                with open(txt_filename, "w") as f:
                    f.write(f"# Episode {i}\n")
                    f.write(f"# Start: {start_idx}, End: {end_idx}, Length: {episode_length}\n")
                    for value_idx, value in enumerate(episode_data):
                        actual_frame_idx = start_idx + value_idx
                        f.write(f"{actual_frame_idx}\t{value}\n")
                
                return None
            
            errors = []
            with ThreadPoolExecutor() as executor:
                results = list(tqdm.tqdm(executor.map(process_episode, range(len(starts))), total=len(starts)))
                for result in results:
                    if result:
                        errors.append(result)
            
            # 结果总结
            valid_episodes = len(starts) - len(errors)
            logger.info(f"成功导出 {valid_episodes} 个episode的stage数据")
            if errors:
                logger.warning(f"遇到 {len(errors)} 个问题:")
                for error in errors:
                    logger.warning(f"  - {error}")
            
            return True
        except Exception as e:
            logger.exception(f"导出stage数据时出错: {str(e)}")
            return False

    def export_episode_start_end_images(self, output_dir="episode_images", camera_keys=None):
        """提取每个episode的首尾图像并保存 - 优化性能"""
        try:
            # 获取episode_ends值
            episode_ends = self.get_episode_ends()
            if episode_ends is None:
                logger.error("无法获取有效的episode_ends数据")
                return False
            
            # 设置默认相机键
            if camera_keys is None:
                camera_keys = ["camera_0", "camera_1"]  # 默认使用这两个相机
            
            # 验证相机键是否存在
            valid_camera_keys = []
            for key in camera_keys:
                full_key = f"data/{key}"
                if full_key in self.store:
                    valid_camera_keys.append(key)
                else:
                    logger.warning(f"未找到相机键 '{full_key}'，已跳过")
            
            if not valid_camera_keys:
                logger.error("没有有效的相机键可用")
                return False
            
            # 计算episode边界
            starts = [0]
            ends = [episode_ends[0] - 1]
            for i in range(1, len(episode_ends)):
                starts.append(episode_ends[i-1])
                ends.append(episode_ends[i] - 1)
            
            logger.info(f"导出episode首尾图像...")
            logger.info(f"检测到 {len(starts)} 个episode")
            logger.info(f"使用的相机: {', '.join(valid_camera_keys)}")
            
            # 确保输出目录存在
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 使用多线程处理
            def process_episode(i):
                start_idx = starts[i]
                end_idx = ends[i]
                
                # 创建当前episode的输出目录
                episode_dir = output_path / f"episode_{i:03d}"
                episode_dir.mkdir(exist_ok=True)
                
                errors = []
                for camera_key in valid_camera_keys:
                    # 获取相机数组
                    try:
                        camera_array = self.store[f"data/{camera_key}"]
                    except KeyError:
                        errors.append(f"episode_{i}: 未找到相机键 'data/{camera_key}'")
                        continue
                    
                    # 检查索引范围
                    if start_idx >= len(camera_array):
                        errors.append(f"episode_{i}: 起始索引 {start_idx} 超出相机 '{camera_key}' 范围")
                        continue
                    
                    if end_idx >= len(camera_array):
                        errors.append(f"episode_{i}: 结束索引 {end_idx} 超出相机 '{camera_key}' 范围")
                        continue
                    
                    # 创建相机目录
                    camera_dir = episode_dir / camera_key
                    camera_dir.mkdir(exist_ok=True)
                    
                    # 读取并保存开始帧
                    try:
                        start_frame = camera_array[start_idx]
                        if start_frame.shape[-1] == 3:  # 假设是RGB格式
                            start_frame = cv2.cvtColor(start_frame, cv2.COLOR_RGB2BGR)
                        start_path = camera_dir / f"start_{camera_key}_episode_{i:03d}.png"
                        cv2.imwrite(str(start_path), start_frame)
                    except Exception as e:
                        errors.append(f"episode_{i}: 保存开始帧失败 - {str(e)}")
                    
                    # 读取并保存结束帧
                    try:
                        end_frame = camera_array[end_idx]
                        if end_frame.shape[-1] == 3:  # 假设是RGB格式
                            end_frame = cv2.cvtColor(end_frame, cv2.COLOR_RGB2BGR)
                        end_path = camera_dir / f"end_{camera_key}_episode_{i:03d}.png"
                        cv2.imwrite(str(end_path), end_frame)
                    except Exception as e:
                        errors.append(f"episode_{i}: 保存结束帧失败 - {str(e)}")
                
                return errors
            
            all_errors = []
            with ThreadPoolExecutor() as executor:
                results = list(tqdm.tqdm(executor.map(process_episode, range(len(starts))), total=len(starts)))
                for errors in results:
                    if errors:
                        all_errors.extend(errors)
            
            # 结果总结
            saved_count = len(starts) * len(valid_camera_keys) * 2 - len(all_errors)
            logger.info(f"成功保存 {saved_count} 张图像")
            if all_errors:
                logger.warning(f"遇到 {len(all_errors)} 个问题:")
                for error in all_errors:
                    logger.warning(f"  - {error}")
            
            return True
        except Exception as e:
            logger.exception(f"导出图像时出错: {str(e)}")
            return False

    def truncate_to_min_length(self, update_episode_ends=False):
        """将所有数组裁剪到最短维度 - 优化性能"""
        try:
            logger.info(f"开始裁剪所有数组到最短有效长度...")
            
            # 在存储上开启可写模式
            write_store = zarr.open(self.zarr_path, mode='r+')
            data_group = write_store['data']
            
            # 1. 找到所有数组中的最小长度
            min_length = self.get_min_length()
            min_length_arrays = []
            
            for key, array in data_group.items():
                if isinstance(array, zarr.Array) and array.shape:
                    length = array.shape[0]
                    if length == min_length:
                        min_length_arrays.append(key)
            
            logger.info(f"确定的最小长度: {min_length} (基于: {', '.join(min_length_arrays[:3])}{'...' if len(min_length_arrays)>3 else ''})")
            
            # 2. 裁剪所有数据数组
            truncated_arrays = []
            for key, array in data_group.items():
                if isinstance(array, zarr.Array) and array.shape:
                    current_length = array.shape[0]
                    
                    if current_length > min_length:
                        # 裁剪数组
                        new_array = array[:min_length]
                        
                        # 删除原数组并替换
                        del data_group[key]
                        data_group.create_dataset(
                            name=key,
                            data=new_array,
                            chunks=array.chunks,
                            compressor=array.compressor,
                            dtype=array.dtype
                        )
                        
                        logger.info(f"裁剪: data/{key} ({current_length} -> {min_length})")
                        truncated_arrays.append(key)
                    elif current_length < min_length:
                        logger.warning(f"data/{key} 长度{current_length} < 最小长度{min_length}")
            
            # 3. 可选更新episode_ends
            updated_episode_ends = False
            if update_episode_ends and 'meta' in write_store and 'episode_ends' in write_store['meta']:
                meta_group = write_store['meta']
                episode_ends = meta_group['episode_ends'][:]
                
                # 找到最后一个有效索引（不超过min_length-1）
                last_valid_index = len(episode_ends) - 1
                while last_valid_index >= 0 and episode_ends[last_valid_index] >= min_length:
                    last_valid_index -= 1
                
                if last_valid_index < len(episode_ends) - 1:
                    # 创建新的episode_ends数组
                    new_episode_ends = episode_ends[:last_valid_index + 1]
                    
                    # 替换episode_ends数组
                    del meta_group['episode_ends']
                    meta_group.create_dataset(
                        name='episode_ends',
                        data=new_episode_ends,
                        chunks=True,
                        dtype='int64'
                    )
                    updated_episode_ends = True
                    logger.info(f"更新: meta/episode_ends ({len(episode_ends)} -> {len(new_episode_ends)})")
            
            # 4. 结果报告
            if truncated_arrays or updated_episode_ends:
                logger.info(f"操作结果:")
                if truncated_arrays:
                    logger.info(f"裁剪了 {len(truncated_arrays)} 个数组")
                if updated_episode_ends:
                    logger.info(f"更新了episode_ends元数据")
                logger.info(f"所有数组已统一到 {min_length} 帧")
                return True
            else:
                logger.info("所有数组已符合最小长度要求，无需裁剪")
                return True
                
        except Exception as e:
            logger.exception(f"裁剪过程中出错: {str(e)}")
            return False

    def view_frame_at_index(self, frame_idx, camera_key="camera_0"):
        """精确查看指定索引的图像帧 - 优化显示"""
        try:
            # 获取相机数组
            camera_path = f"data/{camera_key}"
            if camera_path not in self.store:
                logger.error(f"未找到相机键 '{camera_path}'")
                return False
                
            camera_array = self.store[camera_path]
            array_length = len(camera_array)
            
            # 检查索引范围
            if frame_idx < 0 or frame_idx >= array_length:
                logger.error(f"索引 {frame_idx} 超出范围 (0-{array_length-1})")
                return False
            
            # 读取图像帧
            frame = camera_array[frame_idx]
            
            # 转换为BGR格式
            if frame.ndim == 3 and frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 显示图像
            window_name = f"Frame {frame_idx} - {camera_key}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            cv2.imshow(window_name, frame)
            
            # 添加文本信息
            print(f"显示索引 {frame_idx} 的图像 ({camera_key})")
            print(f"  形状: {frame.shape}")
            print(f"  数据类型: {frame.dtype}")
            
            # 显示位置信息
            frame_position = ""
            if frame_idx == 0:
                frame_position = "第一帧"
            elif frame_idx == array_length - 1:
                frame_position = "最后一帧"
            print(f"  位置: {frame_position} (数据集长度: {array_length})")
            
            # 等待用户按键
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return True
        except Exception as e:
            logger.exception(f"查看图像时出错: {str(e)}")
            return False

    def convert_stage_to_vector(self):
        """
        将data/stage数组从一维(9080,)转换为二维向量形式(9080,1)
        直接在原Zarr文件中修改数据
        """
        try:
            logger.info("开始转换stage数组为向量形式...")
            
            # 在存储上开启可写模式
            write_store = zarr.open(self.zarr_path, mode='r+')
            
            # 检查stage数组是否存在
            if "data" not in write_store or "stage" not in write_store["data"]:
                logger.error("未找到data/stage数组，无法转换")
                return False
            
            stage_array = write_store["data"]["stage"]
            
            # 检查是否已经是向量形式
            if len(stage_array.shape) == 2 and stage_array.shape[1] == 1:
                logger.info("data/stage已经是向量形式，无需转换")
                return True
                
            # 获取数组数据
            stage_data = stage_array[:]
            dtype = stage_array.dtype
            chunks = stage_array.chunks
            compressor = stage_array.compressor
            
            # 转换形状：(n,) -> (n, 1)
            new_shape = (len(stage_data), 1)
            vector_data = stage_data.reshape(new_shape)
            
            # 删除原数组
            del write_store["data"]["stage"]
            
            # 创建新的向量形式数组
            new_array = write_store["data"].create_dataset(
                name="stage",
                shape=new_shape,
                chunks=(chunks[0], 1) if chunks else (1000, 1),
                dtype=dtype,
                compressor=compressor
            )
            new_array[:] = vector_data
            
            # 清除缓存以反映更改
            self._init_cache()
            self.build_tree_structure()
            
            logger.info(f"转换成功: data/stage {stage_array.shape} -> {new_shape}")
            return True
            
        except Exception as e:
            logger.exception(f"转换过程中出错: {str(e)}")
            return False

    def append_stage_to_action(self):
        """
        功能10：将(8090,1)形式的stage向量数据拼接到(8090,)的原action标量数据内
        将原本每个数据7个值增加到8个值
        
        操作步骤：
        1. 检查并确保action数组存在且形状为(n,7)
        2. 检查stage数组存在且形状为(n,1)
        3. 创建新的形状为(n,8)的数组
        4. 将原action数组的前7个值复制到新数组
        5. 将stage数组的值拼接到第8个维度
        6. 替换原action数组
        """
        try:
            logger.info("开始将stage数据拼接到action数组中...")
            
            # 在存储上开启可写模式
            write_store = zarr.open(self.zarr_path, mode='r+')
            
            # 检查action数组
            if "data" not in write_store or "action" not in write_store["data"]:
                logger.error("未找到data/action数组，无法处理")
                return False
            
            action_array = write_store["data"]["action"]
            
            # 检查action数组形状
            if len(action_array.shape) != 2 or action_array.shape[1] != 7:
                logger.error(f"data/action数组形状错误，应为(n,7)，实际为{action_array.shape}")
                return False
            
            # 检查stage数组
            if "data" not in write_store or "stage" not in write_store["data"]:
                logger.error("未找到data/stage数组，无法拼接")
                return False
            
            stage_array = write_store["data"]["stage"]
            
            # 检查stage数组形状
            if len(stage_array.shape) != 2 or stage_array.shape[1] != 1:
                logger.error(f"data/stage数组形状错误，应为(n,1)，实际为{stage_array.shape}")
                return False
            
            # 检查两个数组长度是否匹配
            n = action_array.shape[0]
            if stage_array.shape[0] != n:
                logger.error(f"数组长度不匹配: action={n}, stage={stage_array.shape[0]}")
                return False
            
            logger.info(f"验证通过: 数组长度一致(n={n})")
            
            # 读取数据
            action_data = action_array[:]
            stage_data = stage_array[:]
            
            # 获取数组属性
            dtype = action_array.dtype
            chunks = action_array.chunks
            compressor = action_array.compressor
            
            # 创建新的(8,)动作数组
            new_shape = (n, 8)
            new_data = np.zeros(new_shape, dtype=dtype)
            
            # 复制前7个值
            new_data[:, :7] = action_data
            
            # 将stage数据作为第8个维度 (展平stage数组)
            new_data[:, 7] = stage_data[:, 0]  # 展平(n,1)->(n,)
            
            # 删除原数组
            del write_store["data"]["action"]
            
            # 创建新的动作数组
            new_array = write_store["data"].create_dataset(
                name="action",
                shape=new_shape,
                chunks=(chunks[0], 8) if chunks else (1000, 8),
                dtype=dtype,
                compressor=compressor
            )
            new_array[:] = new_data
            
            # 清除缓存以反映更改
            self._init_cache()
            self.build_tree_structure()
            
            logger.info(f"拼接成功: data/action {action_array.shape} -> {new_shape}")
            return True
            
        except Exception as e:
            logger.exception(f"拼接过程中出错: {str(e)}")
            return False

    # 在ZarrInspector类中添加以下新功能

    def create_minimal_dataset(self, output_path, camera_key="camera_0", camera_key_new="camera0_rgb", robot_id="robot0"):
        """
        创建只包含指定相机和机器人数据的新Zarr文件
        包含以下数据：
        - camera0_rgb: RGB图像 [T, H, W, 3] (uint8)
        - robot0_eef_pos: 末端位置 [T, 3]
        - robot0_eef_rot_axis_angle: 末端旋转(轴角) [T, 3]
        - robot0_gripper_width: 夹爪宽度 [T, 1]
        - robot0_eef_pos_wrt_start: 起始位姿 [T, 3]
        - robot0_eef_rot_axis_angle_wrt_start: 结束位姿 [T, 6]
        - robot0_demo_start_pose: 演示起始位姿 [T, 6]
        - robot0_demo_end_pose: 演示结束位姿 [T, 6]
        - stage: 阶段信息 [T, 1]
        - action: 动作向量 [T, 10]
        """
        try:
            logger.info(f"===== 开始创建精简数据集 =====")
            logger.info(f"输出路径: {output_path}")
            logger.info(f"使用机器人: {robot_id}")
            
            # 打开原始存储
            source_store = zarr.open(self.zarr_path, mode='r')
            
            # 创建新存储
            new_store = zarr.open(output_path, mode='w')
            
            # 复制meta组和episode_ends
            if 'meta' in source_store and 'episode_ends' in source_store['meta']:
                logger.info("复制meta/episode_ends...")
                episode_ends = source_store['meta']['episode_ends'][:]
                new_store.create_group('meta')
                new_store['meta'].array('episode_ends', episode_ends, dtype='int64')
            
            # 创建data组
            data_group = new_store.create_group('data')
            
            # 1. 复制相机数据并重命名为camera0_rgb
            original_camera_key = camera_key
            target_camera_key = camera_key_new
            
            camera_path = f"data/{original_camera_key}"
            if camera_path in source_store:
                logger.info(f"复制相机数据: {camera_path} -> data/{target_camera_key}")
                camera_array = source_store[camera_path]
                
                # 直接创建新数组并复制数据
                data_group.create_dataset(
                    name=target_camera_key,
                    shape=camera_array.shape,
                    chunks=camera_array.chunks,
                    dtype=camera_array.dtype,
                    compressor=camera_array.compressor
                )
                data_group[target_camera_key][:] = camera_array[:]
            else:
                logger.error(f"未找到相机数据: {camera_path}")
                return False
            
            # 2. 创建机器人数据
            # 获取episode边界
            episode_ends = self.get_episode_ends()
            if episode_ends is None:
                logger.error("无法获取episode_ends数据")
                return False
            
            # 计算episode起始点
            starts = [0]
            ends = [episode_ends[0] - 1]
            for i in range(1, len(episode_ends)):
                starts.append(episode_ends[i-1])
                ends.append(episode_ends[i] - 1)
            
            # 检查必要的基础数据是否存在
            required_keys = ['robot_eef_pose', 'robot_gripper', 'action', 'stage']
            for key in required_keys:
                if f"data/{key}" not in source_store:
                    logger.error(f"缺少必要数据字段: data/{key}")
                    return False
            
            # 创建新数据集
            new_datasets = {
                f'{robot_id}_eef_pos': (3,),
                f'{robot_id}_eef_rot_axis_angle': (3,),
                f'{robot_id}_gripper_width': (1,),
                f'{robot_id}_eef_pos_wrt_start': (3,),
                f'{robot_id}_eef_rot_axis_angle_wrt_start': (6,),
                f'{robot_id}_demo_start_pose': (6,),
                f'{robot_id}_demo_end_pose': (6,),
                'stage': (1,),
                'action': (10,)
            }
            
            # 初始化新数组
            total_frames = source_store['data']['robot_eef_pose'].shape[0]
            for name, shape in new_datasets.items():
                logger.info(f"创建字段: data/{name} [形状: ({total_frames}, {shape[0]})]")
                data_group.zeros(
                    name, 
                    shape=(total_frames,) + shape,
                    chunks=(1000,) + shape,
                    dtype='float32'
                )
            
            # 处理每个episode
            for ep_idx in trange(len(starts)):
                start_idx = starts[ep_idx]
                end_idx = ends[ep_idx]
                ep_length = end_idx - start_idx + 1
                
                # 获取当前episode的基础数据
                eef_pose = source_store['data']['robot_eef_pose'][start_idx:end_idx+1]
                gripper = source_store['data']['robot_gripper'][start_idx:end_idx+1]
                action_7d = source_store['data']['action'][start_idx:end_idx+1]
                stage_data = source_store['data']['stage'][start_idx:end_idx+1]
                
                # 提取位置和旋转(轴角)
                positions = eef_pose[:, :3]
                rotations_aa = eef_pose[:, 3:6]
                
                # 计算相对于起始点的位姿
                start_pos = positions[0]
                start_rot = rotations_aa[0]
                
                # 计算相对位置
                rel_positions = positions - start_pos
                
                # 计算相对旋转 (使用pose_util函数)
                rel_rotations = []
                for i in range(ep_length):
                    # 将起始旋转转换为矩阵
                    start_mat = pose_to_mat(np.concatenate([start_pos, start_rot]))
                    # 将当前旋转转换为矩阵
                    current_mat = pose_to_mat(np.concatenate([positions[i], rotations_aa[i]]))
                    # 计算相对旋转矩阵
                    rel_mat = np.linalg.inv(start_mat) @ current_mat
                    # 将相对旋转矩阵转换为6D表示
                    rel_6d = mat_to_pose10d(rel_mat)[3:9]  # 只取旋转部分
                    rel_rotations.append(rel_6d)
                rel_rotations = np.array(rel_rotations)
                
                # 将7维动作转换为10维动作
                action_10d = []
                for i in range(ep_length):
                    # 提取位置和旋转
                    pos = action_7d[i, :3]
                    rot_aa = action_7d[i, 3:6]
                    gripper_val = action_7d[i, 6]
                    
                    # 将旋转转换为6D表示
                    rot_mat = pose_to_mat(np.concatenate([np.zeros(3), rot_aa]))
                    rot_6d = mat_to_pose10d(rot_mat)[3:9]
                    
                    # 组合10维动作
                    action_10d.append(np.concatenate([pos, rot_6d, [gripper_val]]))
                action_10d = np.array(action_10d)
                
                # 获取演示起始和结束位姿
                demo_start_pose = np.concatenate([positions[0], rotations_aa[0]])
                demo_end_pose = np.concatenate([positions[-1], rotations_aa[-1]])
                
                # 修复：创建与episode长度匹配的数组
                demo_start_pose_array = np.tile(demo_start_pose, (ep_length, 1))
                demo_end_pose_array = np.tile(demo_end_pose, (ep_length, 1))
                
                # 存储到新数据集
                data_group[f'{robot_id}_eef_pos'][start_idx:end_idx+1] = positions
                data_group[f'{robot_id}_eef_rot_axis_angle'][start_idx:end_idx+1] = rotations_aa
                data_group[f'{robot_id}_gripper_width'][start_idx:end_idx+1] = gripper
                data_group[f'{robot_id}_eef_pos_wrt_start'][start_idx:end_idx+1] = rel_positions
                data_group[f'{robot_id}_eef_rot_axis_angle_wrt_start'][start_idx:end_idx+1] = rel_rotations
                data_group[f'{robot_id}_demo_start_pose'][start_idx:end_idx+1] = demo_start_pose_array
                data_group[f'{robot_id}_demo_end_pose'][start_idx:end_idx+1] = demo_end_pose_array
                data_group['stage'][start_idx:end_idx+1] = stage_data
                data_group['action'][start_idx:end_idx+1] = action_10d
            
            logger.info("===== 精简数据集创建完成 =====")
            
            # 检查新数据集
            new_inspector = ZarrInspector(output_path)
            new_inspector.print_detailed_storage_report()
            
            return True
        
        except Exception as e:
            logger.exception(f"创建精简数据集时出错: {str(e)}")
            return False



def compare_zarr_structures(store1, store2, path=""):
    """
    递归比较两个Zarr存储结构的兼容性
    返回详细的兼容性报告 - 优化性能
    """
    compatibility = True
    details = {"path": path, "compatible": True, "details": []}
    
    # 比较节点类型
    node_type1 = "group" if isinstance(store1, zarr.Group) else "array"
    node_type2 = "group" if isinstance(store2, zarr.Group) else "array"
    
    if node_type1 != node_type2:
        details["compatible"] = False
        details["details"].append({
            "issue": "节点类型不匹配",
            "dataset1": node_type1,
            "dataset2": node_type2
        })
        return False, details
    
    # 对于数组节点
    if isinstance(store1, zarr.Array):
        # 比较除第一维外的形状
        if store1.shape[1:] != store2.shape[1:]:
            details["compatible"] = False
            details["details"].append({
                "issue": "数组形状不匹配",
                "dataset1_shape": store1.shape,
                "dataset2_shape": store2.shape
            })
        
        # 比较数据类型
        if store1.dtype != store2.dtype:
            details["compatible"] = False
            details["details"].append({
                "issue": "数据类型不匹配",
                "dataset1_dtype": str(store1.dtype),
                "dataset2_dtype": str(store2.dtype)
            })
        
        # 检查压缩器兼容性
        if store1.compressor != store2.compressor:
            details["compatible"] = False
            details["details"].append({
                "issue": "压缩器不匹配",
                "dataset1_compressor": str(store1.compressor),
                "dataset2_compressor": str(store2.compressor)
            })
        
        return details["compatible"], details
    
    # 对于组节点
    keys1 = set(store1.keys())
    keys2 = set(store2.keys())
    
    # 排除episode_ends的特殊处理
    if path == "/meta" or path == "/":
        keys1.discard("episode_ends")
        keys2.discard("episode_ends")
    
    # 检查缺少的键
    missing_in_ds2 = keys1 - keys2
    missing_in_ds1 = keys2 - keys1
    
    if missing_in_ds2:
        details["compatible"] = False
        details["details"].append({
            "issue": "键在数据集2中缺失",
            "missing_keys": list(missing_in_ds2)
        })
    
    if missing_in_ds1:
        details["compatible"] = False
        details["details"].append({
            "issue": "键在数据集1中缺失",
            "missing_keys": list(missing_in_ds1)
        })
    
    # 递归比较子节点
    common_keys = keys1 & keys2
    for key in common_keys:
        sub_path = f"{path}/{key}" if path else key
        sub_compatible, sub_details = compare_zarr_structures(store1[key], store2[key], sub_path)
        details["details"].append(sub_details)
        if not sub_compatible:
            details["compatible"] = False
    
    return details["compatible"], details

def print_compatibility_report(report, indent=0):
    """打印详细的兼容性报告 - 优化输出格式"""
    prefix = "  " * indent
    print(f"{prefix}路径: {report['path']}")
    print(f"{prefix}兼容: {'是' if report['compatible'] else '否'}")
    
    for detail in report["details"]:
        if "path" in detail:
            # 这是子报告
            print_compatibility_report(detail, indent+1)
        else:
            # 这是问题详情
            issue = detail["issue"]
            print(f"{prefix}  - 问题: {issue}")
            for k, v in detail.items():
                if k != "issue":
                    if isinstance(v, list):
                        v_str = ", ".join(map(str, v))
                    else:
                        v_str = str(v)
                    print(f"{prefix}    {k}: {v_str}")

def merge_zarr_datasets(dataset1_path, dataset2_path, output_path, chunk_size=1000):
    """高效合并两个Zarr数据集，生成新数据集"""
    try:
        print(f"\n===== 开始合并数据集 =====")
        print(f"  数据集1: {dataset1_path}")
        print(f"  数据集2: {dataset2_path}")
        print(f"  输出位置: {output_path}")
        
        # 打开两个数据集
        store1 = zarr.open(dataset1_path, 'r')
        store2 = zarr.open(dataset2_path, 'r')
        
        # 1. 比较两个数据集的结构
        print("\n[步骤1] 比较数据集结构...")
        compatible, report = compare_zarr_structures(store1, store2, "/")
        print_compatibility_report(report)
        
        if not compatible:
            print("\n❌ 数据集结构不兼容，无法合并!")
            return False
        
        print("\n✅ 数据集结构兼容，可以合并")
        
        # 2. 确定数据集的长度（偏移量）
        print("\n[步骤2] 确定数据集长度...")
        base_length = 0
        
        # 确保数据集1有data组
        if 'data' not in store1:
            print("❌ 数据集1缺少data组，无法确定基础数组长度")
            return False
            
        data_group1 = store1['data']
        
        # 在data组中查找有效的数组
        valid_keys = ["camera_0", "camera_1", "action", "stage"]
        for key in valid_keys:
            if key in data_group1 and isinstance(data_group1[key], zarr.Array):
                base_length = data_group1[key].shape[0]
                print(f"  从 '{key}' 获取基础长度: {base_length}")
                break
                
        if base_length == 0:
            # 如果以上都没找到，尝试找任何有效数组
            for key, value in data_group1.items():
                if isinstance(value, zarr.Array) and hasattr(value, 'shape') and len(value.shape) > 0:
                    base_length = value.shape[0]
                    print(f"  从 '{key}' 获取基础长度: {base_length}")
                    break
                    
        if base_length == 0:
            print("❌ 无法确定基础数组长度，合并失败")
            return False
            
        print(f"  确定的基础长度(偏移量): {base_length}")
        
        # 3. 创建输出存储
        print("\n[步骤3] 创建输出存储...")
        output_store = zarr.open(output_path, 'w')
        print(f"  已创建输出存储: {output_path}")
        
        # 4. 处理episode_ends
        print("\n[步骤4] 处理episode_ends...")
        new_episode_ends = []
        
        # 添加第一个数据集的episode_ends
        if "meta" in store1 and "episode_ends" in store1["meta"]:
            ends1 = store1["meta/episode_ends"][:]
            new_episode_ends.extend(ends1)
            print(f"  添加数据集1的{len(ends1)}个episode结束点")
        else:
            print("  数据集1中没有episode_ends")
        
        # 添加第二个数据集的episode_ends（应用偏移量）
        if "meta" in store2 and "episode_ends" in store2["meta"]:
            ends2 = store2["meta/episode_ends"][:]
            # 应用偏移量
            adjusted_ends = [end + base_length for end in ends2]
            new_episode_ends.extend(adjusted_ends)
            print(f"  添加数据集2的{len(ends2)}个episode结束点（应用偏移量: +{base_length}）")
        else:
            print("  数据集2中没有episode_ends")
        
        # 确保meta组存在
        if not output_store.get('meta'):
            print("  创建meta组...")
            output_store.create_group('meta')
        
        meta_group = output_store['meta']
        
        # 创建新的episode_ends数组
        if new_episode_ends:
            print(f"  创建episode_ends数组，共{len(new_episode_ends)}个结束点")
            # 检查是否存在并删除旧数组
            if 'episode_ends' in meta_group:
                print("  删除已存在的旧episode_ends数组")
                del meta_group['episode_ends']
            # 创建新数组
            chunk_size = min(len(new_episode_ends), 1000)
            meta_group.array('episode_ends', new_episode_ends, dtype='int64', chunks=(chunk_size,))
            
            # 调试信息：确保数组被正确创建
            print(f"  验证: meta/episode_ends 存在? {'是' if 'episode_ends' in meta_group else '否'}")
            if 'episode_ends' in meta_group:
                print(f"  数组形状: {meta_group['episode_ends'].shape}")
        else:
            print("  没有可用的episode_ends数据")
        
        # 5. 创建数据结构并复制第一个数据集
        print("\n[步骤5] 复制数据集1...")
        # 递归复制第一个数据集的结构和数据
        def copy_tree(src, dst_path=""):
            """递归复制组结构和数组数据"""
            for key, item in src.items():
                full_path = f"{dst_path}/{key}" if dst_path else key
                
                # 跳过episode_ends数组（已在步骤4处理）
                if full_path == "meta/episode_ends" or full_path == "/meta/episode_ends":
                    continue
                
                # 特殊处理：跳过根层的meta组（已单独处理）
                if full_path == "meta" and dst_path == "":
                    print(f"  跳过复制源meta组，已单独处理")
                    continue
                
                if isinstance(item, zarr.Group):
                    # 创建组
                    print(f"  创建组: {full_path}")
                    new_group = output_store.create_group(full_path, overwrite=True)
                    # 递归复制
                    copy_tree(item, full_path)
                else:
                    # 创建数组并复制数据
                    print(f"  复制数组: {full_path} [形状: {item.shape}, 块: {item.chunks}]")
                    zarr.copy(item, output_store, name=full_path)
        
        # 从根开始复制
        copy_tree(store1)
        print("  数据集1复制完成")
        
        # 6. 添加第二个数据集的数据
        print("\n[步骤6] 添加数据集2的数据（应用偏移量）...")

        # 确保data组存在
        if 'data' not in output_store:
            print("  创建data组...")
            output_store.create_group('data')
        
        data_group = output_store['data']
        
        # 遍历第二个数据集的数组
        for key, array in store2['data'].items():
            # 检查目标数组是否存在
            if key not in data_group:
                # 创建目标数组
                print(f"  创建目标数组: data/{key}")
                data_group.zeros(
                    key, 
                    shape=(0,) + array.shape[1:], 
                    chunks=array.chunks,
                    dtype=array.dtype,
                    compressor=array.compressor
                )
            
            target = data_group[key]
            source = array

            # ===== 修复从这里开始 =====
            # 验证维度兼容性
            print(f"  验证维度兼容性...")
            if len(target.shape) != len(source.shape):
                logger.error(f"维度数不匹配! data/{key}: 目标{len(target.shape)}维 vs 源{len(source.shape)}维")
                continue
            
            if target.shape[1:] != source.shape[1:]:
                logger.error(f"维度不匹配! data/{key}: "
                            f"目标维度{target.shape} vs 源维度{source.shape}")
                continue
            print(f"  目标数组: data/{key} [形状: {target.shape}, 块: {target.chunks}]")
            print(f"  源数组: data/{key} [形状: {source.shape}, 块: {source.chunks}]")
            # ===== 修复到这里结束 =====

            # 构建新的完整形状
            current_length = target.shape[0]
            new_shape = (current_length + source.shape[0],) + target.shape[1:]
            
            # 调整目标数组大小 - 使用完整形状元组
            target.resize(new_shape)
            
            # 分块复制数据
            chunks = source.chunks[0] if source.chunks else chunk_size
            # (n+k-1)//k比(n-1)//k+1（更安全）保证全量覆盖
            total_chunks = (source.shape[0] + chunks - 1) // chunks
            
            print(f"  开始追加数据: {source.shape[0]}帧, 块大小={chunks}, 总块数={total_chunks}")
            
            # 使用进度条显示
            try:
                from tqdm import tqdm
                chunk_iter = tqdm(range(0, source.shape[0], chunks), total=total_chunks)
            except ImportError:
                print("  注意: 安装'tqdm'可获得更好的进度显示")
                chunk_iter = range(0, source.shape[0], chunks)
                last_percent = -1
            
            for start in chunk_iter:
                end = min(start + chunks, source.shape[0])
                chunk = source[start:end]
                target_position = current_length + start
                target[target_position:target_position+len(chunk)] = chunk
                
                # 如果没有使用tqdm，手动打印进度
                if "tqdm" not in globals():
                    percent = int((end / source.shape[0]) * 100)
                    if percent != last_percent and percent % 10 == 0:
                        print(f"  进度: {percent}%")
                        last_percent = percent
            logger.info(f"追加 data/{key}: 添加{source.shape[0]}个数据点 (新长度: {new_shape[0]})")
            
        
        # 7. 打印合并结果
        print("\n✅ 合并完成!")
        print(f"新数据集路径: {output_path}")
        
        # 检查合并后结果
        merged_inspector = ZarrInspector(output_path)
        total_frames = 0
        for key in ["camera_0", "camera_1", "action", "stage"]:
            if f"data/{key}" in merged_inspector.store:
                total_frames = merged_inspector.store[f"data/{key}"].shape[0]
                break
        
        print(f"总帧数: {total_frames}")
        print(f"总episode数: {len(new_episode_ends)}")
        
        # 详细的 episode_ends 输出
        print("\n=== 详细 episode_ends 信息 ===")
        if 'meta' in merged_inspector.store and 'episode_ends' in merged_inspector.store['meta']:
            episode_ends = merged_inspector.store['meta']['episode_ends'][:]
            print(f"合并后总 episode 数: {len(episode_ends)}")
            # print(f"前5个值: {episode_ends[:5]}")
            # print(f"后5个值: {episode_ends[-5:]}")
            print(f"episode_ends值: {episode_ends[:]}")
            print(f"数据集结构验证:")
            print(f"  理论总帧数: {base_length + store2['data']['camera_0'].shape[0]}")
            print(f"  实际总帧数: {total_frames}")
            
            # 验证是否所有episode结束点都在有效范围内
            valid = True
            if episode_ends[-1] > total_frames:
                print(f"⚠️ 警告: 最后episode结束点 {episode_ends[-1]} 超出总帧数 {total_frames}")
                valid = False

            if min(episode_ends) <= 0:
                print(f"⚠️ 警告: 最小结束点 {min(episode_ends)} 无效")
                valid = False

            # 修复：添加 np.all()
            if not (np.diff(episode_ends) > 0).all():
                print("⚠️ 警告: episode结束点不是单调递增")
                valid = False

            if valid:
                print("✅ 所有episode结束点验证通过")
        else:
            print("❌ 错误: meta/episode_ends 不存在!")
            print("可能的原因:")
            print("1. 步骤4中创建数组失败")
            print("2. 在后续复制过程中被覆盖")
            print("3. Zarr 存储同步问题")
            
            # 列出所有 meta 组内容进行调试
            if 'meta' in merged_inspector.store:
                print("\nmeta 组内容:")
                for key in merged_inspector.store['meta'].keys():
                    print(f"  - {key}")
            else:
                print("❌ meta组也不存在!")
        
        # 保存合并后的树结构
        print("\n保存合并后的树结构到JSON文件...")
        merged_inspector.build_tree_structure()
        merged_inspector.save_tree_json(output_path + "_tree.json")
        
        return True
    
    except Exception as e:
        print(f"\n❌ 合并数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False




if __name__ == "__main__":
    # 配置参数
    # zarr_path = "/home/hy/Desktop/dp_0314/data/palce_0621/replay_buffer.zarr"
    # zarr_path = "/home/hy/Desktop/dp_0314/data/palce_0622/replay_buffer.zarr"
    # zarr_path = "/home/hy/Desktop/dp_0314/data/place621/replay_buffer.zarr"
    # zarr_path = "/home/hy/Desktop/dp_0314/data/stage_0718/replay_buffer.zarr"
    # zarr_path = "/home/hy/Desktop/dp_0314/data/stage_0718_1/replay_buffer.zarr"
    # zarr_path = "/home/hy/Desktop/dp_0314/data/stage_0718_full/replay_buffer.zarr"
    # zarr_path = "/home/hy/Desktop/dp_0314/data/stage_0718_full_stage/replay_buffer.zarr"
    # zarr_path = "/home/hy/Desktop/dp_0314/data/bcb_0721/replay_buffer.zarr"
    # stages_output_path = "./data/bcb_0721/episode_stages"
    # stages_output_dir="./data/bcb_0721/episode_images"

    target_key = "data/camera_0"  # 示例键
    # zarr_path = "/home/hy/Desktop/dp_0314/data/button_grb_0730/replay_buffer.zarr"
    zarr_path = "/home/hy/Desktop/dp_0314/data/cup_and_saucer0805/replay_buffer.zarr"
    stages_output_path = "./data/cup_and_saucer0805/episode_stages"
    stages_output_dir="./data/cup_and_saucer0805/episode_images"
    try:
        # 创建检查器
        inspector = ZarrInspector(zarr_path)
        
        # 功能1: 打印详细存储报告
        print("\n====== 功能1: 详细存储报告 ======")
        inspector.print_detailed_storage_report()
        
        # 功能2: 构建并打印树形结构
        print("\n====== 功能2: 查看Zarr树形结构 ======")
        tree_structure = inspector.build_tree_structure()
        inspector.print_tree()
        
        # 功能3: 检查特定键（调用时会自动打印信息）
        print(f"\n====== 功能3: 检查键 '{target_key}' ======")
        # 直接调用inspect_key即可自动打印详细信息
        key_info = inspector.inspect_key(target_key)
        
        # 功能4: 查看episode_ends值
        print("\n====== 功能4: 查看episode_ends值 ======")
        episode_ends = inspector.inspect_episode_ends()
        
        # # 可选: 保存树结构到JSON
        # inspector.save_tree_json("zarr_tree.json")

        # 功能5: 导出每个episode的stage数据
        print("\n====== 功能5: 导出每个episode的stage数据 ======")
        export_success = inspector.export_episode_stages(output_dir=stages_output_path)
        
        if export_success:
            print("\nStage数据导出完成!")
        
        # 功能6: 导出每个episode的首尾图像
        print("\n====== 功能6: 导出每个episode的首尾图像 ======")
        camera_keys = ["camera_0", "camera_1"]  # 可以修改为其他相机键
        image_success = inspector.export_episode_start_end_images(camera_keys=camera_keys,output_dir=stages_output_dir)

        if image_success:
            print("\n首尾图像导出完成!")
        
        
        # 精简功能：精确查看指定索引的图像
        print("\n====== 精确查看指定索引的图像 ======")
        
        # 获取最小长度作为参考
        min_length = -1 
        for key in ["camera_0", "camera_0_depth", "camera_1","camera_1_depth","action", "stage"]:
            try:
                array = inspector.store[f"data/{key}"]
                if min_length < 0 or len(array) < min_length:
                    min_length = len(array)
            except KeyError:
                continue
        
        print(f"数据集最小长度: {min_length}")
        print(f"有效索引范围: 0-{min_length-1}")
        
        # 查看可能的边界帧
        frames_to_view = [0, min_length//2, min_length-1]
        
        for frame_idx in frames_to_view:
            camera_key = "camera_0"
            print(f"\n查看索引 {frame_idx} 的图像:")
            view_success = inspector.view_frame_at_index(frame_idx, camera_key)
            if not view_success:
                print(f"无法查看索引 {frame_idx} 的图像")
        
        # 查看最后一个可能的超出索引
        frame_idx = min_length
        camera_key = "camera_0"
        print(f"\n尝试查看超出范围的索引 {frame_idx}:")
        view_success = inspector.view_frame_at_index(frame_idx, camera_key)
        if not view_success:
            print(f"确认: 索引 {frame_idx} 超出有效范围")
        
        # # 功能7: 裁剪所有数组到最小长度
        # print("\n====== 功能7: 裁剪所有数组到最小长度 ======")
        # # 注意: 默认不更新episode_ends
        # truncate_success = inspector.truncate_to_min_length(update_episode_ends=False)
        # if truncate_success:
        #     print("\n裁剪操作完成，重新分析数据结构:")
        #     # 重新初始化检查器以加载更新后的数据
        #     inspector = ZarrInspector(zarr_path)
        #     inspector.print_detailed_storage_report()
        #     print("\n裁剪后树形结构:")
        #     inspector.build_tree_structure()
        #     inspector.print_tree()

        # 新增功能9: 直接转换data/stage为向量形式
        print("\n====== 功能9: 转换data/stage为向量形式 ======")
        
        # 查看转换前的stage数组
        print("转换前的stage数组信息:")
        inspector.inspect_key("data/stage")
        
        # 执行转换 - 直接修改原文件
        conversion_success = inspector.convert_stage_to_vector()
        
        if conversion_success:
            # 重新加载数据查看效果
            inspector = ZarrInspector(zarr_path)
            
            # 检查转换后的stage键
            print("\n查看转换后的stage数组:")
            inspector.inspect_key("data/stage")
            
            # 打印更新后的树形结构
            print("\n更新后的数据结构:")
            inspector.build_tree_structure()
            inspector.print_tree()

        # # 在 __main__ 部分添加
        # print("\n====== 功能10: 将stage数据拼接到action数组 ======")
        # # 查看转换前的action和stage数组
        # print("转换前的action数组信息:")
        # inspector.inspect_key("data/action")
        # print("\n转换前的stage数组信息:")
        # inspector.inspect_key("data/stage")

        # # 执行转换 - 直接修改原文件
        # append_success = inspector.append_stage_to_action()

        # if append_success:
        #     # 重新加载数据查看效果
        #     inspector = ZarrInspector(zarr_path)
            
        #     # 检查转换后的action键
        #     print("\n查看转换后的action数组:")
        #     inspector.inspect_key("data/action")
            
        #     # 打印更新后的树形结构
        #     print("\n更新后的数据结构:")
        #     inspector.build_tree_structure()
        #     inspector.print_tree()

        # 功能11，用于生成所需的额外数据字段。这个功能将基于现有数据计算并添加robot0_eef_pos、robot0_eef_rot_axis_angle、action以及它们的相对位姿表示
        # 将3D轴角旋转转换为6D旋转表示
        # 将7D动作（位置3D + 旋转3D + 夹爪1D）转换为10D动作（位置3D + 旋转6D + 夹爪1D）
        # ​​相对位姿计算​​：
        #     计算每个时间点相对于episode起始点的位置偏移
        #     计算每个时间点相对于episode起始点的旋转偏移（在6D空间中）
        # ​​episode处理​​：
        #     根据episode_ends划分数据
        #     对每个episode独立处理，确保相对位姿计算正确
        # ​​存储优化​​：
        #     使用分块存储提高读写效率
        #     支持覆盖已存在字段（可选）
        print("\n====== 功能11: 生成umi用数据字段 ======")
        # 新功能：创建精简数据集
        print("\n====== 创建精简数据集 =====")
        original_path = zarr_path
        new_path = os.path.join(os.path.dirname(original_path), "replay_buffer_new.zarr")
        
        # 确保新路径不存在
        if os.path.exists(new_path):
            print(f"删除已存在的旧数据集: {new_path}")
            shutil.rmtree(new_path)
        
        create_success = inspector.create_minimal_dataset(
            output_path=new_path,
            camera_key="camera_0",  # 使用YAML中指定的相机键名
            robot_id="robot0"
        )
        
        if create_success:
            print(f"\n✅ 精简数据集创建成功: {new_path}")
            # 检查新数据集
            new_inspector = ZarrInspector(new_path)
            new_inspector.print_detailed_storage_report()
            new_inspector.build_tree_structure()
            new_inspector.print_tree()
        else:
            print("\n❌ 精简数据集创建失败")


    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


    # """
    # 正确复制数据集结构和数据，特别是解决了输出数据集大小为0的问题。
    # 合并后的数据集会包含两个数据集的完整帧序列，以及合并后的episode_ends（第二数据集的索引已正确偏移）
    # """
    # try:
    #     # 新增功能8: 合并数据集
    #     print("\n====== 功能8: 合并两个数据集 ======")
        
    #     # 定义路径
    #     dataset1_path = "/home/hy/Desktop/dp_0314/data/bcb_0721_1/replay_buffer.zarr"
    #     dataset2_path = "/home/hy/Desktop/dp_0314/data/bcb_0721/replay_buffer.zarr"
    #     output_path = "/home/hy/Desktop/dp_0314/data/bcb_0721_full/replay_buffer.zarr"

    #     # dataset1_path = "/home/hy/Desktop/dp_0314/data/stage_0718/replay_buffer.zarr"
    #     # dataset2_path = "/home/hy/Desktop/dp_0314/data/stage_0718_1/replay_buffer.zarr"
    #     # output_path = "/home/hy/Desktop/dp_0314/data/stage_0718_full/replay_buffer.zarr"
        
    #     # 执行合并
    #     merge_success = merge_zarr_datasets(dataset1_path, dataset2_path, output_path)
        
    #     if merge_success:
    #         print("\n===== 检查合并后的数据集 =====")
    #         merged_inspector = ZarrInspector(output_path)
    #         merged_inspector.print_detailed_storage_report()
    #         merged_inspector.build_tree_structure()
    #         merged_inspector.print_tree()
            
    #         print("\n=== 合并后的episode_ends信息 ===")
    #         merged_inspector.inspect_key("meta/episode_ends")
    #         merged_inspector.inspect_episode_ends()

    # except Exception as e:
    #     print(f"发生错误: {str(e)}")
    #     import traceback
    #     traceback.print_exc()

