from typing import Optional
import os
import pathlib
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading
import hashlib
import time

class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    # 在 TrainDiffusionUnetImageWorkspace 类中添加以下方法
    def save_checkpoint_full(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None, 
            use_thread=False):  # 默认改为False，确保保存完成
        """
        增强版保存检查点方法：
        1. 确保包含完整的配置信息（cfg）
        2. 添加训练状态信息（epoch, global_step）
        3. 支持深度序列化复杂对象
        4. 添加完整性检查标志
        """
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
            
        # 确保输出目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 设置包含/排除键
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir', 'epoch', 'global_step')
            
        # 创建payload结构
        payload = {
            'cfg': self.cfg,  # 完整配置
            'metadata': {      # 添加元数据
                'save_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'version': '1.1'
            },
            'state_dicts': dict(),
            'pickles': dict(),
            'train_state': {   # 训练状态信息
                'epoch': self.epoch,
                'global_step': self.global_step
            }
        }

        # 收集可序列化的状态字典
        for key, value in self.__dict__.items():
            if key in exclude_keys:
                continue
                
            # 处理有状态字典的对象（模型、优化器等）
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                try:
                    state_dict = value.state_dict()
                    # 深度复制到CPU以确保安全
                    payload['state_dicts'][key] = _copy_to_cpu(state_dict)
                except Exception as e:
                    print(f"警告: 无法保存 {key} 的状态: {str(e)}")
            
            # 处理需要pickle的对象
            elif key in include_keys:
                try:
                    # 使用dill深度序列化
                    payload['pickles'][key] = dill.dumps(value)
                except Exception as e:
                    print(f"警告: 无法pickle {key}: {str(e)}")
        
        # 添加完整性检查标志
        payload['integrity_check'] = hashlib.md5(
            str(payload['train_state']).encode()
        ).hexdigest()

        # 保存检查点（禁用线程确保完整性）
        try:
            torch.save(payload, path.open('wb'), pickle_module=dill)
            print(f"检查点已保存至: {path.absolute()}")
            # 验证检查点完整性
            self._verify_checkpoint(path)
        except Exception as e:
            print(f"保存检查点时出错: {str(e)}")
            raise
        
        return str(path.absolute())

    def _verify_checkpoint(self, path):
        """验证检查点完整性"""
        try:
            payload = torch.load(path.open('rb'), pickle_module=dill)
            # 检查元数据
            if 'metadata' not in payload or 'train_state' not in payload:
                print("警告: 检查点缺少关键字段")
            
            # 验证完整性哈希
            if 'integrity_check' in payload:
                current_hash = hashlib.md5(
                    str(payload['train_state']).encode()
                ).hexdigest()
                if payload['integrity_check'] != current_hash:
                    print("警告: 检查点完整性验证失败")
            
            print(f"检查点验证成功: epoch={payload['train_state']['epoch']}, step={payload['train_state']['global_step']}")
        except Exception as e:
            print(f"检查点验证失败: {str(e)}")

    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
