from typing import Dict, List, Union
import numbers
from queue import (Empty, Full)
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from diffusion_policy.shared_memory.shared_memory_util import ArraySpec, SharedAtomicCounter
from diffusion_policy.shared_memory.shared_ndarray import SharedNDArray


class SharedMemoryQueue:
    """
    A Lock-Free FIFO Shared Memory Data Structure.
    Stores a sequence of dict of numpy arrays.
    """
    class CallbackGuard:
        def __init__(self, callback, data):
            self.callback = callback
            self.data = data
        
        def __enter__(self):
            return self.data
        
        def __exit__(self, type, value, traceback):
            self.callback()
            del self.data
            del self.callback

    def __init__(self,
            shm_manager: SharedMemoryManager,
            array_specs: List[ArraySpec],
            buffer_size: int
        ):

        # create atomic counter
        write_counter = SharedAtomicCounter(shm_manager)
        read_counter = SharedAtomicCounter(shm_manager)
        
        # allocate shared memory
        shared_arrays = dict()
        for spec in array_specs:
            key = spec.name
            assert key not in shared_arrays
            array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(buffer_size,) + tuple(spec.shape),
                dtype=spec.dtype)
            shared_arrays[key] = array
        
        self.buffer_size = buffer_size
        self.array_specs = array_specs
        self.write_counter = write_counter
        self.read_counter = read_counter
        self.shared_arrays = shared_arrays
    
    
        # # 新增的代码
        # self.shared_arrays = shared_arrays
        # self.shm_manager = shm_manager  # 保存以便动态分配

    @classmethod
    def create_from_examples(cls, 
            shm_manager: SharedMemoryManager,
            examples: Dict[str, Union[np.ndarray, numbers.Number]], 
            buffer_size: int
            ):
        specs = list()
        for key, value in examples.items():
            shape = None
            dtype = None
            if isinstance(value, np.ndarray):
                shape = value.shape
                dtype = value.dtype
                assert dtype != np.dtype('O')
            elif isinstance(value, numbers.Number):
                shape = tuple()
                dtype = np.dtype(type(value))
            else:
                raise TypeError(f'Unsupported type {type(value)}')

            spec = ArraySpec(
                name=key,
                shape=shape,
                dtype=dtype
            )
            specs.append(spec)

        obj = cls(
            shm_manager=shm_manager,
            array_specs=specs,
            buffer_size=buffer_size
            )
        return obj
    
    def qsize(self):
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        return n_data
    
    def empty(self):
        n_data = self.qsize()
        return n_data <= 0
    
    def clear(self):
        self.read_counter.store(self.write_counter.load())
    
    def put(self, data: Dict[str, Union[np.ndarray, numbers.Number]]):
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        if n_data >= self.buffer_size:
            raise Full()
        
        next_idx = write_count % self.buffer_size

        # write to shared memory
        for key, value in data.items():

            if key not in self.shared_arrays:
                print(f'Key {key} not found in shared arrays, available keys: {list(self.shared_arrays.keys())}')
                raise KeyError(f'Key {key} not found in shared arrays，请检查键名是否正确')
                # 强制预先定义数据结构，防止意外内存膨胀，避免运行时内存分配开销 
                # 写入写出快50-100倍
                # # 动态分配新键的内存
                # if isinstance(value, np.ndarray):
                #     spec = ArraySpec(name=key, shape=value.shape, dtype=value.dtype)
                # elif isinstance(value, numbers.Number):
                #     spec = ArraySpec(name=key, shape=(), dtype=np.dtype(type(value)))
                # else:
                #     raise TypeError(f"Unsupported value type {type(value)} for key '{key}'")

                # array = SharedNDArray.create_from_shape(
                #     mem_mgr=self.shm_manager,
                #     shape=(self.buffer_size,) + spec.shape,
                #     dtype=spec.dtype
                # )
                # self.shared_arrays[key] = array

            arr: np.ndarray
            arr = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                arr[next_idx] = value
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)

        # update idx
        self.write_counter.add(1)
    
    def get(self, out=None) -> Dict[str, np.ndarray]:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        if out is None:
            out = self._allocate_empty()

        next_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            np.copyto(out[key], arr[next_idx])
        
        # update idx
        self.read_counter.add(1)
        return out
    
    def get_next_view(self) -> Dict[str, np.ndarray]:
        """获取下一个写入位置的直接引用"""
        """
        Get reference to the next element to write
        for zero-copy writing
        """
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        if n_data >= self.buffer_size:
            raise Full()
        
        next_idx = write_count % self.buffer_size
        out = dict()
        for key, value in self.shared_arrays.items():
            arr = value.get()
            out[key] = arr[next_idx]

        return out

    def put_next_view(self, data: Dict[str, Union[np.ndarray, numbers.Number]]):
        """提交已通过视图写入的数据"""
        # 无需数据拷贝，直接更新计数器
        """
        Used in conjunction with get_next_view
        for zero-copy writing
        """
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        if n_data >= self.buffer_size:
            raise Full()
        
        next_idx = write_count % self.buffer_size
        # write to shared memory
        for key, value in data.items():
            arr: np.ndarray
            arr = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                # assumed already written to the array
                pass
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)

        # update idx
        self.write_counter.add(1)

    # 安全访问模式 回调上下文管理器
    def get_view(self) -> CallbackGuard:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()
        
        next_idx = read_count % self.buffer_size
        data = dict()
        for key, value in self.shared_arrays.items():
            arr = value.get()
            data[key] = arr[next_idx]
        
        return self.CallbackGuard(
            callback=lambda: self.read_counter.add(1), # 自动回调
            data=data)

    def get_k(self, k, out=None) -> Dict[str, np.ndarray]:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()
        assert k <= n_data

        out = self._get_k_impl(k, read_count, out=out)
        self.read_counter.add(k)
        return out

    def get_all(self, out=None) -> Dict[str, np.ndarray]:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        out = self._get_k_impl(n_data, read_count, out=out)
        self.read_counter.add(n_data)
        return out
    
    def _get_k_impl(self, k, read_count, out=None) -> Dict[str, np.ndarray]:
        if out is None:
            out = self._allocate_empty(k)

        curr_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            target = out[key]

            start = curr_idx
            end = min(start + k, self.buffer_size)
            target_start = 0
            target_end = (end - start)
            target[target_start: target_end] = arr[start:end]

            remainder = k - (end - start)
            if remainder > 0:
                # wrap around
                start = 0
                end = start + remainder
                target_start = target_end
                target_end = k
                target[target_start: target_end] = arr[start:end]

        return out
    
    def _allocate_empty(self, k=None):
        result = dict()
        for spec in self.array_specs:
            shape = spec.shape
            if k is not None:
                shape = (k,) + shape
            result[spec.name] = np.empty(
                shape=shape, dtype=spec.dtype)
        return result
