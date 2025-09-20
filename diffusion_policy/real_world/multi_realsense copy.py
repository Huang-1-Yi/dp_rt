from typing import List, Optional, Union, Dict, Callable
import numbers
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import pyrealsense2 as rs
from diffusion_policy.real_world.single_realsense import SingleRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder, VideoRecorder_new

# repeat_to_list 从末端前移
def repeat_to_list(x, n: int, cls):
    """
    功能：将输入参数统一转换为长度为n的列表
    """
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n
    return x

class MultiRealsense:
    """
    ​多摄像头管理：支持同时控制多个RealSense相机（自动检测或指定序列号）
    ​硬件加速：使用pyrealsense2库直接操作相机硬件
    ​共享内存：通过SharedMemoryManager实现进程间大数据共享（比如图像帧）
    ​视频录制：支持多路视频同步录制（每个摄像头独立视频文件）
    ​参数配置：可批量设置曝光、白平衡、分辨率等相机参数
    ​数据获取：提供两种数据格式（原始数据get()和可视化数据get_vis()）
    """
    def __init__(self,
        serial_numbers: Optional[List[str]]=None,
        shm_manager: Optional[SharedMemoryManager]=None,
        resolution=(640,480),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        enable_color=True,
        enable_depth=False,
        enable_infrared=False,
        enable_sam2 =False,
        get_max_k=30,
        advanced_mode_config: Optional[Union[dict, List[dict]]]=None,
        transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        recording_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        # video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]]=None,
        video_recorder: Optional[Union[VideoRecorder_new, List[VideoRecorder_new]]]=None,
        
        verbose=False
        ):
        self.enable_sam2 = enable_sam2
        # 若未提供shm_manager则自动创建，用于高效传递摄像头数据
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        
        # 若serial_numbers=None，调用SingleRealsense.get_connected_devices_serial()自动发现设备
        if serial_numbers is None:
            serial_numbers = SingleRealsense.get_connected_devices_serial()
        
        n_cameras = len(serial_numbers)
        # 通过repeat_to_list将配置参数扩展为与摄像头数量匹配的列表
        advanced_mode_config = repeat_to_list(advanced_mode_config, n_cameras, dict)
        transform = repeat_to_list(transform, n_cameras, Callable)
        vis_transform = repeat_to_list(vis_transform, n_cameras, Callable)
        recording_transform = repeat_to_list(recording_transform, n_cameras, Callable)

        # video_recorder = repeat_to_list(video_recorder, n_cameras, VideoRecorder)
        video_recorder = repeat_to_list(video_recorder, n_cameras, VideoRecorder_new)

        cameras = dict()
        # 为每个摄像头创建SingleRealsense对象，保存到self.cameras字典
        for i, serial in enumerate(serial_numbers):
            print(f"相机{i}编号为{serial}")
            cameras[serial] = SingleRealsense(
                shm_manager=shm_manager,
                serial_number=serial,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                enable_color=enable_color,
                enable_depth=enable_depth,
                enable_infrared=enable_infrared,
                enable_sam2 = self.enable_sam2,
                get_max_k=get_max_k,
                advanced_mode_config=advanced_mode_config[i],
                transform=transform[i],
                vis_transform=vis_transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorder[i],
                verbose=verbose
            )
        
        self.cameras = cameras
        self.shm_manager = shm_manager

    # 通过with MultiRealsense(...) as cameras:语法自动管理资源
    # 进入时启动所有摄像头，退出时自动停止并释放资源
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def n_cameras(self):
        return len(self.cameras)
    
    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False
        return is_ready

    # 异步启动/停止所有摄像头
    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
        
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)
        
        if wait:
            self.stop_wait()

    # 阻塞直到所有摄像头完成启动/停止
    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()
    
    # 创建一个get_obs_resize函数，传入k和out的同时，传入一个shape，函数会像get一样得到最新几帧
    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        按时间顺序返回多帧数据（T表示帧数）
        原始数据保留：直接返回摄像头捕获的原始分辨率、原始帧率数据。
        ​时间序列：包含时间维度T，可用于时序分析。
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
        return out

    # 创建一个get_obs_resize函数，传入k和out的同时，传入一个shape，函数会像get一样得到最新几帧
    def get_sam2(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        按时间顺序返回多帧数据（T表示帧数）
        原始数据保留：直接返回摄像头捕获的原始分辨率、原始帧率数据。
        ​时间序列：包含时间维度T，可用于时序分析。
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get_sam2(k=k, out=this_out)
            out[i] = this_out
        return out

    def get_vis(self, out=None):
        """
        返回适合实时可视化的数据格式
        降采样：降低分辨率以适应显示窗口
        ​格式转换：BGR → RGB、深度图伪彩色化
        ​维度压缩：移除时间维度（只保留最新帧）
        """
        results = list()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = dict()
                for key, v in out.items():
                    # use the slicing trick to maintain the array
                    # when v is 1D
                    this_out[key] = v[i:i+1].reshape(v.shape[1:])
            this_out = camera.get_vis(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = dict()
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
        return out

    def get_vis_sam2(self, out=None):
        """
        返回适合实时可视化的数据格式
        降采样：降低分辨率以适应显示窗口
        ​格式转换：BGR → RGB、深度图伪彩色化
        ​维度压缩：移除时间维度（只保留最新帧）
        """
        results = list()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = dict()
                for key, v in out.items():
                    # use the slicing trick to maintain the array
                    # when v is 1D
                    this_out[key] = v[i:i+1].reshape(v.shape[1:])
            this_out = camera.get_vis_sam2(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = dict()
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
        return out

    def set_color_option(self, option, value):
        """
        rs.option.exposure       # 曝光时间（微秒单位）
        rs.option.gain           # 感光增益
        rs.option.white_balance  # 白平衡色温
        rs.option.brightness     # 亮度
        """
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_color_option(option, value[i])

    # # 同时设置曝光和增益
    # exposures = [1000, 2000]
    # gains = [64, 80]
    # multi_cams.set_exposure(exposure=exposures, gain=gains)
    def set_exposure(self, exposure=None, gain=None):
        """
        手动/自动设置曝光时间和增益
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)
    
    def set_white_balance(self, white_balance=None):
        """
        设置白平衡（自动或固定值）
        """
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)
    
    def get_intrinsics(self):
        """
        获取摄像头内参矩阵（用于3D重建）
        """
        return np.array([c.get_intrinsics() for c in self.cameras.values()])
    
    def get_depth_scale(self):
        return np.array([c.get_depth_scale() for c in self.cameras.values()])

    # 为每个摄像头创建MP4文件
    def start_recording(self, video_path: Union[str, List[str]], start_time: float):
        if isinstance(video_path, str):
            # directory
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = list()
            for i in range(self.n_cameras):
                video_path.append(
                    str(video_dir.joinpath(f'{i}.mp4').absolute()))
        assert len(video_path) == self.n_cameras

        for i, camera in enumerate(self.cameras.values()):
            camera.start_recording(video_path[i], start_time)

    # 停止所有录制
    def stop_recording(self):
        for i, camera in enumerate(self.cameras.values()):
            camera.stop_recording()
    
    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)

