# 已实现保存rgb+mask结果到内存映射文件
from typing import Optional, Callable, Dict
import os
import enum
import time
import json
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from diffusion_policy.shared_memory.shared_ndarray import SharedNDArray
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from diffusion_policy.real_world.video_recorder import VideoRecorder, VideoRecorder_new

# from diffusion_policy.real_world.stagepredictorprocess import EnhancedStagePredictorProcess


class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4

class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            serial_number,
            resolution=(1280,720),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            enable_sam2 = False,
            get_max_k=30,
            advanced_mode_config=None,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            # video_recorder: Optional[VideoRecorder] = None,
            video_recorder: Optional[VideoRecorder_new] = None,

            verbose=False,
        ):
        super().__init__()

        # sam2
        self.enable_sam2 = enable_sam2
        self.serial_number = serial_number
        if self.enable_sam2:
            self.flag = 90
            if self.serial_number == '241122074919':
                self.sam_server_address = "tcp://127.0.0.1:4245"
                print(f"self.serial_number is {self.serial_number},set server address to {self.sam_server_address}")
            elif self.serial_number == '243522074975':
                self.sam_server_address = "tcp://127.0.0.1:4246"
                print(f"self.serial_number is {self.serial_number},set server address to {self.sam_server_address}")
            elif self.serial_number == '241122306029':
                self.sam_server_address = "tcp://127.0.0.1:4247"
                print(f"self.serial_number is {self.serial_number},set server address to {self.sam_server_address}")
            else:
                print("Invalid serial number for SAM client. serial ==",serial_number)

        if put_fps is None:
            put_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples['color'] = np.empty(
                shape=shape+(3,), dtype=np.uint8)
        if enable_depth:
            examples['depth'] = np.empty(
                shape=shape, dtype=np.uint16)
        if enable_infrared:
            examples['infrared'] = np.empty(
                shape=shape, dtype=np.uint8)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0
        # sam2
        if self.enable_sam2:
            examples['color_mask'] = np.empty(
                shape=shape, dtype=np.uint8)
            examples['timestamp_mask'] = 0.0
        
        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None 
                else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': rs.option.exposure.value,
            'option_value': 0.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(7,),
                dtype=np.float64)
        intrinsics_array.get()[:] = 0

        # create video recorder
        if video_recorder is None:
            # video_recorder = VideoRecorder.create_h264(
            #     fps=capture_fps, 
            #     codec='h264',
            #     input_pix_fmt='bgr24', 
            #     crf=18,
            #     thread_type='FRAME',
            #     thread_count=1)
            # # 使用VideoRecorder_new的创建方法 default to nvenc GPU encoder
            try:
                # 尝试使用GPU编码
                video_recorder = VideoRecorder_new.create_hevc_nvenc(
                    shm_manager=shm_manager, # 需要传入共享内存管理器
                    fps=capture_fps, 
                    input_pix_fmt='bgr24', 
                    bit_rate=3000 * 1000
                )
            except Exception as e:
                print(f"无法初始化HEVC_NVENC编码器: {e}, 回退到CPU编码")
                # 使用CPU编码
                video_recorder = VideoRecorder_new.create_h264(
                    shm_manager=shm_manager,
                    fps=capture_fps,
                    input_pix_fmt='bgr24',
                    bit_rate=3000 * 1000
                )
        assert video_recorder.fps == capture_fps

        # copied variables
        self.shm_manager = shm_manager

        # copied variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None

        self.color_ts_buffer_path = f"/dev/shm/{self.serial_number}_color_ts.npy"
        self.index_buffer_path = f"/dev/shm/{self.serial_number}_index.npy"

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array
    
    @staticmethod
    def get_connected_devices_serial():
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                if product_line == 'D400':
                    # only works with D400 series
                    serials.append(serial)
        serials = sorted(serials)
        return serials

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop() # 确保总是调用stop

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time

        # # 创建数据示例用于初始化VideoRecorder_new
        # shape = self.resolution[::-1]
        # data_example = np.empty(shape=shape+(3,), dtype=np.uint8)
        
        # # 启动视频记录器
        # self.video_recorder.start(
        #     shm_manager=self.shm_manager, 
        #     data_example=data_example)
        # # must start video recorder first to create share memories
        # 添加初始化（必须传入数据形状）
        if isinstance(self.video_recorder, VideoRecorder_new):
            data_example = np.empty(
                shape=self.resolution[::-1] + (3,),  # (h, w, 3)
                dtype=np.uint8
            )
            self.video_recorder.start(
                shm_manager=self.shm_manager,
                data_example=data_example
            )


        # 启动相机进程
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        # 先停止相机子进程
        self.stop_event.set()

        # 再停止视频录制器
        if isinstance(self.video_recorder, VideoRecorder_new):
            self.video_recorder.stop()  # 会触发进程退出

        if wait:
            self.end_wait()

    def start_wait(self):
        if isinstance(self.video_recorder, VideoRecorder_new):
            self.video_recorder.start_wait()  # 等待录制器就绪

        self.ready_event.wait()

    def end_wait(self):
        self.join()

        if isinstance(self.video_recorder, VideoRecorder_new):
            self.video_recorder.end_wait()  # 等待录制器结束
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    
    
    
    
    def get_sam2(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)
    
    # ========= user API ===========
    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put({
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': option.value,
            'option_value': value
        })
    
    def set_exposure(self, exposure=None, gain=None):
        """
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
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0,0] = fx
        mat[1,1] = fy
        mat[0,2] = ppx
        mat[1,2] = ppy
        return mat

    def get_depth_scale(self):
        assert self.ready_event.is_set()
        scale = self.intrinsics_array.get()[-1]
        return scale
    
    def start_recording(self, video_path: str, start_time: float=-1):
        assert self.enable_color

        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')

        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })

        # # 使用VideoRecorder_new的API启动录制
        # self.video_recorder.start_recording(video_path, start_time)
    
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })

        # # 使用VideoRecorder_new的API停止录制
        # self.video_recorder.stop_recording()
    
    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })
    
    # ========= interval API ===========
    def run(self):

        # # # 使用上下文管理器安全初始化CUDA
        # # ========= 安全初始化 CUDA =========
        # try:
        #     import torch
        #     if torch.cuda.is_available():
        #         # 使用设备上下文管理器安全初始化
        #         with torch.cuda.device(0):
        #             # 创建虚拟张量初始化 CUDA 上下文
        #             dummy_tensor = torch.zeros(1).cuda()
        #             print(f"[{self.serial_number}] CUDA context initialized in process")
        # except Exception as e:
        #     print(f"CUDA 初始化失败: {e}")
        #     # 即使 CUDA 初始化失败，也继续运行
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        w, h = self.resolution
        fps = self.capture_fps
        align = rs.align(rs.stream.color)
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        if self.enable_color:
            rs_config.enable_stream(rs.stream.color, 
                w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            rs_config.enable_stream(rs.stream.depth, 
                w, h, rs.format.z16, fps)
        if self.enable_infrared:
            rs_config.enable_stream(rs.stream.infrared,
                w, h, rs.format.y8, fps)

        # # 启动预测器进程
        # if self.enable_stage_predictor:
        #     self.predictor_process.start()

        if self.enable_sam2:
            ################################ sam2输入 ###############################
            color_mmap = None
            timestamp_mmap = None
            index_mmap = None
            sam_input_current_index = -1
            color_ts_dtype = np.dtype([
                ('timestamp', np.float64),          # 时间戳
                ('color', np.uint8, (480, 640, 3))  # (h, w, 3)BGR三通道
            ], align=True)
            index_dtype = np.dtype([
                ('index', np.int64), 
                ('timestamp', np.float64)
            ], align=True)

            # Create or overwrite memory-mapped files

            if os.path.exists(self.color_ts_buffer_path):
                print(f"File {self.color_ts_buffer_path} exists, keeping it.")# deleting
            #     os.remove(self.color_ts_buffer_path)
            if os.path.exists(self.index_buffer_path):
                print(f"File {self.index_buffer_path} exists, keeping it.")
            #     os.remove(self.index_buffer_path)
            # 初始化内存映射文件
            if not os.path.exists(self.color_ts_buffer_path):
                color_ts_mmap = np.lib.format.open_memmap(
                    self.color_ts_buffer_path,
                    dtype=color_ts_dtype,
                    mode='w+',
                    shape=(128,)
                )
                print(f"创建颜色时间戳缓冲区: {self.color_ts_buffer_path}, shape: {color_ts_mmap.shape}")
            else:
                color_ts_mmap = np.load(self.color_ts_buffer_path, mmap_mode='r+')
                assert color_ts_mmap.dtype == color_ts_dtype, "输入缓冲区数据类型不匹配"
                assert color_ts_mmap.shape == (128,), "输入缓冲区形状不匹配"
            if not os.path.exists(self.index_buffer_path):
                index_mmap = np.lib.format.open_memmap(
                    self.index_buffer_path,
                    dtype=index_dtype,
                    mode='w+',
                    shape=(1,)
                )
                print(f"创建索引缓冲区: {self.index_buffer_path}, shape: {index_mmap.shape}")
            else:
                index_mmap = np.load(self.index_buffer_path, mmap_mode='r+')
                assert index_mmap.dtype == index_dtype, "索引缓冲区数据类型不匹配"
                assert index_mmap.shape == (1,), "索引缓冲区形状不匹配"
            # 定义输入索引
            sam_input_current_index = -1
            
            ################################ sam2输出 ###############################

            mask_ts_path = f"/dev/shm/{self.serial_number}_mask_ts.npy"
            mask_index_path = f"/dev/shm/{self.serial_number}_mask_index.npy"
            print(f"内存映射文件路径: {mask_ts_path}, {mask_index_path}")

            # 定义合并后的数据结构
            mask_ts_dtype = np.dtype([
                ('timestamp', np.float64),   # 单通道mask尺寸
                ('mask', np.uint8, (h, w))
            ], align=True)
            mask_index_dtype = np.dtype([
                ('index', np.int64), 
                ('timestamp', np.float64)
            ], align=True)

            # 加载合并后的内存映射文件路径
            if not os.path.exists(mask_ts_path):
                print(f"请先创建内存映射文件: {mask_ts_path}")
            if not os.path.exists(mask_index_path):
                print(f"请先创建内存映射文件: {mask_index_path}")
            mask_ts_mmap = np.load(
                mask_ts_path, 
                mmap_mode='r'# 如果文件不存在，则创建新的

            )
            mask_index_mmap = np.load(
                mask_index_path,
                mmap_mode='r'# 必须r+模式才能同步？no，将r+改r只读模式，加速25%读取速度
                
            )
            # 验证合并后的数据结构
            assert mask_ts_mmap.dtype == mask_ts_dtype, "合并缓冲区数据类型不匹配"
            assert mask_ts_mmap.shape == (128,), "合并缓冲区形状不匹配"
            assert mask_index_mmap.dtype == mask_index_dtype, "索引缓冲区数据类型不匹配"
            assert mask_index_mmap.shape == (1,), "索引缓冲区形状不匹配"
            
            ################################ sam2输出读取 ###############################
            def read_sam2_data():
                # 原子读取索引
                index_entry = np.copy(mask_index_mmap[0])
                current_idx = index_entry['index']
                read_timestamp = index_entry['timestamp']

                # 使用双重校验保证数据一致性
                data_entry = mask_ts_mmap[current_idx]
                if data_entry['timestamp'] == read_timestamp:
                    timestamp = data_entry['timestamp']
                    mask = np.copy(data_entry['mask'])
                else:
                    try:
                        print(f"数据不一致，当前索引: {current_idx}, 当前时间戳: {read_timestamp}, 数据时间戳: {data_entry['timestamp']}")
                        # 回退机制：直接使用索引中的时间戳
                        prev_idx = current_idx-1 if current_idx > 0 else 127
                        timestamp = mask_ts_mmap[prev_idx]['timestamp']
                        mask = np.copy(mask_ts_mmap[prev_idx]['mask'])
                        print(f"尝试回退，当前索引: {prev_idx}, 当前时间戳: {read_timestamp}, 数据时间戳: {timestamp}")
                    except Exception as e:
                        print(f"读取失败: Error reading from memory map: {e}")
                        # 添加异常恢复机制
                        mask = np.zeros((h, w), dtype=np.uint8)

                # print("single中mask的形状:",mask.shape,"self.resolution[::-1]",self.resolution[::-1])
                # print("rgb的形状:",data['color'].shape)
                # if mask.shape != self.resolution[::-1]:
                #     mask = cv2.resize(
                #         mask, 
                #         self.resolution[::-1],  # (width, height)
                #         interpolation=cv2.INTER_NEAREST
                #     )
                #     raise ValueError(f"Mask shape mismatch: expected {(h, w)}, got {mask.shape}")
                return mask, timestamp, current_idx




        try:
            rs_config.enable_device(self.serial_number)

            # start pipeline
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)

            # # report global time
            # # https://github.com/IntelRealSense/librealsense/pull/3909
            # d = pipeline_profile.get_device().first_color_sensor()
            # d.set_option(rs.option.global_time_enabled, 1)

            # # setup advanced mode
            # if self.advanced_mode_config is not None:
            #     json_text = json.dumps(self.advanced_mode_config)
            #     device = pipeline_profile.get_device()
            #     advanced_mode = rs.rs400_advanced_mode(device)
            #     advanced_mode.load_json(json_text)

            # ==== 修改1：使用通用传感器查询方式替换 first_color_sensor() ====
            device = pipeline_profile.get_device()
            sensors = device.query_sensors()
            color_sensor = None
            for sensor in sensors:
                # 使用D405兼容的传感器识别方式
                if hasattr(sensor, 'is_color_sensor') and sensor.is_color_sensor():
                    color_sensor = sensor
                    break
                    
            if color_sensor is not None:
                # 尝试设置全局时间选项
                try:
                    color_sensor.set_option(rs.option.global_time_enabled, 1)
                except Exception as e:
                    print(f"设置全局时间选项失败: {e}")
            else:
                print(f"警告: 未找到彩色传感器，跳过时间戳设置 ({self.serial_number})")
            
            # ==== 修改2：确保应用高级配置 ====
            if self.advanced_mode_config is not None:
                try:
                    advanced_mode = rs.rs400_advanced_mode(device)
                    json_text = json.dumps(self.advanced_mode_config)
                    advanced_mode.load_json(json_text)
                except Exception as e:
                    print(f"应用高级模式配置失败: {e}")


            # get
            color_stream = pipeline_profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ['fx', 'fy', 'ppx', 'ppy', 'height', 'width']
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = getattr(intr, name)

            if self.enable_depth:
                depth_sensor = pipeline_profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                self.intrinsics_array.get()[-1] = depth_scale
            
            # one-time setup (intrinsics etc, ignore for now)
            if self.verbose:
                print(f'[SingleRealsense {self.serial_number}] Main loop started.')

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                # wait for frames to come in
                frameset = pipeline.wait_for_frames()
                receive_time = time.time()
                # align frames to color
                frameset = align.process(frameset)

                # grab data
                data = dict()
                data['camera_receive_timestamp'] = receive_time
                # realsense report in ms
                data['camera_capture_timestamp'] = frameset.get_timestamp() / 1000
                if self.enable_color:
                    color_frame = frameset.get_color_frame()
                    data['color'] = np.asarray(color_frame.get_data())
                    t = color_frame.get_timestamp() / 1000
                    data['camera_capture_timestamp'] = t
                    # print('device', time.time() - t)
                    # print(color_frame.get_frame_timestamp_domain())
                if self.enable_depth:
                    data['depth'] = np.asarray(
                        frameset.get_depth_frame().get_data())
                if self.enable_infrared:
                    data['infrared'] = np.asarray(
                        frameset.get_infrared_frame().get_data())



                if self.enable_sam2:
                    # 先写入再读取
                    try:
                        # 原子化操作：读取当前最新索引
                        sam_input_current_index += 1
                        sam_input_current_index = (sam_input_current_index) % 128
                        
                        # 1. 先写入数据缓冲区, 原子化操作
                        # 强制持久化避免缓存: /dev/shm是内存文件系统，fsync会强制刷盘，完全违背共享内存设计初衷_os.fsync(color_mmap._mmap.fileno())
                        entry = np.array(
                            ( data['camera_capture_timestamp'], data['color']),
                            dtype=color_ts_dtype
                        )
                        color_ts_mmap[sam_input_current_index] = entry
                        color_ts_mmap.flush()       # 先刷新数据
                        
                        # 2. 最后更新索引
                        new_entry = np.array(
                            (sam_input_current_index, data['camera_capture_timestamp']), 
                            dtype=index_dtype
                        )
                        index_mmap[0] = new_entry   # 写入新索引
                        index_mmap.flush()          # 最后刷新索引
                    except Exception as e:
                        print(f"写入失败: Error writing to memory map: {e}")
                        # 添加异常恢复机制
                        sam_input_current_index = (sam_input_current_index + 127) % 128  # 回滚索引
                    
                    if index_mmap is not None:
                        mask, timestamp, newest_idx = read_sam2_data()
                        # 优化方案（保持单通道） 效果：减少约15%的CPU占用，消除OpenCV转换开销
                        # mask_resize = cv2.resize(
                        #         mask.copy(), 
                        #         (240, 320)  # (width, height)
                        #         interpolation=cv2.INTER_NEAREST
                        #     )
                        # data['color_mask'] = mask_resize  # 下游按需转换
                        data['color_mask'] = mask.copy()
                        data['timestamp_mask'] = timestamp
                    else:
                        data['color_mask'] = np.zeros((h, w), dtype=np.uint8)
                        data['timestamp_mask'] = data['camera_capture_timestamp']


                
                # apply transform
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:                
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                            timestamps=[receive_time],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            next_global_idx=put_idx,
                            allow_negative=True
                        )

                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        # put_data['timestamp'] = put_start_time + step_idx / self.put_fps
                        put_data['timestamp'] = receive_time
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False)

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()
                
                # put to vis
                vis_data = data
                if self.vis_transform == self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                self.vis_ring_buffer.put(vis_data, wait=False)
                
                # record frame
                rec_data = data
                if self.recording_transform == self.transform:
                    rec_data = put_data
                elif self.recording_transform is not None:
                    rec_data = self.recording_transform(dict(data))

                # 修改后 - 使用零拷贝接口
                if isinstance(self.video_recorder, VideoRecorder_new) and self.video_recorder.is_ready():
                    img_buffer = self.video_recorder.get_img_buffer()
                    np.copyto(img_buffer, rec_data['color'])  # 零拷贝复制
                    self.video_recorder.write_img_buffer(img_buffer, frame_time=receive_time)
                elif self.video_recorder.is_ready():  # 旧版兼容
                    self.video_recorder.write_frame(rec_data['color'], 
                        frame_time=receive_time)

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] FPS {frequency}')

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.SET_COLOR_OPTION.value:
                        sensor = pipeline_profile.get_device().first_color_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        sensor = pipeline_profile.get_device().first_depth_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path'])
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                        if isinstance(self.video_recorder, VideoRecorder_new):
                            # 使用新API
                            self.video_recorder.start_recording(video_path, start_time)
                        else:  # 旧版兼容
                            self.video_recorder.start(video_path, start_time)
                        # self.video_recorder.start(video_path, start_time=start_time)
                        # self.video_recorder.start_recording(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        if isinstance(self.video_recorder, VideoRecorder_new):
                            self.video_recorder.stop_recording()
                        else:
                            self.video_recorder.stop()
                        # self.video_recorder.stop()
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']
                        # self.ring_buffer.clear()

                iter_idx += 1
        finally:


            # 统一资源清理
            if isinstance(self.video_recorder, VideoRecorder_new):
                self.video_recorder.stop_recording()
            else:
                self.video_recorder.stop()
            # self.video_recorder.stop()
            
            # 关闭 RealSense 管道
            pipeline.stop()
            rs_config.disable_all_streams()
            self.ready_event.set()
            # if self.enable_color:
            if self.enable_sam2:
                # Ensure all data is flushed
                if color_mmap is not None:
                    color_mmap.flush()
                if timestamp_mmap is not None:
                    timestamp_mmap.flush()
                if index_mmap is not None:
                    index_mmap.flush()
        
        if self.verbose:
            print(f'[SingleRealsense {self.serial_number}] Exiting worker process.')
