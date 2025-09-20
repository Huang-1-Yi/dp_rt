# 版本：2025.05.26
# 作者：黄一
from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.franka_interpolation_controller import FrankaInterpolationController

from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator, 
    TimestampActionAccumulator,
    align_timestamps,
    TimestampImageAccumulator,
    TimestampMaskAccumulator,
    TimestampDepthAccumulator,
    TimestampPointCloudAccumulator
)

from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)

import psutil # 内存显示
import cv2

DEFAULT_OBS_KEY_MAP = {    # 默认观测键映射                 建立观测键和键映射的关系
    'ActualTCPPose': 'robot_eef_pose',                  # 实际TCP姿势 机器人末端执行器姿势 
    'ActualQ': 'robot_joint',                           # 实际Q 机器人关节
    'ActualQd': 'robot_joint_vel',                      # 实际Qd 机器人关节速度
    'ActualGripperstate': 'robot_gripper',              # 新增,与gripper不同,这个是夹爪状态             
    'step_idx': 'step_idx',                             # 步骤索引    
    'timestamp': 'timestamp'                            # 时间戳
}
class RealEnvFranka:
    def __init__(self, 
            # required params
            output_dir,                                 # 输出目录
            robot_ip,                                   # 机器人IP地址
            # env params
            verbose             = False,
            frequency           = 10,                   # 控制频率
            n_obs_steps         = 2,                    # 观察步数
            # obs
            obs_image_resolution= (640,480),            # 观察图像分辨率
            max_obs_buffer_size = 30,                   # 最大观察缓冲区大小
            camera_serial_numbers=None,                 # 相机序列号
            obs_key_map         =DEFAULT_OBS_KEY_MAP,   # 观察键映射
            obs_float32         =False,                 # 观察数据类型
            # action
            max_pos_speed       = 0.2,                 # 最大位置速度0.25
            max_rot_speed       = 0.5,                  # 最大旋转速度0.6
            # robot
            robot_obs_latency   = 0.005,                # franka用，机器人观测延迟
            # tcp_offset          = 0.13,                 # ur用，TCP偏移
            init_joints=False,                          # 是否初始化关节
            # video capture params
            video_capture_fps   = 30,                   # 视频捕捉帧率
            video_capture_resolution=(640,480),         # 视频捕捉分辨率
            # saving params
            record_raw_video    = True,                 # 是否记录原始视频，默认是
            thread_per_video    = 2,                    # 每个视频的线程数
            video_crf           = 21,                   # 视频质量
            # vis params
            enable_multi_cam_vis=True,                  # 是否启用多相机可视化
            multi_cam_vis_resolution=(640,480),         # 多相机可视化分辨率
            # shared memory
            shm_manager         = None,                 # 共享内存管理器
            
            #########深度新增参数################
            enable_depth        = False,                 # True False 新增参数：是否启用深度
            depth_scale         = 1000.0,               # 深度数据缩放因子
            depth_dtype=np.uint16,                      # 深度数据类型
            enable_pointcloud   = False,
            # enable_pointcloud=True                    # 新增参数：是否启用点云
            enable_sam2         = False,                       # 新增参数：是否启用sam2
            ):
        
        self.enable_sam2 = enable_sam2           # 是否启用sam2

        ###########################################################
        ##########相机的参数设置和可视化
        ###########################################################
        assert frequency <= video_capture_fps           # 确保控制频率不超过视频捕捉帧率
        output_dir = pathlib.Path(output_dir)           # 将输出目录转换为Path对象
        assert output_dir.parent.is_dir()               # 确保输出目录的父目录存在
        video_dir = output_dir.joinpath('videos')       # 创建视频目录
        video_dir.mkdir(parents=True, exist_ok=True)    # 确保视频目录存在
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())       # 创建重放缓冲区路径
        # 如果内存不足,在ReplayBuffer初始化时启用压缩,ReplayBuffer.create_from_path设置compressor=Blosc(cname='zstd', clevel=3)
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')              # 创建重放缓冲区

        if shm_manager is None:                         # 如果没有提供共享内存管理器
            shm_manager = SharedMemoryManager()         # 创建一个新的共享内存管理器
            shm_manager.start()                         # 启动共享内存管理器
        if camera_serial_numbers is None:               # 如果没有提供相机序列号
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()  # 获取连接的设备序列号

        # obs output rgb，需要由bgr转换为rgb
        color_tf = get_image_transform(
            input_res=video_capture_resolution,         # 图像的分辨率
            output_res=obs_image_resolution,            # 观测的大小
            bgr_to_rgb=True)                            # 获取图像转换函数
        color_transform = color_tf
        if obs_float32:                                 # 如果观察数据类型为float32
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255        # 将图像转换为float32类型并归一化

        def transform(data):                            # 转换颜色数据
            data['color'] = color_transform(data['color'])
            return data
        
        rw, rh, col, row = optimal_row_cols(            # 计算最佳行列数
            n_cameras=len(camera_serial_numbers),       # 相机序列号的长度
            in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution
        ) 
        # 视频捕获参数
                # 4.可视化
        vis_color_transform = get_image_transform(      # 获取可视化颜色转换函数
            input_res=video_capture_resolution,
            output_res=(rw,rh),
            bgr_to_rgb=False
        ) 
        def vis_transform(data):                        # 转换可视化颜色数据
            data['color'] = vis_color_transform(data['color']) 
            return data

        recording_transfrom = None                      # 初始化录制转换函数
        recording_fps       = video_capture_fps         # 初始化录制帧率
        recording_pix_fmt   = 'bgr24'                   # 初始化录制像素格式
        if not record_raw_video:                        # 如果不记录原始视频
            recording_transfrom = transform             # 使用转换函数
            recording_fps       = frequency             # 使用控制频率作为录制帧率
            recording_pix_fmt   = 'rgb24'               # 使用RGB24像素格式

        video_recorder = VideoRecorder.create_h264(
            fps             = recording_fps, 
            codec           = 'h264',
            input_pix_fmt   = recording_pix_fmt, 
            crf             = video_crf,
            thread_type     = 'FRAME',
            thread_count    = thread_per_video
            )                                           # 创建视频录制器

        realsense = MultiRealsense(
            serial_numbers  =camera_serial_numbers,     # 相机序列号
            shm_manager     =shm_manager,               # 共享内存管理器
            resolution      =video_capture_resolution,  # 视频捕获分辨率
            capture_fps     =video_capture_fps,         # 视频捕获帧率
            put_fps         =video_capture_fps,         # 放帧率
            # 在每一帧到达后立即发送 send every frame immediately after arrival
            # 不考虑设置的帧率限制 ignores put_fps
            put_downsample  =False,                     # 放，是否需要下采样
            # record_fps      =recording_fps,             # 记录帧率
            enable_color    =True,#启用颜色
            enable_depth    =True,                        # 3D保存
            # enable_depth    =False,                     # 3D保存
            enable_infrared =False,                     # 启用红外线
            enable_sam2     =self.enable_sam2,                     # 启用sam2识别mask
            get_max_k       =max_obs_buffer_size,       # 获取最大k
            transform       =transform,                 # 转换
            vis_transform   =vis_transform,             # 可视化转换
            recording_transform=recording_transfrom,    # 记录转换
            video_recorder  =video_recorder,            # 视频记录器
            verbose         =False                      # 是否显示详细信息
            ) # 创建Realsense多相机管理器
        
        multi_cam_vis = None                            # 初始化多相机可视化
        if enable_multi_cam_vis:                        # 如果启用多相机可视化
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            )                                           # 创建多相机可视化器



        ########################################################################
        #############机器人的参数设置
        ########################################################################
        j_init = np.array([-0.165, -0.059, 0.167, -1.693, 0.002, 1.642, 0.751]) # 初始化关节角度
        if not init_joints:                                     # 如果不初始化关节
            j_init = None                                       # 关节初始化设置为None

        robot = FrankaInterpolationController(
            shm_manager=shm_manager,                            # 共享内存管理器
            robot_ip=robot_ip,                                  # 机器人IP
            frequency=200,                                      # 频率
            Kx_scale=1.0,                                       # Kx比例
            Kxd_scale=np.array([2.0, 1.5, 2.0, 1.0, 1.0, 1.0]), # Kxd比例
            joints_init=j_init,                                 # 关节初始化
            joints_init_duration=3.0,                           # 关节初始化持续时间
            verbose=False,                                      # 是否显示详细信息
            receive_latency=robot_obs_latency                   # 机器人观测延迟
        )
        # self.SAM = sam
        self.realsense          = realsense                     # 设置Realsense
        self.robot              = robot                         # 设置机器人

        self.verbose            = verbose
        self.multi_cam_vis      = multi_cam_vis                 # 设置多相机可视化
        self.video_capture_fps  = video_capture_fps             # 设置视频捕捉帧率

        # 新代码
        self.obs_image_resolution = obs_image_resolution        # 观测图像分辨率
        self.frequency          = frequency                     # 设置控制频率
        self.n_obs_steps        = n_obs_steps                   # 设置观察步数
        self.max_obs_buffer_size = max_obs_buffer_size          # 设置最大观察缓冲区大小
        self.max_pos_speed      = max_pos_speed                 # 设置最大位置速度
        self.max_rot_speed      = max_rot_speed                 # 设置最大旋转速度
        self.obs_key_map        = obs_key_map                   # 设置观察键映射
        # recording
        self.output_dir         = output_dir                    # 设置输出目录
        self.video_dir          = video_dir                     # 设置视频目录
        self.replay_buffer      = replay_buffer                 # 设置重放缓冲区
        # temp memory buffers
        self.last_realsense_data = None                         # 初始化最后一次Realsense数据
        
        self.last_robot_data    = None                          # 初始化最后一次机器人数据
        # recording buffers
        self.obs_accumulator    = None                          # 初始化观察累加器
        self.action_accumulator = None                          # 初始化动作累加器
        self.stage_accumulator  = None                          # 初始化阶段累加器

        # 图像累加器统一风格初始化
        print("self.realsense.n_cameras ==",self.realsense.n_cameras)
        self.image_accumulators = dict()                        # 每个摄像头一个累加器
        if self.enable_sam2:  # 如果启用sam2
            self.image_mask_accumulators = dict()                   # 每个摄像头一个累加器
        self._is_recording = False                              # 显式录制状态标志

        # 为了获取深度图像新增
        self.enable_depth = enable_depth
        if self.enable_depth:  # 如果启用深度图像
            self.depth_scale = depth_scale
            self.depth_dtype = depth_dtype
            self.depth_accumulators = dict()                    # 深度图像累加器
        self.enable_pointcloud = enable_pointcloud              # 是否启用点云
        if self.enable_pointcloud:
            self.point_cloud_accumulators = dict()              # 点云累加器 
        
        self.start_time = None                                  # 初始化开始时间
        self.is_saving = False                                  # 新增标志位
        

    
    # ======== start-stop API =============
    @property
    # 检查 Realsense 相机和机器人是否准备好
    def is_ready(self):
        ready_flag = self.realsense.is_ready
        if not ready_flag:
            print('ready_flag_realsense==', ready_flag)
        ready_flag = ready_flag and self.robot.is_ready
        if not ready_flag:
            print('ready_flag_robot==', ready_flag)
        return ready_flag
    
    # 启动 Realsense 相机和机器人控制器，如果启用了多相机可视化，也启动多相机可视化
    def start(self, wait=True):
        self.realsense.start(wait=False)        # 启动Realsense
        self.robot.start(wait=False)            # 启动机器人
        if self.multi_cam_vis is not None:      # 如果启用了多相机可视化
            self.multi_cam_vis.start(wait=False)# 启动多相机可视化
        if wait:
            self.start_wait()                   # 等待启动完成

    # 停止 Realsense 相机和机器人控制器，如果启用了多相机可视化，也停止多相机可视化
    def stop(self, wait=True):
        if self._is_recording:                  # 如果正在记录
            self.end_episode()                  # 结束当前记录
        if self.multi_cam_vis is not None:      # 如果启用了多相机可视化
            self.multi_cam_vis.stop(wait=False) # 停止多相机可视化
        self.robot.stop(wait=False)             # 停止机器人
        self.realsense.stop(wait=False)         # 停止Realsense
        if wait:
            self.stop_wait()                    # 等待停止完成

    # 等待 Realsense 相机和机器人控制器启动完成，如果启用了多相机可视化，也等待其启动完成
    def start_wait(self):
        self.realsense.start_wait()             # 等待Realsense启动完成
        self.robot.start_wait()                 # 等待机器人启动完成
        if self.multi_cam_vis is not None:      # 如果启用了多相机可视化
            self.multi_cam_vis.start_wait()     # 等待多相机可视化启动完成
    
    # 等待 Realsense 相机和机器人控制器停止完成，如果启用了多相机可视化，也等待其停止完成
    def stop_wait(self):
        self.robot.stop_wait()                  # 等待机器人停止完成
        self.realsense.stop_wait()              # 等待Realsense停止完成
        if self.multi_cam_vis is not None:      # 如果启用了多相机可视化
            self.multi_cam_vis.stop_wait()      # 等待多相机可视化停止完成

    # ========= context manager ===========
    # 在上下文管理器使用时，启动环境
    def __enter__(self):
        self.start()
        return self
    
    # 在上下文管理器结束时，停止环境
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_recording:                  # 仅在需要时停止
            self.stop()

    # ========= async env API ===========
    # 获取当前的观察数据，包括相机图像和机器人状态
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready                    # 确保环境已准备好
        
        # ===================== 1. 获取原始数据 =====================
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        if self.enable_sam2:
            self.last_realsense_data = self.realsense.get_sam2(k=k, out=self.last_realsense_data)   # 获取Realsense数
        else:
            self.last_realsense_data = self.realsense.get(k=k, out=self.last_realsense_data)
        
        self.last_robot_data = self.robot.get_all_state()                                       # 获取机器人所有状态

        # ===================== 2. 生成统一对齐时间戳 =====================
        dt = 1 / self.frequency
        # 获取最后一个时间戳
        last_camera_timestamps = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()]) 
        obs_align_timestamps = last_camera_timestamps - (np.arange(self.n_obs_steps)[::-1] * dt)    # 对齐观察时间戳

        # 强制时间戳长度与 n_obs_steps 一致
        assert len(obs_align_timestamps) == self.n_obs_steps, "时间轴长度错误"

        # ===================== 3. 数据对齐核心逻辑 =====================
        camera_obs = dict()                                         # 初始化相机观察字典
        if self.enable_depth:
            depth_obs = dict()                                      # 新增深度观测字典
        if self.enable_pointcloud:
            point_cloud_obs = dict()                                # 新增点云观测字典

        for camera_idx, value in self.last_realsense_data.items():  # 遍历每个相机的数据

            # 处理深度图像
            if self.enable_depth and 'depth' in value:
                resized_depths = []
                for depth in value['depth']:
                    # 深度图像缩放和类型转换
                    resized_depth = cv2.resize(depth, self.obs_image_resolution)  # 缩放深度图像到 640x480
                    if self.depth_dtype == np.uint16:
                        resized_depth = (resized_depth * self.depth_scale).astype(np.uint16)
                    resized_depths.append(resized_depth)
                resized_depths = np.array(resized_depths)
            
            # 处理点云数据
            if self.enable_pointcloud and 'point_cloud' in value:
                point_cloud_data = value['point_cloud']  # 假设点云数据在这里
                point_cloud_data = np.array(point_cloud_data)  # 转为NumPy数组


            # 3.1 获取历史帧索引（用于观测返回）
            this_timestamps = value['timestamp']                    # 获取当前相机的时间戳
            aligned_idxs = list()                                   # 初始化索引列表
            for t in obs_align_timestamps:                          # 遍历对齐的时间戳
                is_before_idxs = np.nonzero(this_timestamps < t)[0] # 找到所有小于当前时间戳的索引
                if len(is_before_idxs) > 0:                         # 如果存在小于当前时间戳的索引
                    aligned_idxs.append(is_before_idxs[-1])         # 取最后一个小于当前时间戳的索引，添加到索引列表中
                else:
                    aligned_idxs.append(0) # 如果没有小于当前时间戳的索引，将0添加到索引列表中
            # 对齐rgb数据
            camera_obs[f'camera_{camera_idx}'] = value['color'][aligned_idxs] # 将颜色数据映射到相机观察字典中，(n_obs_steps, H, W, 3)
            if self.enable_sam2:
                # mask_img =cv2.resize(value['color_mask'][aligned_idxs], (320, 240), interpolation=cv2.INTER_NEAREST)
                camera_obs[f'camera_{camera_idx}_mask'] = value['color_mask'][aligned_idxs] # 将颜色数据映射到相机观察字典中，(n_obs_steps, H, W, 3)

            if self.enable_depth:
                # 对齐深度数据对齐
                depth_obs[f'camera_{camera_idx}_depth'] = resized_depths[aligned_idxs]
            if self.enable_pointcloud:
                # 对齐点云数据对齐
                point_cloud_obs[f'camera_{camera_idx}_point_cloud'] = point_cloud_data[aligned_idxs]
            
        # 机器人数据对齐 align robot obs
        robot_timestamps = self.last_robot_data['robot_receive_timestamp']  # 获取机器人的时间戳
        aligned_idxs = list()                                               # 初始化索引列表
        for t in obs_align_timestamps:                                      # 使用历史时间轴，遍历对齐的时间戳
            is_before_idxs = np.nonzero(robot_timestamps < t)[0]            # 找到所有小于当前时间戳的索引
            if len(is_before_idxs) > 0:
                aligned_idxs.append(is_before_idxs[-1])
            else:
                aligned_idxs.append(0)                                      # 如果没有小于当前时间戳的索引，将0添加到索引列表中
        
        # ===================== 4.观察字典映射 ====================
        robot_obs_raw = dict()                                      # 初始化机器人原始观察字典
        for k, v in self.last_robot_data.items():                   # 遍历机器人的数据
            if k in self.obs_key_map:                               # 如果键在观察键映射中
                robot_obs_raw[self.obs_key_map[k]] = v              # 将值映射到机器人原始观察字典中

        robot_obs = dict()                                          # 初始化机器人观察字典
        for k, v in robot_obs_raw.items():                          # 遍历机器人原始观察字典
            robot_obs[k] = v[aligned_idxs]                          # 将数据映射到机器人观察字典中


        # ===================== 7. 构建返回数据 ====================
        obs_data = dict(camera_obs)                                 # 初始化观察数据为相机观察数据
        obs_data.update(robot_obs)                                  # 更新观察数据为机器人观察数据
        obs_data['timestamp'] = obs_align_timestamps                # 设置观察数据的时间戳
        # obs_data['stage'] = np.array([[0], [0]], dtype=np.int64)  # 设置观察数据的阶段信息
        return obs_data                                             # 返回观察数据


    # 执行给定的动作序列
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray, 
            stages: Optional[np.ndarray] = None,                    # 阶段默认值改为 None
            # return_flag: np.ndarray =[0]
            ):
        assert self.is_ready                                        # 确保环境已准备好

        # 使用np.asarray 是零拷贝转换
        actions = np.array(actions)                             # 将动作转换为numpy数组, dtype=np.float32
        timestamps = np.array(timestamps)                       # 将时间戳转换为numpy数组, dtype=np.float64

        # convert action to pose
        # 检查每个时间戳是否晚于当前时间（receive_time），如果是，则认为这是一个新的动作。提取新的动作和时间戳。convert action to pose
        receive_time = time.time()                                  # 获取当前时间
        is_new = timestamps > receive_time                          # 找到所有新时间戳
        new_actions = actions[is_new]                               # 获取新动作
        new_timestamps = timestamps[is_new]                         # 获取新时间戳

        # schedule waypoints
        for i in range(len(new_actions)):                           # 遍历新动作
            robot_action = new_actions[i,:6]
            gripper_action = new_actions[i,6]
            self.robot.schedule_waypoint(                           # 为每个动作调度路径点
                pose=robot_action,
                target_time=new_timestamps[i],
                state=gripper_action,                               # 爪子信息
                width=gripper_action,
                force=15,
                speed=0.3)                                          # 0.2

    # 获取当前机器人的状态
    def get_robot_state(self):
        return self.robot.get_state()                               # 获取机器人状态

    # recording API
    # 开始一个新的记录集，初始化观察和动作累加器，并开始记录视频
    def start_episode(self, start_time=None):
        "Start recording and return first obTimestampActionAccumulators"
        if start_time is None:                                      # 如果开始时间为空
            start_time = time.time()                                # 获取当前时间作为开始时间
        self.start_time = start_time                                # 设置开始时间
        
        if self._is_recording:
            print("✅Recording continue!")                          # 打印开始消息
            return                                                  # 如果不在录制状态，直接返回
        else:
            print("✅Recording started!")                           # 打印开始消息
            self._is_recording = True                               # 开始录制时设为True

        assert self.is_ready                                        # 确保环境已准备好

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes                  # 获取当前集ID
        this_video_dir = self.video_dir.joinpath(str(episode_id))   # 创建视频目录
        this_video_dir.mkdir(parents=True, exist_ok=True)           # 确保视频目录存在
        n_cameras = self.realsense.n_cameras                        # 获取相机数量

        # 使用列表推导式 + f-string (提速3倍)
        # 最佳实践：在父路径初始化时处理绝对路径
        abs_video_dir = this_video_dir.resolve()  # 提前解析← 关键修改点
        # 使用高效列表推导式 + 保留绝对路径
        video_paths = [str(abs_video_dir / f"{i}.mp4") for i in range(n_cameras)]

        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)           # 重新启动Realsense
        self.realsense.start_recording(video_path=video_paths, start_time=start_time) # 开始记录视频

        print(f'Episode {episode_id} started!')                     # 打印开始消息


    # 结束当前的记录集，停止记录视频，并将记录的数据保存到重放缓冲区
    def end_episode(self):
        "Stop recording"
        
        if not self._is_recording:
            print("✅Recording closed before!")                     # 打印开始消息
            return                                                  # 如果不在录制状态，直接返回
        else:
            print("✅Recording ended!")
            self._is_recording = False                              # 结束录制后设为False

        assert self.is_ready                                        # 确保环境已准备好
        self.is_saving = True                                       # 进入保存状态

        # 内存使用监控
        process = psutil.Process()
        mem_before = process.memory_info().rss
        print(f"[Memory] Pre-processing: {mem_before / 1024**2:.2f} MB")

        self.realsense.stop_recording()                             # 停止记录视频


        # 退出保存状态
        self.is_saving = False      
        # 保存完毕
        print(f'Episode {self.replay_buffer.n_episodes} 清理完毕!')
        # 监控点3：清理完成后
        mem_final = process.memory_info().rss
        print(f"[Memory] Final: {mem_final / 1024**2:.2f} MB")

    # 删除最近的记录集，包括删除视频文件和缓冲区中的数据
    def drop_episode(self):
        self.end_episode()                                          # 结束当前记录
        self.replay_buffer.drop_episode()                           # 删除最近的记录
        episode_id = self.replay_buffer.n_episodes                  # 获取当前集ID
        this_video_dir = self.video_dir.joinpath(str(episode_id))   # 获取视频目录路径
        if this_video_dir.exists():                                 # 如果视频目录存在
            shutil.rmtree(str(this_video_dir))                      # 删除视频目录
        print(f'Episode {episode_id} dropped!')                     # 打印删除消息

