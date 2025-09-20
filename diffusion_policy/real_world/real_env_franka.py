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
    StrictTimestampActionAccumulator,
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

from protas.test18 import EnhancedStagePredictor
from diffusion_policy.real_world.action_processing import process_actions
DEFAULT_OBS_KEY_MAP = {                                 # 默认观测键映射 建立观测键和键映射的关系
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
            max_pos_speed       = 0.2,                  # 最大位置速度0.25
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
            enable_depth        = True,                 # True False 新增参数：是否启用深度
            depth_scale         = 1000.0,               # 深度数据缩放因子
            depth_dtype=np.uint16,                      # 深度数据类型
            enable_pointcloud   = False,
            # enable_pointcloud=True                    # 新增参数：是否启用点云
            enable_sam2         = False,                # 新增参数：是否启用sam2
            enable_predictor    = False,                # False True
            ):
        self.enable_depth = enable_depth                # 是否启用深度
        self.enable_sam2 = enable_sam2                  # 是否启用sam2
        self.enable_predictor = enable_predictor        # 是否启用阶段预测器
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
        
        # 视频捕获参数
        rw, rh, col, row = optimal_row_cols(            # 计算最佳行列数
            n_cameras=len(camera_serial_numbers),       # 相机序列号的长度
            in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution
        ) 
        
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
        )                                               # 创建视频录制器

        realsense = MultiRealsense(
            serial_numbers  =camera_serial_numbers,     # 相机序列号
            shm_manager     =shm_manager,               # 共享内存管理器
            resolution      =video_capture_resolution,  # 视频捕获分辨率
            capture_fps     =video_capture_fps,         # 视频捕获帧率
            put_fps         =video_capture_fps,         # 放帧率
            # 在每一帧到达后立即发送 send every frame immediately after arrival
            # 不考虑设置的帧率限制 ignores put_fps
            put_downsample  =False,                     # 放，是否需要下采样
            record_fps      =recording_fps,             # 记录帧率
            enable_color    =True,                      # 启用颜色
            enable_depth    =self.enable_depth,                      # 3D保存
            # enable_depth    =False,                     # 3D保存
            enable_infrared =False,                     # 启用红外线
            enable_sam2     =self.enable_sam2,          # 启用sam2识别mask
            get_max_k       =max_obs_buffer_size,       # 获取最大k
            transform       =transform,                 # 转换
            vis_transform   =vis_transform,             # 可视化转换
            recording_transform=recording_transfrom,    # 记录转换
            video_recorder  =video_recorder,            # 视频记录器
            verbose         =False                      # 是否显示详细信息
        )   # 创建Realsense多相机管理器
        
        multi_cam_vis = None                            # 初始化多相机可视化
        if enable_multi_cam_vis:                        # 如果启用多相机可视化
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            )                                           # 创建多相机可视化器
        
        if self.enable_predictor:  # 如果启用阶段预测器
            model_path = "/home/hy/Desktop/dp_0314/protas/models/0626/data_camera_0_custom_i3d/split_1/epoch-200.model"
            mapping_file = "/home/hy/Desktop/dp_0314/protas/data/data_camera_0_custom_i3d/mapping.txt"
            graph_path = "/home/hy/Desktop/dp_0314/protas/data/data_camera_0_custom_i3d/graph/graph.pkl"
            device = 'cuda'
            history_size = 64
            self.predictor = EnhancedStagePredictor(
                model_path=model_path,
                mapping_file=mapping_file,
                graph_path=graph_path,
                device=device,
                history_size=history_size, 
            )
            model_path1 = "/home/hy/Desktop/dp_0314/protas/models/0626/data_camera_1_custom_i3d/split_1/epoch-200.model"
            mapping_file1 = "/home/hy/Desktop/dp_0314/protas/data/data_camera_1_custom_i3d/mapping.txt"
            graph_path1 = "/home/hy/Desktop/dp_0314/protas/data/data_camera_1_custom_i3d/graph/graph.pkl"
            self.predictor1 = EnhancedStagePredictor(
                model_path=model_path1,
                mapping_file=mapping_file1,
                graph_path=graph_path1,
                device=device,
                history_size=history_size,
            )

        ########################################################################
        #############机器人的参数设置
        ########################################################################
        j_init = np.array([-0.165, -0.059, 0.167, -1.693, 0.002, 1.642, 0.751]) # 初始化关节角度
        if not init_joints:                                     # 如果不初始化关节
            j_init = None                                       # 关节初始化设置为None

        robot = FrankaInterpolationController(
            shm_manager=shm_manager,                            # 共享内存管理器
            robot_ip=robot_ip,                                  # 机器人IP
            frequency=1000,                                      # 频率
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
        # get data获取数据,both have more than n_obs_steps data
        # 当前情况下，每次从相机获取6帧数据，但实际只需要对齐n_obs_steps=2帧
        # 注意，umi修改了k的定义
        # n_obs_steps=2, video_capture_fps=30, frequency=10 → k=6
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        # self.last_realsense_data = self.realsense.get(k=k, out=self.last_realsense_data)      # 获取Realsense数
        if self.enable_sam2:
            self.last_realsense_data = self.realsense.get_sam2(k=k, out=self.last_realsense_data)   # 获取Realsense数
        else:
            self.last_realsense_data = self.realsense.get(k=k, out=self.last_realsense_data)
        
        self.last_robot_data = self.robot.get_all_state()                                       # 获取机器人所有状态

        # ===================== 2. 生成统一对齐时间戳 =====================
        # align camera obs timestamps
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
        predictor_obs = dict()                                  # 初始化预测器观察字典


        for camera_idx, value in self.last_realsense_data.items():  # 遍历每个相机的数据

            # print("value['color'].shape==",value['color'].shape,"value['color_mask'].shape==",value['color_mask'].shape)

            # ==================================================
            # ======= 目前eval时需要注释这段图像累加器代码 ========
            # ==================================================
            if self.obs_accumulator is not None:
                self.image_accumulators[camera_idx].put(
                    images=value['color'],                          # shape (N, 480, 640, 3)
                    timestamps=value['timestamp']                   # shape (N,)
                )
                if self.enable_sam2:
                    self.image_mask_accumulators[camera_idx].put(
                        masks=value['color_mask'],                  # shape (N, 480, 640, 3)
                        timestamps=value['timestamp_mask']          # shape (N,)
                    )
            
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

                if self.obs_accumulator is not None:
                    self.depth_accumulators[camera_idx].put(
                        depths=resized_depths,
                        timestamps=value['timestamp']
                    )
            
            # 处理点云数据
            if self.enable_pointcloud and 'point_cloud' in value:   # 新增点云的数据处理（确保条件正确）
                point_cloud_data = value['point_cloud']  # 假设点云数据在这里
                point_cloud_data = np.array(point_cloud_data)  # 转为NumPy数组
                # 将点云数据放入累加器
                if self.obs_accumulator is not None:
                    self.point_cloud_accumulators[camera_idx].put(
                        point_clouds=point_cloud_data,
                        timestamps=value['timestamp']
                    )

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

            if self.enable_predictor:
                if camera_idx == 0:  # 仅对第一个相机进行阶段预测
                    # print(f"[RealEnv] Camera0帧形状: {camera_obs[f'camera_{camera_idx}'][-1].shape}")  # 应该是 (H, W, 3)
                    # predictor_obs[f'camera_{camera_idx}_stage'], predictor_obs[f'camera_{camera_idx}_progress'], predictor_obs[f'camera_{camera_idx}_stage_changed'] = self.predictor.process_frame(camera_obs[f'camera_{camera_idx}'][-1])
                    # 调用预测器处理当前帧
                    stage, progress, changed = self.predictor.process_frame(camera_obs[f'camera_{camera_idx}'][-1])
                else:
                    stage, progress, changed = self.predictor1.process_frame(camera_obs[f'camera_{camera_idx}'][-1])
                # === 关键修复: 将结果封装为列表 ===
                predictor_obs[f'camera_{camera_idx}_stage'] = [stage]
                predictor_obs[f'camera_{camera_idx}_progress'] = [progress]
                predictor_obs[f'camera_{camera_idx}_stage_changed'] = [int(changed)]
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

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                robot_obs_raw,
                robot_timestamps
            )

        # ===================== 7. 构建返回数据 ====================
        obs_data = dict(camera_obs)                                 # 初始化观察数据为相机观察数据
        obs_data.update(robot_obs)                                  # 更新观察数据为机器人观察数据
        if self.enable_predictor:
            obs_data.update(predictor_obs)                            # 更新观察数据为预测器观察数据
        obs_data['timestamp'] = obs_align_timestamps                # 设置观察数据的时间戳
        
        # obs_data['stage'] = np.array([[0], [0]], dtype=np.int64)  # 设置观察数据的阶段信息
        return obs_data                                             # 返回观察数据


    # 执行给定的动作序列
    # actions，动作序列；timestamps，动作对应的时间戳；stages，动作的阶段信息（可选）。
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray, 
            stages: Optional[np.ndarray] = None,                    # 阶段默认值改为 None
            # return_flag: np.ndarray =[0]
            ):
        if self.is_ready:                                     # 确保环境已准备好

            # 使用np.asarray 是零拷贝转换
            actions = np.array(actions)                             # 将动作转换为numpy数组, dtype=np.float32
            timestamps = np.array(timestamps)                       # 将时间戳转换为numpy数组, dtype=np.float64

            # convert action to pose
            # 检查每个时间戳是否晚于当前时间（receive_time），如果是，则认为这是一个新的动作。提取新的动作和时间戳。convert action to pose
            receive_time = time.time()                                  # 获取当前时间
            is_new = timestamps > receive_time                          # 找到所有新时间戳
            new_actions = actions[is_new]                               # 获取新动作
            new_timestamps = timestamps[is_new]                         # 获取新时间戳
            new_stages = stages[is_new]                                 # 获取新阶段

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
                
            # 如果存在 action_accumulator，则将新的动作和时间戳记录下来
            if self.action_accumulator is not None:                     # 如果动作累加器不为空
                self.action_accumulator.put(                            # 将新动作和新时间戳放入动作累加器
                    new_actions,
                    new_timestamps
                )
            if self.stage_accumulator is not None:                      # 如果阶段累加器不为空
                self.stage_accumulator.put(                             # 将新阶段和新时间戳放入阶段累加器
                    new_stages,
                    new_timestamps
                )
        else:
            print("Environment is not ready, cannot execute actions.")

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
        # 最佳实践：在父路径初始化时处理绝对路径
        abs_video_dir = this_video_dir.resolve()  # 提前解析← 关键修改点
        # 使用高效列表推导式 + 保留绝对路径
        video_paths = [str(abs_video_dir / f"{i}.mp4") for i in range(n_cameras)]

        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)           # 重新启动Realsense
        self.realsense.start_recording(video_path=video_paths, start_time=start_time) # 开始记录视频

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(             # 创建观察累加器
            start_time=start_time,
            dt=1/self.frequency
        )
        self.action_accumulator = StrictTimestampActionAccumulator(  # 创建严格动作累加器
            start_time=start_time,
            dt=1/self.frequency
        )
        self.stage_accumulator = StrictTimestampActionAccumulator(        # 创建阶段累加器
            start_time=start_time,
            dt=1/self.frequency
        )
        # self.action_accumulator = TimestampActionAccumulator(       # 创建动作累加器
        #     start_time=start_time,
        #     dt=1/self.frequency
        # )
        # self.stage_accumulator = TimestampActionAccumulator(        # 创建阶段累加器
        #     start_time=start_time,
        #     dt=1/self.frequency
        # )
        # 创建相机数据观测累加器
        for cam_idx in range(n_cameras):
            self.image_accumulators[cam_idx] = TimestampImageAccumulator(
                start_time=start_time,
                dt=1 / self.frequency,
                image_shape=self.obs_image_resolution[::-1] + (3,)  # (H, W, C)
                )
            if self.enable_sam2:
                ###################新增rgb图像mask累加器###################
                self.image_mask_accumulators[cam_idx] = TimestampMaskAccumulator(
                    start_time=start_time,
                    dt=1 / self.frequency,
                    mask_shape=self.obs_image_resolution[::-1],     # (H, W)  (480,640)
                    mask_dtype=np.uint8,                            # mask数据类型
                    )
            
            ###################新增深度图像累加器#####################
            if self.enable_depth:
                self.depth_accumulators[cam_idx] = TimestampDepthAccumulator(
                    start_time=start_time,
                    dt=1 / self.frequency,
                    depth_shape=self.obs_image_resolution[::-1],    # 深度图像形状
                    depth_dtype=self.depth_dtype
                )
            print(f"image_shape: {self.obs_image_resolution[::-1] + (3,)}")

            # 为每个摄像头初始化累加器
            if self.enable_pointcloud:
                for cam_idx in range(n_cameras):
                    self.point_cloud_accumulators[cam_idx] = TimestampPointCloudAccumulator(
                        start_time=start_time,
                        dt=1 / self.frequency,
                        point_shape=(10000, 6)
                    )
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

        if self.obs_accumulator is not None:                        # 如果观察累加器不为空，结束记录
            assert self.action_accumulator is not None              # 确保动作累加器不为空
            assert self.stage_accumulator is not None               # 确保阶段累加器不为空

            # 获取数据
            obs_data = self.obs_accumulator.data                    # 获取观察数据
            obs_timestamps = self.obs_accumulator.timestamps        # 获取观察时间戳

            actions = self.action_accumulator.actions               # 获取动作数据
            action_timestamps = self.action_accumulator.timestamps  # 获取动作时间戳
            stages = self.stage_accumulator.actions                 # 获取阶段数据

            # #修正代码
            # ###################################################
            # print("开始处理动作数据...")
            # try:
            #     # 确保动作数据格式正确
            #     if actions.ndim == 1:
            #         actions = actions.reshape(-1, 1)
                    
            #     # 处理动作数据 (XYZ + RPY + Gripper)
            #     processed_actions = process_actions(actions.copy())
                
            #     # 保留其他维度不变（如果有）
            #     if processed_actions.shape[1] < actions.shape[1]:
            #         processed_actions = np.hstack((
            #             processed_actions,
            #             actions[:, processed_actions.shape[1]:]
            #         ))
                    
            #     actions = processed_actions  # 使用处理后的动作
            #     print("动作数据处理完成")
            # except Exception as e:
            #     print(f"动作处理失败: {str(e)}")
            #     import traceback
            #     traceback.print_exc()
            # ##################################################

            print(f"obs_timestamps_length: {len(obs_timestamps)}")
            print(f"action_timestamps_length: {len(action_timestamps)}")

            camera_timestamps = {}
            for cam_idx, accum in self.image_accumulators.items():
                print(f"depth_{cam_idx} timestamps: {len(accum.timestamps)}")
                print(f"camera_{cam_idx} images: {len(accum.images)}")
                # 将每个相机的时间戳存储在字典中，key 为相机的索引，value 为对应的时间戳数组
                camera_timestamps[cam_idx] = accum.timestamps
            
            if self.enable_sam2:
                image_timestamps = {}
                for cam_idx, accum in self.image_mask_accumulators.items():
                    print(f"mask_{cam_idx} timestamps: {len(accum.timestamps)}")
                    print(f"mask_{cam_idx} images: {len(accum.masks)}")
                    # 将每个相机的时间戳存储在字典中，key 为相机的索引，value 为对应的时间戳数组
                    image_timestamps[cam_idx] = accum.timestamps
            
            if self.enable_depth:
                # 获取深度数据的时间戳
                depth_timestamps = {}
                for cam_idx, accum in self.depth_accumulators.items():
                    print(f"depth_{cam_idx} timestamps: {len(accum.timestamps)}")
                    print(f"depth_{cam_idx} depths: {len(accum.depths)}")
                    depth_timestamps[cam_idx] = accum.timestamps
            
            # 获取点云数据的时间戳
            if self.enable_pointcloud:
                point_cloud_timestamps = {}
                for cam_idx, accum in self.point_cloud_accumulators.items():
                    print(f"point_cloud_{cam_idx} timestamps: {len(accum.timestamps)}")
                    print(f"point_cloud_{cam_idx} point_clouds: {len(accum.point_clouds)}")
                    point_cloud_timestamps[cam_idx] = accum.timestamps

            if self.verbose:                            # 使用实际理论长度校验
                # 计算各累积器的理论长度
                obs_len = len(self.obs_accumulator)
                action_len = len(self.action_accumulator)
                stage_len = len(self.stage_accumulator)
                # 获取所有有效图像累积器的实际长度
                image_acc_actual_lengths = [acc.actual_length for acc in self.image_accumulators if acc is not None]
                    
                # 打印所有中间值（调试关键）
                print("[DEBUG] Accumulator Lengths:")
                print(f"  - Obs: {obs_len}")
                print(f"  - Action: {action_len}")
                print(f"  - Stage: {stage_len}")
                print(f"  - Image Actual Lengths: {image_acc_actual_lengths}")

                image_min = min(image_acc_actual_lengths)
                if self.enable_sam2:
                    image_mask_acc_actual_lengths = [acc.actual_length for acc in self.image_mask_accumulators if acc is not None]
                    print(f"  - Image Pred Actual Lengths: {image_mask_acc_actual_lengths}") 
                    image_mask_min = min(image_mask_acc_actual_lengths)
                    n_steps = min(obs_len, action_len, stage_len, image_min, image_mask_min)
                else:
                    n_steps = min(obs_len, action_len, stage_len, image_min)
                print(f"[INFO] Final n_steps = {n_steps}")

            # 获取所有时间戳：双星号在Python中用于字典解包，将键值对合并到外层字典中
            data_lengths = {
                "obs": len(self.obs_accumulator),
                "action": len(self.action_accumulator),
                "stage": len(self.stage_accumulator),
                **{f"camera_{cam_idx}": len(accum.images) for cam_idx, accum in self.image_accumulators.items()},
                # **{f"camera_{cam_idx}_mask": len(accum.masks) for cam_idx, accum in self.image_mask_accumulators.items()}
            }
            
            if self.enable_sam2:
                data_lengths.update(
                    **{f"camera_{cam_idx}_mask": len(accum.masks) for cam_idx, accum in self.image_mask_accumulators.items()}
                )
            if self.enable_depth:
                data_lengths.update(
                    **{f"depth_{cam_idx}": len(accum.depths) for cam_idx, accum in self.depth_accumulators.items()}
                )
            if self.enable_pointcloud:
                data_lengths.update(
                    **{f"point_cloud_{cam_idx}": len(accum.point_clouds) for cam_idx, accum in self.point_cloud_accumulators.items()}
                )
            # 找到最小公共长度
            n_steps = min(data_lengths.values())
            if n_steps == 0:
                raise RuntimeError("Insufficient data to save episode!")

            # 基础字段
            episode = {
                'timestamp': obs_timestamps[:n_steps],
                'action_timestamps': action_timestamps[:n_steps],
                'action': actions[:n_steps],
                'stage': stages[:n_steps],
                **{key: value[:n_steps] for key, value in obs_data.items()}
            }

            # 摄像头数据（带错误处理）
            for cam_idx, accum in self.image_accumulators.items():
                try:
                    episode[f'camera_{cam_idx}'] = accum.images[:n_steps]
                    episode[f'camera_{cam_idx}_timestamps'] = accum.timestamps[:n_steps]
                except Exception as e:
                    print(f"Camera {cam_idx} 数据异常: {e}")
                    episode[f'camera_{cam_idx}'] = np.zeros((n_steps, *self.obs_image_resolution, 3), dtype=np.uint8)

            # 添加预测数据
            if self.enable_sam2:
                for cam_idx, accum in self.image_mask_accumulators.items():
                    try:
                        episode[f'camera_{cam_idx}_mask'] = self.image_mask_accumulators[cam_idx].masks[:n_steps]
                    except Exception as e:
                        print(f"Camera {cam_idx} 预测数据异常: {e}")
                        episode[f'camera_{cam_idx}_mask'] = np.zeros((n_steps, *self.obs_image_resolution), dtype=np.uint8)
                
            # 添加深度数据（含独立时间戳）
            if self.enable_depth:
                episode.update({
                    # 深度数据
                    **{f'camera_{cam_idx}_depth': accum.depths[:n_steps]
                    for cam_idx, accum in self.depth_accumulators.items()},
                    
                    # 深度时间戳
                    **{f'camera_{cam_idx}_depth_timestamps': accum.timestamps[:n_steps]
                    for cam_idx, accum in self.depth_accumulators.items()}
                })
            
            # # 添加点云数据（含独立时间戳）
            if self.enable_pointcloud:
                episode.update({
                    # 点云数据
                    **{f'camera_{cam_idx}_point_cloud': accum.point_clouds[:n_steps]
                    for cam_idx, accum in self.point_cloud_accumulators.items()},
                    
                    # 点云时间戳
                    **{f'camera_{cam_idx}_point_cloud_timestamps': accum.timestamps[:n_steps]
                    for cam_idx, accum in self.point_cloud_accumulators.items()}
                })
            
            # 数据一致性校验
            assert len({len(v) for v in episode.values()}) == 1, "数据长度不一致!"
            
            # 监控点2：数据保存前 
            mem_before_save = process.memory_info().rss
            print(f"[Memory] before-saving: {mem_before_save / 1024**2:.2f} MB")
            
            # 同步保存（阻塞主线程）
            try:
                self.replay_buffer.add_episode(episode, compressors='disk')
                print(f'Episode {self.replay_buffer.n_episodes} saved!')
            except Exception as e:
                print(f'保存失败: {e}')

            episode_id = self.replay_buffer.n_episodes - 1                  # 获取当前集ID
            print(f'Episode {episode_id} 保存完毕!')                         # 保存完毕

            # 监控点3：数据保存后 
            mem_after_save = process.memory_info().rss
            print(f"[Memory] Post-saving: {mem_after_save / 1024**2:.2f} MB")

            # 清理累加器
            self._is_recording = False                                      # 确保状态标志被清除
            self.obs_accumulator = None                                     # 清空观察累加器
            self.action_accumulator = None                                  # 清空动作累加器
            self.stage_accumulator = None                                   # 清空阶段累加器
            # 强制释放图像累加器等
            for accum in self.image_accumulators.values():
                accum.clear()
            if self.enable_sam2:
                for accum in self.image_mask_accumulators.values():
                    accum.clear()
            # 清空相机数据累加器
            if self.enable_depth:
                for accum in self.depth_accumulators.values():
                    accum.clear()
            # 新增点云清理
            if self.enable_pointcloud:
                for accum in self.point_cloud_accumulators.values():       
                    accum.clear()
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


