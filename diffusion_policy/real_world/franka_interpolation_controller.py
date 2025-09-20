import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait
import torch
from diffusion_policy.common.pose_util import pose_to_mat, mat_to_pose
import zerorpc

from enum import Enum


from multiprocessing import Value  # 添加到文件顶部

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    GRIPPER_COMMAND = 3


tx_flangerot90_tip = np.identity(4)
# X轴负方向偏移3.36cm，Z轴正方向偏移24.7cm
tx_flangerot90_tip[:3, 3] = np.array([0, 0.0336, 0.274])
# Y轴正方向偏移3.36cm，Z轴正方向偏移27.4cm
tx_flangerot90_tip[:3, 3] = np.array([0, 0.0336, 0.274])


tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3, :3] = st.Rotation.from_euler('x', [np.pi / 2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3, :3] = st.Rotation.from_euler('z', [np.pi / 4]).as_matrix()

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @ tx_flangerot90_tip
tx_tip_flange = np.linalg.inv(tx_flange_tip)


class FrankaInterface:
    def __init__(self, ip='192.168.0.168', port=4242):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        ee_pose = np.array(self.server.get_ee_pose())
        return ee_pose
        # 法兰位置转换为末端执行器位置
        # flange_pose = np.array(self.server.get_ee_pose())
        # tip_pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
        # return tip_pose

    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())

    def get_joint_velocities(self):
        return np.array(self.server.get_joint_velocities())

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.start_cartesian_impedance(
            Kx.tolist(),
            Kxd.tolist()
        )

    # to do
    def get_gripper_width(self):
        # gripper_flag = np.array([self.server.get_gripper_width()], dtype=np.float32)
        gripper_flag = np.array(self.server.get_gripper_width())
        return gripper_flag

    # def open_gripper(self, width: float, speed: float = 0.02):
    #     if width < 0 or speed <= 0:
    #         raise ValueError("宽度和速度必须为正值。")
    #     # print(f"Opening gripper: width={width}, speed={speed}")
    #     self.server.open_gripper(width, speed)

    # def close_gripper(self, width: float = 0.0, force: float = 20.0, speed: float = 0.02):
    #     if force <= 0 or speed <= 0:
    #         raise ValueError("力和速度必须为正值。")
    #     # print(f"Closing gripper: width={width} force={force}, speed={speed}")
    #     self.server.close_gripper(width, speed, force)

    def update_desired_gripper_width(self, width: float = 0.0, force: float = 20.0, speed: float = 0.02):
        if force <= 0 or speed <= 0:
            raise ValueError("力和速度必须为正值。")
        # print(f"Updating gripper: width={width} force={force}, speed={speed}")
        self.server.update_desired_gripper_width(width, speed, force)

    def move_to_start(self):
        self.server.move_to_start()
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        # 转换为末端执行器位置进行执行：将工具尖端位姿转换为齐次矩阵，转换为法兰坐标系，转换回位姿向量
        # pose = mat_to_pose(pose_to_mat(pose) @ tx_tip_flange)
        
        self.server.update_desired_ee_pose(pose.tolist())

        
        
    def terminate_current_policy(self):
        self.server.terminate_current_policy()

    def close(self):
        self.server.close()



class FrankaInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """

    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 robot_ip,
                 robot_port=4242,
                 frequency =1000,
                 Kx_scale  =1.0,
                 Kxd_scale =1.0,
                 launch_timeout=3,
                 joints_init=None,
                 joints_init_duration=None,
                 soft_real_time=False,
                 verbose=False,
                 get_max_k=None,
                 receive_latency=0.0
                 ):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FrankaPositionalController")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.frequency = frequency
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,

            'target_time': 0.0,
            'state_value': 0.0,
            'width_value': 0.0,                            
            'force_value': 0.0,                               
            'speed_value': 0.0,                               
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd', 'get_joint_velocities'),
            ('ActualGripperstate', 'get_gripper_width'),
        ]# 机器人状态
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)
            elif 'gripper' in func_name:
                example[key] = np.zeros(1)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

        # # self.return_flag = False
        # self.return_flag = Value('b', False)  # 使用共享内存布尔值
        # print("初始化self.return_flag = False")

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    # real env 没用，可能是ur用的
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert (duration >= (1 / self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        # print('ServoL scheduled')
        # print(message)
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time, state, width, force, speed):
        pose = np.array(pose)   #转换 pose 为 NumPy 数组，确保后续处理一致性
        assert pose.shape == (6,)   #确保 pose 的形状为 (6,)
        state = np.array(state)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,         # 指令为 Command.SCHEDULE_WAYPOINT.value
            'target_pose': pose,                            # 目标姿态为 pose
            'target_time': target_time,                     # 目标时间为 target_time

            'state_value': state,                           # 夹爪状态  
            'width_value': width,                           # 夹爪目标开合宽度
            'force_value': force,                           # 夹爪施加的力
            'speed_value': speed,                           # 夹爪运动速度
        }
        # print(message)
        self.input_queue.put(message)                       # 将指令放入 input_queue

    def clear_queue(self):
        """清空指令队列"""
        self.input_queue.clear()

    # def schedule_waypoint(self, pose, target_time):
    #     pose = np.array(pose)   #转换 pose 为 NumPy 数组，确保后续处理一致性
    #     assert pose.shape == (6,)   #确保 pose 的形状为 (6,)

    #     message = {
    #         'cmd': Command.SCHEDULE_WAYPOINT.value,     #指令为 Command.SCHEDULE_WAYPOINT.value
    #         'target_pose': pose,                        #目标姿态为 pose
    #         'target_time': target_time                  #目标时间为 target_time
    #     }
    #     # print('Waypoint scheduled')
    #     # print(message)
    #     self.input_queue.put(message)                   #将指令放入 input_queue


    # #夹抓打开，利用input_queue.put(message)  修改12月12日16.26
    # def schedule_gripper_command(self, state, width, force, speed):
    #     # assert width >= 0.0, "Gripper width must be non-negative."
    #     # assert force > 0.0, "Gripper force must be positive."
    #     # assert speed > 0.0, "Gripper speed must be positive."
    #     state = np.array(state)
    #     message = {
    #         'cmd': Command.GRIPPER_COMMAND.value,           # 指令类型为 GRIPPER_COMMAND
    #         'state_value': state,
    #         'width_value': width,                           # 夹爪目标开合宽度
    #         'force_value': force,                           # 夹爪施加的力
    #         'speed_value': speed,                           # 夹爪运动速度
    #         }
    #     # print('Gripper command scheduled')
    #     # print(message)
    #     self.input_queue.put(message)

    def move_to_start(self):
        with self.return_flag.get_lock():  # 保证原子操作
            if not self.return_flag.value:
                self.return_flag.value = True
                print(f"标志符设置为: {self.return_flag.value}")
            else:
                print("无需修改")

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    #robot1 = FrankaInterface()
    def run(self):
        # 软实时控制，确保控制循环不会后退
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # 连接到机器人，实例化 FrankaInterface
        robot = FrankaInterface(self.robot_ip, self.robot_port)

        try:
            if self.verbose:
                print(f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")   #打印连接到机器人的信息

            # 初始化机器人关节
            if self.joints_init is not None:     #如果关节初始化不为空
                robot.move_to_joint_positions(               #移动到关节位置
                    positions=np.asarray(self.joints_init),     #关节位置为 self.joints_init
                    time_to_go=self.joints_init_duration           #移动时间为 self.joints_init_duration
                )

            # 这部分是机械臂轨迹规划的核心代码
            dt = 1. / self.frequency                # 计算周期
            flag_number = self.frequency * 4        # 4s回到初始位置
            curr_pose = robot.get_ee_pose()         # 获取当前末端执行器姿态

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()               # 获取当前时间
            last_waypoint_time = curr_t             # 上一个路点时间
            pose_interp = PoseTrajectoryInterpolator(    #实例化 PoseTrajectoryInterpolator 一个轨迹插值器  
                times=[curr_t],                     # 时间为当前时间
                poses=[curr_pose]                   # 姿态为当前姿态
            )

            # 笛卡尔阻抗控制
            robot.start_cartesian_impedance(
                Kx=self.Kx,   # 刚度矩阵系数，用于控制力控
                Kxd=self.Kxd  # 阻尼矩阵系数，用于控制速度
            )

            t_start = time.monotonic()      # 获取当前时间
            iter_idx = 0                    # 迭代次数
            keep_running = True             # 是否继续运行
            while keep_running:             # 循环

                # send command to robot
                t_now = time.monotonic()     # 获取当前时间
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                tip_pose = pose_interp(t_now)    # 获取当前时间的姿态

                # print("self.return_flag:",self.return_flag,"flag_number:",flag_number)
                # send command to robot
                # if self.return_flag:
                # if self.return_flag.value: 
                #     # Step 1:发送返回指令，移动到初始位置
                #     robot.move_to_start()
                #     print("move_to_start_success")
                #     print("tip_pose",tip_pose)# 
                #     curr_pose = robot.get_ee_pose()         # 获取当前末端执行器姿态
                #     print("curr_pose",curr_pose)
                #     robot.update_desired_ee_pose(curr_pose)   # 更新机器人末端执行器姿态
                # else:
                #     robot.update_desired_ee_pose(tip_pose)   # 更新机器人末端执行器姿态
                #     # print("update_desired_ee_pose")
                robot.update_desired_ee_pose(tip_pose)   # 更新机器人末端执行器姿态

                # update robot state
                state = dict()              # 创建一个字典
                for key, func_name in self.receive_keys:
                    state[key] = getattr(robot, func_name)()

                t_recv = time.time()        # 获取当前时间
                state['robot_receive_timestamp'] = t_recv   #机器人接收时间戳
                state['robot_timestamp'] = t_recv - self.receive_latency    #机器人时间戳
                self.ring_buffer.put(state)    # 将状态放入 ring_buffer

                # fetch command from queue
                try:
                    # commands = self.input_queue.get_all()
                    # n_cmd = len(commands['cmd'])
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)     #获取一个指令
                    # print(f"Commands fetched: {commands}")
                    n_cmd = len(commands['cmd'])    #获取指令数量         
                except Empty:
                    n_cmd = 0

                # execute commands
                # 根据命令去执行
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break

                    # # 钟浩
                    # elif cmd == Command.GRIPPER_COMMAND.value:
                    #     # print('Gripper command received')
                    #     # print(f"Command dictionary: {command}")

                    #     # 确保命令中包含必要的字段
                    #     if 'width_value' in command and 'force_value' in command and 'speed_value' in command:
                    #         width = command['width_value']
                    #         force = command['force_value']
                    #         speed = command['speed_value']
                    #         robot.update_desired_gripper_width(width, force, speed)
                    #         # if width == 0:
                    #         #     robot.close_gripper(width, force, speed)
                    #         # else:
                    #         #     robot.open_gripper(width, speed)  # 假设机器人接口中有此方法
                            
                    #         if self.verbose:
                    #             print(f"[FrankaPositionalController] Gripper command executed: width={width}, force={force}, speed={speed}")
                    #     else:
                    #         print(f"Error: Missing gripper command fields in {command}")
                    #         continue

                    # 一般常用，优先判断
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])

                        # 爪子
                        width = command['width_value']
                        force = command['force_value']
                        speed = command['speed_value']
                        robot.update_desired_gripper_width(width, force, speed)

                        # 机械臂
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[FrankaPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[FrankaPositionalController] Actual frequency {1 / (time.monotonic() - t_now)}")

        finally:
            # manditory cleanup
            # terminate
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            robot.terminate_current_policy()
            del robot
            self.ready_event.set()

            if self.verbose:
                print(f"[FrankaPositionalController] Disconnected from robot: {self.robot_ip}")
 