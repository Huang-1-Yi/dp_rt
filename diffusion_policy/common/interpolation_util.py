# umi使用的一维线性插值函数

# 6自由度位姿（位置+旋转）插值器、
    # t：时间点序列
    # x：位姿序列（N×6数组，前3维位置，后3维旋转向量）
    # ​​核心组件​​：

    # ​​位置插值​​：pos_interp = get_interp1d(t, pos)（线性插值）
    # ​​旋转插值​​：rot_interp = st.Slerp(t, rot)（球面线性插值）
    # ​​__call__方法​​：

    # 输入时间点 t
    # 返回该时间点的插值位姿（6维向量）
    # ​​使用场景​​：

    # 机器人轨迹平滑插值

# 夹爪校准插值器、
    # aruco_measured_width：Aruco标记测量的宽度序列
    # aruco_actual_width：Aruco标记实际宽度序列
    # ​​核心组件​​：

    # ​​计算夹爪实际宽度​​：gripper_actual_width = aruco_actual_width - aruco_min_width
    # ​​线性插值​​：interp = get_interp1d(aruco_measured_width, gripper_actual_width)
    # ​​使用场景​​：

    # 夹爪校准

import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def get_interp1d(t, x):
    gripper_interp = si.interp1d(
        t, x, 
        axis=0, bounds_error=False, 
        fill_value=(x[0], x[-1]))
    return gripper_interp


class PoseInterpolator:
    def __init__(self, t, x):
        pos = x[:,:3]
        rot = st.Rotation.from_rotvec(x[:,3:])
        self.pos_interp = get_interp1d(t, pos)
        self.rot_interp = st.Slerp(t, rot)
    
    @property
    def x(self):
        return self.pos_interp.x
    
    def __call__(self, t):
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)

        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        rvec = rot.as_rotvec()
        pose = np.concatenate([pos, rvec], axis=-1)
        return pose

def get_gripper_calibration_interpolator(
        aruco_measured_width, 
        aruco_actual_width):
    """
    Assumes the minimum width in aruco_actual_width
    is measured when the gripper is fully closed
    and maximum width is when the gripper is fully opened
    """
    aruco_measured_width = np.array(aruco_measured_width)
    aruco_actual_width = np.array(aruco_actual_width)
    assert len(aruco_measured_width) == len(aruco_actual_width)
    assert len(aruco_actual_width) >= 2
    aruco_min_width = np.min(aruco_actual_width)
    gripper_actual_width = aruco_actual_width - aruco_min_width
    interp = get_interp1d(aruco_measured_width, gripper_actual_width)
    return interp
