import numpy as np

def detect_spikes(data, position_threshold=0.08, velocity_threshold=0.04,
                  acceleration_threshold=0.02):
    """
    检测突变点：仅基于位置、速度、加速度的阈值（已移除角度变化判断）
    """
    spikes = []
    if len(data) < 3:
        return spikes

    positions = data[:, :3]
    # 计算速度、加速度
    velocities = np.diff(positions, axis=0)
    accelerations = np.diff(velocities, axis=0)

    for i in range(2, len(data)):
        pos_jump = np.linalg.norm(positions[i] - positions[i - 1])
        vel_mag = np.linalg.norm(velocities[i - 1]) if i - 1 < len(velocities) else 0
        acc_mag = np.linalg.norm(accelerations[i - 2]) if i - 2 < len(accelerations) else 0

        print(f"帧 {i}: 位置突变={pos_jump:.4f}, 速度={vel_mag:.4f}, 加速度={acc_mag:.4f}")

        if (pos_jump > position_threshold or 
            vel_mag > velocity_threshold or 
            acc_mag > acceleration_threshold):
            print(f"  -> 检测到突变点于帧 {i}")
            spikes.append(i)
    return spikes

def interpolate_spikes(data, spikes, method='mean'):
    """
    修复突变点：
    - 位置：插值（均值或线性）
    - 欧拉角：统一使用段前一帧的角度
    """
    fixed = data.copy()
    if not spikes:
        print("未检测到突变点，无需插值。")
        return fixed

    print(f"插值修复以下帧的突变点: {spikes}")
    spikes_set = set(spikes)
    i = 0
    while i < len(spikes):
        start = spikes[i]
        end = start
        while end + 1 in spikes_set:
            end += 1
        i = spikes.index(end) + 1

        i_prev = start - 1
        while i_prev in spikes_set and i_prev > 0:
            i_prev -= 1
        i_next = end + 1
        while i_next in spikes_set and i_next < len(data) - 1:
            i_next += 1

        print(f"修复突变段从帧 {start} 到帧 {end}，插值参考帧为 {i_prev} 和 {i_next}。")

        if 0 <= i_prev < len(data) and 0 <= i_next < len(data):
            for j in range(start, end + 1):
                if method == 'mean':
                    fixed[j, :3] = (data[i_prev, :3] + data[i_next, :3]) / 2
                elif method == 'linear':
                    alpha = (j - i_prev) / (i_next - i_prev)
                    fixed[j, :3] = (1 - alpha) * data[i_prev, :3] + alpha * data[i_next, :3]
                fixed[j, 3:6] = data[i_prev, 3:6]  # 保持角度不变
                print(f"  帧 {j}: 位置修复为 {fixed[j, :3]}, 角度修复为 {fixed[j, 3:6]}")
    return fixed

def fix_gripper_jitters(data, gripper_threshold=0.04, min_duration=5):
    """
    修复夹爪跳变：检测夹爪的突变并进行平滑处理
    """
    fixed = data.copy()
    gripper = fixed[:, 6].copy()
    n = len(gripper)
    i = 0

    while i < n - 1:
        if abs(gripper[i] - gripper[i + 1]) > gripper_threshold:
            prev_state = gripper[i]
            start_index = i + 1
            j = start_index
            while j < n - 1 and abs(gripper[j] - gripper[j + 1]) <= gripper_threshold:
                j += 1
            end_index = j
            duration = end_index - start_index + 1
            if duration < min_duration:
                print(f"修复夹爪跳变：从帧 {start_index} 到帧 {end_index}，持续 {duration} 帧")
                fixed[start_index:end_index + 1, 6] = prev_state
                i = end_index
            else:
                i = end_index
        else:
            i += 1
    return fixed

def smooth_data(data, window_size=5):
    """
    平滑数据：使用滑动窗口平滑位置（角度不处理）
    """
    if len(data) == 0:
        return data.copy()

    smoothed = data.copy()
    half_window = window_size // 2

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]

        smoothed[i, :3] = np.mean(window[:, :3], axis=0)
        smoothed[i, 3:6] = data[i, 3:6]  # 保留原始角度

    return smoothed

def process_actions(actions):
    """
    主处理函数：修复和平滑动作数据（角度保持不变）
    """
    if len(actions) == 0:
        return actions

    print("开始检测突变点...")
    spikes = detect_spikes(actions)
    print(f"检测到的突变点帧号: {spikes}")

    fixed_actions = interpolate_spikes(actions, spikes)

    print("开始修复夹爪跳变...")
    fixed_actions = fix_gripper_jitters(fixed_actions)

    print("开始位置平滑处理...")
    smoothed_actions = smooth_data(fixed_actions)
    
    

    print("动作处理完成。")
    return smoothed_actions
    
    
    
    
    
#  ######使用代码

#             # 获取数据
#             obs_data = self.obs_accumulator.data
#             obs_timestamps = self.obs_accumulator.timestamps
#             actions = self.action_accumulator.actions
#             action_timestamps = self.action_accumulator.timestamps
#             stages = self.stage_accumulator.actions
#             stages_timestamps = self.stage_accumulator.timestamps
            
            
#             #修正代码
#             ###################################################
#             print("开始处理动作数据...")
#             try:
#                 # 确保动作数据格式正确
#                 if actions.ndim == 1:
#                     actions = actions.reshape(-1, 1)
                    
#                 # 处理动作数据 (XYZ + RPY + Gripper)
#                 processed_actions = process_actions(actions.copy())
                
#                 # 保留其他维度不变（如果有）
#                 if processed_actions.shape[1] < actions.shape[1]:
#                     processed_actions = np.hstack((
#                         processed_actions,
#                         actions[:, processed_actions.shape[1]:]
#                     ))
                    
#                 actions = processed_actions  # 使用处理后的动作
#                 print("动作数据处理完成")
#             except Exception as e:
#                 print(f"动作处理失败: {str(e)}")
#                 import traceback
#                 traceback.print_exc()
#             ##################################################

