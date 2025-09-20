import numpy as np
import scipy.spatial.transform as st
# 处理和转换与姿态和变换相关的数学操作。下面是每个函数的简要说明：
# 这段代码提供了一系列的函数，用于处理和转换与姿态和变换相关的数学操作。下面是每个函数的简要说明：
# 1. `pos_rot_to_mat(pos, rot)`:
#    - 接受一个位置 `pos` 和一个旋转 `rot`，并返回一个4x4的变换矩阵。
# 2. `mat_to_pos_rot(mat)`:
#    - 接受一个4x4的变换矩阵 `mat`，并返回位置和旋转。
# 3. `pos_rot_to_pose(pos, rot)`:
#    - 接受一个位置 `pos` 和一个旋转 `rot`，并返回一个包含位置和旋转的6D姿态。
# 4. `pose_to_pos_rot(pose)`:
#    - 接受一个6D姿态 `pose`，并返回位置和旋转。
# 5. `pose_to_mat(pose)`:
#    - 接受一个6D姿态 `pose`，并返回一个4x4的变换矩阵。
# 6. `mat_to_pose(mat)`:
#    - 接受一个4x4的变换矩阵 `mat`，并返回一个6D姿态。
# 7. `transform_pose(tx, pose)`:
#    - 接受一个变换矩阵 `tx` 和一个姿态 `pose`，并返回变换后的姿态。
# 8. `transform_point(tx, point)`:
#    - 接受一个变换矩阵 `tx` 和一个点 `point`，并返回变换后的点。
# 9. `project_point(k, point)`:
#    - 接受一个相机内参矩阵 `k` 和一个点 `point`，并返回投影后的图像坐标。
# 10. `apply_delta_pose(pose, delta_pose)`:
#     - 接受一个姿态 `pose` 和一个姿态增量 `delta_pose`，并返回新的姿态。
# 11. `rot_from_directions(from_vec, to_vec)`:
#     - 接受两个方向向量 `from_vec` 和 `to_vec`，并返回从 `from_vec` 到 `to_vec` 的旋转。
# 12. `normalize(vec, tol=1e-7)`:
#     - 接受一个向量 `vec` 和一个容差 `tol`，并返回归一化的向量。
# 13. `rot6d_to_mat(d6)`:
#     - 接受一个6D旋转 `d6`，并返回一个4x4的旋转矩阵。
# 14. `mat_to_rot6d(mat)`:
#     - 接受一个4x4的旋转矩阵 `mat`，并返回一个6D旋转。
# 15. `mat_to_pose10d(mat)`:
#     - 接受一个4x4的变换矩阵 `mat`，并返回一个10D姿态。
# 16. `pose10d_to_mat(d10)`:
#     - 接受一个10D姿态 `d10`，并返回一个4x4的变换矩阵。
# 这些函数提供了从不同的坐标空间和表示形式转换的能力，以及进行变换和投影的基本操作。它们通常用于机器人、计算机视觉和3D图形等领域，其中需要处理和转换不同类型的姿态和变换。

def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])
    return pos, rot

def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))

def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose

def transform_point(tx, point):
    return point @ tx[:3,:3].T + tx[:3,3]

def project_point(k, point):
    x = point @ k.T
    uv = x[...,:2] / x[...,[2]]
    return uv

def apply_delta_pose(pose, delta_pose):
    new_pose = np.zeros_like(pose)

    # simple add for position
    new_pose[:3] = pose[:3] + delta_pose[:3]

    # matrix multiplication for rotation
    rot = st.Rotation.from_rotvec(pose[3:])
    drot = st.Rotation.from_rotvec(delta_pose[3:])
    new_pose[3:] = (drot * rot).as_rotvec()

    return new_pose

# def normalize(vec, tol=1e-7):
#     return vec / np.maximum(np.linalg.norm(vec), tol)

def rot_from_directions(from_vec, to_vec):
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    axis = np.cross(from_vec, to_vec)
    axis = normalize(axis)
    angle = np.arccos(np.dot(from_vec, to_vec))
    rotvec = axis * angle
    rot = st.Rotation.from_rotvec(rotvec)
    return rot

# 应删除重复定义，保留更安全的版本
def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    # return vec / norm
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose10d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def pose10d_to_mat(d10):
    pos = d10[...,:3]
    d6 = d10[...,3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1]+(4,4), dtype=d10.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out