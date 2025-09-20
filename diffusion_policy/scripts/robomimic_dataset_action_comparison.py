"""
从两个 HDF5 文件中读取数据（动作数据），
计算它们的位移误差（pos_dist）和旋转误差（rot_dist），并打印出最大的位置和旋转误差
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import pathlib
import h5py
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation

def read_all_actions(hdf5_file, metric_skip_steps=1):
    """
    该函数的作用是从 HDF5 文件中读取所有的动作数据，并将它们合并为一个大的数组
    每个演示的动作存储在路径 f'data/demo_{i}/actions' 中
    """
    n_demos = len(hdf5_file['data'])                        # 获取数据集中的演示数量
    all_actions = list()                                    # 存储所有动作数据的列表
    for i in tqdm(range(n_demos)):                          # 遍历所有演示
        actions = hdf5_file[f'data/demo_{i}/actions'][:]    # 获取每个演示的动作数据
        all_actions.append(actions[metric_skip_steps:])     # 从动作数据中跳过指定的步骤，并将其添加到列表中
    all_actions = np.concatenate(all_actions, axis=0)       # 合并所有动作数据为一个大的数组
    return all_actions


@click.command()
@click.option('-i', '--input', required=True, help='input hdf5 path')
@click.option('-o', '--output', required=True, help='output hdf5 path. Parent directory must exist')
def main(input, output):
    # 从命令行读取输入文件和输出文件的路径，并执行相关的计算
    # process inputs
    input = pathlib.Path(input).expanduser()    # 扩展用户目录（如 ~）为完整路径
    assert input.is_file()                      # 确保输入路径是一个有效的文件

    output = pathlib.Path(output).expanduser()  # 扩展输出路径
    assert output.is_file()                     # 确保输出路径是一个有效的文件

    # 打开输入和输出 HDF5 文件
    input_file = h5py.File(str(input), 'r')
    output_file = h5py.File(str(output), 'r')

    # 读取所有动作数据
    input_all_actions = read_all_actions(input_file)
    output_all_actions = read_all_actions(output_file)
    # 计算位置误差（pos_dist）和旋转误差（rot_dist）
    pos_dist = np.linalg.norm(input_all_actions[:,:3] - output_all_actions[:,:3], axis=-1)# 计算两个动作的前三个坐标值（即位置坐标）之间的欧几里得距离
    rot_dist = (Rotation.from_rotvec(input_all_actions[:,3:6]
        ) * Rotation.from_rotvec(output_all_actions[:,3:6]).inv()# 将旋转向量转换为旋转对象，然后计算两个旋转对象的相对旋转（inv() 表示旋转的逆）
        ).magnitude()

    # 打印最大的位置误差和旋转误差
    print(f'max pos dist: {pos_dist.max()}')
    print(f'max rot dist: {rot_dist.max()}')

if __name__ == "__main__":
    main()
