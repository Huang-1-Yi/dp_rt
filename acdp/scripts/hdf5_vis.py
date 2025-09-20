import h5py
import numpy as np

def inspect_hdf5(file_path):
    print(f"Inspecting: {file_path}")
    with h5py.File(file_path, 'r') as f:
        # 获取演示集列表
        demos = list(f['data'].keys())
        print(f"数据集中包含 {len(demos)} 个演示 (demos)")
        
        total_samples = 0
        demo_info = []
        
        for i, demo_name in enumerate(demos):
            demo_group = f[f'data/{demo_name}']
            
            # 获取状态数据
            states = demo_group['states'][:]
            actions = demo_group['actions'][:]
            
            # 检测观察数据类型
            obs_types = []
            if 'obs' in demo_group:
                for obs_name in demo_group['obs']:
                    obs_data = demo_group[f'obs/{obs_name}']
                    if isinstance(obs_data, h5py.Dataset):
                        obs_types.append(obs_name)
            
            # 记录信息
            samples = states.shape[0]
            total_samples += samples
            demo_info.append({
                'name': demo_name,
                'samples': samples,
                'state_shape': states.shape,
                'action_shape': actions.shape,
                'obs_types': obs_types
            })
        
        print(f"\n总样本数 (transitions): {total_samples}")
        print(f"平均每demo样本数: {total_samples/len(demos):.1f}")
        
        # 打印元数据
        print("\n元数据:")
        for k, v in f.attrs.items():
            print(f"- {k}: {v}")
            
        # 打印前3个demo的详细信息
        print("\n前3个demo详情:")
        for info in demo_info[:3]:
            print(f"{info['name']}:")
            print(f"  样本数: {info['samples']}")
            print(f"  状态数据维度: {info['state_shape']}")
            print(f"  动作数据维度: {info['action_shape']}")
            print(f"  包含观察数据类型: {info['obs_types']}")

if __name__ == "__main__":
    # file_path = "/home/robot/Desktop/stagedp/data/robomimic/datasets/stack_d1/stack_d1.hdf5"
    file_path = "/home/robot/Desktop/stagedp/data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5"
    inspect_hdf5(file_path)