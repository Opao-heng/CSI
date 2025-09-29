import h5py
import numpy as np
import os

def load_mat_data(file_path):
    """
    读取 v7.3 格式的 .mat 文件，并返回 amp、id 和 env 数据。
    参数: file_path (str): .mat 文件的路径。
    返回: tuple: 包含 amp_data, id_data, env_data 的 numpy 数组。
    """
    with h5py.File(file_path, 'r') as f:
        # 进入 data 组
        data_group = f['data']

        # 提取数据
        amp_data = data_group['amp'][()]
        id_data = data_group['id'][()]
        env_data = data_group['env'][()]

    return amp_data, id_data, env_data


def group_amp_data(amp_data, id_data, env_data):
    """
    按照 id 和 env 对 amp 数据进行分组

    参数:
        amp_data (numpy.ndarray): 幅度数据，形状为 (num_subcarriers, num_tx_rx, num_time, num_samples)
        id_data (numpy.ndarray): ID数据，形状为 (num_samples,)
        env_data (numpy.ndarray): 环境数据，形状为 (num_samples,)

    返回:
        dict: 按照 (id, env) 分组的 amp 数据字典
    """
    # 确保数据形状一致
    num_samples = amp_data.shape[3]  # CSI数据在第4个维度

    # 创建分组字典
    grouped_data = {}

    # 遍历所有样本，按 id 和 env 分组
    for i in range(num_samples):
        # 获取当前样本的 id 和 env
        current_id = int(id_data[i])
        current_env = int(env_data[i])

        # 创建分组键
        group_key = (current_id, current_env)

        # 如果该分组不存在，创建一个新的列表
        if group_key not in grouped_data:
            grouped_data[group_key] = []

        # 将当前样本的 amp 数据添加到对应分组
        # 提取第i个样本的数据，保持前3个维度不变
        sample_data = amp_data[:, :, :, i]
        grouped_data[group_key].append(sample_data)

    # 将列表转换为 numpy 数组
    for key in grouped_data:
        # 转换为数组，形状为 (num_samples_in_group, num_subcarriers, num_tx_rx, num_time)
        grouped_data[key] = np.array(grouped_data[key])
        # 转置为 (num_subcarriers, num_tx_rx, num_time, num_samples_in_group)
        grouped_data[key] = np.transpose(grouped_data[key], (1, 2, 3, 0))

    return grouped_data


def save_grouped_data(grouped_data, output_dir='grouped_data'):
    """
    保存分组后的数据到文件

    参数:
        grouped_data (dict): 分组后的数据
        output_dir (str): 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存每个分组的数据
    for (id_val, env_val), data in grouped_data.items():
        # 创建文件名
        filename = f"id_{id_val}_env_{env_val}.npy"
        filepath = os.path.join(output_dir, filename)

        # 保存数据
        np.save(filepath, data)
        print(f"保存分组数据: ID={id_val}, ENV={env_val}, 样本数={data.shape[3]}, 形状={data.shape}")


if __name__ == '__main__':
    file_path = 'data/v1/test_legal.mat'
    amp_data, id_data, env_data = load_mat_data(file_path)
    print(f"原始数据形状: amp={amp_data.shape}, id={id_data.shape}, env={env_data.shape}")

    # 按 id 和 env 分组 amp 数据
    grouped_data = group_amp_data(amp_data, id_data, env_data)

    # 输出分组信息
    print("\n数据分组信息:")
    for (id_val, env_val), data in grouped_data.items():
        print(f"ID={id_val}, ENV={env_val}, 样本数={data.shape[3]}, 数据形状={data.shape}")

    # 保存分组数据
    save_grouped_data(grouped_data)

    print(f"\n总共创建了 {len(grouped_data)} 个分组")
