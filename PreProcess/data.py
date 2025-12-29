import h5py
import numpy as np
import os


def load_mat_data(file_path):
    """
    从HDF5格式的.mat文件中读取CSI幅度数据、用户ID和环境标签
    参数: file_path (str) - v7.3格式的.mat文件路径
    返回: tuple - (amp_data, id_data, env_data) 三个numpy数组
    """
    # 以只读模式打开HDF5文件
    with h5py.File(file_path, 'r') as f:
        # 步骤1: 访问data组
        data_group = f['RawData']
        # 步骤2: 从data组中提取幅度、ID和环境数据
        amp_data = data_group['amp'][()]
        id_data = data_group['id'][()]
        env_data = data_group['env'][()]
    # 步骤3: 返回三个数据数组
    return amp_data, id_data, env_data


def group_amp_data(amp_data, id_data, env_data):
    """
    根据用户ID和环境标签对CSI幅度数据进行分组
    参数:
      amp_data (numpy.ndarray): 幅度数据，形状为 (num_subcarriers, num_tx_rx, num_time, num_samples)
      id_data (numpy.ndarray): 用户ID数组，形状为 (num_samples,)
      env_data (numpy.ndarray): 环境标签数组，形状为 (num_samples,)
    返回: dict - 键为(id, env)元组，值为该分组下的amp数据numpy数组
    """
    # 步骤1: 获取样本总数
    num_samples = amp_data.shape[3]  # CSI数据在第4个维度
    grouped_data = {}

    # 步骤2: 遍历所有样本，按(id, env)组合分组
    for i in range(num_samples):
        current_id = int(id_data[i])
        current_env = int(env_data[i])
        group_key = (current_id, current_env)

        # 初始化分组列表
        if group_key not in grouped_data:
            grouped_data[group_key] = []

        # 提取第i个样本的amp数据并添加到分组
        sample_data = amp_data[:, :, :, i]
        grouped_data[group_key].append(sample_data)

    # 步骤3: 将列表形式的数据转换为numpy数组，并调整维度顺序
    for key in grouped_data:
        # 转换为数组，形状为 (num_samples_in_group, num_subcarriers, num_tx_rx, num_time)
        grouped_data[key] = np.array(grouped_data[key])
        # 转置为 (num_subcarriers, num_tx_rx, num_time, num_samples_in_group)
        grouped_data[key] = np.transpose(grouped_data[key], (1, 2, 3, 0))

    return grouped_data


def save_grouped_data(grouped_data, output_dir='grouped_data'):
    """
    将分组后的数据保存为.npy文件
    参数:
      grouped_data (dict) - 分组数据字典，键为(id, env)元组
      output_dir (str) - 输出目录路径，默认为'grouped_data'
    返回: 无返回值，直接将数据写入文件
    """
    # 步骤1: 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 步骤2: 遍历每个分组，逐个保存为.npy文件
    for (id_val, env_val), data in grouped_data.items():
        # 根据id和env生成文件名
        filename = f"id_{id_val}_env_{env_val}.npy"
        filepath = os.path.join(output_dir, filename)

        # 步骤3: 保存数据并输出日志信息
        np.save(filepath, data)
        print(f"保存分组数据: ID={id_val}, ENV={env_val}, 样本数={data.shape[3]}, 形状={data.shape}")


if __name__ == '__main__':
    file_path = 'RawData/v1/test_legal.mat'
    amp_data, id_data, env_data = load_mat_data(file_path)
    print(f"原始数据形状: amp={amp_data.shape}, id={id_data.shape}, env={env_data.shape}")

    # 步骤1: 按 id 和 env 分组 amp 数据
    grouped_data = group_amp_data(amp_data, id_data, env_data)

    # 步骤2: 输出分组信息
    print("\n数据分组信息:")
    for (id_val, env_val), data in grouped_data.items():
        print(f"ID={id_val}, ENV={env_val}, 样本数={data.shape[3]}, 数据形状={data.shape}")

    # 步骤3: 保存分组数据
    save_grouped_data(grouped_data)

    print(f"\n总共创建了 {len(grouped_data)} 个分组")
