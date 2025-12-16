import os
import torch
import numpy as np
import re

def load_env_data(data_dir):
    """
    读取env文件夹下的所有数据文件，合并成数据和标签张量

    Args:
        data_dir (str): 包含数据文件的目录路径

    Returns:
        tuple: (data_tensor, label_tensor)
            - data_tensor: 形状为(总样本数, 3, 6000, N)的张量
            - label_tensor: 形状为(总样本数,)的标签张量
    """

    # 存储所有数据和标签
    all_data = []
    all_labels = []

    # 遍历目录下的所有文件
    for filename in os.listdir(data_dir):
        if filename.endswith('.npy') or filename.endswith('.npz'):
            # 从文件名中提取id
            # 假设文件名格式为 id_0_env_2.npy 或类似格式
            match = re.search(r'id_(\d+)_env_\d+', filename)
            if match:
                file_id = int(match.group(1))

                # 加载文件数据
                file_path = os.path.join(data_dir, filename)
                if filename.endswith('.npy'):
                    data = np.load(file_path)
                else:  # .npz格式
                    with np.load(file_path) as loaded:
                        data = loaded['arr_0']  # 假设数据在arr_0中

                # 确保数据形状正确 (56, 3, 6000, N)
                assert data.shape[0] == 56, f"数据第0维应该是56，但得到{data.shape[0]}"
                assert data.shape[1] == 3, f"数据第1维应该是3，但得到{data.shape[1]}"
                assert data.shape[2] == 6000, f"数据第2维应该是6000，但得到{data.shape[2]}"

                # 转换为张量
                data_tensor = torch.from_numpy(data)

                # 获取样本数量N
                N = data_tensor.shape[3]

                # 重塑数据为 (N, 56, 3, 6000)
                data_reshaped = data_tensor.permute(3, 0, 1, 2).contiguous()

                # 创建标签张量，所有标签都是file_id
                labels = torch.full((N,), file_id, dtype=torch.long)

                # 添加到总数据中
                all_data.append(data_reshaped)
                all_labels.append(labels)

                print(f"处理文件: {filename}, id: {file_id}, 样本数: {N}")

    # 合并所有数据
    if all_data:
        # 按第0维(样本数)拼接
        final_data = torch.cat(all_data, dim=0)
        final_labels = torch.cat(all_labels, dim=0)

        print(f"总样本数: {final_data.shape[0]}")
        print(f"数据形状: {final_data.shape}")
        print(f"标签形状: {final_labels.shape}")

        return final_data, final_labels
    else:
        print("未找到符合条件的数据文件")
        return None, None


# 使用示例
if __name__ == "__main__":
    # 指定数据目录
    data_directory = "../data/env2_ldentify"

    # 加载数据
    target_data, target_labels = load_env_data(data_directory)

    if target_data is not None:
        # 现在target_data和target_labels可以作为目标域数据使用
        print("数据加载完成！")
        print(f"目标域数据形状: {target_data.shape}")
        print(f"目标域标签形状: {target_labels.shape}")

    # 可选：保存合并后的数据
    torch.save(target_data, '../data/target_env2_data.pt')
    torch.save(target_labels, '../data/target_env2_labels.pt')
