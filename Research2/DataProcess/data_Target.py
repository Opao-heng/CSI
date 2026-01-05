import os
import torch
import numpy as np
import re

"""
读取环境文件夹下的所有.npy/.npz数据文件，合并成整体数据和标签张量
参数: data_dir (str) - 包含数据文件的目录路径
返回: tuple - (data_tensor, label_tensor)
  data_tensor: 合并后的数据张量，形状为(total_samples, 56, 3, 6000)
  label_tensor: 合并后的标签张量，形状为(total_samples,)
"""
def load_env_data(data_dir):
    # 步骤1: 初始化数据和标签存储列表
    all_data = []
    all_labels = []

    # 步骤2: 遍历目录下的所有.npy和.npz文件
    for filename in os.listdir(data_dir):
        if filename.endswith('.npy') or filename.endswith('.npz'):
            # 步骤3: 从文件名中提取用户ID（文件名格式：id_0_env_2.npy）
            match = re.search(r'id_(\d+)_env_\d+', filename)
            if match:
                file_id = int(match.group(1))

                # 步骤4: 加载数据文件
                file_path = os.path.join(data_dir, filename)
                if filename.endswith('.npy'):
                    data = np.load(file_path)
                else:  # .npz格式
                    with np.load(file_path) as loaded:
                        data = loaded['arr_0']  # 数据存储在arr_0中

                # 步骤5: 验证数据形状是否正确 (56, 3, 6000, N)
                assert data.shape[0] == 56, f"数据第0维应该是56，但得到{data.shape[0]}"
                assert data.shape[1] == 3, f"数据第1维应该是3，但得到{data.shape[1]}"
                assert data.shape[2] == 6000, f"数据第2维应该是6000，但得到{data.shape[2]}"

                # 步骤6: 将numpy数组转换为PyTorch张量
                data_tensor = torch.from_numpy(data)

                # 步骤7: 获取样本数量N
                N = data_tensor.shape[3]

                # 步骤8: 重塑数据为 (N, 56, 3, 6000)
                data_reshaped = data_tensor.permute(3, 0, 1, 2).contiguous()

                # 步骤9: 创建标签张量，所有样本标签都是file_id
                labels = torch.full((N,), file_id, dtype=torch.long)

                # 步骤10: 添加处理后的数据和标签到列表
                all_data.append(data_reshaped)
                all_labels.append(labels)

                print(f"处理文件: {filename}, id: {file_id}, 样本数: {N}")

    # 步骤11: 合并所有数据并输出信息
    if all_data:
        # 按第0维(样本数)拼接
        final_data = torch.cat(all_data, dim=0)
        final_labels = torch.cat(all_labels, dim=0)

        print(f"总样本数: {final_data.shape[0]}")
        print(f"数据形状: {final_data.shape}")
        print(f"标签形状: {final_labels.shape}")

        return final_data, final_labels
    else:
        # 步骤12: 处理未找到数据的情况
        print("未找到符合条件的数据文件")
        return None, None


"""
主程序入口：加载目标域数据並保存
功能：
  1. 调用load_env_data加载数据
  2. 验评数据加载是否成功
  3. 保存合并后的数据为PyTorch格式
"""
if __name__ == "__main__":
    # 步骤1: 指定数据目录
    data_directory = "../../RawData/env2_ldentify"

    # 步骤2: 调用函数加载数据
    target_data, target_labels = load_env_data(data_directory)

    # 步骤3: 检查数据是否加载成功
    if target_data is not None:
        print("数据加载完成！")
        print(f"目标域数据形状: {target_data.shape}")
        print(f"目标域标签形状: {target_labels.shape}")

    # 步骤4: 可选：保存合并后的数据为PyTorch格式
    torch.save(target_data, '../Data/target_env2_data.pt')
    torch.save(target_labels, '../Data/target_env2_labels.pt')
