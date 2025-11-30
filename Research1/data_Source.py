import os
import torch
import numpy as np
import re

"""
读取单个环境文件夹下的所有.npy/.npz数据文件并转换为张量
参数: data_dir (str) - 包含数据文件的目录路径
返回: tuple - (all_data, all_labels) 数据张量列表和标签张量列表
"""
def load_single_env_data(data_dir):
    # 步骤1: 初始化数据和标签列表
    all_data = []
    all_labels = []
    
    # 步骤2: 遍历目录下的所有.npy和.npz文件
    for filename in os.listdir(data_dir):
        if filename.endswith('.npy') or filename.endswith('.npz'):
            # 步骤3: 从文件名中提取用户ID
            match = re.search(r'id_(\d+)_env_\d+', filename)
            if match:
                file_id = int(match.group(1))
                
                # 步骤4: 加载数据文件
                file_path = os.path.join(data_dir, filename)
                if filename.endswith('.npy'):
                    data = np.load(file_path)
                else:  # .npz格式
                    with np.load(file_path) as loaded:
                        data = loaded['arr_0']
                
                # 步骤5: 验证数据形状 (56, 3, 6000, N)
                assert data.shape[0] == 56, f"数据第0维应该是56，但得到{data.shape[0]}"
                assert data.shape[1] == 3, f"数据第1维应该是3，但得到{data.shape[1]}"
                assert data.shape[2] == 6000, f"数据第2维应该是6000，但得到{data.shape[2]}"
                
                # 步骤6: 将numpy数组转换为PyTorch张量
                data_tensor = torch.from_numpy(data)
                
                # 步骤7: 获取样本数量N并重塑数据为 (N, 56, 3, 6000)
                N = data_tensor.shape[3]
                data_reshaped = data_tensor.permute(3, 0, 1, 2).contiguous()
                
                # 步骤8: 创建标签张量，所有样本标签都是file_id
                labels = torch.full((N,), file_id, dtype=torch.long)
                
                # 步骤9: 添加处理后的数据和标签到列表中
                all_data.append(data_reshaped)
                all_labels.append(labels)
                
                print(f"处理文件: {filename}, id: {file_id}, 样本数: {N}")
    
    return all_data, all_labels


"""
加载并合并两个环境(env0和env1)的数据，形成源域数据集
参数:
  env0_dir (str) - env0数据目录路径
  env1_dir (str) - env1数据目录路径
返回: tuple - (source_data, source_labels)
  source_data: 合并后的数据张量，形状为 (total_samples, 56, 3, 6000)
  source_labels: 合并后的标签张量，形状为 (total_samples,)
"""
def load_combined_env_data(env0_dir, env1_dir):
    # 步骤1: 加载env0环境的数据
    print("加载env0数据...")
    env0_data_list, env0_labels_list = load_single_env_data(env0_dir)
    
    # 步骤2: 加载env1环境的数据
    print("\n加载env1数据...")
    env1_data_list, env1_labels_list = load_single_env_data(env1_dir)
    
    # 步骤3: 合并两个环境的数据和标签列表
    all_data = env0_data_list + env1_data_list
    all_labels = env0_labels_list + env1_labels_list
    
    # 步骤4: 检查是否有数据，有则沿样本维度(dim=0)拼接
    if all_data:
        source_data = torch.cat(all_data, dim=0)
        source_labels = torch.cat(all_labels, dim=0)
        
        # 步骤5: 输出最终数据信息
        print(f"\n总样本数: {source_data.shape[0]}")
        print(f"数据形状: {source_data.shape}")
        print(f"标签形状: {source_labels.shape}")
        
        return source_data, source_labels
    else:
        print("未找到符合条件的数据文件")
        return None, None


"""
主程序入口：加载源域数据并保存
"""
if __name__ == "__main__":
    # 步骤1: 指定两个环境的数据目录
    env0_directory = "../data/env0_ldentify"
    env1_directory = "../data/env1_ldentify"
    
    # 步骤2: 加载并合并两个环境的数据
    source_data, source_labels = load_combined_env_data(env0_directory, env1_directory)
    
    # 步骤3: 如果数据加载成功，输出信息并保存
    if source_data is not None:
        print(f"源域数据形状: {source_data.shape}")
        print(f"源域标签形状: {source_labels.shape}")
        
        # 步骤4: 可选保存合并后的数据为PyTorch格式
        torch.save(source_data, 'Data/source_env0_env1_data.pt')
        torch.save(source_labels, 'Data/source_env0_env1_labels.pt')
