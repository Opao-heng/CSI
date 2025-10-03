"""
入侵者检测模块数据加载器
根据研究内容二的要求进行数据集划分
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class IntruderDataset(Dataset):
    """入侵者检测数据集"""
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_and_split_data():
    """
    加载源域和目标域数据，并进行数据集划分
    
    数据集划分：
    ○ 源域训练集：env0 和 env1 中 10 名已知用户的 CSI 数据
    ○ 目标域辅助集：env2 中 10 名已知用户的少量数据（用于域适应）
    ○ 验证集：从目标域辅助集中划分部分数据，并加入少量模拟入侵者数据（用于阈值确定）
    ○ 测试集：env2 中 10 名已知用户（300 样本）和 3 名入侵者（90 样本）的混合数据
    """
    
    # 加载源域数据（env0 和 env1 中 10 名已知用户的 CSI 数据）
    print("加载源域数据...")
    source_data = torch.load('../data/SourceData/source_data.pt')
    source_labels = torch.load('../data/SourceData/source_labels.pt')
    
    print(f"源域数据形状: {source_data.shape}")
    print(f"源域标签形状: {source_labels.shape}")
    
    # 加载目标域已知用户数据（env2 中 10 名已知用户）
    print("加载目标域已知用户数据...")
    target_legal_data = torch.load('../data/TargetData/target_data.pt')
    target_legal_labels = torch.load('../data/TargetData/target_labels.pt')
    
    print(f"目标域已知用户数据形状: {target_legal_data.shape}")
    print(f"目标域已知用户标签形状: {target_legal_labels.shape}")
    
    # 加载入侵者数据（env2 中 3 名入侵者），入侵者数据在 env2_intruder 目录中
    print("加载入侵者数据...")
    intruder_data_list = []
    intruder_labels_list = []
    
    # 为入侵者分配标签 -1
    for i in range(3):
        intruder_id = 10 + i  # 入侵者ID从10开始
        intruder_file = f'../data/env2_intruder/id_{intruder_id}_env_2.npy'

        intruder_data = np.load(intruder_file)
        # 转换为张量并重塑
        intruder_tensor = torch.from_numpy(intruder_data)
        N = intruder_tensor.shape[3]
        intruder_reshaped = intruder_tensor.permute(3, 0, 1, 2).contiguous()
        intruder_labels = torch.full((N,), -1, dtype=torch.long)  # 入侵者标签为-1

        intruder_data_list.append(intruder_reshaped)
        intruder_labels_list.append(intruder_labels)

    intruder_data = torch.cat(intruder_data_list, dim=0)
    intruder_labels = torch.cat(intruder_labels_list, dim=0)
    print(f"入侵者数据形状: {intruder_data.shape}")
    print(f"入侵者标签形状: {intruder_labels.shape}")
    
    # 数据集划分
    print("\n进行数据集划分...")
    
    # 1. 源域训练集：全部源域数据
    source_train_data = source_data
    source_train_labels = source_labels

    print(f"源域训练集数据形状: {source_train_data.shape}")
    print(f"源域训练集标签形状: {source_train_labels.shape}")
    
    # 2. 目标域辅助训练集：从目标域已知用户数据中抽取少量数据用于域适应
    target_indices = np.arange(len(target_legal_data))
    target_aux_indices, target_remaining_indices = train_test_split(
        target_indices, test_size=0.7, random_state=42, stratify=target_legal_labels.numpy()
    )
    # 随机抽取30%的数据作为目标域辅助训练集
    target_aux_data = target_legal_data[target_aux_indices]
    target_aux_labels = target_legal_labels[target_aux_indices]
    
    # 剩余的目标域已知用户数据用于测试集
    target_test_legal_data = target_legal_data[target_remaining_indices]
    target_test_legal_labels = target_legal_labels[target_remaining_indices]
    
    print(f"目标域辅助训练集数据形状: {target_aux_data.shape}")
    print(f"目标域辅助训练集标签形状: {target_aux_labels.shape}")
    
    # 3. 验证集：从目标域辅助集中划分部分数据，并加入少量模拟入侵者数据
    # 从目标域辅助集中抽取50个样本作为验证集
    val_indices = np.random.choice(len(target_aux_data), size=min(50, len(target_aux_data)), replace=False)
    val_legal_data = target_aux_data[val_indices]
    val_legal_labels = target_aux_labels[val_indices]

    # 训练时没有真实入侵者数据，创建一些模拟入侵者数据
    val_data = val_legal_data
    val_labels = val_legal_labels
    num_simulated_intruders_per_method = 50  # 每种方法生成50个样本
    total_simulated_intruders = 200  # 总共生成200个模拟入侵者样本

    simulated_intruder_list = []
    simulated_labels_list = []

    # 方法1: 在合法用户数据基础上添加噪声
    base_indices_1 = np.random.choice(len(target_aux_data), size=num_simulated_intruders_per_method, replace=True)
    simulated_data_1 = target_aux_data[base_indices_1].clone()
    noise_1 = torch.randn_like(simulated_data_1) * 0.5
    simulated_data_1 = simulated_data_1 + noise_1
    simulated_intruder_list.append(simulated_data_1)

    # 方法2: 使用幅度缩放
    base_indices_2 = np.random.choice(len(target_aux_data), size=num_simulated_intruders_per_method, replace=True)
    simulated_data_2 = target_aux_data[base_indices_2].clone()
    scale_factors = torch.FloatTensor(num_simulated_intruders_per_method, 1, 1, 1).uniform_(0.3, 3.0)
    simulated_data_2 = simulated_data_2 * scale_factors
    simulated_intruder_list.append(simulated_data_2)

    # 方法3: 时间序列偏移
    base_indices_3 = np.random.choice(len(target_aux_data), size=num_simulated_intruders_per_method, replace=True)
    simulated_data_3 = target_aux_data[base_indices_3].clone()
    # 对时间维度(最后一个维度)进行循环移位
    shifts = np.random.randint(0, 6000, size=num_simulated_intruders_per_method)
    for i in range(num_simulated_intruders_per_method):
        shift = shifts[i]
        simulated_data_3[i] = torch.roll(simulated_data_3[i], shifts=shift, dims=2)
    simulated_intruder_list.append(simulated_data_3)

    # 方法4: 组合方法 - 噪声+缩放
    base_indices_4 = np.random.choice(len(target_aux_data), size=num_simulated_intruders_per_method, replace=True)
    simulated_data_4 = target_aux_data[base_indices_4].clone()
    # 添加噪声
    noise_4 = torch.randn_like(simulated_data_4) * 0.3
    simulated_data_4 = simulated_data_4 + noise_4
    # 幅度缩放
    scale_factors_4 = torch.FloatTensor(num_simulated_intruders_per_method, 1, 1, 1).uniform_(0.7, 1.5)
    simulated_data_4 = simulated_data_4 * scale_factors_4
    simulated_intruder_list.append(simulated_data_4)

    # 合并所有模拟入侵者数据
    simulated_intruder_data = torch.cat(simulated_intruder_list, dim=0)
    simulated_intruder_labels = torch.full((total_simulated_intruders,), -1, dtype=torch.long)

    val_data = torch.cat([val_data, simulated_intruder_data], dim=0)
    val_labels = torch.cat([val_labels, simulated_intruder_labels], dim=0)

    print(f"验证集数据形状: {val_data.shape}")
    print(f"验证集标签形状: {val_labels.shape}")
    
    # 4. 测试集：目标域已知用户剩余数据 + 入侵者数据
    # 目标域已知用户数据
    test_known_data = target_test_legal_data
    test_known_labels = target_test_legal_labels
    
    # 入侵者数据
    test_intruder_data = intruder_data
    test_intruder_labels = intruder_labels

    # 合并测试集数据
    test_data = torch.cat([test_known_data, test_intruder_data], dim=0)
    test_labels = torch.cat([test_known_labels, test_intruder_labels], dim=0)
    
    print(f"测试集数据形状: {test_data.shape}")
    print(f"测试集标签形状: {test_labels.shape}")
    
    # 创建数据集对象
    source_train_dataset = IntruderDataset(source_train_data, source_train_labels)
    target_aux_dataset = IntruderDataset(target_aux_data, target_aux_labels)
    val_dataset = IntruderDataset(val_data, val_labels)
    test_dataset = IntruderDataset(test_data, test_labels)
    
    return {
        'source_train': source_train_dataset,
        'target_aux': target_aux_dataset,
        'validation': val_dataset,
        'test': test_dataset,
        'source_train_raw': (source_train_data, source_train_labels),
        'target_aux_raw': (target_aux_data, target_aux_labels),
        'validation_raw': (val_data, val_labels),
        'test_raw': (test_data, test_labels)
    }


def create_data_loaders(datasets, batch_size=32):
    """
    创建数据加载器
    """
    data_loaders = {}
    
    for key, dataset in datasets.items():
        if key.endswith('_raw'):
            continue  # 跳过原始数据
            
        data_loaders[key] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(key != 'test'),  # 测试集不打乱
            num_workers=0  # Windows兼容性
        )
        
    return data_loaders


if __name__ == "__main__":
    # 测试数据加载和划分
    try:
        datasets = load_and_split_data()
        data_loaders = create_data_loaders(datasets)
        
        print("\n数据加载器创建完成:")
        for key, loader in data_loaders.items():
            print(f"{key} 数据加载器: {len(loader)} 个批次")
            
        # 显示一些样本信息
        print("\n样本信息:")
        for key, (data, labels) in datasets.items():
            if key.endswith('_raw'):
                print(f"{key}: 数据形状 {data.shape}, 标签形状 {labels.shape}")
                # 显示标签分布
                unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
                print(f"  标签分布: {dict(zip(unique_labels, counts))}")
                
    except Exception as e:
        print(f"数据加载过程中出现错误: {e}")
        import traceback
        traceback.print_exc()