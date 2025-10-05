import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import os


class IntruderDataset(Dataset):
    """身份识别数据集"""
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # 确保返回的数据格式正确
        data = self.data[idx]
        label = self.labels[idx]
        
        # 检查数据维度并确保是float32类型
        if data.dtype != torch.float32:
            data = data.float()
            
        # 确保标签是long类型
        if label.dtype != torch.long:
            label = label.long()
            
        # 添加数据标准化
        # 对数据进行归一化处理，使其均值为0，标准差为1
        data = (data - data.mean()) / (data.std() + 1e-8)
            
        return data, label


def load_identify_data(save_datasets=True):
    """
    ○ 源域训练集：env0 和 env1 中 10 名已知用户的 CSI 数据
    ○ 目标域训练集：env2 中 10 名已知用户的少量数据（用于域适应）
    ○ 验证集：源域和目标域训练集的部分合法用户数据
    ○ 测试集：剩余的源域和目标域训练集的部分合法用户数据
    """

    # 加载源域数据（env0 和 env1 中 10 名已知用户的 CSI 数据）
    print("\n----------------------------------------------------\n")
    print("研究内容二---身份识别---加载数据...")
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
    print("\n----------------------------------------------------\n")

    # 数据集划分
    print("进行数据集划分...")

    # 1. 源域数据按比例划分：70%训练集，15%验证集，15%测试集
    src_indices = np.arange(len(source_data))
    src_train_indices, src_temp_indices = train_test_split(
        src_indices, test_size=0.3, random_state=42, stratify=source_labels.numpy()
    )
    src_val_indices, src_test_indices = train_test_split(
        src_temp_indices, test_size=0.5, random_state=42, stratify=source_labels[src_temp_indices].numpy()
    )

    src_train_data = source_data[src_train_indices]
    src_train_labels = source_labels[src_train_indices]
    src_val_data = source_data[src_val_indices]
    src_val_labels = source_labels[src_val_indices]
    src_test_data = source_data[src_test_indices]
    src_test_labels = source_labels[src_test_indices]

    print(f"源域训练集数据形状: {src_train_data.shape}")
    print(f"源域训练集标签形状: {src_train_labels.shape}")
    print(f"源域验证集数据形状: {src_val_data.shape}")
    print(f"源域验证集标签形状: {src_val_labels.shape}")
    print(f"源域测试集数据形状: {src_test_data.shape}")
    print(f"源域测试集标签形状: {src_test_labels.shape}")

    # 2. 目标域辅助训练集：从目标域已知用户数据中抽取少量数据用于域适应（30%）
    target_indices = np.arange(len(target_legal_data))
    target_aux_indices, target_remaining_indices = train_test_split(
        target_indices, test_size=0.7, random_state=42, stratify=target_legal_labels.numpy()
    )
    # 随机抽取30%的数据作为目标域辅助训练集
    target_aux_data = target_legal_data[target_aux_indices]
    target_aux_labels = target_legal_labels[target_aux_indices]

    # 目标域剩余数据按比例划分：50%验证集，50%测试集
    target_remaining_labels = target_legal_labels[target_remaining_indices]
    target_val_indices, target_test_indices = train_test_split(
        target_remaining_indices,
        test_size=0.5,
        random_state=42,
        stratify=target_remaining_labels.numpy()
    )

    target_val_data = target_legal_data[target_val_indices]
    target_val_labels = target_legal_labels[target_val_indices]
    target_test_data = target_legal_data[target_test_indices]
    target_test_labels = target_legal_labels[target_test_indices]

    print(f"目标域辅助训练集数据形状: {target_aux_data.shape}")
    print(f"目标域辅助训练集标签形状: {target_aux_labels.shape}")
    print(f"目标域验证集数据形状: {target_val_data.shape}")
    print(f"目标域验证集标签形状: {target_val_labels.shape}")
    print(f"目标域测试集数据形状: {target_test_data.shape}")
    print(f"目标域测试集标签形状: {target_test_labels.shape}")

    # 3. 身份识别数据集组合
    # 身份识别训练集：源域训练集 + 目标域辅助训练集
    identity_train_data = torch.cat([src_train_data, target_aux_data], dim=0)
    identity_train_labels = torch.cat([src_train_labels, target_aux_labels], dim=0)

    # 身份识别验证集：源域验证集 + 目标域验证集
    identity_val_data = torch.cat([src_val_data, target_val_data], dim=0)
    identity_val_labels = torch.cat([src_val_labels, target_val_labels], dim=0)

    # 身份识别测试集：源域测试集 + 目标域测试集
    identity_test_data = torch.cat([src_test_data, target_test_data], dim=0)
    identity_test_labels = torch.cat([src_test_labels, target_test_labels], dim=0)

    # 分别保存源域和目标域测试集，用于单独评估
    src_identity_test_data = src_test_data
    src_identity_test_labels = src_test_labels
    tgt_identity_test_data = target_test_data
    tgt_identity_test_labels = target_test_labels

    # 打乱数据集
    # 身份识别训练集打乱
    identity_train_indices = np.random.permutation(len(identity_train_data))
    identity_train_data = identity_train_data[identity_train_indices]
    identity_train_labels = identity_train_labels[identity_train_indices]

    # 身份识别验证集打乱
    identity_val_indices = np.random.permutation(len(identity_val_data))
    identity_val_data = identity_val_data[identity_val_indices]
    identity_val_labels = identity_val_labels[identity_val_indices]

    # 身份识别测试集打乱
    identity_test_indices = np.random.permutation(len(identity_test_data))
    identity_test_data = identity_test_data[identity_test_indices]
    identity_test_labels = identity_test_labels[identity_test_indices]

    print(f"\n-------------------------------------------------------")
    print(f"身份识别训练集数据形状: {identity_train_data.shape}")
    print(f"身份识别训练集标签形状: {identity_train_labels.shape}")
    print(f"身份识别验证集数据形状: {identity_val_data.shape}")
    print(f"身份识别验证集标签形状: {identity_val_labels.shape}")
    print(f"身份识别测试集数据形状: {identity_test_data.shape}")
    print(f"身份识别测试集标签形状: {identity_test_labels.shape}")
    print(f"-------------------------------------------------------\n")

    # 创建数据集对象
    identity_train_dataset = IntruderDataset(identity_train_data, identity_train_labels)
    target_aux_dataset = IntruderDataset(target_aux_data, target_aux_labels)
    identity_val_dataset = IntruderDataset(identity_val_data, identity_val_labels)
    identity_test_dataset = IntruderDataset(identity_test_data, identity_test_labels)

    # 分别创建源域和目标域测试数据集
    src_identity_test_dataset = IntruderDataset(src_identity_test_data, src_identity_test_labels)
    tgt_identity_test_dataset = IntruderDataset(tgt_identity_test_data, tgt_identity_test_labels)

    return {
        'identity_train': identity_train_dataset,      # 用于身份识别模型训练
        'target_aux': target_aux_dataset,              # 用于域适应
        'identity_validation': identity_val_dataset,   # 用于身份识别模型验证
        'identity_test': identity_test_dataset,        # 用于身份识别模型测试（合并）
        'src_identity_test': src_identity_test_dataset,  # 源域身份识别测试
        'tgt_identity_test': tgt_identity_test_dataset,  # 目标域身份识别测试
        'identity_train_raw': (identity_train_data, identity_train_labels),
        'target_aux_raw': (target_aux_data, target_aux_labels),
        'identity_validation_raw': (identity_val_data, identity_val_labels),
        'identity_test_raw': (identity_test_data, identity_test_labels),
        'src_identity_test_raw': (src_identity_test_data, src_identity_test_labels),
        'tgt_identity_test_raw': (tgt_identity_test_data, tgt_identity_test_labels)
    }


def create_data_loaders(datasets, batch_size=8):
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
            shuffle=(key != 'test' and not key.endswith('_test')),  # 测试集不打乱
            num_workers=0,  # Windows兼容性
            drop_last=(key == 'source_train' or key == 'target_aux')  # 避免不完整的batch
        )
        
    return data_loaders


if __name__ == "__main__":
    datasets = load_identify_data()
