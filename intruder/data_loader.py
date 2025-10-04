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


def load_and_split_data():
    """
    加载源域和目标域数据，并进行数据集划分
    
    数据集划分：
    ○ 源域训练集：env0 和 env1 中 10 名已知用户的 CSI 数据
    ○ 目标域辅助集：env2 中 10 名已知用户的少量数据（用于域适应）
    ○ 身份识别验证集：源域和目标域辅助集的部分合法用户数据（用于身份识别模型验证）
    ○ 入侵者检测验证集：身份识别验证集数据 + 模拟入侵者数据（用于入侵者检测器训练）
    ○ 测试集：env2 中 10 名已知用户（300 样本）和 3 名入侵者（90 样本）的混合数据
    """
    
    try:
        # 加载源域数据（env0 和 env1 中 10 名已知用户的 CSI 数据）
        print("加载源域数据...")
        source_data = torch.load('../data/SourceData/source_data.pt')
        source_labels = torch.load('../data/SourceData/source_labels.pt')
        
        # 确保数据类型正确
        if source_data.dtype != torch.float32:
            source_data = source_data.float()
        if source_labels.dtype != torch.long:
            source_labels = source_labels.long()
        
        print(f"源域数据形状: {source_data.shape}")
        print(f"源域标签形状: {source_labels.shape}")
        
        # 加载目标域已知用户数据（env2 中 10 名已知用户）
        print("加载目标域已知用户数据...")
        target_legal_data = torch.load('../data/TargetData/target_data.pt')
        target_legal_labels = torch.load('../data/TargetData/target_labels.pt')
        
        # 确保数据类型正确
        if target_legal_data.dtype != torch.float32:
            target_legal_data = target_legal_data.float()
        if target_legal_labels.dtype != torch.long:
            target_legal_labels = target_legal_labels.long()
        
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
            intruder_tensor = torch.from_numpy(intruder_data).float()
            N = intruder_tensor.shape[3]
            intruder_reshaped = intruder_tensor.permute(3, 0, 1, 2).contiguous()
            intruder_labels = torch.full((N,), -1, dtype=torch.long)  # 入侵者标签为-1

            intruder_data_list.append(intruder_reshaped)
            intruder_labels_list.append(intruder_labels)

        intruder_data = torch.cat(intruder_data_list, dim=0)
        intruder_labels = torch.cat(intruder_labels_list, dim=0)
        
        # 确保数据类型正确
        if intruder_data.dtype != torch.float32:
            intruder_data = intruder_data.float()
        if intruder_labels.dtype != torch.long:
            intruder_labels = intruder_labels.long()
            
        print(f"入侵者数据形状: {intruder_data.shape}")
        print(f"入侵者标签形状: {intruder_labels.shape}")
        
        # 数据集划分
        print("\n进行数据集划分...")
        
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

        print(f"-------------------------------------------------------")
        print(f"身份识别训练集数据形状: {identity_train_data.shape}")
        print(f"身份识别训练集标签形状: {identity_train_labels.shape}")
        print(f"身份识别验证集数据形状: {identity_val_data.shape}")
        print(f"身份识别验证集标签形状: {identity_val_labels.shape}")
        print(f"身份识别测试集数据形状: {identity_test_data.shape}")
        print(f"身份识别测试集标签形状: {identity_test_labels.shape}")
        print(f"-------------------------------------------------------")
        
        # 4. 入侵者检测验证集：身份识别验证集数据 + 模拟入侵者数据
        # 创建模拟入侵者数据（多种方法）
        num_simulated_intruders_per_method = 20
        total_simulated_intruders = 80

        simulated_intruder_list = []
        simulated_labels_list = []

        # 方法1: 在合法用户数据基础上添加噪声
        base_indices_1 = np.random.choice(len(identity_val_data), size=num_simulated_intruders_per_method, replace=True)
        simulated_data_1 = identity_val_data[base_indices_1].clone()
        noise_1 = torch.randn_like(simulated_data_1) * 0.2
        simulated_data_1 = simulated_data_1 + noise_1
        simulated_intruder_list.append(simulated_data_1)

        # 方法2: 使用幅度缩放
        base_indices_2 = np.random.choice(len(identity_val_data), size=num_simulated_intruders_per_method, replace=True)
        simulated_data_2 = identity_val_data[base_indices_2].clone()
        scale_factors = torch.FloatTensor(num_simulated_intruders_per_method, 1, 1, 1).uniform_(0.5, 2.0)
        simulated_data_2 = simulated_data_2 * scale_factors
        simulated_intruder_list.append(simulated_data_2)

        # 方法3: 时间序列偏移
        base_indices_3 = np.random.choice(len(identity_val_data), size=num_simulated_intruders_per_method, replace=True)
        simulated_data_3 = identity_val_data[base_indices_3].clone()
        # 对时间维度(最后一个维度)进行循环移位
        shifts = np.random.randint(0, 6000, size=num_simulated_intruders_per_method)
        for i in range(num_simulated_intruders_per_method):
            shift = shifts[i]
            simulated_data_3[i] = torch.roll(simulated_data_3[i], shifts=shift, dims=2)
        simulated_intruder_list.append(simulated_data_3)

        # 方法4: 组合方法 - 噪声+缩放
        base_indices_4 = np.random.choice(len(identity_val_data), size=num_simulated_intruders_per_method, replace=True)
        simulated_data_4 = identity_val_data[base_indices_4].clone()
        # 添加噪声
        noise_4 = torch.randn_like(simulated_data_4) * 0.1
        simulated_data_4 = simulated_data_4 + noise_4
        # 幅度缩放
        scale_factors_4 = torch.FloatTensor(num_simulated_intruders_per_method, 1, 1, 1).uniform_(0.8, 1.2)
        simulated_data_4 = simulated_data_4 * scale_factors_4
        simulated_intruder_list.append(simulated_data_4)

        # 合并所有模拟入侵者数据
        simulated_intruder_data = torch.cat(simulated_intruder_list, dim=0)
        simulated_intruder_labels = torch.full((total_simulated_intruders,), -1, dtype=torch.long)

        # 合并身份识别验证集和模拟入侵者数据形成入侵者检测验证集
        intruder_val_data = torch.cat([identity_val_data, simulated_intruder_data], dim=0)
        intruder_val_labels = torch.cat([identity_val_labels, simulated_intruder_labels], dim=0)
        
        # 打乱入侵者检测验证集数据
        intruder_val_indices = np.random.permutation(len(intruder_val_data))
        intruder_val_data = intruder_val_data[intruder_val_indices]
        intruder_val_labels = intruder_val_labels[intruder_val_indices]

        print(f"入侵者检测验证集数据形状: {intruder_val_data.shape}")
        print(f"入侵者检测验证集标签形状: {intruder_val_labels.shape}")
        
        # 5. 入侵者检测测试集：身份识别测试集 + 真实入侵者数据
        intruder_test_data = torch.cat([identity_test_data, intruder_data], dim=0)
        intruder_test_labels = torch.cat([identity_test_labels, intruder_labels], dim=0)
        
        # 打乱入侵者检测测试集数据
        intruder_test_indices = np.random.permutation(len(intruder_test_data))
        intruder_test_data = intruder_test_data[intruder_test_indices]
        intruder_test_labels = intruder_test_labels[intruder_test_indices]

        print(f"入侵者检测测试集数据形状: {intruder_test_data.shape}")
        print(f"入侵者检测测试集标签形状: {intruder_test_labels.shape}")
        print(f"-------------------------------------------------------")
        
        # 创建数据集对象
        identity_train_dataset = IntruderDataset(identity_train_data, identity_train_labels)
        target_aux_dataset = IntruderDataset(target_aux_data, target_aux_labels)
        identity_val_dataset = IntruderDataset(identity_val_data, identity_val_labels)
        intruder_val_dataset = IntruderDataset(intruder_val_data, intruder_val_labels)
        identity_test_dataset = IntruderDataset(identity_test_data, identity_test_labels)
        intruder_test_dataset = IntruderDataset(intruder_test_data, intruder_test_labels)
        
        return {
            'identity_train': identity_train_dataset,      # 用于身份识别模型训练
            'target_aux': target_aux_dataset,              # 用于域适应
            'identity_validation': identity_val_dataset,   # 用于身份识别模型验证
            'intruder_validation': intruder_val_dataset,   # 用于入侵者检测器训练
            'identity_test': identity_test_dataset,        # 用于身份识别模型测试
            'intruder_test': intruder_test_dataset,        # 用于入侵者检测器测试
            'identity_train_raw': (identity_train_data, identity_train_labels),
            'target_aux_raw': (target_aux_data, target_aux_labels),
            'identity_validation_raw': (identity_val_data, identity_val_labels),
            'intruder_validation_raw': (intruder_val_data, intruder_val_labels),
            'identity_test_raw': (identity_test_data, identity_test_labels),
            'intruder_test_raw': (intruder_test_data, intruder_test_labels)
        }
    except Exception as e:
        print(f"数据加载过程中出现错误: {e}")
        raise e


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
            shuffle=(key != 'test'),  # 测试集不打乱
            num_workers=0,  # Windows兼容性
            drop_last=(key == 'source_train' or key == 'target_aux')  # 避免不完整的batch
        )
        
    return data_loaders


if __name__ == "__main__":
    # 测试数据加载和划分
    try:
        datasets = load_and_split_data()
        data_loaders = create_data_loaders(datasets, batch_size=4)  # 使用更小的batch size
        
        print("\n数据加载器创建完成:")
        for key, loader in data_loaders.items():
            if not key.endswith('_raw'):
                print(f"{key} 数据加载器: {len(loader)} 个批次")
            
        # 显示一些样本信息
        print("\n样本信息:")
        for key, (data, labels) in datasets.items():
            if key.endswith('_raw'):
                print(f"{key}: 数据形状 {data.shape}, 标签形状 {labels.shape}")
                # 显示标签分布
                unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
                print(f"  标签分布: {dict(zip(unique_labels, counts))}")
                
        # 测试数据加载
        print("\n测试数据加载:")
        for key, loader in data_loaders.items():
            if not key.endswith('_raw') and len(loader) > 0:
                try:
                    data, labels = next(iter(loader))
                    print(f"{key}: 数据形状 {data.shape}, 标签形状 {labels.shape}")
                    print(f"  数据类型: {data.dtype}, 标签类型: {labels.dtype}")
                    print(f"  数据范围: [{data.min():.4f}, {data.max():.4f}]")
                    print(f"  标签范围: [{labels.min()}, {labels.max()}]")
                    break
                except Exception as e:
                    print(f"  加载测试样本时出错: {e}")
                
    except Exception as e:
        print(f"数据加载过程中出现错误: {e}")
        import traceback
        traceback.print_exc()