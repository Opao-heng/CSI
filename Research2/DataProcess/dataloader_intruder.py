import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class IntruderDetectionDataset(Dataset):
    """入侵者检测数据集"""

    def __init__(self, data, labels, identity_labels=None):
        self.data = data
        self.labels = labels  # 二分类标签：0-合法用户，1-入侵者
        self.identity_labels = identity_labels  # 身份标签：0-9表示合法用户ID，-1表示入侵者

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 确保返回的数据格式正确
        data = self.data[idx]
        label = self.labels[idx]

        # 检查数据维度并确保是float32类型
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        elif data.dtype != torch.float32:
            data = data.float()

        # 确保标签是long类型
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        elif label.dtype != torch.long:
            label = label.long()
        
        # === 数据增强：对训练集添加微小噪声 ===
        # 只对训练集进行数据增强
        if torch.rand(1).item() < 0.5:  # 50%概率添加噪声
            noise = torch.randn_like(data) * 0.02  # 微小高斯噪声
            data = data + noise

        # 处理身份标签
        if self.identity_labels is not None:
            identity_label = self.identity_labels[idx]
            if not isinstance(identity_label, torch.Tensor):
                identity_label = torch.tensor(identity_label, dtype=torch.long)
            elif identity_label.dtype != torch.long:
                identity_label = identity_label.long()
            return data, label, identity_label
        else:
            return data, label


def load_intruder_data():
    """
    加载入侵者检测专用数据集 - 闭集学习策略
    训练集、验证集、测试集：都使用合法用户 + 真实入侵者数据
    标签定义：0-合法用户，1-入侵者
    """

    # 加载源域数据（env0 和 env1 中 10 名已知用户的 CSI 数据）
    print("\n=====================================================")
    print("研究内容二 - 入侵检测 - 开放集数据集加载")
    print("=====================================================")
    print("加载源域数据（合法用户）...")

    source_data = torch.load('C:\\Users\\USER\\Desktop\\liuheng\\Research2\\Data\\source_env0_env1_data.pt')
    source_labels = torch.load('C:\\Users\\USER\\Desktop\\liuheng\\Research2\\Data\\/source_env0_env1_labels.pt')
    print(f"源域合法用户数据形状: {source_data.shape}")
    print(f"源域合法用户标签形状: {source_labels.shape}")

    # 加载目标域用户数据（env2 中 10 名已知用户包含研究内容一 TFGAN 生成的 CSI 数据）
    print("加载目标域数据（合法用户）...")
    target_legal_data = torch.load('C:\\Users\\USER\\Desktop\\liuheng\\Research2\\Data\\target_env2_gan_data.pt')
    target_legal_labels = torch.load('C:\\Users\\USER\\Desktop\\liuheng\\Research2\\Data\\target_env2_gan_labels.pt')
    print(f"目标域合法用户数据形状: {target_legal_data.shape}")
    print(f"目标域合法用户标签形状: {target_legal_labels.shape}")

    # 加载入侵者数据（env2 中 3 名入侵者）- 用于训练、验证和测试
    print("加载真实入侵者数据（用于训练、验证和测试）...")
    intruder_data_list = []

    # 加载真实入侵者数据
    for i in range(3):
        intruder_id = 10 + i  # 入侵者ID从10开始
        intruder_file = f'C:\\Users\\USER\\Desktop\\liuheng\\RawData\\env2_Intruder\\id_{intruder_id}_env_2.npy'

        intruder_data = np.load(intruder_file)
        # 转换为张量并重塑
        intruder_tensor = torch.from_numpy(intruder_data).float()
        N = intruder_tensor.shape[3]
        intruder_reshaped = intruder_tensor.permute(3, 0, 1, 2).contiguous()

        intruder_data_list.append(intruder_reshaped)

    real_intruder_data = torch.cat(intruder_data_list, dim=0)
    # 真实入侵者标签为1
    real_intruder_labels = torch.full((real_intruder_data.size(0),), 1, dtype=torch.long)
    # 真实入侵者身份标签为-1
    real_intruder_identity_labels = torch.full((real_intruder_data.size(0),), -1, dtype=torch.long)

    print(f"真实入侵者数据形状: {real_intruder_data.shape}")
    print(f"真实入侵者标签形状: {real_intruder_labels.shape}")
    print("=====================================================")

    # ======== 闭集学习数据划分策略 ========
    print("\n进行数据集划分（闭集学习）...")
    print("策略：训练集、验证集、测试集都使用合法用户 + 真实入侵者数据")
    print("注意：直接使用真实入侵者数据，不再动态生成伪入侵者")
    
    # 先划分入侵者数据：20%训练集，20%验证集，60%测试集
    intruder_indices = np.arange(len(real_intruder_data))
    intruder_train_indices, intruder_temp_indices = train_test_split(
        intruder_indices, test_size=0.8, random_state=42
    )
    intruder_val_indices, intruder_test_indices = train_test_split(
        intruder_temp_indices, test_size=0.75, random_state=42
    )
    
    real_intruder_train_data = real_intruder_data[intruder_train_indices]
    real_intruder_train_labels = real_intruder_labels[intruder_train_indices]
    real_intruder_train_identity = real_intruder_identity_labels[intruder_train_indices]
    
    real_intruder_val_data = real_intruder_data[intruder_val_indices]
    real_intruder_val_labels = real_intruder_labels[intruder_val_indices]
    real_intruder_val_identity = real_intruder_identity_labels[intruder_val_indices]
    
    real_intruder_test_data = real_intruder_data[intruder_test_indices]
    real_intruder_test_labels = real_intruder_labels[intruder_test_indices]
    real_intruder_test_identity = real_intruder_identity_labels[intruder_test_indices]
    
    print(f"[入侵者数据] 训练集: {real_intruder_train_data.shape}, 验证集: {real_intruder_val_data.shape}, 测试集: {real_intruder_test_data.shape}")

    # 1. 源域数据按比例划分：20%训练集，20%验证集，60%测试集
    src_indices = np.arange(len(source_data))
    src_train_indices, src_temp_indices = train_test_split(
        src_indices, test_size=0.8, random_state=42, stratify=source_labels.numpy()
    )
    src_val_indices, src_test_indices = train_test_split(
        src_temp_indices, test_size=0.75, random_state=42, stratify=source_labels[src_temp_indices].numpy()
    )

    src_train_data = source_data[src_train_indices]
    # 源域训练集标签转换为0（合法用户）
    src_train_labels = torch.zeros(len(src_train_indices), dtype=torch.long)
    # 源域训练集身份标签保持原标签
    src_train_identity_labels = source_labels[src_train_indices]
    src_val_data = source_data[src_val_indices]
    # 源域验证集标签转换为0（合法用户）
    src_val_labels = torch.zeros(len(src_val_indices), dtype=torch.long)
    # 源域验证集身份标签保持原标签
    src_val_identity_labels = source_labels[src_val_indices]
    src_test_data = source_data[src_test_indices]
    # 源域测试集标签转换为0（合法用户）
    src_test_labels = torch.zeros(len(src_test_indices), dtype=torch.long)
    # 源域测试集身份标签保持原标签
    src_test_identity_labels = source_labels[src_test_indices]

    print(f"[源域] 训练集: {src_train_data.shape}, 验证集: {src_val_data.shape}, 测试集: {src_test_data.shape}")

    # 2. 目标域训练集：从目标域已知用户数据中抽取少量数据用于域适应（30%）
    target_indices = np.arange(len(target_legal_data))
    target_aux_indices, target_remaining_indices = train_test_split(
        target_indices, test_size=0.7, random_state=42, stratify=target_legal_labels.numpy()
    )
    # 随机抽取30%的数据作为目标域辅助训练集
    target_aux_data = target_legal_data[target_aux_indices]
    # 目标域辅助训练集标签转换为0（合法用户）
    target_aux_labels = torch.zeros(len(target_aux_indices), dtype=torch.long)
    # 目标域辅助训练集身份标签保持原标签
    target_aux_identity_labels = target_legal_labels[target_aux_indices]

    # 目标域剩余数据按比例划分：50%验证集，50%测试集
    target_remaining_labels = target_legal_labels[target_remaining_indices]
    target_val_indices, target_test_indices = train_test_split(
        np.arange(len(target_remaining_indices)),
        test_size=0.5,
        random_state=42,
        stratify=target_remaining_labels.numpy()
    )

    target_val_data = target_legal_data[target_remaining_indices][target_val_indices]
    # 目标域验证集标签转换为0（合法用户）
    target_val_labels = torch.zeros(len(target_val_indices), dtype=torch.long)
    # 目标域验证集身份标签保持原标签
    target_val_identity_labels = target_remaining_labels[target_val_indices]
    target_test_data = target_legal_data[target_remaining_indices][target_test_indices]
    # 目标域测试集标签转换为0（合法用户）
    target_test_labels = torch.zeros(len(target_test_indices), dtype=torch.long)
    # 目标域测试集身份标签保持原标签
    target_test_identity_labels = target_remaining_labels[target_test_indices]

    print(
        f"[目标域] 训练集: {target_aux_data.shape}, 验证集: {target_val_data.shape}, 测试集: {target_test_data.shape}")

    # 3. 构建训练集和验证集（合法用户 + 真实入侵者）
    # 合并合法用户数据
    legal_train_data = torch.cat([src_train_data, target_aux_data], dim=0)
    legal_train_labels = torch.cat([src_train_labels, target_aux_labels], dim=0)
    legal_train_identity_labels = torch.cat([src_train_identity_labels, target_aux_identity_labels], dim=0)
    
    legal_val_data = torch.cat([src_val_data, target_val_data], dim=0)
    legal_val_labels = torch.cat([src_val_labels, target_val_labels], dim=0)
    legal_val_identity_labels = torch.cat([src_val_identity_labels, target_val_identity_labels], dim=0)
    
    # === 对合法用户进行下采样以平衡类别比例 ===
    # 目标：让合法用户:入侵者 = 1.5:1 (更平衡的比例)
    train_intruder_count = len(real_intruder_train_data)  # 61个入侵者
    val_intruder_count = len(real_intruder_val_data)      # 62个入侵者
    
    # 训练集：保留1.5倍入侵者数量的合法用户（61*1.5≈92）
    target_legal_train_count = int(train_intruder_count * 1.5)
    if len(legal_train_data) > target_legal_train_count:
        # 随机采样，保持身份分布
        sample_indices = np.random.choice(len(legal_train_data), target_legal_train_count, replace=False)
        legal_train_data = legal_train_data[sample_indices]
        legal_train_labels = legal_train_labels[sample_indices]
        legal_train_identity_labels = legal_train_identity_labels[sample_indices]
        print(f"[训练集采样] 合法用户从 {len(sample_indices)} 减少到 {target_legal_train_count}")
    
    # 验证集：保留1.5倍入侵者数量的合法用户（62*1.5≈93）
    target_legal_val_count = int(val_intruder_count * 1.5)
    if len(legal_val_data) > target_legal_val_count:
        sample_indices = np.random.choice(len(legal_val_data), target_legal_val_count, replace=False)
        legal_val_data = legal_val_data[sample_indices]
        legal_val_labels = legal_val_labels[sample_indices]
        legal_val_identity_labels = legal_val_identity_labels[sample_indices]
        print(f"[验证集采样] 合法用户从 {len(sample_indices)} 减少到 {target_legal_val_count}")
    
    # 合并合法用户和真实入侵者
    intruder_train_data = torch.cat([legal_train_data, real_intruder_train_data], dim=0)
    intruder_train_labels = torch.cat([legal_train_labels, real_intruder_train_labels], dim=0)
    intruder_train_identity_labels = torch.cat([legal_train_identity_labels, real_intruder_train_identity], dim=0)

    intruder_val_data = torch.cat([legal_val_data, real_intruder_val_data], dim=0)
    intruder_val_labels = torch.cat([legal_val_labels, real_intruder_val_labels], dim=0)
    intruder_val_identity_labels = torch.cat([legal_val_identity_labels, real_intruder_val_identity], dim=0)

    # 打乱训练集和验证集数据
    train_indices = np.random.permutation(len(intruder_train_data))
    intruder_train_data = intruder_train_data[train_indices]
    intruder_train_labels = intruder_train_labels[train_indices]
    intruder_train_identity_labels = intruder_train_identity_labels[train_indices]

    val_indices = np.random.permutation(len(intruder_val_data))
    intruder_val_data = intruder_val_data[val_indices]
    intruder_val_labels = intruder_val_labels[val_indices]
    intruder_val_identity_labels = intruder_val_identity_labels[val_indices]

    train_legal_count = len(legal_train_data)
    train_intruder_count = len(real_intruder_train_data)
    val_legal_count = len(legal_val_data)
    val_intruder_count = len(real_intruder_val_data)
    
    print(f"\n[训练集] 数据: {intruder_train_data.shape}")
    print(f"  - 合法用户: {train_legal_count} ({train_legal_count / len(intruder_train_data) * 100:.1f}%)")
    print(f"  - 真实入侵者: {train_intruder_count} ({train_intruder_count / len(intruder_train_data) * 100:.1f}%)")
    print(f"[验证集] 数据: {intruder_val_data.shape}")
    print(f"  - 合法用户: {val_legal_count} ({val_legal_count / len(intruder_val_data) * 100:.1f}%)")
    print(f"  - 真实入侵者: {val_intruder_count} ({val_intruder_count / len(intruder_val_data) * 100:.1f}%)")

    # 4. 构建测试集：目标域测试集数据（合法用户）+ 真实入侵者数据
    intruder_test_data = torch.cat([target_test_data, real_intruder_test_data], dim=0)
    intruder_test_labels = torch.cat([target_test_labels, real_intruder_test_labels], dim=0)
    intruder_test_identity_labels = torch.cat([target_test_identity_labels, real_intruder_test_identity], dim=0)

    # 打乱测试集数据
    test_indices = np.random.permutation(len(intruder_test_data))
    intruder_test_data = intruder_test_data[test_indices]
    intruder_test_labels = intruder_test_labels[test_indices]
    intruder_test_identity_labels = intruder_test_identity_labels[test_indices]

    legal_test_count = len(target_test_data)
    intruder_test_count = len(real_intruder_test_data)
    print(f"[测试集] 数据: {intruder_test_data.shape}")
    print(f"  - 合法用户: {legal_test_count} ({legal_test_count / len(intruder_test_data) * 100:.1f}%)")
    print(f"  - 真实入侵者: {intruder_test_count} ({intruder_test_count / len(intruder_test_data) * 100:.1f}%)")
    print("=====================================================")

    # 5. 创建数据集对象
    intruder_train_dataset = IntruderDetectionDataset(intruder_train_data, intruder_train_labels,
                                                      intruder_train_identity_labels)
    intruder_val_dataset = IntruderDetectionDataset(intruder_val_data, intruder_val_labels,
                                                    intruder_val_identity_labels)
    intruder_test_dataset = IntruderDetectionDataset(intruder_test_data, intruder_test_labels,
                                                     intruder_test_identity_labels)

    print(f"\n数据加载完成！")

    return {
        'intruder_train': intruder_train_dataset,  # 用于入侵者检测模型训练（包含合法用户+真实入侵者）
        'intruder_validation': intruder_val_dataset,  # 用于入侵者检测模型验证（包含合法用户+真实入侵者）
        'intruder_test': intruder_test_dataset,  # 用于入侵者检测模型测试（包含合法用户+真实入侵者）
        'intruder_train_raw': (intruder_train_data, intruder_train_labels, intruder_train_identity_labels),
        'intruder_validation_raw': (intruder_val_data, intruder_val_labels, intruder_val_identity_labels),
        'intruder_test_raw': (intruder_test_data, intruder_test_labels, intruder_test_identity_labels)
    }


def create_intruder_data_loaders(datasets, batch_size=32):
    """
    创建入侵者检测专用数据加载器
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
            drop_last=False,  # 不丢弃不完整的batch
            collate_fn=custom_collate_fn  # 使用自定义的collate函数处理不规则数据
        )

    return data_loaders


def custom_collate_fn(batch):
    """
    自定义的collate函数，处理可能不规则的数据
    """
    # 分离数据和标签
    if len(batch[0]) == 3:  # 包含身份标签
        data_list, labels_list, identity_labels_list = zip(*batch)
    else:  # 不包含身份标签
        data_list, labels_list = zip(*batch)
        identity_labels_list = None

    # 转换为张量
    try:
        # 尝试直接堆叠（如果所有数据形状一致）
        data = torch.stack(data_list, dim=0)
        labels = torch.stack(labels_list, dim=0) if isinstance(labels_list[0], torch.Tensor) else torch.tensor(
            labels_list)
        if identity_labels_list is not None:
            identity_labels = torch.stack(identity_labels_list, dim=0) if isinstance(identity_labels_list[0],
                                                                                     torch.Tensor) else torch.tensor(
                identity_labels_list)
        else:
            identity_labels = None
    except RuntimeError:
        # 如果形状不一致，返回列表形式
        data = list(data_list)
        labels = list(labels_list)
        identity_labels = list(identity_labels_list) if identity_labels_list is not None else None

    if identity_labels is not None:
        return data, labels, identity_labels
    else:
        return data, labels


if __name__ == "__main__":
    datasets = load_intruder_data()
    data_loaders = create_intruder_data_loaders(datasets, batch_size=32)