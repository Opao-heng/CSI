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
    加载入侵者检测专用数据集
    重新构建训练集，加入模拟的入侵者数据
    标签定义：0-已知用户，1-入侵者
    """

    # 加载源域数据（env0 和 env1 中 10 名已知用户的 CSI 数据）
    print("\n----------------------------------------------------\n")
    print("研究内容二---入侵检测---加载数据...")
    print("加载源域数据...")
    source_data = torch.load('C:\\Users\\USER\\Desktop\\liuheng\\Research2\\Data\\source_env0_env1_data.pt')
    source_labels = torch.load('C:\\Users\\USER\\Desktop\\liuheng\\Research2\\Data\\source_env0_env1_labels.pt')

    print(f"源域数据形状: {source_data.shape}")
    print(f"源域标签形状: {source_labels.shape}")

    # 加载目标域已知用户数据（env2 中 10 名已知用户）
    print("加载目标域已知用户数据...")
    target_legal_data = torch.load('C:\\Users\\USER\\Desktop\\liuheng\\Research2\\Data\\target_env2_gan_data.pt')
    target_legal_labels = torch.load('C:\\Users\\USER\\Desktop\\liuheng\\Research2\\Data\\target_env2_gan_labels.pt')

    print(f"目标域已知用户数据形状: {target_legal_data.shape}")
    print(f"目标域已知用户标签形状: {target_legal_labels.shape}")

    # 加载入侵者数据（env2 中 3 名入侵者），入侵者数据在 env2_intruder 目录中
    print("加载真实入侵者数据...")
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

    print(f"源域训练集数据形状: {src_train_data.shape}")
    print(f"源域训练集标签形状: {src_train_labels.shape}")
    print(f"源域验证集数据形状: {src_val_data.shape}")
    print(f"源域验证集标签形状: {src_val_labels.shape}")
    print(f"源域测试集数据形状: {src_test_data.shape}")
    print(f"源域测试集标签形状: {src_test_labels.shape}")

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

    print(f"目标域训练集数据形状: {target_aux_data.shape}")
    print(f"目标域训练集标签形状: {target_aux_labels.shape}")
    print(f"目标域验证集数据形状: {target_val_data.shape}")
    print(f"目标域验证集标签形状: {target_val_labels.shape}")
    print(f"目标域测试集数据形状: {target_test_data.shape}")
    print(f"目标域测试集标签形状: {target_test_labels.shape}")

    # 3. 构建合法用户数据集（用于生成模拟入侵者数据）
    # 合法用户数据：源域训练集 + 目标域辅助训练集 + 源域验证集 + 目标域验证集
    legal_user_data = torch.cat([src_train_data, target_aux_data, src_val_data, target_val_data], dim=0)
    legal_user_labels = torch.cat([src_train_labels, target_aux_labels, src_val_labels, target_val_labels], dim=0)
    legal_user_identity_labels = torch.cat([src_train_identity_labels, target_aux_identity_labels,
                                            src_val_identity_labels, target_val_identity_labels], dim=0)

    # 4. 创建模拟入侵者数据（多种方法），数量为合法用户数据的50%
    num_legal_samples = len(legal_user_data)
    total_simulated_intruders = max(1, num_legal_samples // 2)  # 模拟入侵者数据数量为合法用户数据的50%

    simulated_intruder_list = []

    # 方法1: 在合法用户数据基础上添加噪声（25%）
    num_method1 = max(1, total_simulated_intruders // 4)
    base_indices_1 = np.random.choice(len(legal_user_data), size=num_method1, replace=True)
    simulated_data_1 = legal_user_data[base_indices_1].clone()
    noise_1 = torch.randn_like(simulated_data_1) * 0.2
    simulated_data_1 = simulated_data_1 + noise_1
    simulated_intruder_list.append(simulated_data_1)

    # 方法2: 使用幅度缩放（25%）
    num_method2 = max(1, total_simulated_intruders // 4)
    base_indices_2 = np.random.choice(len(legal_user_data), size=num_method2, replace=True)
    simulated_data_2 = legal_user_data[base_indices_2].clone()
    scale_factors = torch.FloatTensor(num_method2, 1, 1, 1).uniform_(0.5, 2.0)
    simulated_data_2 = simulated_data_2 * scale_factors
    simulated_intruder_list.append(simulated_data_2)

    # 方法3: 时间序列偏移（25%）
    num_method3 = max(1, total_simulated_intruders // 4)
    base_indices_3 = np.random.choice(len(legal_user_data), size=num_method3, replace=True)
    simulated_data_3 = legal_user_data[base_indices_3].clone()
    # 对时间维度(最后一个维度)进行循环移位
    shifts = np.random.randint(0, 6000, size=num_method3)
    for i in range(num_method3):
        shift = shifts[i]
        simulated_data_3[i] = torch.roll(simulated_data_3[i], shifts=shift, dims=2)
    simulated_intruder_list.append(simulated_data_3)

    # 方法4: 组合方法 - 噪声+缩放（25%）
    num_method4 = max(1, total_simulated_intruders - num_method1 - num_method2 - num_method3)
    base_indices_4 = np.random.choice(len(legal_user_data), size=num_method4, replace=True)
    simulated_data_4 = legal_user_data[base_indices_4].clone()
    # 添加噪声
    noise_4 = torch.randn_like(simulated_data_4) * 0.1
    simulated_data_4 = simulated_data_4 + noise_4
    # 幅度缩放
    scale_factors_4 = torch.FloatTensor(num_method4, 1, 1, 1).uniform_(0.8, 1.2)
    simulated_data_4 = simulated_data_4 * scale_factors_4
    simulated_intruder_list.append(simulated_data_4)

    # 合并所有模拟入侵者数据
    simulated_intruder_data = torch.cat(simulated_intruder_list, dim=0)
    # 模拟入侵者标签为1
    simulated_intruder_labels = torch.full((total_simulated_intruders,), 1, dtype=torch.long)
    # 模拟入侵者身份标签为-1
    simulated_intruder_identity_labels = torch.full((total_simulated_intruders,), -1, dtype=torch.long)

    print(f"\n-------------------------------------------------------")
    print(f"总共生成模拟入侵者数据形状: {simulated_intruder_data.shape}")
    print(f"-------------------------------------------------------\n")

    # 5. 将合法用户数据和模拟入侵者数据分别划分为训练集和验证集
    # 合法用户数据划分
    legal_train_data = torch.cat([src_train_data, target_aux_data], dim=0)
    legal_train_labels = torch.cat([src_train_labels, target_aux_labels], dim=0)
    legal_train_identity_labels = torch.cat([src_train_identity_labels, target_aux_identity_labels], dim=0)
    legal_val_data = torch.cat([src_val_data, target_val_data], dim=0)
    legal_val_labels = torch.cat([src_val_labels, target_val_labels], dim=0)
    legal_val_identity_labels = torch.cat([src_val_identity_labels, target_val_identity_labels], dim=0)

    # 模拟入侵者数据划分（与合法用户数据采用相同的划分比例）
    train_ratio = len(legal_train_data) / len(legal_user_data)

    # 划分模拟入侵者数据
    sim_train_size = int(len(simulated_intruder_data) * train_ratio)
    sim_train_indices = np.random.choice(len(simulated_intruder_data), size=sim_train_size, replace=False)
    sim_val_indices = np.setdiff1d(np.arange(len(simulated_intruder_data)), sim_train_indices)

    sim_train_data = simulated_intruder_data[sim_train_indices]
    sim_train_labels = simulated_intruder_labels[sim_train_indices]
    sim_train_identity_labels = simulated_intruder_identity_labels[sim_train_indices]
    sim_val_data = simulated_intruder_data[sim_val_indices]
    sim_val_labels = simulated_intruder_labels[sim_val_indices]
    sim_val_identity_labels = simulated_intruder_identity_labels[sim_val_indices]

    # 入侵者检测训练集：合法用户训练数据 + 模拟入侵者训练数据（合法用户:模拟入侵者 = 10:1）
    intruder_train_data = torch.cat([legal_train_data, sim_train_data], dim=0)
    intruder_train_labels = torch.cat([legal_train_labels, sim_train_labels], dim=0)
    intruder_train_identity_labels = torch.cat([legal_train_identity_labels, sim_train_identity_labels], dim=0)

    # 打乱入侵者检测训练集数据
    intruder_train_indices = np.random.permutation(len(intruder_train_data))
    intruder_train_data = intruder_train_data[intruder_train_indices]
    intruder_train_labels = intruder_train_labels[intruder_train_indices]
    intruder_train_identity_labels = intruder_train_identity_labels[intruder_train_indices]

    print(f"入侵者检测训练集数据形状: {intruder_train_data.shape}")
    print(f"入侵者检测训练集标签形状: {intruder_train_labels.shape}")
    print(f"  - 合法用户训练样本数: {len(legal_train_data)}")
    print(f"  - 模拟入侵者训练样本数: {len(sim_train_data)}")

    # 入侵者检测验证集：合法用户验证数据 + 模拟入侵者验证数据（合法用户:模拟入侵者 = 10:1）
    intruder_val_data = torch.cat([legal_val_data, sim_val_data], dim=0)
    intruder_val_labels = torch.cat([legal_val_labels, sim_val_labels], dim=0)
    intruder_val_identity_labels = torch.cat([legal_val_identity_labels, sim_val_identity_labels], dim=0)

    # 打乱入侵者检测验证集数据
    intruder_val_indices = np.random.permutation(len(intruder_val_data))
    intruder_val_data = intruder_val_data[intruder_val_indices]
    intruder_val_labels = intruder_val_labels[intruder_val_indices]
    intruder_val_identity_labels = intruder_val_identity_labels[intruder_val_indices]

    print(f"入侵者检测验证集数据形状: {intruder_val_data.shape}")
    print(f"入侵者检测验证集标签形状: {intruder_val_labels.shape}")
    print(f"  - 合法用户验证样本数: {len(legal_val_data)}")
    print(f"  - 模拟入侵者验证样本数: {len(sim_val_data)}")

    # 6. 入侵者检测测试集：目标域测试集数据（合法用户）+ 真实入侵者数据
    # 入侵者检测测试集只使用目标域测试集数据（合法用户）+ 真实入侵者数据
    intruder_test_data = torch.cat([target_test_data, real_intruder_data], dim=0)
    intruder_test_labels = torch.cat([target_test_labels, real_intruder_labels], dim=0)
    intruder_test_identity_labels = torch.cat([target_test_identity_labels, real_intruder_identity_labels], dim=0)

    # 打乱入侵者检测测试集数据
    intruder_test_indices = np.random.permutation(len(intruder_test_data))
    intruder_test_data = intruder_test_data[intruder_test_indices]
    intruder_test_labels = intruder_test_labels[intruder_test_indices]
    intruder_test_identity_labels = intruder_test_identity_labels[intruder_test_indices]

    print(f"入侵者检测测试集数据形状: {intruder_test_data.shape}")
    print(f"入侵者检测测试集标签形状: {intruder_test_labels.shape}")
    print(f"  - 合法用户测试样本数: {len(target_test_data)}")
    print(f"  - 真实入侵者测试样本数: {len(real_intruder_data)}")
    print(f"-------------------------------------------------------")

    # 创建数据集对象
    intruder_train_dataset = IntruderDetectionDataset(intruder_train_data, intruder_train_labels,
                                                      intruder_train_identity_labels)
    intruder_val_dataset = IntruderDetectionDataset(intruder_val_data, intruder_val_labels,
                                                    intruder_val_identity_labels)
    intruder_test_dataset = IntruderDetectionDataset(intruder_test_data, intruder_test_labels,
                                                     intruder_test_identity_labels)

    return {
        'intruder_train': intruder_train_dataset,  # 用于入侵者检测模型训练
        'intruder_validation': intruder_val_dataset,  # 用于入侵者检测模型验证
        'intruder_test': intruder_test_dataset,  # 用于入侵者检测模型测试
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