import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 从目标域数据中为每个标签选择最多10个样本
def select_samples_by_label(data, labels, samples_per_label=10):
    """
    为每个标签选择指定数量的样本
    
    Args:
        data: 输入数据，形状为(N, 56, 3, 6000)
        labels: 标签数据，形状为(N, 1)
        samples_per_label: 每个标签需要的样本数
    
    Returns:
        selected_data: 选择的数据
        selected_labels: 对应的标签
    """
    # 转换标签为一维数组以便处理
    labels_flat = labels.squeeze() if labels.dim() > 1 else labels
    
    # 获取唯一标签
    unique_labels = torch.unique(labels_flat)
    print(f"Unique labels in target domain: {unique_labels}")
    
    selected_data_list = []
    selected_labels_list = []
    
    # 为每个标签选择样本
    for label in unique_labels:
        # 找到该标签的所有索引
        label_indices = torch.where(labels_flat == label)[0]
        
        # 随机选择指定数量的样本（如果样本数不足则选择所有）
        num_samples = min(len(label_indices), samples_per_label)
        selected_indices = label_indices[torch.randperm(len(label_indices))[:num_samples]]
        
        # 添加到结果列表
        selected_data_list.append(data[selected_indices])
        selected_labels_list.append(labels[selected_indices])
        
        print(f"Label {label}: selected {num_samples} samples")
    
    # 合并所有选择的样本
    if selected_data_list:
        selected_data = torch.cat(selected_data_list, dim=0)
        selected_labels = torch.cat(selected_labels_list, dim=0)
    else:
        selected_data = torch.empty(0, *data.shape[1:])
        selected_labels = torch.empty(0, *labels.shape[1:])
    
    return selected_data, selected_labels


if __name__ == "__main__":
    # 加载数据文件
    source_data = torch.load('../data/SourceData/source_data.pt')
    source_labels = torch.load('../data/SourceData/source_labels.pt')
    target_data = torch.load('../data/TargetData/target_data.pt')
    target_labels = torch.load('../data/TargetData/target_labels.pt')

    # 查看数据形状
    print(f"Source data shape: {source_data.shape}")
    print(f"Source labels shape: {source_labels.shape}")
    print(f"Target data shape: {target_data.shape}")
    print(f"Target labels shape: {target_labels.shape}")

    # 创建源域数据集和数据加载器
    source_dataset = CustomDataset(source_data, source_labels)
    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)

    # 从目标域数据中选择每个标签10个样本
    selected_target_data, selected_target_labels = select_samples_by_label(
        target_data, target_labels, samples_per_label=10
    )

    # 查看数据形状
    print(f"Selected target data shape: {selected_target_data.shape}")
    print(f"Selected target labels shape: {selected_target_labels.shape}")

    # 创建目标域数据集和数据加载器
    target_dataset = CustomDataset(selected_target_data, selected_target_labels)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)
