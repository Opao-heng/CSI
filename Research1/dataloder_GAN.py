import torch
from torch.utils.data import Dataset, DataLoader


"""
自定义数据集类，用于载入数据和标签
参数:
  data - 整个数据集的张量
  labels - 整个数据集的标签张量
"""
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        # 步骤1: 存储数据和标签
        self.data = data
        self.labels = labels

    def __len__(self):
        # 返回数据集整整余数
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引idx返回数据和对应标签
        return self.data[idx], self.labels[idx]


"""
为目标域数据每个标签选择固定数量的样本，用于生成GAN数据
参数:
  data - 整个数据的张量，形状为(N, 56, 3, 6000)
  labels - 整个数据的标签，形状为(N,) 或 (N, 1)
  samples_per_label - 每个标签需要选择的样本数，默认10
返回: tuple - (selected_data, selected_labels)
  selected_data: 选择后的数据张量，形状为(M, 56, 3, 6000)，其中M是总选择样本数
  selected_labels: 选择后的标签张量，形状为(M,)
"""
def select_samples_by_label(data, labels, samples_per_label=10):
    # 步骤1: 仆平整化标签数组，确保是一维数组
    labels_flat = labels.squeeze() if labels.dim() > 1 else labels
    
    # 步骤2: 获取整个数据集中的不同标签
    unique_labels = torch.unique(labels_flat)
    print(f"Unique labels in target domain: {unique_labels}")
    
    selected_data_list = []
    selected_labels_list = []
    
    # 步骤3: 为每个标签选择样本
    for label in unique_labels:
        # 找到该标签的所有样本的索引
        label_indices = torch.where(labels_flat == label)[0]
        
        # 随机选择指定数量的样本（如果样本数不足则选择所有）
        num_samples = min(len(label_indices), samples_per_label)
        selected_indices = label_indices[torch.randperm(len(label_indices))[:num_samples]]
        
        # 添加选择后的样本到结果列表
        selected_data_list.append(data[selected_indices])
        selected_labels_list.append(labels[selected_indices])
        
        print(f"Label {label}: selected {num_samples} samples")
    
    # 步骤4: 合并所有选择的样本
    if selected_data_list:
        selected_data = torch.cat(selected_data_list, dim=0)
        selected_labels = torch.cat(selected_labels_list, dim=0)
    else:
        # 如果不有样本被选择，创建空张量
        selected_data = torch.empty(0, *data.shape[1:])
        selected_labels = torch.empty(0, *labels.shape[1:])
    
    return selected_data, selected_labels

