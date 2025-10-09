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


if __name__ == "__main__":
    # 加载数据文件示例
    source_data = torch.load('../data/SourceData/source_data.pt')
    source_labels = torch.load('../data/SourceData/source_labels.pt')
    target_data = torch.load('../data/TargetData_hat/target_data.pt')
    target_labels = torch.load('../data/TargetData_hat/target_labels.pt')

    # 查看数据形状
    print(f"Source data shape: {source_data.shape}")
    print(f"Source labels shape: {source_labels.shape}")
    print(f"Target data shape: {target_data.shape}")
    print(f"Target labels shape: {target_labels.shape}")

    # 创建数据集
    source_dataset = CustomDataset(source_data, source_labels)
    target_dataset = CustomDataset(target_data, target_labels)

    # 创建数据加载器
    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)
