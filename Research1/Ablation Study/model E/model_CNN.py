# model_CNN.py - Model E: 简单CNN分类器（移除交叉注意力模块）
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """专为CSI数据设计的特征提取器 - 输入形状: [batch, 56, 3, 6000]"""

    def __init__(self, feature_dim=512):
        super(FeatureExtractor, self).__init__()
        
        # CSI数据形状: [batch_size, 56_subcarriers, 3_antennas, 6000_time]
        # 我们将其重塑为适合卷积的格式
        
        # 第一阶段：处理天线维度和时间维度
        self.conv1d_layers = nn.Sequential(
            # 沿时间轴的1D卷积，保持子载波和天线维度
            nn.Conv1d(3, 16, kernel_size=15, stride=4, padding=7),  # [batch*56, 3, 6000] -> [batch*56, 16, 1500]
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),  # -> [batch*56, 16, 375]
            
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),  # -> [batch*56, 32, 188]
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),  # -> [batch*56, 32, 47]
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),  # -> [batch*56, 64, 47]
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(16)  # -> [batch*56, 64, 16]
        )
        
        # 第二阶段：处理子载波维度
        self.conv2d_layers = nn.Sequential(
            # 输入: [batch, 56, 64*16] -> [batch, 56, 1024]
            # 重塑为2D: [batch, 1, 56, 1024]
            nn.Conv2d(1, 32, kernel_size=(7, 9), stride=(2, 4), padding=(3, 4)),  # -> [batch, 32, 28, 256]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(2, 2), padding=(2, 3)),  # -> [batch, 64, 14, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2)),  # -> [batch, 128, 7, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))  # -> [batch, 128, 1, 1]
        )
        
        # 特征映射层
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # 输入检查: x shape应为 [batch_size, 56, 3, 6000]
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D input with shape {x.shape}")
        
        batch_size, num_subcarriers, num_antennas, time_length = x.shape
        
        # 第一阶段：处理每个子载波的时间序列
        # 重塑为 [batch*56, 3, 6000] 以便1D卷积处理
        x_reshaped = x.view(batch_size * num_subcarriers, num_antennas, time_length)
        
        # 1D卷积处理时间维度
        x_conv1d = self.conv1d_layers(x_reshaped)  # [batch*56, 64, 16]
        
        # 重塑回子载波维度: [batch, 56, 64*16]
        x_conv1d = x_conv1d.view(batch_size, num_subcarriers, -1)  # [batch, 56, 1024]
        
        # 第二阶段：2D卷积处理子载波维度
        # 添加通道维度: [batch, 1, 56, 1024]
        x_2d = x_conv1d.unsqueeze(1)
        
        # 2D卷积处理
        x_conv2d = self.conv2d_layers(x_2d)  # [batch, 128, 1, 1]
        
        # 展平并通过全连接层
        x_flat = x_conv2d.view(batch_size, -1)  # [batch, 128]
        features = self.fc(x_flat)  # [batch, 512]
        
        return features


class Classifier(nn.Module):
    """简单的分类器"""

    def __init__(self, in_dim=512, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)  # [batch_size, num_classes]


class SimpleCNNModel(nn.Module):
    """
    Model E: 简单CNN分类器（移除交叉注意力模块）
    只使用特征提取器 + 分类器，直接进行身份识别
    """

    def __init__(self, num_classes=10):
        super(SimpleCNNModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, data):
        """
        Model E的简化前向传播：
        只需要单个域的数据输入，直接提取特征并分类
        """
        # 特征提取 - 输出维度 [batch_size, 512]
        features = self.feature_extractor(data)
        
        # 分类预测 - 输出维度 [batch_size, num_classes]
        pred = self.classifier(features)
        
        return pred, features
