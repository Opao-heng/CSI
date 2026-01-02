"""
Model A (Baseline1): w/o TFGAN & CAL
基线模型 - 仅使用源域数据训练CNN分类器,不使用生成器和交叉注意力
"""

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """专为CSI数据设计的特征提取器 - 输入形状: [batch, 56, 3, 6000]"""

    def __init__(self, feature_dim=512):
        super(FeatureExtractor, self).__init__()
        
        # 第一阶段：处理天线维度和时间维度
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(16)
        )
        
        # 第二阶段：处理子载波维度
        self.conv2d_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 9), stride=(2, 4), padding=(3, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(2, 2), padding=(2, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
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
        batch_size, num_subcarriers, num_antennas, time_length = x.shape
        
        # 第一阶段：处理每个子载波的时间序列
        x_reshaped = x.view(batch_size * num_subcarriers, num_antennas, time_length)
        x_conv1d = self.conv1d_layers(x_reshaped)
        
        # 重塑回子载波维度
        x_conv1d = x_conv1d.view(batch_size, num_subcarriers, -1)
        
        # 第二阶段：2D卷积处理子载波维度
        x_2d = x_conv1d.unsqueeze(1)
        x_conv2d = self.conv2d_layers(x_2d)
        
        # 展平并通过全连接层
        x_flat = x_conv2d.view(batch_size, -1)
        features = self.fc(x_flat)
        
        return features


class Classifier(nn.Module):
    """分类器"""

    def __init__(self, in_dim=512, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class BaselineModel(nn.Module):
    """基线模型：仅特征提取 + 分类"""

    def __init__(self, num_classes=10):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, data):
        # 特征提取
        features = self.feature_extractor(data)
        
        # 分类预测
        pred = self.classifier(features)
        
        return pred, features
