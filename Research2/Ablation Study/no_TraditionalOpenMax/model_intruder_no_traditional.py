import torch
import torch.nn as nn
import numpy as np


class LearnableThresholdDetector(nn.Module):
    """
    可学习的阈值检测器
    利用身份识别模型提取的特征进行训练，学习区分合法用户和入侵者的阈值
    """
    
    def __init__(self, feature_dim=128):
        super(LearnableThresholdDetector, self).__init__()
        self.feature_dim = feature_dim
        
        # 简化特征编码器 - 减少复杂性以防止过拟合
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        
        # 简化分类器
        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1)
        )
        
    def forward(self, features):
        """
        前向传播
        """
        # 处理单个样本的情况
        if features.dim() == 1:
            features = features.unsqueeze(0)
            
        # 特征编码
        encoded_features = self.feature_encoder(features)
        
        # 分类
        logits = self.classifier(encoded_features).squeeze()
        probabilities = torch.sigmoid(logits)  # 明确使用sigmoid函数
        
        # 确保输出维度正确
        if probabilities.dim() == 0:
            probabilities = probabilities.unsqueeze(0)
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        
        # 预测：概率>0.5为入侵者(1)，否则为合法用户(0)
        predictions = (probabilities > 0.5).float()
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'logits': logits
        }


class LearnableIntruderDetectorNoTraditional(nn.Module):
    """
    可学习的入侵者检测模型（无传统OpenMax组件）- 专门用于二分类任务（合法用户 vs 入侵者）
    只使用可学习的阈值检测器进行入侵者检测
    输出：0表示合法用户，1表示入侵者
    """
    
    def __init__(self, feature_dim=128):
        super(LearnableIntruderDetectorNoTraditional, self).__init__()
        self.feature_dim = feature_dim
        
        # 初始化可学习阈值检测器
        self.learnable_threshold_detector = LearnableThresholdDetector(feature_dim)
        
        # 简化的融合层 - 使用特征和logits信息进行检测
        # 输入维度：features + logits
        fusion_input_dim = feature_dim + 10  # 128维特征 + 10维logits
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, features, logits, identity_labels=None):
        """
        前向传播
        Args:
            features: 身份识别模型提取的特征 (batch_size, feature_dim)
            logits: 身份识别模型的输出logits (batch_size, num_known_users)
            identity_labels: 身份标签（可选，用于兼容性）
        Returns:
            predictions: 入侵者检测预测结果 (0:合法用户, 1:入侵者)
            probabilities: 检测概率
        """
        # 处理单个样本的情况
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # 获取可学习阈值检测器的结果
        learnable_result = self.learnable_threshold_detector(features)
        learnable_logits = learnable_result['logits']  # 获取logits
        learnable_probs = learnable_result['probabilities']

        # 确保所有张量的batch维度一致
        batch_size = features.size(0)
        if learnable_probs.size(0) != batch_size:
            learnable_probs = learnable_probs[:batch_size]
        if logits.size(0) != batch_size:
            logits = logits[:batch_size]

        # 使用特征和logits信息进行融合
        # 将特征和logits拼接
        combined_input = torch.cat([features, logits], dim=1)
        
        # 通过融合层得到最终的logits
        final_logits = self.fusion_layer(combined_input).squeeze()
        # 应用sigmoid函数得到概率
        final_probabilities = torch.sigmoid(final_logits)

        # 确保输出维度正确
        if final_probabilities.dim() == 0:
            final_probabilities = final_probabilities.unsqueeze(0)
        if final_logits.dim() == 0:
            final_logits = final_logits.unsqueeze(0)

        # 预测：概率>0.5为入侵者(1)，否则为合法用户(0)
        predictions = (final_probabilities > 0.5).float()
        
        return {
            'predictions': predictions,
            'probabilities': final_probabilities,
            'logits': final_logits,
            'learnable_probs': learnable_probs
        }