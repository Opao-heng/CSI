import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """特征提取器 - 输入CSI数据，输出128维判别性特征向量（适度简化版用于消融实验）"""
    
    def __init__(self, feature_dim=128):
        super(FeatureExtractor, self).__init__()
        # 适度简化的时间维度1D卷积
        self.time_conv = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=15, stride=2, padding=7),  # 保持16通道
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),  # 保持32通道
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),  # 保持64通道
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)  # 保持32长度
        )
        
        # 适度简化的子载波维度处理
        self.subcarrier_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, stride=2, padding=3),  # 保持32通道
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)  # 保持16长度
        )
        
        # 特征融合和映射
        self.feature_fusion = nn.Sequential(
            nn.Linear(32 * 16, 128),  # 保持输入维度512
            nn.ReLU(),
            nn.Dropout(0.3),  # 适度dropout率
            nn.Linear(128, feature_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: [batch_size, 56, 3, 6000]
        batch_size, num_subcarriers, num_antennas, time_length = x.shape
        
        # 1. 沿时间维度处理每个子载波-天线组合
        x_time = x.view(batch_size * num_subcarriers, num_antennas, time_length)
        x_time = self.time_conv(x_time)  # [batch*56, 64, 32]
        
        # 2. 沿子载波维度处理
        x_sub = x_time.view(batch_size, num_subcarriers, 64, -1)
        x_sub = x_sub.permute(0, 2, 1, 3).contiguous()
        x_sub = x_sub.view(batch_size, 64, -1)
        x_sub = self.subcarrier_conv(x_sub)  # [batch, 32, 16]
        
        # 3. 特征融合
        x_flat = x_sub.view(batch_size, -1)
        features = self.feature_fusion(x_flat)  # [batch, 128]
        
        return features


class AttentionModule(nn.Module):
    """注意力模块 - 增强特征表示（适度简化版用于消融实验）"""
    
    def __init__(self, feature_dim=128):
        super(AttentionModule, self).__init__()
        self.feature_dim = feature_dim
        # 适度简化注意力权重计算
        self.attention_weights = nn.Sequential(
            nn.Linear(feature_dim, 32),  # 保持32中间维度
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        # 保持LayerNorm以维持一定复杂度
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        # x shape: [batch_size, feature_dim]
        attention_scores = self.attention_weights(x)
        attention_weights = torch.softmax(attention_scores, dim=0)
        
        weighted_features = x * attention_weights
        out = self.layer_norm(x + weighted_features)  # 恢复LayerNorm
        
        return out


class ProjectionHead(nn.Module):
    """投影头 - 将特征映射到32维空间用于对比学习"""
    
    def __init__(self, input_dim=128, projection_dim=32):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, projection_dim)
        )
        
    def forward(self, x):
        return self.projection(x)


class IdentityClassifier(nn.Module):
    """身份分类器 - 对已知用户进行身份识别（适度简化版用于消融实验）"""
    
    def __init__(self, feature_dim=128, num_classes=10):
        super(IdentityClassifier, self).__init__()
        # 适度简化分类器结构
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),  # 保持32中间维度
            nn.ReLU(),
            nn.Dropout(0.3),  # 适度dropout率
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)


class IdentifyDetectionSystemNoContrastive(nn.Module):
    """ 身份识别模型（无对比损失版本） """
    
    def __init__(self, num_classes=10, feature_dim=128, projection_dim=32):
        super(IdentifyDetectionSystemNoContrastive, self).__init__()
        self.feature_extractor = FeatureExtractor(feature_dim=feature_dim)
        self.attention = AttentionModule(feature_dim=feature_dim)
        self.projection_head = ProjectionHead(feature_dim, projection_dim)
        self.identity_classifier = IdentityClassifier(feature_dim, num_classes)
        
    def forward(self, x_source, x_target=None):
        # 特征提取
        features_source = self.feature_extractor(x_source)
        features_source = self.attention(features_source)
        
        # 投影到32维空间用于对比学习（保留但不用于训练）
        proj_source = self.projection_head(features_source)
        
        # 身份分类
        logits_source = self.identity_classifier(features_source)
        
        if x_target is not None:
            features_target = self.feature_extractor(x_target)
            features_target = self.attention(features_target)
            
            # 投影到32维空间用于对比学习（保留但不用于训练）
            proj_target = self.projection_head(features_target)
            
            # 身份分类
            logits_target = self.identity_classifier(features_target)
            
            # 注意：这里不返回对比学习相关的特征，因为我们要消除对比损失
            return {
                'features_source': features_source,
                'features_target': features_target,
                'logits_source': logits_source,
                'logits_target': logits_target
            }
        else:
            # 仅推理模式
            return {
                'features': features_source,
                'logits': logits_source
            }