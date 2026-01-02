import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """1D残差块 - 提升特征提取能力"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 如果输入输出维度不同，使用1x1卷积进行映射
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ChannelAttention(nn.Module):
    """通道注意力机制 - 增强重要特征通道"""
    
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = self.sigmoid(avg_out + max_out).view(b, c, 1)
        return x * out


class FeatureExtractor(nn.Module):
    """优化的特征提取器 - 输入CSI数据，输出128维判别性特征向量"""
    
    def __init__(self, feature_dim=128):
        super(FeatureExtractor, self).__init__()
        
        # 改进的时间维度卷积 - 使用残差块
        self.time_conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            ResidualBlock1D(32, 64, kernel_size=9, stride=2),
            nn.MaxPool1d(2),
            
            ResidualBlock1D(64, 128, kernel_size=5, stride=1),
            ChannelAttention(128),  # 添加通道注意力
            nn.AdaptiveAvgPool1d(64)
        )
        
        # 改进的子载波维度处理 - 使用残差块和注意力
        self.subcarrier_conv = nn.Sequential(
            ResidualBlock1D(128, 96, kernel_size=7, stride=2),
            ChannelAttention(96),
            ResidualBlock1D(96, 64, kernel_size=5, stride=1),
            nn.AdaptiveAvgPool1d(32)
        )
        
        # 特征融合和映射 - 增加深度
        self.feature_fusion = nn.Sequential(
            nn.Linear(64 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: [batch_size, 56, 3, 6000]
        batch_size, num_subcarriers, num_antennas, time_length = x.shape
        
        # 1. 沿时间维度处理每个子载波-天线组合
        x_time = x.view(batch_size * num_subcarriers, num_antennas, time_length)
        x_time = self.time_conv(x_time)  # [batch*56, 128, 64]
        
        # 2. 沿子载波维度处理
        x_sub = x_time.view(batch_size, num_subcarriers, 128, -1)
        x_sub = x_sub.permute(0, 2, 1, 3).contiguous()
        x_sub = x_sub.view(batch_size, 128, -1)
        x_sub = self.subcarrier_conv(x_sub)  # [batch, 64, 32]
        
        # 3. 特征融合
        x_flat = x_sub.view(batch_size, -1)
        features = self.feature_fusion(x_flat)  # [batch, 128]
        
        return features


class AttentionModule(nn.Module):
    """注意力模块 - 增强特征表示"""
    
    def __init__(self, feature_dim=128):
        super(AttentionModule, self).__init__()
        self.feature_dim = feature_dim
        self.attention_weights = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        # x shape: [batch_size, feature_dim]
        attention_scores = self.attention_weights(x)
        attention_weights = torch.softmax(attention_scores, dim=0)
        
        weighted_features = x * attention_weights
        out = self.layer_norm(x + weighted_features)
        
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
    """身份分类器 - 对已知用户进行身份识别"""
    
    def __init__(self, feature_dim=128, num_classes=10):
        super(IdentityClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)


class IdentifyDetectionSystem(nn.Module):
    """ 身份识别模型 """
    
    def __init__(self, num_classes=10, feature_dim=128, projection_dim=32):
        super(IdentifyDetectionSystem, self).__init__()
        self.feature_extractor = FeatureExtractor(feature_dim=feature_dim)
        self.attention = AttentionModule(feature_dim=feature_dim)
        self.projection_head = ProjectionHead(feature_dim, projection_dim)
        self.identity_classifier = IdentityClassifier(feature_dim, num_classes)
        
    def forward(self, x_source, x_target=None):
        # 特征提取
        features_source = self.feature_extractor(x_source)
        features_source = self.attention(features_source)
        
        if x_target is not None:
            features_target = self.feature_extractor(x_target)
            features_target = self.attention(features_target)
            
            # 投影到32维空间用于对比学习
            proj_source = self.projection_head(features_source)
            proj_target = self.projection_head(features_target)
            
            # 身份分类
            logits_source = self.identity_classifier(features_source)
            logits_target = self.identity_classifier(features_target)
            
            return {
                'features_source': features_source,
                'features_target': features_target,
                'proj_source': proj_source,
                'proj_target': proj_target,
                'logits_source': logits_source,
                'logits_target': logits_target
            }
        else:
            # 仅推理模式
            logits = self.identity_classifier(features_source)
            return {
                'features': features_source,
                'logits': logits
            }