"""
Model B: w/o Generator, Only CAL
仅使用交叉注意力机制,不使用生成器进行数据增强
直接在少量目标域数据和源域数据上训练
"""

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """专为CSI数据设计的特征提取器"""

    def __init__(self, feature_dim=512):
        super(FeatureExtractor, self).__init__()
        
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
        
        x_reshaped = x.view(batch_size * num_subcarriers, num_antennas, time_length)
        x_conv1d = self.conv1d_layers(x_reshaped)
        x_conv1d = x_conv1d.view(batch_size, num_subcarriers, -1)
        
        x_2d = x_conv1d.unsqueeze(1)
        x_conv2d = self.conv2d_layers(x_2d)
        
        x_flat = x_conv2d.view(batch_size, -1)
        features = self.fc(x_flat)
        
        return features


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, dim=512, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim) ** -0.5

    def forward(self, q, k, v):
        original_q_shape = q.shape

        if q.dim() == 2:
            q = q.unsqueeze(1)
        if k.dim() == 2:
            k = k.unsqueeze(1)
        if v.dim() == 2:
            v = v.unsqueeze(1)

        B, N_q, C = q.shape
        N_k = k.shape[1]

        head_dim = C // self.num_heads

        q = q.view(B, N_q, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, N_k, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, N_k, self.num_heads, head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, N_q, C)

        if len(original_q_shape) == 2:
            out = out.squeeze(1)

        return out


class CrossAttentionModule(nn.Module):
    """交叉注意力模块"""

    def __init__(self, dim=512, num_heads=8):
        super(CrossAttentionModule, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(dim, num_heads)

        self.q_s_linear = nn.Linear(dim, dim)
        self.k_s_linear = nn.Linear(dim, dim)
        self.v_s_linear = nn.Linear(dim, dim)

        self.q_t_linear = nn.Linear(dim, dim)
        self.k_t_linear = nn.Linear(dim, dim)
        self.v_t_linear = nn.Linear(dim, dim)

        self.q_c_linear = nn.Linear(dim, dim)
        self.k_c_linear = nn.Linear(dim, dim)
        self.v_c_linear = nn.Linear(dim, dim)

    def forward(self, feat_s, feat_t):
        # 源域自注意力
        Q_s = self.q_s_linear(feat_s)
        K_s = self.k_s_linear(feat_s)
        V_s = self.v_s_linear(feat_s)
        F_s = self.attention(Q_s, K_s, V_s)

        # 目标域自注意力
        Q_t = self.q_t_linear(feat_t)
        K_t = self.k_t_linear(feat_t)
        V_t = self.v_t_linear(feat_t)
        F_t = self.attention(Q_t, K_t, V_t)

        # 跨域交叉注意力
        Q_c = self.q_c_linear(F_s)
        K_c = self.k_c_linear(F_t)
        V_c = self.v_c_linear(F_t)
        F_c = self.attention(Q_c, K_c, V_c)

        return F_s, F_t, F_c


class Classifier(nn.Module):
    """分类器"""

    def __init__(self, in_dim=512, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class ModelB_OnlyCAL(nn.Module):
    """Model B: 只有交叉注意力,无生成器"""

    def __init__(self, num_classes=10):
        super(ModelB_OnlyCAL, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.cross_attention = CrossAttentionModule()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, src_data, tgt_data):
        # 特征提取
        feat_s = self.feature_extractor(src_data)
        feat_t = self.feature_extractor(tgt_data)

        # 交叉注意力处理
        F_s, F_t, F_c = self.cross_attention(feat_s, feat_t)

        # 分类预测
        pred_s = self.classifier(F_s)
        pred_t = self.classifier(F_t)

        return pred_s, pred_t, F_s, F_t, F_c
