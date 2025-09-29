# attention.py
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

class MultiHeadAttention(nn.Module):
    """多头注意力机制实现"""

    def __init__(self, dim=512, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim) ** -0.5

    def forward(self, q, k, v):
        # 处理输入维度，确保为3维张量 [B, N, C]
        original_q_shape = q.shape
        original_k_shape = k.shape
        original_v_shape = v.shape

        if q.dim() == 2:
            q = q.unsqueeze(1)  # [B, C] -> [B, 1, C]
        if k.dim() == 2:
            k = k.unsqueeze(1)  # [B, C] -> [B, 1, C]
        if v.dim() == 2:
            v = v.unsqueeze(1)  # [B, C] -> [B, 1, C]

        # 验证输入维度一致性
        if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
            raise ValueError(f"Inconsistent feature dimensions: q:{q.shape[-1]}, k:{k.shape[-1]}, v:{v.shape[-1]}")

        B, N_q, C = q.shape
        N_k = k.shape[1]

        # 验证特征维度
        if C != self.dim:
            raise ValueError(f"Input feature dimension {C} doesn't match expected dimension {self.dim}")
        if C % self.num_heads != 0:
            raise ValueError(f"Feature dimension {C} is not divisible by num_heads {self.num_heads}")

        head_dim = C // self.num_heads

        # 多头注意力计算
        q = q.view(B, N_q, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, N_k, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, N_k, self.num_heads, head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, N_q, C)

        # 确保输出维度与输入维度一致
        # 如果原始输入是2维的，输出也保持2维
        if len(original_q_shape) == 2:
            out = out.squeeze(1)  # [B, 1, C] -> [B, C]

        return out


class CrossAttentionModule(nn.Module):
    """交叉注意力模块：包含源域自注意力、目标域自注意力、跨域交叉注意力"""

    def __init__(self, dim=512, num_heads=8):
        super(CrossAttentionModule, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(dim, num_heads)

        # 源域自注意力线性投影层
        self.q_s_linear = nn.Linear(dim, dim)
        self.k_s_linear = nn.Linear(dim, dim)
        self.v_s_linear = nn.Linear(dim, dim)

        # 目标域自注意力线性投影层
        self.q_t_linear = nn.Linear(dim, dim)
        self.k_t_linear = nn.Linear(dim, dim)
        self.v_t_linear = nn.Linear(dim, dim)

        # 交叉注意力线性投影层
        self.q_c_linear = nn.Linear(dim, dim)
        self.k_c_linear = nn.Linear(dim, dim)
        self.v_c_linear = nn.Linear(dim, dim)

    def forward(self, feat_s, feat_t):
        # 源域自注意力计算
        Q_s = self.q_s_linear(feat_s)
        K_s = self.k_s_linear(feat_s)
        V_s = self.v_s_linear(feat_s)
        F_s = self.attention(Q_s, K_s, V_s)

        # 目标域自注意力计算
        Q_t = self.q_t_linear(feat_t)
        K_t = self.k_t_linear(feat_t)
        V_t = self.v_t_linear(feat_t)
        F_t = self.attention(Q_t, K_t, V_t)

        # 跨域交叉注意力计算（源域为Query，目标域为Key和Value）
        Q_c = self.q_c_linear(F_s)
        K_c = self.k_c_linear(F_t)
        V_c = self.v_c_linear(F_t)
        F_c = self.attention(Q_c, K_c, V_c)

        return F_s, F_t, F_c


class Classifier(nn.Module):

    def __init__(self, in_dim=512, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        # 移除softmax，因为CrossEntropyLoss内部已经包含了softmax操作

    def forward(self, x):
        return self.fc(x)  # [batch_size, num_classes]


class CrossAttentionModel(nn.Module):
    """完整模型结构：特征提取 -> 交叉注意力 -> 分类"""

    def __init__(self, num_classes=10):
        super(CrossAttentionModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.cross_attention = CrossAttentionModule()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, src_data, tgt_data):
        # 特征提取 - 输出维度 [batch_size, 512]
        feat_s = self.feature_extractor(src_data)
        feat_t = self.feature_extractor(tgt_data)

        # 交叉注意力处理 - 输出维度 [batch_size, 512] for each
        F_s, F_t, F_c = self.cross_attention(feat_s, feat_t)

        # 分类预测 - 输出维度 [batch_size, num_classes]
        pred_s = self.classifier(F_s)
        pred_t = self.classifier(F_t)

        return pred_s, pred_t, F_s, F_t, F_c
