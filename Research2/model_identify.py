# model_identify.py - 基于Research1的CrossAttentionModel增强的身份识别模型
# 设计逻辑：继承第三章的特征提取器和交叉注意力机制，增加流形投影头用于入侵检测
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """专为CSI数据设计的特征提取器 - 继承自Research1 (第三章)
    输入形状: [batch, 56, 3, 6000]
    输出形状: [batch, 512]
    """

    def __init__(self, feature_dim=512):
        super(FeatureExtractor, self).__init__()
        
        # CSI数据形状: [batch_size, 56_subcarriers, 3_antennas, 6000_time]
        # 我们将其重塑为适合卷积的格式
        
        # 第一阶段：处理天线维度和时间维度
        self.conv1d_layers = nn.Sequential(
            # 沿时间轴的1D卷积,保持子载波和天线维度
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
    """多头注意力机制实现 - 继承自Research1 (第三章)"""

    def __init__(self, dim=512, num_heads=6):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim) ** -0.5

    def forward(self, q, k, v):
        # 处理输入维度,确保为3维张量 [B, N, C]
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
        # 如果原始输入是2维的,输出也保持2维
        if len(original_q_shape) == 2:
            out = out.squeeze(1)  # [B, 1, C] -> [B, C]

        return out


class CrossAttentionModule(nn.Module):
    """交叉注意力模块 - 继承自Research1 (第三章)
    包含源域自注意力、目标域自注意力、跨域交叉注意力
    """

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

        # 跨域交叉注意力计算(源域为Query,目标域为Key和Value)
        Q_c = self.q_c_linear(F_s)
        K_c = self.k_c_linear(F_t)
        V_c = self.v_c_linear(F_t)
        F_c = self.attention(Q_c, K_c, V_c)

        return F_s, F_t, F_c


class ManifoldProjectionHead(nn.Module):
    """流形投影头 (第四章新增) - 将交叉注意力特征压缩到32维流形空间
    设计目标：为入侵检测提供紧凑、判别性强的特征表示
    输入: F_s或F_t (512维交叉注意力特征)
    输出: 32维流形空间特征
    """
    
    def __init__(self, input_dim=512, projection_dim=32):
        super(ManifoldProjectionHead, self).__init__()
        
        # 三层渐进式降维投影
        self.projection = nn.Sequential(
            # 第一层: 512 -> 256
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # 第二层: 256 -> 96
            nn.Linear(256, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            
            # 第三层: 96 -> 32 (流形空间)
            nn.Linear(96, projection_dim)
        )
        
    def forward(self, x):
        # 投影到32维流形空间
        return self.projection(x)


class IdentityClassifier(nn.Module):
    """身份分类器 - 基于交叉注意力特征进行身份识别
    输入: F_s或F_t (512维交叉注意力特征)
    输出: 身份分类logits (num_classes维)
    """
    
    def __init__(self, feature_dim=512, num_classes=10):
        super(IdentityClassifier, self).__init__()
        
        # 两层分类器结构
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)


class IdentifyDetectionSystem(nn.Module):
    """身份识别系统 (第四章) - 基于Research1的CrossAttentionModel增强
    
    继承关系:
    - 特征提取器 (FeatureExtractor): 完全继承自第三章
    - 交叉注意力模块 (CrossAttentionModule): 完全继承自第三章
    
    新增组件:
    - 流形投影头 (ManifoldProjectionHead): 将F_s和F_t压缩到32维流形空间,为入侵检测准备
    - 身份分类器 (IdentityClassifier): 基于F_s和F_t进行身份识别
    
    设计意图:
    第三章解决跨域身份识别,第四章在此基础上增加流形投影,
    将学到的跨域对齐特征压缩到低维流形空间,为后续入侵检测提供紧凑、判别性强的特征表示
    """
    
    def __init__(self, num_classes=10, feature_dim=512, projection_dim=32):
        super(IdentifyDetectionSystem, self).__init__()
        
        # 继承自第三章的核心组件
        self.feature_extractor = FeatureExtractor(feature_dim=feature_dim)
        self.cross_attention = CrossAttentionModule(dim=feature_dim, num_heads=8)
        
        # 第四章新增组件
        self.manifold_projection = ManifoldProjectionHead(feature_dim, projection_dim)
        self.identity_classifier = IdentityClassifier(feature_dim, num_classes)
        
    def forward(self, x_source, x_target=None):
        if x_target is not None:
            # 训练模式：处理源域和目标域数据
            
            # 1. 特征提取 (继承自第三章)
            feat_s = self.feature_extractor(x_source)  # [batch, 512]
            feat_t = self.feature_extractor(x_target)  # [batch, 512]
            
            # 2. 交叉注意力处理 (继承自第三章)
            F_s, F_t, F_c = self.cross_attention(feat_s, feat_t)
            # F_s: 源域自注意力特征 [batch, 512]
            # F_t: 目标域自注意力特征 [batch, 512]
            # F_c: 跨域交叉注意力特征 [batch, 512]
            
            # 3. 流形投影 (第四章新增) - 为入侵检测准备
            proj_source = self.manifold_projection(F_s)  # [batch, 32]
            proj_target = self.manifold_projection(F_t)  # [batch, 32]
            
            # 4. 身份分类
            logits_source = self.identity_classifier(F_s)  # [batch, num_classes]
            logits_target = self.identity_classifier(F_t)  # [batch, num_classes]
            
            return {
                'features_source': F_s,      # 源域交叉注意力特征,用于对比学习
                'features_target': F_t,      # 目标域交叉注意力特征,用于对比学习
                'proj_source': proj_source,  # 源域流形投影,用于入侵检测
                'proj_target': proj_target,  # 目标域流形投影,用于入侵检测
                'logits_source': logits_source,  # 源域身份预测
                'logits_target': logits_target   # 目标域身份预测
            }
        else:
            # 推理模式：仅处理源域数据
            feat_s = self.feature_extractor(x_source)
            
            # 由于没有目标域,使用源域自身进行交叉注意力(退化为自注意力)
            F_s, _, _ = self.cross_attention(feat_s, feat_s)
            
            proj = self.manifold_projection(F_s)
            logits = self.identity_classifier(F_s)
            
            return {
                'features': F_s,
                'proj': proj,
                'logits': logits
            }