import torch
import torch.nn as nn
import torch.nn.functional as F

# model_Identify.py - 基于Research1的CrossAttentionModel增强的身份识别模型
# 设计逻辑：继承第三章的特征提取器和交叉注意力机制，增加流形投影头用于入侵检测


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


class AnomalyOrientedProjection(nn.Module):
    """面向异常检测的流形投影 (第四章核心创新) - 增强版
    
    设计动机:
    - 第三章: 512维特征针对分类任务优化(类间分离)
    - 第四章: 32维特征针对异常检测优化(类内紧凑+距离度量)
    
    技术创新 (v2.0 - 网络结构优化):
    1. 深层投影网络: 更平滑的维度压缩 + 残差连接
    2. 软分配Center机制: 多中心加权融合,提升流形质量
    3. 强归一化: LayerNorm + BatchNorm,增强训练稳定性
    
    输入: F_s或F_t (512维交叉注意力特征)
    输出: 32维紧凑特征 + 到类中心的距离信息
    """
    
    def __init__(self, input_dim=512, projection_dim=32, num_classes=10):
        super(AnomalyOrientedProjection, self).__init__()
        
        # 更深的投影网络: 512 -> 256 -> 128 -> 64 -> 32 (更平滑的压缩)
        # 使用残差连接缓解梯度消失
        self.proj_stage1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),  # LayerNorm更适合小batch
            nn.ReLU(inplace=True),
            nn.Dropout(0.15)
        )
        
        self.proj_stage2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.proj_stage3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.08)
        )
        
        # 最终投影到32维
        self.proj_final = nn.Linear(64, projection_dim)
        
        # 可学习的类中心原型 (为OpenMax距离计算准备)
        # 使用Xavier初始化,更稳定
        self.class_centers = nn.Parameter(
            torch.randn(num_classes, projection_dim) * (2.0 / (num_classes + projection_dim)) ** 0.5
        )
        
        # 软分配机制: 计算样本到所有中心的权重
        self.soft_assignment = nn.Sequential(
            nn.Linear(projection_dim, num_classes),
            nn.Softmax(dim=1)  # 输出[B, num_classes],表示对每个中心的权重
        )
        
        # Center-aware调整层 (增强版): 多中心加权融合
        self.center_fusion = nn.Sequential(
            nn.Linear(projection_dim * 2, 128),  # 拼接[投影特征, 加权中心]
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(128, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        # 残差门控: 自动学习是否需要Center调整
        self.residual_gate = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.Sigmoid()  # 门控权重[0, 1]
        )
        
    def forward(self, x, return_distances=False):
        """
        Args:
            x: [B, 512] - 第三章的交叉注意力特征
            return_distances: 是否返回距离信息(训练时需要)
        
        Returns:
            proj_adjusted: [B, 32] - 面向异常检测的紧凑特征
            min_distances: [B] - 到最近类中心的距离(可选)
            nearest_centers: [B] - 最近的类中心索引(可选)
        """
        # 1. 深层投影: 512 -> 256 -> 128 -> 64 -> 32 (更平滑的压缩)
        h1 = self.proj_stage1(x)  # [B, 256]
        h2 = self.proj_stage2(h1)  # [B, 128]
        h3 = self.proj_stage3(h2)  # [B, 64]
        proj = self.proj_final(h3)  # [B, 32]
        
        # L2归一化: 统一特征尺度,提升距离度量质量
        proj = F.normalize(proj, p=2, dim=1)
        
        # 2. 计算到所有类中心的距离
        # proj: [B, 32], class_centers: [10, 32]
        # 同样归一化类中心
        centers_normalized = F.normalize(self.class_centers, p=2, dim=1)
        distances = torch.cdist(proj, centers_normalized)  # [B, 10]
        
        # 3. 软分配: 计算样本对每个中心的权重
        assignment_weights = self.soft_assignment(proj)  # [B, 10]
        
        # 4. 加权融合中心: 不只用最近的,而是加权组合多个中心
        weighted_center = torch.matmul(assignment_weights, centers_normalized)  # [B, 32]
        
        # 5. Center-aware调整: 融合原始投影和加权中心
        combined = torch.cat([proj, weighted_center], dim=1)  # [B, 64]
        center_adjusted = self.center_fusion(combined)  # [B, 32]
        
        # 6. 残差门控: 自适应地混合原始投影和center调整
        gate = self.residual_gate(proj)  # [B, 32]
        proj_adjusted = gate * proj + (1 - gate) * center_adjusted  # [B, 32]
        
        # 7. 最终L2归一化
        proj_adjusted = F.normalize(proj_adjusted, p=2, dim=1)
        
        if return_distances:
            # 返回到最近中心的距离(用于损失计算)
            min_distances, nearest_centers = torch.min(distances, dim=1)  # [B]
            return proj_adjusted, min_distances, nearest_centers
        else:
            return proj_adjusted


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
    - 面向异常检测的流形投影 (AnomalyOrientedProjection): 
      将512维特征压缩到32维,并显式建模类中心,为OpenMax入侵检测准备
    - 身份分类器 (IdentityClassifier): 基于F_s和F_t进行身份识别
    
    设计意图:
    第三章解决跨域身份识别(分类任务),第四章针对入侵检测(异常检测任务)优化特征空间:
    - 降维: 512维→32维,降低OpenMax计算复杂度,缓解维度灾难
    - 度量优化: 显式建模类中心,优化欧式距离度量
    - 紧凑性: Center-aware机制引导特征向类中心靠拢,增强正常样本聚集
    """
    
    def __init__(self, num_classes=10, feature_dim=512, projection_dim=32):
        super(IdentifyDetectionSystem, self).__init__()
        
        # 继承自第三章的核心组件
        self.feature_extractor = FeatureExtractor(feature_dim=feature_dim)
        self.cross_attention = CrossAttentionModule(dim=feature_dim, num_heads=8)
        
        # 第四章新增组件
        self.manifold_projection = AnomalyOrientedProjection(feature_dim, projection_dim, num_classes)
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
            
            # 3. 流形投影 (第四章新增) - 面向异常检测优化
            # 返回32维紧凑特征 + 距离信息(用于center-aware损失)
            if self.training:
                proj_source, dist_s, center_s = self.manifold_projection(F_s, return_distances=True)
                proj_target, dist_t, center_t = self.manifold_projection(F_t, return_distances=True)
            else:
                proj_source = self.manifold_projection(F_s)  # [batch, 32]
                proj_target = self.manifold_projection(F_t)  # [batch, 32]
                dist_s = dist_t = center_s = center_t = None
            
            # 4. 身份分类
            logits_source = self.identity_classifier(F_s)  # [batch, num_classes]
            logits_target = self.identity_classifier(F_t)  # [batch, num_classes]
            
            return {
                'features_source': F_s,      # 源域交叉注意力特征,用于对比学习
                'features_target': F_t,      # 目标域交叉注意力特征,用于对比学习
                'proj_source': proj_source,  # 源域32维投影,用于入侵检测
                'proj_target': proj_target,  # 目标域32维投影,用于入侵检测
                'logits_source': logits_source,  # 源域身份预测
                'logits_target': logits_target,   # 目标域身份预测
                # 训练时返回距离信息(用于center-aware损失)
                'dist_source': dist_s if self.training else None,
                'dist_target': dist_t if self.training else None,
                'center_source': center_s if self.training else None,
                'center_target': center_t if self.training else None
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