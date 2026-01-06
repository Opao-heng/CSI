import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
import math


class TraditionalOpenMax:
    """
    传统OpenMax入侵者检测器 - 基于极值理论建模已知用户特征分布
    用于入侵者检测：合法用户 vs 入侵者（二分类）
    """
    
    def __init__(self, num_known_users=10, alpha=3):
        self.num_known_users = num_known_users  # 已知用户数量
        self.alpha = alpha  # Weibull分布的尾部大小参数
        self.weibull_models = {}
        self.mean_vectors = {}
        
    def fit(self, features, labels, identity_labels):
        """
        在已知用户数据上拟合Weibull分布参数
        注意：这里只使用合法用户数据（标签为0）来建模
        """
        # 只使用合法用户数据（标签为0）来建模已知用户特征分布
        legal_user_mask = (labels == 0)
        legal_user_features = features[legal_user_mask]
        legal_user_identity_labels = identity_labels[legal_user_mask]
        
        # 为每个身份分别计算平均特征向量
        for user_id in range(self.num_known_users):
            # 确保索引不会越界
            if len(legal_user_identity_labels) > 0:
                user_mask = (legal_user_identity_labels == user_id)
                # 确保掩码不为空且至少有一个True值
                if np.any(user_mask) and len(legal_user_features) > 0:
                    user_features = legal_user_features[user_mask]
                    if len(user_features) > 0:
                        # 确保user_features是2D数组
                        if user_features.ndim == 1:
                            user_features = user_features.reshape(1, -1)
                        self.mean_vectors[user_id] = np.mean(user_features, axis=0)
                        
                        # 计算每个合法用户样本到中心的距离
                        mean_vector = self.mean_vectors[user_id].reshape(1, -1)
                        distances = cdist(user_features, mean_vector, metric='euclidean').flatten()
                        
                        # 为每个用户拟合Weibull分布
                        if len(distances) > 0:
                            self._fit_weibull_model(user_id, distances)
    
    def _fit_weibull_model(self, class_idx, distances):
        """
        为每个用户拟合Weibull分布参数 - 使用简化但稳定的方法
        """
        if len(distances) > 1:
            # 使用简化但更稳定的方法估计Weibull参数
            mean_dist = np.mean(distances)
            # 使用经验方法估计形状参数
            std_dist = np.std(distances)
            if std_dist > 0 and mean_dist > 0:
                # 形状参数的近似估计
                shape = (mean_dist / std_dist) ** 1.5
                # 限制形状参数范围以提高稳定性
                shape = np.clip(shape, 0.5, 5.0)
            else:
                shape = 1.0
            
            # 计算尺度参数
            scale = mean_dist / math.gamma(1 + 1/shape)
            
            self.weibull_models[class_idx] = {'shape': shape, 'scale': scale}
    
    def predict(self, features):
        """
        使用OpenMax进行入侵者检测（二分类）
        返回：0表示合法用户，1表示入侵者
        """
        # 处理单个样本的情况
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        batch_size = len(features)
        predictions = np.zeros(batch_size, dtype=int)
        scores = np.zeros(batch_size)
        
        for i in range(batch_size):
            feature = features[i]
            
            # 计算到每个合法用户中心的距离和尾部概率
            max_tail_prob = 0
            for user_id in range(self.num_known_users):
                if user_id in self.mean_vectors and user_id in self.weibull_models:
                    # 计算到用户中心的距离
                    distance = np.linalg.norm(feature - self.mean_vectors[user_id])
                    
                    # 计算尾部概率，增加数值稳定性
                    model = self.weibull_models[user_id]
                    # 防止除零和数值溢出
                    if model['scale'] > 0 and np.isfinite(model['shape']) and np.isfinite(distance):
                        try:
                            power = (distance / model['scale']) ** model['shape']
                            if np.isfinite(power) and power < 700:  # 防止exp溢出
                                tail_prob = np.exp(-power)
                            else:
                                tail_prob = 0.0
                        except:
                            tail_prob = 0.0
                    else:
                        tail_prob = 0.0
                    
                    if tail_prob > max_tail_prob:
                        max_tail_prob = tail_prob
            
            # 转换为入侵者检测的分数
            scores[i] = max_tail_prob
            
            # 阈值判断：如果尾部概率低于某个阈值，则认为是入侵者
            predictions[i] = 0 if max_tail_prob > 0.5 else 1  # 0:合法用户, 1:入侵者
            
        return predictions, scores

class LearnableThresholdDetector(nn.Module):
    """
    入侵者检测器 - 基于32维流形空间的轻量级异常检测模型
    
    设计理念（优化版）：
    1. 直接在32维流形空间上工作，避免过度压缩
    2. 浅层网络结构，防止过拟合
    3. 利用流形投影已有的紧凑性，专注于决策边界学习
    """
    
    def __init__(self, feature_dim=32):
        super(LearnableThresholdDetector, self).__init__()
        self.feature_dim = feature_dim
        
        # 优化的异常度估计器 - 增强特征提取能力
        # 32 -> 24 -> 12 -> 1 (适度增加容量)
        self.detector = nn.Sequential(
            nn.Linear(feature_dim, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(24, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(12, 1)  # 输出：异常分数
        )
        
    def forward(self, features):
        """
        前向传播
        Args:
            features: 32维流形投影特征 (batch_size, 32)
        Returns:
            dict: {
                'predictions': 二分类预测 (0:合法用户, 1:入侵者),
                'probabilities': 入侵者概率,
                'logits': 原始分数
            }
        """
        # 处理单个样本
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # 异常分数估计
        logits = self.detector(features).squeeze(-1)  # [batch]
        
        # 转换为概率
        probabilities = torch.sigmoid(logits)
        
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


class LearnableComprehensiveIntruderDetector(nn.Module):
    """
    可学习的综合入侵者检测模型 - 专门用于二分类任务（合法用户 vs 入侵者）
    核心创新：OpenMax（传统统计） + 深度学习检测器 + 自适应融合
    输出：0表示合法用户，1表示入侵者
    """
    
    def __init__(self, num_known_users=10, feature_dim=32, alpha=3):
        super(LearnableComprehensiveIntruderDetector, self).__init__()
        self.num_known_users = num_known_users
        self.feature_dim = feature_dim
        
        # 初始化TraditionalOpenMax组件
        self.traditional_openmax = TraditionalOpenMax(num_known_users, alpha)
        
        # 初始化可学习阈值检测器（32维流形空间）
        self.learnable_threshold_detector = LearnableThresholdDetector(feature_dim)
        
        # 增强的融合层 - 融合4维特征（2个概率 + 差异 + 均值）
        # 输入：openmax_prob + learnable_prob + |diff| + mean = 4维
        self.fusion_layer = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(12, 6),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(6, 1)
        )
        
        # 移除openmax_fitted标志位，每次都会更新模型
    
    def fit_traditional_openmax(self, features, labels, identity_labels):
        """
        拟合TraditionalOpenMax模型
        """
        self.traditional_openmax.fit(features, labels, identity_labels)
    
    def forward(self, features, logits, identity_labels):
        """
        前向传播
        Args:
            features: 32维流形投影特征 (batch_size, 32)
            logits: 身份识别模型的输出logits (batch_size, num_known_users)
            identity_labels: 身份标签 (用于OpenMax)
        Returns:
            predictions: 入侵者检测预测结果 (0:合法用户, 1:入侵者)
            probabilities: 检测概率
        """
        # 处理单个样本的情况
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # 1. 获取可学习阈值检测器的结果（直接在32维流形空间判别）
        learnable_result = self.learnable_threshold_detector(features)
        learnable_probs = learnable_result['probabilities']

        # 2. 使用预先拟合好的 TraditionalOpenMax 进行打分
        features_np = features.detach().cpu().numpy()
        _, openmax_scores = self.traditional_openmax.predict(features_np)
        # 转换为入侵者概率（OpenMax分数越低越可能是入侵者）
        openmax_probs = torch.from_numpy(1 - openmax_scores).float().to(features.device)

        # 确保所有张量维度一致
        if openmax_probs.dim() == 0:
            openmax_probs = openmax_probs.unsqueeze(0)
        if learnable_probs.dim() == 0:
            learnable_probs = learnable_probs.unsqueeze(0)
            
        # 确保batch维度一致
        batch_size = features.size(0)
        if openmax_probs.size(0) != batch_size:
            openmax_probs = openmax_probs[:batch_size]
        if learnable_probs.size(0) != batch_size:
            learnable_probs = learnable_probs[:batch_size]

        # 3. 增强融合：融合两个概率值和置信度差异
        prob_diff = torch.abs(learnable_probs - openmax_probs).unsqueeze(1)  # 差异信息
        prob_mean = ((learnable_probs + openmax_probs) / 2).unsqueeze(1)  # 平均置信度
        combined_input = torch.cat([
            learnable_probs.unsqueeze(1), 
            openmax_probs.unsqueeze(1),
            prob_diff,
            prob_mean
        ], dim=1)  # [batch, 4]
        
        # 通过融合层得到最终的logits
        final_logits = self.fusion_layer(combined_input).squeeze(-1)
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
            'openmax_probs': openmax_probs,
            'learnable_probs': learnable_probs
        }