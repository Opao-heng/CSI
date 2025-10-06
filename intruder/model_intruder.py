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
            user_mask = (legal_user_identity_labels == user_id)
            user_features = legal_user_features[user_mask]
            if len(user_features) > 0:
                self.mean_vectors[user_id] = np.mean(user_features, axis=0)
                
                # 计算每个合法用户样本到中心的距离
                distances = cdist(user_features, [self.mean_vectors[user_id]], metric='euclidean').flatten()
                
                # 为每个用户拟合Weibull分布
                if len(distances) > 0:
                    self._fit_weibull_model(user_id, distances)
    
    def _fit_weibull_model(self, class_idx, distances):
        """
        为每个用户拟合Weibull分布参数
        """
        if len(distances) > 1:
            # 估计形状参数和尺度参数
            mean_dist = np.mean(distances)
            shape = 1.0  # 简化处理
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
                    
                    # 计算尾部概率
                    model = self.weibull_models[user_id]
                    tail_prob = np.exp(-((distance / model['scale']) ** model['shape']))
                    
                    if tail_prob > max_tail_prob:
                        max_tail_prob = tail_prob
            
            # 转换为入侵者检测的分数
            scores[i] = max_tail_prob
            
            # 阈值判断：如果尾部概率低于某个阈值，则认为是入侵者
            predictions[i] = 0 if max_tail_prob > 0.5 else 1  # 0:合法用户, 1:入侵者
            
        return predictions, scores

class LearnableThresholdDetector(nn.Module):
    """
    可学习的阈值检测器
    利用身份识别模型提取的特征进行训练，学习区分合法用户和入侵者的阈值
    """
    
    def __init__(self, feature_dim=128):
        super(LearnableThresholdDetector, self).__init__()
        self.feature_dim = feature_dim
        
        # 特征编码器 - 增强复杂性
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 分类器 - 增强复杂性
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
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
        probabilities = self.classifier(encoded_features).squeeze()
        
        # 确保输出维度正确
        if probabilities.dim() == 0:
            probabilities = probabilities.unsqueeze(0)
        
        # 预测：概率>0.5为入侵者(1)，否则为合法用户(0)
        predictions = (probabilities > 0.5).float()
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }


class LearnableComprehensiveIntruderDetector(nn.Module):
    """
    可学习的综合入侵者检测模型 - 专门用于二分类任务（合法用户 vs 入侵者）
    结合 TraditionalOpenMax 和 LearnableThresholdDetector进行综合检测
    输出：0表示合法用户，1表示入侵者
    """
    
    def __init__(self, num_known_users=10, feature_dim=128, alpha=3):
        super(LearnableComprehensiveIntruderDetector, self).__init__()
        self.num_known_users = num_known_users
        self.feature_dim = feature_dim
        
        # 初始化TraditionalOpenMax组件
        self.traditional_openmax = TraditionalOpenMax(num_known_users, alpha)
        
        # 初始化可学习阈值检测器
        self.learnable_threshold_detector = LearnableThresholdDetector(feature_dim)
        
        # 融合层 - 结合两种检测方法的结果
        self.fusion_layer = nn.Sequential(
            nn.Linear(2, 16),  # 输入两个检测分数
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 添加一个标志位，表示是否已经拟合了TraditionalOpenMax模型
        self.openmax_fitted = False
    
    def fit_traditional_openmax(self, features, labels, identity_labels):
        """
        拟合TraditionalOpenMax模型
        """
        self.traditional_openmax.fit(features, labels, identity_labels)
        self.openmax_fitted = True
    
    def forward(self, features, logits, identity_labels):
        """
        前向传播
        Args:
            features: 身份识别模型提取的特征 (batch_size, feature_dim)
            logits: 身份识别模型的输出logits (batch_size, num_known_users)
        Returns:
            predictions: 入侵者检测预测结果 (0:合法用户, 1:入侵者)
            probabilities: 检测概率
        """
        # 处理单个样本的情况
        if features.dim() == 1:
            features = features.unsqueeze(0)

        # 获取可学习阈值检测器的结果
        learnable_result = self.learnable_threshold_detector(features)
        learnable_probs = learnable_result['probabilities']

        # 将torch tensor转换为numpy array以供TraditionalOpenMax使用
        features_np = features.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
        
        # 从logits生成二分类标签（0表示合法用户，1表示入侵者）
        # 这里我们假设logits中最大值对应的是合法用户预测，其他为入侵者
        # 但实际上在训练过程中，我们已经有了真实的二分类标签
        # 所以我们需要从identity_labels生成正确的二分类标签
        # 合法用户标签为0，入侵者标签为1
        # 在数据加载器中，入侵者的identity_labels为-1
        
        # 生成正确的二分类标签
        binary_labels = np.where(identity_labels == -1, 1, 0)
        
        # 只有在TraditionalOpenMax尚未拟合时才进行拟合
        if not self.openmax_fitted:
            self.fit_traditional_openmax(features_np, binary_labels, identity_labels)
            
        _, openmax_scores = self.traditional_openmax.predict(features_np)
        # 将分数转换为tensor并调整范围到[0,1]，其中0表示入侵者，1表示合法用户
        openmax_probs = torch.from_numpy(1 - openmax_scores).float().to(features.device)  # 转换为入侵者概率

        combined_input = torch.stack([learnable_probs, openmax_probs], dim=1)
        
        # 通过融合层得到最终的概率
        final_probabilities = self.fusion_layer(combined_input).squeeze()

        # 预测：概率>0.5为入侵者(1)，否则为合法用户(0)
        predictions = (final_probabilities > 0.5).float()
        
        return {
            'predictions': predictions,
            'probabilities': final_probabilities,
            'openmax_probs': openmax_probs,
            'learnable_probs': learnable_probs
        }