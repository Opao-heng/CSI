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


class TraditionalIntruderDetectorOnly(nn.Module):
    """
    传统的入侵者检测模型（仅使用传统OpenMax组件）- 专门用于二分类任务（合法用户 vs 入侵者）
    只使用传统的OpenMax进行入侵者检测
    输出：0表示合法用户，1表示入侵者
    """
    
    def __init__(self, num_known_users=10, alpha=3):
        super(TraditionalIntruderDetectorOnly, self).__init__()
        self.num_known_users = num_known_users
        
        # 初始化TraditionalOpenMax组件
        self.traditional_openmax = TraditionalOpenMax(num_known_users, alpha)
    
    def fit_traditional_openmax(self, features, labels, identity_labels):
        """
        拟合TraditionalOpenMax模型
        """
        self.traditional_openmax.fit(features, labels, identity_labels)
    
    def forward(self, features, logits=None, identity_labels=None):
        """
        前向传播
        Args:
            features: 身份识别模型提取的特征 (batch_size, feature_dim)
            logits: 身份识别模型的输出logits (batch_size, num_known_users) - 仅用于兼容性
            identity_labels: 身份标签
        Returns:
            predictions: 入侵者检测预测结果 (0:合法用户, 1:入侵者)
            probabilities: 检测概率
        """
        # 处理单个样本的情况
        if features.dim() == 1:
            features = features.unsqueeze(0)

        # 将torch tensor转换为numpy array以供TraditionalOpenMax使用
        features_np = features.detach().cpu().numpy()
        
        # 生成正确的二分类标签
        # 确保identity_labels是numpy数组
        if identity_labels is not None:
            if torch.is_tensor(identity_labels):
                identity_labels_np = identity_labels.detach().cpu().numpy()
            else:
                identity_labels_np = identity_labels
                
            binary_labels = np.where(identity_labels_np == -1, 1, 0)
        else:
            # 如果没有身份标签，默认所有为合法用户
            binary_labels = np.zeros(features_np.shape[0])
        
        # 拟合TraditionalOpenMax模型
        if identity_labels is not None:
            self.fit_traditional_openmax(features_np, binary_labels, identity_labels_np)
        else:
            # 如果没有身份标签，创建默认的身份标签
            default_identity_labels = np.zeros(features_np.shape[0])
            self.fit_traditional_openmax(features_np, binary_labels, default_identity_labels)
            
        predictions, scores = self.traditional_openmax.predict(features_np)
        # 将分数转换为tensor并调整范围到[0,1]，其中0表示入侵者，1表示合法用户
        probabilities = torch.from_numpy(1 - scores).float().to(features.device)  # 转换为入侵者概率

        # 确保所有张量维度一致
        if probabilities.dim() == 0:
            probabilities = probabilities.unsqueeze(0)
            
        # 确保所有张量的batch维度一致
        batch_size = features.size(0)
        if probabilities.size(0) != batch_size:
            probabilities = probabilities[:batch_size]

        # 转换预测结果为tensor
        predictions_tensor = torch.from_numpy(predictions).float().to(features.device)
        if predictions_tensor.dim() == 0:
            predictions_tensor = predictions_tensor.unsqueeze(0)
        if predictions_tensor.size(0) != batch_size:
            predictions_tensor = predictions_tensor[:batch_size]

        return {
            'predictions': predictions_tensor,
            'probabilities': probabilities,
            'openmax_probs': probabilities
        }