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
    
    def __init__(self, num_known_users=10, alpha=4):
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
            
            # 【开放集识别】动态阈值调整：降低阈值，更敏感地检测异常
            # 如果尾部概率低于某个阈值，则认为是入侵者
            predictions[i] = 0 if max_tail_prob > 0.05 else 1  # 0:合法用户, 1:入侵者，降低阈值
            
        return predictions, scores


class OneClassIntruderDetector(nn.Module):
    """
    单类分类器 - 基于深度一类分类（Deep One-Class Classification）
    核心策略：直接利用身份识别模型已训练好的32维流形空间
    推理：计算样本到合法用户中心的距离，距离大 → 入侵者
    """
    
    def __init__(self, feature_dim=32, num_known_users=10):
        super(OneClassIntruderDetector, self).__init__()
        self.feature_dim = feature_dim
        self.num_known_users = num_known_users
        
        # 简化的距离到异常分数映射（单层非线性）
        # 通过伪入侵者训练，学习"距离越大，分数越高"的单调映射
        self.distance_to_score = nn.Sequential(
            nn.Linear(1, 8),  # 输入：最小距离
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)  # 输出异常分数 (logit，会通过sigmoid转为概率)
        )
        
        # 可学习的类中心（从训练数据初始化）
        self.register_buffer('class_centers', torch.zeros(num_known_users, feature_dim))
        self.centers_initialized = False
        
    def initialize_centers(self, features, labels):
        """
        从训练数据初始化类中心（只调用一次）
        """
        if self.centers_initialized:
            return
        
        with torch.no_grad():
            for c in range(self.num_known_users):
                mask = (labels == c)
                if mask.sum() > 0:
                    self.class_centers[c] = features[mask].mean(dim=0)
        
        self.centers_initialized = True
        print(f"✅ 类中心初始化完成")
        
    def forward(self, features):
        """
        前向传播
        Args:
            features: 32维流形投彡特征 (batch_size, 32)
        Returns:
            dict: {
                'anomaly_scores': 异常分数 (batch_size,),
                'probabilities': 入侵者概率 (batch_size,),
                'distances': 到每个类中心的距离 (batch_size, num_known_users),
                'min_distance': 到最近中心的距离 (batch_size,)
            }
        """
        # 处理单个样本
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # 计算到每个类中心的欧式距离
        distances = torch.cdist(features, self.class_centers, p=2)  # (batch_size, num_known_users)
        
        # 找到最小距离（到最近的合法用户中心）
        min_distance, _ = torch.min(distances, dim=1)  # (batch_size,)
        
        # 将最小距离映射到异常分数
        anomaly_logits = self.distance_to_score(min_distance.unsqueeze(1)).squeeze(-1)  # (batch_size,)
        
        # 转换为概率
        anomaly_probs = torch.sigmoid(anomaly_logits)  # (batch_size,)
        
        return {
            'anomaly_scores': anomaly_logits,  # 异常分数 (logits)
            'probabilities': anomaly_probs,     # 入侵者概率
            'distances': distances,             # 到每个类中心的距离
            'min_distance': min_distance        # 到最近中心的距离
        }


class LearnableComprehensiveIntruderDetector(nn.Module):
    """
    开放集学习框架 - 结合统计方法和深度学习
    核心策略：
    1. OneClassDetector: 学习合法用户的紧凑流形
    2. TraditionalOpenMax: 统计方法建模已知用户分布
    3. 动态融合：自适应权重调整
    """
    
    def __init__(self, num_known_users=10, feature_dim=32, alpha=4):
        super(LearnableComprehensiveIntruderDetector, self).__init__()
        self.num_known_users = num_known_users
        self.feature_dim = feature_dim
        
        # 初始化单类分类器
        self.one_class_detector = OneClassIntruderDetector(feature_dim, num_known_users)
        
        # 初始化TraditionalOpenMax组件
        self.traditional_openmax = TraditionalOpenMax(num_known_users, alpha)
        
        # 动态融合网络：自适应调整两个检测器的权重
        self.fusion_network = nn.Sequential(
            nn.Linear(3, 24),  # 3: oneclass_score, openmax_score, min_distance
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(24, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)  # 最终异常分数
        )
        
    def fit_traditional_openmax(self, features, labels, identity_labels):
        """
        拟合TraditionalOpenMax模型
        """
        self.traditional_openmax.fit(features, labels, identity_labels)
    
    def forward(self, features, logits=None, identity_labels=None, use_openmax=True):
        """
        前向传播
        Args:
            features: 32维流形投彡特征 (batch_size, 32)
            logits: 身份识别模型的输出logits (可选)
            identity_labels: 身份标签 (可选)
            use_openmax: 是否使用OpenMax
        Returns:
            dict: {
                'predictions': 入侵者检测预测结果 (0:合法用户, 1:入侵者),
                'probabilities': 检测概率,
                'logits': 原始分数
            }
        """
        # 处理单个样本的情况
        if features.dim() == 1:
            features = features.unsqueeze(0)

        # 1. 获取单类分类器的结果
        oneclass_outputs = self.one_class_detector(features)
        oneclass_probs = oneclass_outputs['probabilities']
        min_distance = oneclass_outputs['min_distance']

        # 2. 使用预先拟合好的 TraditionalOpenMax 进行打分（可选）
        if use_openmax:
            features_np = features.detach().cpu().numpy()
            _, openmax_scores = self.traditional_openmax.predict(features_np)
            # 转换为入侵者概率（OpenMax分数越低越可能是入侵者）
            openmax_probs = torch.from_numpy(1 - openmax_scores).float().to(features.device)
        else:
            # 训练初期不使用OpenMax，使用oneclass_probs的副本
            openmax_probs = oneclass_probs.clone().detach()

        # 确保所有张量维度一致
        if openmax_probs.dim() == 0:
            openmax_probs = openmax_probs.unsqueeze(0)
        if oneclass_probs.dim() == 0:
            oneclass_probs = oneclass_probs.unsqueeze(0)
        if min_distance.dim() == 0:
            min_distance = min_distance.unsqueeze(0)
            
        # 确保batch维度一致
        batch_size = features.size(0)
        if openmax_probs.size(0) != batch_size:
            openmax_probs = openmax_probs[:batch_size]
        if oneclass_probs.size(0) != batch_size:
            oneclass_probs = oneclass_probs[:batch_size]

        # 3. 动态融合：结合两个检测器的结果
        fusion_input = torch.stack([
            oneclass_probs,
            openmax_probs,
            min_distance
        ], dim=1)  # (batch_size, 3)
        
        final_logits = self.fusion_network(fusion_input).squeeze(-1)
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
            'oneclass_probs': oneclass_probs,
            'openmax_probs': openmax_probs,
            'min_distance': min_distance
        }