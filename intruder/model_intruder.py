import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
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
        
    def fit(self, features, labels):
        """
        在已知用户数据上拟合Weibull分布参数
        注意：这里只使用合法用户数据（标签为0）来建模
        """
        # 只使用合法用户数据（标签为0）来建模已知用户特征分布
        legal_user_features = features[labels == 0]
        
        # 计算合法用户的平均特征向量
        if len(legal_user_features) > 0:
            self.mean_vectors[0] = np.mean(legal_user_features, axis=0)
        
        # 计算每个合法用户样本到中心的距离
        if 0 in self.mean_vectors:
            distances = cdist(legal_user_features, [self.mean_vectors[0]], metric='euclidean').flatten()
            
            # 为合法用户拟合Weibull分布
            if len(distances) > 0:
                self._fit_weibull_model(0, distances)
    
    def _fit_weibull_model(self, class_idx, distances):
        """
        为合法用户群体拟合Weibull分布参数
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
        
        if 0 not in self.mean_vectors or 0 not in self.weibull_models:
            return predictions, scores
            
        for i in range(batch_size):
            feature = features[i]
            
            # 计算到合法用户中心的距离
            distance = np.linalg.norm(feature - self.mean_vectors[0])
            
            # 计算尾部概率（越接近合法用户中心，尾部概率越高）
            model = self.weibull_models[0]
            tail_prob = np.exp(-((distance / model['scale']) ** model['shape']))
            
            # 转换为入侵者检测的分数（尾部概率越高，越可能是合法用户）
            scores[i] = tail_prob
            
            # 阈值判断：如果尾部概率低于某个阈值，则认为是入侵者
            predictions[i] = 0 if tail_prob > 0.5 else 1  # 0:合法用户, 1:入侵者
            
        return predictions, scores

class TraditionalEnergyDetector:
    """
    传统能量检测器 - 基于模型输出总置信度的能量检测方法
    """
    
    def __init__(self, energy_threshold=1.5):
        self.energy_threshold = energy_threshold
    
    def compute_energy(self, logits):
        """
        计算模型输出的总能量（置信度）
        """
        # 使用softmax计算概率分布
        probs = torch.softmax(logits, dim=-1)  # 使用-1维度确保正确处理
        # 计算总能量：概率分布的负熵
        energy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return energy
    
    def detect(self, logits):
        """
        基于能量阈值检测入侵者
        返回：0表示合法用户，1表示入侵者
        """
        energy = self.compute_energy(logits)
        # 能量高于阈值表示是入侵者
        predictions = (energy > self.energy_threshold).long()
        return predictions, energy

class LearnableThresholdDetector(nn.Module):
    """
    可学习的阈值检测器
    利用身份识别模型提取的特征进行训练，学习区分合法用户和入侵者的阈值
    """
    
    def __init__(self, feature_dim=128):
        super(LearnableThresholdDetector, self).__init__()
        self.feature_dim = feature_dim
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 分类器 - 输出入侵者检测概率
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
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

class LearnableEnergyDetector(nn.Module):
    """
    可学习的能量检测器
    """
    
    def __init__(self, input_dim=10):  # 10个已知用户的logits
        super(LearnableEnergyDetector, self).__init__()
        
        # 可学习的能量计算网络
        self.energy_network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # 可学习的阈值参数
        self.energy_threshold = nn.Parameter(torch.tensor(1.5))
    
    def forward(self, logits):
        """
        前向传播
        """
        # 处理单个样本的情况
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            
        # 计算能量
        energy = self.energy_network(logits)
        energy = energy.squeeze()
        
        # 确保输出维度正确
        if energy.dim() == 0:
            energy = energy.unsqueeze(0)
        
        # 基于可学习阈值进行检测：能量高于阈值为入侵者
        predictions = (energy > self.energy_threshold).float()
        
        return {
            'predictions': predictions,
            'energy': energy,
            'threshold': self.energy_threshold
        }

class ComprehensiveIntruderDetector(nn.Module):
    """
    综合入侵者检测模型 - 专门用于二分类任务（合法用户 vs 入侵者）
    输出：0表示合法用户，1表示入侵者
    """
    
    def __init__(self, num_known_users=10, feature_dim=128):
        super(ComprehensiveIntruderDetector, self).__init__()
        self.num_known_users = num_known_users
        self.feature_dim = feature_dim
        
        # 传统OpenMax检测器
        self.traditional_openmax = TraditionalOpenMax(num_known_users)
        
        # 传统能量检测器
        self.traditional_energy = TraditionalEnergyDetector()
        
        # 可学习的阈值检测器
        self.learnable_threshold = LearnableThresholdDetector(feature_dim)
        
        # 可学习的能量检测器
        self.learnable_energy = LearnableEnergyDetector(num_known_users)
        
        # 融合网络 - 输入4个检测器的输出，输出最终的入侵者检测概率
        self.fusion_network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, logits):
        """
        前向传播
        Args:
            features: 身份识别模型提取的特征 (batch_size, feature_dim)
            logits: 身份识别模型的输出logits (batch_size, num_known_users)
        Returns:
            predictions: 入侵者检测预测结果 (0:合法用户, 1:入侵者)
            fusion_prob: 融合后的检测概率
        """
        # 处理单个样本的情况
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            
        batch_size = features.size(0)
        
        # 1. 传统OpenMax检测
        features_np = features.detach().cpu().numpy()
        openmax_predictions, openmax_scores = self.traditional_openmax.predict(features_np)
        openmax_scores = torch.tensor(openmax_scores, dtype=torch.float32, device=features.device)
        
        # 2. 传统能量检测
        energy_predictions, energy_scores = self.traditional_energy.detect(logits)
        
        # 3. 可学习阈值检测
        threshold_outputs = self.learnable_threshold(features)
        threshold_predictions = threshold_outputs['predictions']
        threshold_probabilities = threshold_outputs['probabilities']
        
        # 4. 可学习能量检测
        energy_outputs = self.learnable_energy(logits)
        learnable_energy_predictions = energy_outputs['predictions']
        learnable_energy_scores = energy_outputs['energy']
        
        # 确保所有输出的维度一致
        if openmax_scores.dim() == 0:
            openmax_scores = openmax_scores.unsqueeze(0)
        if energy_scores.dim() == 0:
            energy_scores = energy_scores.unsqueeze(0)
        if threshold_probabilities.dim() == 0:
            threshold_probabilities = threshold_probabilities.unsqueeze(0)
        if learnable_energy_scores.dim() == 0:
            learnable_energy_scores = learnable_energy_scores.unsqueeze(0)
        
        # 融合所有检测结果
        # 注意：对于概率型输出，我们直接使用；对于二分类输出，转换为概率
        fusion_input = torch.stack([
            openmax_scores,                    # 传统OpenMax分数
            torch.sigmoid(energy_scores),      # 传统能量分数转为概率
            threshold_probabilities,           # 可学习阈值检测概率
            torch.sigmoid(learnable_energy_scores)  # 可学习能量检测分数转为概率
        ], dim=1)
        
        # 通过融合网络得到最终的入侵者检测概率
        fusion_prob = self.fusion_network(fusion_input).squeeze()
        
        # 确保输出维度正确
        if fusion_prob.dim() == 0:
            fusion_prob = fusion_prob.unsqueeze(0)
        
        # 最终预测：概率>0.5为入侵者(1)，否则为合法用户(0)
        final_predictions = (fusion_prob > 0.5).float()  # 使用float类型
        
        return {
            'predictions': final_predictions,
            'fusion_prob': fusion_prob,
            'openmax_scores': openmax_scores,
            'energy_scores': energy_scores,
            'threshold_prob': threshold_probabilities,
            'learnable_energy_scores': learnable_energy_scores
        }