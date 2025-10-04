import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import weibull_min

class FeatureExtractor(nn.Module):
    """简化版特征提取器 - 输入CSI数据，输出128维判别性特征向量"""
    
    def __init__(self, feature_dim=128):
        super(FeatureExtractor, self).__init__()
        # 简化的时间维度1D卷积
        self.time_conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )
        
        # 简化的子载波维度处理
        self.subcarrier_conv = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        
        # 特征融合和映射
        self.feature_fusion = nn.Sequential(
            nn.Linear(64 * 32, 256),
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


class IntruderDetectionSystem(nn.Module):
    """ 1、身份识别模型 """
    
    def __init__(self, num_classes=10, feature_dim=128, projection_dim=32):
        super(IntruderDetectionSystem, self).__init__()
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

class OpenMaxIntruderDetector:
    """
    OpenMax入侵者检测器 - 基于极值理论检测未知入侵者
    """
    
    def __init__(self, num_classes=10, feature_dim=128):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_means = np.zeros((num_classes, feature_dim))
        self.weibull_models = {}
        self.fitted = False
    
    def fit(self, features, labels):
        """
        使用已知用户数据拟合OpenMax参数
        """
        # 确保输入是numpy数组
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
            
        # 计算每个类别的均值特征向量
        for i in range(self.num_classes):
            class_features = features[labels == i]
            if len(class_features) > 0:
                self.class_means[i] = np.mean(class_features, axis=0)
        
        # 为每个类别拟合Weibull分布
        distances = cdist(features, self.class_means, metric='euclidean')
        
        for i in range(self.num_classes):
            class_distances = distances[labels == i, i]
            if len(class_distances) >= 10:  # 需要足够的样本
                # 取最大的20个距离值进行拟合
                sorted_distances = np.sort(class_distances)[::-1]
                tail_values = sorted_distances[:min(20, len(sorted_distances))]
                
                try:
                    # 拟合Weibull分布
                    shape, loc, scale = weibull_min.fit(tail_values, floc=0)
                    self.weibull_models[i] = {
                        'shape': shape,
                        'scale': scale,
                        'loc': loc
                    }
                except:
                    # 拟合失败时使用默认参数
                    self.weibull_models[i] = {
                        'shape': 1.0,
                        'scale': 1.0,
                        'loc': 0.0
                    }
            else:
                # 样本不足时使用默认参数
                self.weibull_models[i] = {
                    'shape': 1.0,
                    'scale': 1.0,
                    'loc': 0.0
                }
                
        self.fitted = True
    
    def predict(self, features, threshold=0.8):
        """
        检测入侵者
        返回: predictions (-1表示入侵者, >=0表示已知用户ID)
        """
        # 确保输入是numpy数组
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
            
        # 计算到各类别中心的距离
        distances = cdist(features, self.class_means, metric='euclidean')
        
        # 计算激活分数（负距离）
        activations = -distances
        
        # OpenMax变换
        N = activations.shape[0]
        revised_activations = np.zeros((N, self.num_classes + 1))
        
        for i in range(N):
            # 计算每个类别的alpha值
            sorted_indices = np.argsort(distances[i])
            alpha = np.zeros(self.num_classes)
            
            # 对最近的10个类别分配alpha值
            for rank, class_idx in enumerate(sorted_indices[:min(10, len(sorted_indices))]):
                alpha[class_idx] = 1.0 - (rank / 10.0)
                
            # 计算修正量
            revision = np.zeros(self.num_classes)
            for j in range(self.num_classes):
                if j in self.weibull_models:
                    model = self.weibull_models[j]
                    # 计算尾部概率
                    tail_prob = weibull_min.sf(distances[i, j], 
                                             model['shape'], 
                                             model['loc'], 
                                             model['scale'])
                    revision[j] = alpha[j] * tail_prob
                    
            # 修正激活分数
            total_revision = np.sum(revision)
            for j in range(self.num_classes):
                revised_activations[i, j] = activations[i, j] * (1 - revision[j])
                
            # 未知类别激活分数
            revised_activations[i, self.num_classes] = total_revision
        
        # 转换为概率分布
        exp_activations = np.exp(revised_activations - np.max(revised_activations, axis=1, keepdims=True))
        probabilities = exp_activations / np.sum(exp_activations, axis=1, keepdims=True)
        
        # 判断是否为入侵者
        unknown_probs = probabilities[:, self.num_classes]
        predictions = np.where(unknown_probs > threshold, -1, np.argmax(probabilities[:, :-1], axis=1))
        
        return predictions, probabilities

class EnergyIntruderDetector:
    """
    能量检测器 - 基于模型置信度检测入侵者
    """
    
    def __init__(self, threshold=1.5):
        self.threshold = threshold
    
    def compute_energy(self, probabilities):
        """
        计算能量分数（负熵）
        """
        # 添加小值避免log(0)
        epsilon = 1e-8
        energy_scores = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
        return energy_scores
    
    def predict(self, probabilities):
        """
        基于能量分数检测入侵者
        """
        energy_scores = self.compute_energy(probabilities)
        # 能量分数越高表示越不确定，越可能是入侵者
        predictions = np.where(energy_scores > self.threshold, -1, np.argmax(probabilities, axis=1))
        return predictions, energy_scores


class ComprehensiveIntruderDetector:
    """ 2、入侵者检测模型 """
    
    def __init__(self, num_classes=10, feature_dim=128):
        self.openmax_detector = OpenMaxIntruderDetector(num_classes, feature_dim)
        self.energy_detector = EnergyIntruderDetector(threshold=1.5)
        self.fitted = False
    
    def fit(self, features, labels):
        """
        拟合检测器参数
        """
        print("拟合综合入侵者检测器...")
        self.openmax_detector.fit(features, labels)
        self.fitted = True
        print("入侵者检测器参数拟合完成")
    
    def predict(self, features, logits, openmax_threshold=0.8, energy_threshold=1.5):
        """
        综合检测入侵者
        """
        if not self.fitted:
            raise RuntimeError("Comprehensive intruder detector has not been fitted yet!")
        
        # OpenMax检测
        openmax_predictions, openmax_probs = self.openmax_detector.predict(
            features, threshold=openmax_threshold)
        
        # 能量检测
        if isinstance(logits, torch.Tensor):
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        energy_predictions, energy_scores = self.energy_detector.predict(probabilities)
        
        # 决策融合：如果任一检测器认为是入侵者，则判定为入侵者
        final_predictions = np.where(
            (openmax_predictions == -1) | (energy_predictions == -1),
            -1,  # 入侵者
            openmax_predictions  # 已知用户
        )
        
        return {
            'predictions': final_predictions,
            'openmax_predictions': openmax_predictions,
            'energy_predictions': energy_predictions,
            'openmax_probabilities': openmax_probs,
            'energy_scores': energy_scores
        }