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


class IdentifyDetectionSystem(nn.Module):
    """ 1、身份识别模型 """
    
    def __init__(self, num_classes=10, feature_dim=128, projection_dim=32):
        super(IdentifyDetectionSystem, self).__init__()
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

class LearnableOpenMax(nn.Module):
    """
    可学习的OpenMax入侵者检测器 - 结合传统OpenMax方法与深度学习
    利用源域和目标域身份特征进行训练
    """
    
    def __init__(self, num_classes=10, feature_dim=128):
        super(LearnableOpenMax, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # 可学习的类别中心
        self.class_centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        
        # 可学习的Weibull分布参数
        self.weibull_shapes = nn.Parameter(torch.ones(num_classes))
        self.weibull_scales = nn.Parameter(torch.ones(num_classes))
        
        # 特征变换网络，用于增强特征表示
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 注意力机制，用于动态调整类别权重
        self.attention_weights = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.Tanh(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )
        
        # 域适应模块，处理源域和目标域特征
        self.domain_adapter = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),  # 拼接源域和目标域特征
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, features_source, features_target=None, threshold=0.8):
        """
        前向传播，计算入侵者检测结果
        """
        batch_size = features_source.size(0)
        
        # 如果提供了目标域特征，则进行域适应处理
        if features_target is not None:
            # 拼接源域和目标域特征
            combined_features = torch.cat([features_source, features_target], dim=1)
            # 域适应处理
            adapted_features = self.domain_adapter(combined_features)
        else:
            # 仅使用源域特征
            adapted_features = features_source
        
        # 特征变换
        transformed_features = self.feature_transform(adapted_features)
        
        # 计算到各类别中心的距离
        distances = torch.cdist(transformed_features, self.class_centers, p=2)
        
        # 计算注意力权重
        attention_scores = self.attention_weights(transformed_features)
        
        # 计算激活分数（负距离）
        activations = -distances
        
        # OpenMax变换
        revised_activations = torch.zeros(batch_size, self.num_classes + 1, device=adapted_features.device)
        
        # 计算修正量
        revision = torch.zeros(batch_size, self.num_classes, device=adapted_features.device)
        for i in range(self.num_classes):
            # 计算尾部概率（简化版）
            tail_prob = torch.exp(-distances[:, i] / (self.weibull_scales[i] + 1e-8))
            revision[:, i] = attention_scores[:, i] * tail_prob
            
        # 修正激活分数
        total_revision = torch.sum(revision, dim=1, keepdim=True)
        for i in range(self.num_classes):
            revised_activations[:, i] = activations[:, i] * (1 - revision[:, i])
            
        # 未知类别激活分数
        revised_activations[:, self.num_classes] = total_revision.squeeze()
        
        # 转换为概率分布
        probabilities = torch.softmax(revised_activations, dim=1)
        
        # 判断是否为入侵者
        unknown_probs = probabilities[:, self.num_classes]
        known_user_probs = probabilities[:, :self.num_classes]
        predictions = torch.where(unknown_probs > threshold, 
                                torch.tensor(-1, device=adapted_features.device), 
                                torch.argmax(known_user_probs, dim=1))
        
        return predictions, probabilities, {
            'distances': distances,
            'attention_scores': attention_scores,
            'revision': revision,
            'adapted_features': adapted_features
        }

class LearnableEnergyDetector(nn.Module):
    """
    可学习的能量检测器 - 结合传统能量检测方法与深度学习
    利用源域和目标域身份特征进行训练
    """
    
    def __init__(self, input_dim=10, threshold=1.5):
        super(LearnableEnergyDetector, self).__init__()
        self.threshold = threshold
        
        # 可学习的能量计算网络
        self.energy_network = nn.Sequential(
            nn.Linear(input_dim * 2, 64),  # 考虑源域和目标域概率分布
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # 确保能量分数为正
        )
        
        # 特征注意力机制
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, input_dim * 2),
            nn.Sigmoid()
        )
        
        # 域适应模块
        self.domain_adapter = nn.Sequential(
            nn.Linear(input_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
    def compute_energy(self, probs_source, probs_target=None):
        """
        计算能量分数（负熵的可学习版本）
        """
        # 如果提供了目标域概率，则拼接源域和目标域概率
        if probs_target is not None:
            combined_probs = torch.cat([probs_source, probs_target], dim=1)
        else:
            # 仅使用源域概率，目标域部分用零填充
            batch_size = probs_source.size(0)
            zero_padding = torch.zeros(batch_size, probs_source.size(1), device=probs_source.device)
            combined_probs = torch.cat([probs_source, zero_padding], dim=1)
        
        # 应用注意力机制
        attention_weights = self.feature_attention(combined_probs)
        weighted_probs = combined_probs * attention_weights
        
        # 域适应处理
        adapted_probs = self.domain_adapter(weighted_probs)
        
        # 计算能量分数
        energy_scores = self.energy_network(combined_probs)
        return energy_scores.squeeze()
    
    def forward(self, probs_source, probs_target=None):
        """
        前向传播，基于能量分数检测入侵者
        """
        energy_scores = self.compute_energy(probs_source, probs_target)
        # 能量分数越高表示越不确定，越可能是入侵者
        predictions = torch.where(energy_scores > self.threshold, 
                                torch.tensor(-1, device=probs_source.device), 
                                torch.argmax(probs_source, dim=1))
        return predictions, energy_scores

class LearnableComprehensiveIntruderDetector(nn.Module):
    """ 
    可学习的综合入侵者检测模型 - 结合OpenMax和能量检测的优点
    利用身份识别模型的源域和目标域特征进行训练
    """
    
    def __init__(self, num_classes=10, feature_dim=128, identity_classes=10):
        super(LearnableComprehensiveIntruderDetector, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.identity_classes = identity_classes
        
        # 可学习的OpenMax检测器
        self.openmax_detector = LearnableOpenMax(num_classes, feature_dim)
        
        # 可学习的能量检测器
        self.energy_detector = LearnableEnergyDetector(identity_classes, threshold=1.5)
        
        # 融合网络，用于决策融合
        self.fusion_network = nn.Sequential(
            nn.Linear(4, 32),  # 输入：openmax_unknown_prob, energy_score, identity_confidence_source, identity_confidence_target
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # 特征增强网络
        self.feature_enhancer = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 域间关系建模
        self.inter_domain_modeling = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        
    def forward(self, features_source, features_target=None, logits_source=None, logits_target=None, 
                openmax_threshold=0.8, energy_threshold=1.5):
        """
        前向传播，综合检测入侵者
        """
        # 特征增强
        enhanced_features_source = self.feature_enhancer(features_source)
        
        # 如果提供了目标域特征，则进行域间关系建模
        if features_target is not None:
            enhanced_features_target = self.feature_enhancer(features_target)
            # 域间关系建模
            combined_features = torch.cat([enhanced_features_source, enhanced_features_target], dim=1)
            inter_domain_features = self.inter_domain_modeling(combined_features)
        else:
            enhanced_features_target = None
            inter_domain_features = enhanced_features_source
        
        # OpenMax检测
        openmax_predictions, openmax_probs, openmax_details = self.openmax_detector(
            enhanced_features_source, enhanced_features_target, openmax_threshold)
        
        # 身份分类概率
        identity_probs_source = torch.softmax(logits_source, dim=1) if logits_source is not None else torch.ones(features_source.size(0), self.identity_classes, device=features_source.device) / self.identity_classes
        identity_probs_target = torch.softmax(logits_target, dim=1) if logits_target is not None else None
        
        # 能量检测
        energy_predictions, energy_scores = self.energy_detector(identity_probs_source, identity_probs_target)
        
        # 决策融合
        openmax_unknown_probs = openmax_probs[:, self.num_classes]  # 未知类别的概率
        identity_confidence_source = torch.max(identity_probs_source, dim=1)[0]   # 源域身份分类的置信度
        
        # 融合特征
        if identity_probs_target is not None:
            identity_confidence_target = torch.max(identity_probs_target, dim=1)[0]   # 目标域身份分类的置信度
            fusion_features = torch.stack([
                openmax_unknown_probs,
                energy_scores,
                identity_confidence_source,
                identity_confidence_target
            ], dim=1)
        else:
            # 如果没有目标域概率，使用零填充
            fusion_features = torch.stack([
                openmax_unknown_probs,
                energy_scores,
                identity_confidence_source,
                torch.zeros_like(identity_confidence_source)
            ], dim=1)
        
        # 融合决策
        fusion_weights = self.fusion_network(fusion_features).squeeze()
        
        # 综合预测：根据融合权重决定是否为入侵者
        final_predictions = torch.where(fusion_weights > 0.5,
                                      torch.tensor(-1, device=features_source.device),
                                      torch.argmax(identity_probs_source, dim=1))
        
        return {
            'predictions': final_predictions,
            'openmax_predictions': openmax_predictions,
            'energy_predictions': energy_predictions,
            'openmax_probabilities': openmax_probs,
            'energy_scores': energy_scores,
            'fusion_weights': fusion_weights,
            'openmax_details': openmax_details,
            'inter_domain_features': inter_domain_features
        }

# 保持原有的传统检测器类，用于对比和初始化
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
    """ 2、传统入侵者检测模型 """
    
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