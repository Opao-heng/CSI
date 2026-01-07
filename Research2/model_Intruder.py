import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
import math


class TraditionalOpenMax:
    """
    传统OpenMax入侵者检测器 - 基于极值理论建模已知用户特征分布（增强版）
    用于入侵者检测：合法用户 vs 入侵者（二分类）
    
    增强功能：
    1. 动态调整尾部大小（alpha）
    2. 多尺度距离度量（欧氏距离 + 余弦相似度）
    """
    
    def __init__(self, num_known_users=10, alpha=4, distance_metric='hybrid'):
        self.num_known_users = num_known_users
        self.alpha = alpha  # Weibull分布的尾部大小参数
        self.distance_metric = distance_metric  # 'euclidean', 'cosine', 'hybrid'
        self.weibull_models = {}
        self.mean_vectors = {}
        self.best_alpha = alpha  # 记录最佳alpha
        
    def fit(self, features, labels, identity_labels, search_alpha=False, val_features=None, val_labels=None, val_identity_labels=None):
        """
        在已知用户数据上拟合Weibull分布参数
        注意：这里只使用合法用户数据（标签为0）来建模
        
        Args:
            features: 训练特征
            labels: 训练标签
            identity_labels: 训练身份标签
            search_alpha: 是否搜索最佳alpha
            val_features: 验证集特征（用于alpha搜索）
            val_labels: 验证集标签
            val_identity_labels: 验证集身份标签
        """
        # 只使用合法用户数据（标签为0）来建模已知用户特征分布
        legal_user_mask = (labels == 0)
        legal_user_features = features[legal_user_mask]
        legal_user_identity_labels = identity_labels[legal_user_mask]
        
        # 如果启用alpha搜索，使用验证集寻找最佳参数
        if search_alpha and val_features is not None:
            best_alpha = self._search_best_alpha(
                legal_user_features, legal_user_identity_labels,
                val_features, val_labels, val_identity_labels
            )
            self.alpha = best_alpha
            self.best_alpha = best_alpha
            print(f"✓ 最佳 alpha = {best_alpha:.2f}")
        
        # 为每个身份分别计算平均特征向量
        for user_id in range(self.num_known_users):
            if len(legal_user_identity_labels) > 0:
                user_mask = (legal_user_identity_labels == user_id)
                if np.any(user_mask) and len(legal_user_features) > 0:
                    user_features = legal_user_features[user_mask]
                    if len(user_features) > 0:
                        if user_features.ndim == 1:
                            user_features = user_features.reshape(1, -1)
                        self.mean_vectors[user_id] = np.mean(user_features, axis=0)
                        
                        # 计算每个合法用户样本到中心的距离（使用选定的度量）
                        mean_vector = self.mean_vectors[user_id].reshape(1, -1)
                        distances = self._compute_distances(user_features, mean_vector)
                        
                        # 为每个用户拟合Weibull分布
                        if len(distances) > 0:
                            self._fit_weibull_model(user_id, distances)
    
    def _search_best_alpha(self, train_features, train_identity_labels, val_features, val_labels, val_identity_labels, alpha_range=(1, 10, 0.5)):
        """
        在验证集上搜索最佳的alpha值
        
        Args:
            train_features: 训练特征
            train_identity_labels: 训练身份标签
            val_features: 验证特征
            val_labels: 验证标签
            val_identity_labels: 验证身份标签
            alpha_range: (start, end, step) alpha搜索范围
        
        Returns:
            best_alpha: 最佳alpha值
        """
        start, end, step = alpha_range
        alpha_candidates = np.arange(start, end + step, step)
        
        best_score = -1
        best_alpha = self.alpha
        
        for alpha_candidate in alpha_candidates:
            # 临时设置alpha
            temp_alpha = self.alpha
            self.alpha = alpha_candidate
            
            # 重新拟合模型
            self.mean_vectors = {}
            self.weibull_models = {}
            
            for user_id in range(self.num_known_users):
                user_mask = (train_identity_labels == user_id)
                if np.any(user_mask):
                    user_features = train_features[user_mask]
                    if len(user_features) > 0:
                        if user_features.ndim == 1:
                            user_features = user_features.reshape(1, -1)
                        self.mean_vectors[user_id] = np.mean(user_features, axis=0)
                        mean_vector = self.mean_vectors[user_id].reshape(1, -1)
                        distances = self._compute_distances(user_features, mean_vector)
                        if len(distances) > 0:
                            self._fit_weibull_model(user_id, distances)
            
            # 在验证集上评估
            predictions, _ = self.predict(val_features)
            
            # 计算F1分数（对于合法用户检测）
            from sklearn.metrics import f1_score
            val_legal_mask = (val_labels == 0)
            if val_legal_mask.sum() > 0:
                score = f1_score(val_labels[val_legal_mask], predictions[val_legal_mask], zero_division=0)
                if score > best_score:
                    best_score = score
                    best_alpha = alpha_candidate
            
            # 恢复alpha
            self.alpha = temp_alpha
        
        return best_alpha
    
    def _compute_distances(self, features, center, metric=None):
        """
        计算特征到中心的距离（支持多种度量）
        
        Args:
            features: 特征向量 (n, d)
            center: 中心向量 (1, d)
            metric: 距离度量类型，如果为None则使用self.distance_metric
        
        Returns:
            distances: 距离数组 (n,)
        """
        if metric is None:
            metric = self.distance_metric
        
        if metric == 'euclidean':
            # 欧氏距离
            distances = cdist(features, center, metric='euclidean').flatten()
        elif metric == 'cosine':
            # 余弦距离 = 1 - 余弦相似度
            distances = cdist(features, center, metric='cosine').flatten()
        elif metric == 'hybrid':
            # 混合度量：欧氏距离和余弦距离的加权平均
            euclidean_dist = cdist(features, center, metric='euclidean').flatten()
            cosine_dist = cdist(features, center, metric='cosine').flatten()
            # 归一化到相同尺度
            euclidean_norm = euclidean_dist / (np.max(euclidean_dist) + 1e-8)
            cosine_norm = cosine_dist / (np.max(cosine_dist) + 1e-8)
            # 加权平均（0.6欧氏 + 0.4余弦）
            distances = 0.6 * euclidean_norm + 0.4 * cosine_norm
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return distances
    
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
            feature = features[i].reshape(1, -1)
            
            # 计算到每个合法用户中心的距离和尾部概率（使用混合度量）
            max_tail_prob = 0
            for user_id in range(self.num_known_users):
                if user_id in self.mean_vectors and user_id in self.weibull_models:
                    # 计算到用户中心的距离（使用选定的度量）
                    center = self.mean_vectors[user_id].reshape(1, -1)
                    distance = self._compute_distances(feature, center)[0]
                    
                    # 计算尾部概率，增加数值稳定性
                    model = self.weibull_models[user_id]
                    if model['scale'] > 0 and np.isfinite(model['shape']) and np.isfinite(distance):
                        try:
                            power = (distance / model['scale']) ** model['shape']
                            if np.isfinite(power) and power < 700:
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
            
            # 动态阈值调整：降低阈值，更敏感地检测异常
            predictions[i] = 0 if max_tail_prob > 0.05 else 1
            
        return predictions, scores


class OneClassIntruderDetector(nn.Module):
    """
    单类分类器 - 基于深度一类分类（Deep One-Class Classification）
    增强版：利用到所有类中心的距离信息
    核心策略：直接利用身份识别模型已训练好的32维流形空间
    推理：计算样本到所有合法用户中心的距离，让网络学习“类间真空区”模式
    """
    
    def __init__(self, feature_dim=32, num_known_users=10):
        super(OneClassIntruderDetector, self).__init__()
        self.feature_dim = feature_dim
        self.num_known_users = num_known_users
        
        # 注册类中心（从训练数据初始化）
        self.register_buffer('class_centers', torch.zeros(num_known_users, feature_dim))
        self.centers_initialized = False
        
        # 优化版：简化网络结构，防止过拟合
        # 输入: [原始特征(32) + 到所有中心距离(10) + 距离统计(3)] = 45维
        self.distance_processor = nn.Sequential(
            nn.Linear(feature_dim + num_known_users + 3, 128),
            nn.LayerNorm(128),  # LayerNorm对小batch更稳定
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)  # 输出异常 logits
        )
        
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
        
        # 1. 计算到所有类中心的距离 [B, num_known_users]
        # 使用余弦距离可能比欧氏距离在归一化空间更好，但这里先保持欧氏距离
        all_distances = torch.cdist(features, self.class_centers, p=2)  # (batch_size, num_known_users)
        
        # 余弦相似度转距离
        features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
        centers_norm = torch.nn.functional.normalize(self.class_centers, p=2, dim=1)
        cosine_sim = torch.mm(features_norm, centers_norm.t())
        cosine_distances = 1 - cosine_sim
        
        # 混合距离: 60%欧氏 + 40%余弦
        all_distances = 0.6 * all_distances + 0.4 * cosine_distances
        
        # 2. 找到最小距离 (保留用于后续逻辑)
        min_distance, _ = torch.min(all_distances, dim=1)
        max_distance, _ = torch.max(all_distances, dim=1)
        mean_distance = torch.mean(all_distances, dim=1)
        std_distance = torch.std(all_distances, dim=1)
        distance_stats = torch.stack([min_distance, std_distance, mean_distance], dim=1)
        
        combined_input = torch.cat([features, all_distances, distance_stats], dim=1)
        anomaly_logits = self.distance_processor(combined_input).squeeze(-1)
        
        anomaly_probs = torch.sigmoid(anomaly_logits)  # (batch_size,)
        
        return {
            'anomaly_scores': anomaly_logits,  # 异常分数 (logits)
            'probabilities': anomaly_probs,     # 入侵者概率
            'distances': all_distances,         # 返回所有距离
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
    
    def __init__(self, num_known_users=10, feature_dim=32, alpha=4, distance_metric='hybrid'):
        super(LearnableComprehensiveIntruderDetector, self).__init__()
        self.num_known_users = num_known_users
        self.feature_dim = feature_dim
        
        # 初始化单类分类器
        self.one_class_detector = OneClassIntruderDetector(feature_dim, num_known_users)
        
        # 初始化TraditionalOpenMax组件（使用混合距离度量）
        self.traditional_openmax = TraditionalOpenMax(num_known_users, alpha, distance_metric=distance_metric)
        
        # 优化版：简化融合网络，专注关键特征
        # 输入: [oneclass_score, openmax_score, min_distance, distance_std, nearest_ratio, max_distance] = 6维
        self.fusion_network = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(32, 1)  # 最终异常分数
        )
        
    def fit_traditional_openmax(self, features, labels, identity_labels, search_alpha=False, val_features=None, val_labels=None, val_identity_labels=None):
        """
        拟合TraditionalOpenMax模型
        
        Args:
            features: 训练特征
            labels: 训练标签
            identity_labels: 训练身份标签
            search_alpha: 是否搜索最佳alpha
            val_features: 验证集特征（用于alpha搜索）
            val_labels: 验证集标签
            val_identity_labels: 验证集身份标签
        """
        self.traditional_openmax.fit(features, labels, identity_labels, 
                                     search_alpha=search_alpha,
                                     val_features=val_features,
                                     val_labels=val_labels,
                                     val_identity_labels=val_identity_labels)
    
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
        all_distances = oneclass_outputs['distances']  # (batch_size, num_known_users)

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

        # 3. 增强融合：加入距离分布的统计特征
        # 计算距离的方差（高方差可能意味着在类间区域）
        distance_std = torch.std(all_distances, dim=1)  # (batch_size,)
        # 计算到最近和次近中心的距离比（比值接近1说明在边界）
        sorted_distances, _ = torch.sort(all_distances, dim=1)
        nearest_ratio = sorted_distances[:, 0] / (sorted_distances[:, 1] + 1e-8)  # (batch_size,)
        # 新增: 最大距离（远离所有中心可能是入侵者）
        max_distance, _ = torch.max(all_distances, dim=1)
        
        # 4. 动态融合：结合多个检测器的结果和距离统计特征
        fusion_input = torch.stack([
            oneclass_probs,
            openmax_probs,
            min_distance,
            distance_std,      # 距离方差
            nearest_ratio,     # 最近/次近比值
            max_distance       # 新增: 最大距离
        ], dim=1)  # (batch_size, 6)
        
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