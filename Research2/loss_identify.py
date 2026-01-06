import torch
import torch.nn.functional as F


class ManifoldLoss:
    """
    流形优化损失 V3.0 - 面向OpenMax+深度学习联合入侵检测的32维流形空间优化
    
    针对model_intruder.py的优化目标:
    1. OpenMax需要: 类内紧凑(小方差) + 类中心分离(大欧氏距离) + 规整分布(适合Weibull)
    2. 深度检测器需要: 判别性特征(合法用户vs入侵者边界清晰)
    
    核心改进:
    1. 欧氏距离优化: 直接优化到类中心的欧氏距离(OpenMax使用欧氏距离)
    2. 类内方差约束: 让距离分布更规整,适合Weibull建模
    3. Triplet Loss: 同类拉近,异类推远,增强判别边界
    4. 类中心分离margin: 确保不同身份有足够的分离度
    """
    
    def __init__(self, intra_weight=1.0, inter_weight=0.5, margin=0.5, 
                 variance_weight=0.3, triplet_weight=0.5):
        """
        Args:
            intra_weight: 类内紧凑性损失权重
            inter_weight: 类间分离性损失权重  
            margin: 类间分离的margin (欧氏距离)
            variance_weight: 类内方差约束权重 (让分布更规整)
            triplet_weight: Triplet损失权重
        """
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
        self.margin = margin
        self.variance_weight = variance_weight
        self.triplet_weight = triplet_weight
        
        # 损失记录
        self.loss_history = {
            'intra_loss_src': [],
            'intra_loss_tgt': [],
            'inter_loss_src': [],
            'inter_loss_tgt': [],
            'total_intra_loss': [],
            'total_inter_loss': [],
            'total_loss': []
        }
    
    def compute_center_loss(self, proj_features, labels):
        """
        Center Loss: 最小化样本到其类中心的欧氏距离
        这是OpenMax的核心需求 - 使用欧氏距离计算到类中心的距离
        """
        device = proj_features.device
        batch_size = proj_features.size(0)
        
        if batch_size == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # 计算每个类的中心
        unique_labels = torch.unique(labels)
        centers = {}
        for c in unique_labels:
            mask = (labels == c)
            if mask.sum() > 0:
                centers[c.item()] = proj_features[mask].mean(dim=0)
        
        if len(centers) == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # 计算每个样本到其类中心的欧氏距离
        center_loss = torch.tensor(0.0, device=device)
        variance_loss = torch.tensor(0.0, device=device)
        
        for c in unique_labels:
            c_item = c.item()
            if c_item not in centers:
                continue
            mask = (labels == c)
            if mask.sum() > 1:
                class_features = proj_features[mask]
                center = centers[c_item]
                
                # 欧氏距离到中心
                distances = torch.norm(class_features - center.unsqueeze(0), dim=1)
                
                # Center Loss: 平均距离 (越小越紧凑)
                center_loss += distances.mean()
                
                # Variance Loss: 距离的方差 (越小分布越规整,适合Weibull)
                if distances.size(0) > 1:
                    variance_loss += distances.var()
        
        num_classes = len(unique_labels)
        if num_classes > 0:
            center_loss = center_loss / num_classes
            variance_loss = variance_loss / num_classes
        
        return center_loss, variance_loss
    
    def compute_separation_loss(self, proj_features, labels):
        """
        类间分离损失: 最大化不同类中心之间的欧氏距离
        使用Hinge Loss确保最小距离大于margin
        """
        device = proj_features.device
        
        # 计算每个类的中心
        unique_labels = torch.unique(labels)
        centers = []
        for c in unique_labels:
            mask = (labels == c)
            if mask.sum() > 0:
                centers.append(proj_features[mask].mean(dim=0))
        
        if len(centers) < 2:
            return torch.tensor(0.0, device=device)
        
        centers_tensor = torch.stack(centers)  # [num_classes, 32]
        
        # 计算所有类中心对之间的欧氏距离
        distances = torch.cdist(centers_tensor, centers_tensor)  # [K, K]
        
        # 提取上三角(不包括对角线)的距离
        num_centers = len(centers)
        separation_loss = torch.tensor(0.0, device=device)
        pair_count = 0
        
        for i in range(num_centers):
            for j in range(i + 1, num_centers):
                dist = distances[i, j]
                # Hinge Loss: max(0, margin - distance)
                # 希望distance > margin
                separation_loss += F.relu(self.margin - dist)
                pair_count += 1
        
        if pair_count > 0:
            separation_loss = separation_loss / pair_count
        
        # 返回正值，外部不需要取负
        return separation_loss
    
    def compute_triplet_loss(self, proj_features, labels, margin=0.3):
        """
        Triplet Loss: 同类拉近,异类推远
        anchor-positive距离 < anchor-negative距离 - margin
        """
        device = proj_features.device
        batch_size = proj_features.size(0)
        
        if batch_size < 3:
            return torch.tensor(0.0, device=device)
        
        triplet_loss = torch.tensor(0.0, device=device)
        triplet_count = 0
        
        # 遍历所有可能的triplet
        for i in range(batch_size):
            anchor = proj_features[i]
            anchor_label = labels[i]
            
            # 找到同类样本(positive)
            pos_mask = (labels == anchor_label)
            pos_mask[i] = False  # 排除自己
            
            # 找到异类样本(negative)
            neg_mask = (labels != anchor_label)
            
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            
            # 计算到所有正样本的距离
            pos_features = proj_features[pos_mask]
            pos_distances = torch.norm(anchor.unsqueeze(0) - pos_features, dim=1)
            
            # 计算到所有负样本的距离
            neg_features = proj_features[neg_mask]
            neg_distances = torch.norm(anchor.unsqueeze(0) - neg_features, dim=1)
            
            # 使用hard mining: 最远的正样本和最近的负样本
            hardest_pos_dist = pos_distances.max()
            hardest_neg_dist = neg_distances.min()
            
            # Triplet loss: max(0, pos_dist - neg_dist + margin)
            loss = F.relu(hardest_pos_dist - hardest_neg_dist + margin)
            triplet_loss += loss
            triplet_count += 1
        
        if triplet_count > 0:
            triplet_loss = triplet_loss / triplet_count
        
        if torch.isnan(triplet_loss) or torch.isinf(triplet_loss):
            return torch.tensor(0.0, device=device)
        
        return triplet_loss
    
    def __call__(self, proj_source, labels_source, proj_target, labels_target, verbose=True):
        """
        计算总损失 - 面向OpenMax+深度学习联合入侵检测
        """
        device = proj_source.device
        
        # ===== 源域损失 =====
        # 1. Center Loss + Variance Loss (OpenMax核心需求)
        center_loss_src, variance_loss_src = self.compute_center_loss(proj_source, labels_source)
        
        # 2. 类间分离损失
        separation_loss_src = self.compute_separation_loss(proj_source, labels_source)
        
        # 3. Triplet Loss (增强判别边界)
        triplet_loss_src = self.compute_triplet_loss(proj_source, labels_source)
        
        # 组合源域损失
        intra_loss_src = center_loss_src + self.variance_weight * variance_loss_src + self.triplet_weight * triplet_loss_src
        inter_loss_src = separation_loss_src  # separation_loss已经是正值
        
        # ===== 目标域损失 =====
        center_loss_tgt, variance_loss_tgt = self.compute_center_loss(proj_target, labels_target)
        separation_loss_tgt = self.compute_separation_loss(proj_target, labels_target)
        triplet_loss_tgt = self.compute_triplet_loss(proj_target, labels_target)
        
        intra_loss_tgt = center_loss_tgt + self.variance_weight * variance_loss_tgt + self.triplet_weight * triplet_loss_tgt
        inter_loss_tgt = separation_loss_tgt
        
        # ===== 总损失 =====
        total_intra_loss = (intra_loss_src + intra_loss_tgt) / 2.0
        total_inter_loss = (inter_loss_src + inter_loss_tgt) / 2.0
        
        # 总损失 = 类内紧凑性（越小越好） + 类间分离性（Hinge loss，越小说明分离度越好）
        total_loss = self.intra_weight * total_intra_loss + self.inter_weight * total_inter_loss
        
        # 数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 构建损失字典
        loss_dict = {
            'intra_loss_src': intra_loss_src.item() if isinstance(intra_loss_src, torch.Tensor) else intra_loss_src,
            'intra_loss_tgt': intra_loss_tgt.item() if isinstance(intra_loss_tgt, torch.Tensor) else intra_loss_tgt,
            'inter_loss_src': inter_loss_src.item() if isinstance(inter_loss_src, torch.Tensor) else inter_loss_src,
            'inter_loss_tgt': inter_loss_tgt.item() if isinstance(inter_loss_tgt, torch.Tensor) else inter_loss_tgt,
            'total_intra_loss': total_intra_loss.item() if isinstance(total_intra_loss, torch.Tensor) else total_intra_loss,
            'total_inter_loss': total_inter_loss.item() if isinstance(total_inter_loss, torch.Tensor) else total_inter_loss,
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        }
        
        # 记录损失历史
        for key, value in loss_dict.items():
            self.loss_history[key].append(value)
        
        # 打印损失
        if verbose:
            print(f'    [损失] '
                  f'类内:源{loss_dict["intra_loss_src"]:.4f}+目{loss_dict["intra_loss_tgt"]:.4f}=总{loss_dict["total_intra_loss"]:.4f} | '
                  f'类间:源{loss_dict["inter_loss_src"]:.4f}+目{loss_dict["inter_loss_tgt"]:.4f}=总{loss_dict["total_inter_loss"]:.4f} | '
                  f'总:{loss_dict["total_loss"]:.4f}')
        
        return total_loss, loss_dict
    
    def get_loss_history(self):
        """获取损失历史"""
        return self.loss_history
    
    def reset_history(self):
        """重置损失历史"""
        for key in self.loss_history:
            self.loss_history[key] = []


def compute_manifold_loss(proj_source, labels_source, proj_target, labels_target):
    """
    兼容性函数
    """
    loss_fn = ManifoldLoss(intra_weight=1.0, inter_weight=0.5, margin=0.5)
    total_loss, _ = loss_fn(proj_source, labels_source, proj_target, labels_target, verbose=False)
    return total_loss