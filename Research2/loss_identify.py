import torch


class ManifoldLoss:
    """
    流形优化损失集合 - 面向OpenMax入侵检测的32维流形空间优化
    
    设计目标:
    1. 类内紧凑性: 同一身份的特征聚集在一起（OpenMax需要紧凑分布）
    2. 类间分离性: 不同身份之间相互远离（避免混淆）
    3. 自监督方式: 不依赖身份分类损失（预训练模型已保证身份识别）
    """
    
    def __init__(self, intra_weight=1.0, inter_weight=0.3):
        """
        Args:
            intra_weight: 类内紧凑性损失权重 (默认1.0)
            inter_weight: 类间分离性损失权重 (默认0.3)
        """
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
        
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
    
    def compute_intra_class_loss(self, proj_features, labels, domain_name=''):
        """
        计算类内紧凑性损失
        
        Args:
            proj_features: [batch, 32] 投影特征
            labels: [batch] 标签
            domain_name: 域名称 (用于打印)
        
        Returns:
            intra_loss: 类内紧凑性损失
        """
        device = proj_features.device
        intra_loss = 0.0
        unique_labels = torch.unique(labels)
        
        for c in unique_labels:
            mask = (labels == c)
            if mask.sum() > 1:
                class_features = proj_features[mask]
                center = class_features.mean(dim=0, keepdim=True)
                # 类内方差（越小越紧凑）
                intra_loss += ((class_features - center) ** 2).sum() / mask.sum()
        
        intra_loss = intra_loss / len(unique_labels) if len(unique_labels) > 0 else 0.0
        
        # 数值稳定性检查
        if torch.isnan(intra_loss) or torch.isinf(intra_loss):
            intra_loss = torch.tensor(0.0, device=device)
        
        return intra_loss
    
    def compute_inter_class_loss(self, proj_features, labels, domain_name=''):
        """
        计算类间分离性损失
        
        Args:
            proj_features: [batch, 32] 投影特征
            labels: [batch] 标签
            domain_name: 域名称 (用于打印)
        
        Returns:
            inter_loss: 类间分离性损失 (负值，最小化负值=最大化距离)
        """
        device = proj_features.device
        
        # 计算所有类中心
        centers = []
        unique_labels = torch.unique(labels)
        for c in unique_labels:
            mask = (labels == c)
            if mask.sum() > 0:
                centers.append(proj_features[mask].mean(dim=0))
        
        # 需要至少两个类才能计算类间距离
        if len(centers) <= 1:
            return torch.tensor(0.0, device=device)
        
        # 计算类中心之间的距离
        centers_tensor = torch.stack(centers)  # [num_classes, 32]
        distances = torch.cdist(centers_tensor, centers_tensor)  # [num_classes, num_classes]
        
        # 只考虑非对角线元素（不同类之间）
        mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=device)
        inter_distances = distances[mask]
        
        # 负号：最小化负距离 = 最大化距离
        inter_loss = -inter_distances.mean()
        
        # 数值稳定性检查
        if torch.isnan(inter_loss) or torch.isinf(inter_loss):
            inter_loss = torch.tensor(0.0, device=device)
        
        return inter_loss
    
    def __call__(self, proj_source, labels_source, proj_target, labels_target, verbose=True):
        """
        计算总损失
        
        Args:
            proj_source: 源域32维投影特征 [batch, 32]
            labels_source: 源域标签 [batch]
            proj_target: 目标域32维投影特征 [batch, 32]
            labels_target: 目标域标签 [batch]
            verbose: 是否打印损失详情
        
        Returns:
            total_loss: 总损失
            loss_dict: 损失字典 (用于记录)
        """
        device = proj_source.device
        
        # 1. 源域类内紧凑性损失
        intra_loss_src = self.compute_intra_class_loss(proj_source, labels_source, '源域')
        
        # 2. 目标域类内紧凑性损失
        intra_loss_tgt = self.compute_intra_class_loss(proj_target, labels_target, '目标域')
        
        # 3. 源域类间分离性损失
        inter_loss_src = self.compute_inter_class_loss(proj_source, labels_source, '源域')
        
        # 4. 目标域类间分离性损失
        inter_loss_tgt = self.compute_inter_class_loss(proj_target, labels_target, '目标域')
        
        # 5. 总类内损失
        total_intra_loss = (intra_loss_src + intra_loss_tgt) / 2.0
        
        # 6. 总类间损失
        total_inter_loss = (inter_loss_src + inter_loss_tgt) / 2.0
        
        # 7. 总损失 = 类内紧凑性 * 权重 + 类间分离性 * 权重
        total_loss = self.intra_weight * total_intra_loss + self.inter_weight * total_inter_loss
        
        # 数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=device)
        
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
        
        # 打印损失 (简洁版，根据用户偏好)
        if verbose:
            print(f'    [损失] '
                  f'类内:源{loss_dict["intra_loss_src"]:.6f}+目{loss_dict["intra_loss_tgt"]:.6f}=总{loss_dict["total_intra_loss"]:.6f} | '
                  f'类间:源{loss_dict["inter_loss_src"]:.6f}+目{loss_dict["inter_loss_tgt"]:.6f}=总{loss_dict["total_inter_loss"]:.6f} | '
                  f'总:{loss_dict["total_loss"]:.6f}')
        
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
    兼容性函数 - 保持与train_identify.py中的调用兼容
    直接调用ManifoldLoss类
    """
    loss_fn = ManifoldLoss(intra_weight=1.0, inter_weight=0.3)
    total_loss, _ = loss_fn(proj_source, labels_source, proj_target, labels_target, verbose=False)
    return total_loss