import torch
import torch.nn.functional as F


class ManifoldLoss:
    """
    流形优化损失 - 面向32维流形空间优化
    
    核心目标:
    1. 类内紧凑: 样本靠近类中心 (欧氏距离)
    2. 类间分离: 不同类中心相互远离
    """
    
    def __init__(self, intra_weight=1.0, inter_weight=0.3):
        """
        Args:
            intra_weight: 类内紧凑性权重
            inter_weight: 类间分离性权重
        """
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
    
    def compute_intra_loss(self, proj_features, labels):
        """
        类内紧凑性: 最小化样本到类中心的欧氏距离
        """
        device = proj_features.device
        
        if proj_features.size(0) == 0:
            return torch.tensor(0.0, device=device)
        
        # 计算每个类的中心和类内距离
        unique_labels = torch.unique(labels)
        intra_loss = torch.tensor(0.0, device=device)
        
        for c in unique_labels:
            mask = (labels == c)
            if mask.sum() > 1:
                class_features = proj_features[mask]
                center = class_features.mean(dim=0)
                
                # 欧氏距离到中心
                distances = torch.norm(class_features - center.unsqueeze(0), dim=1)
                intra_loss += distances.mean()
        
        num_classes = len(unique_labels)
        if num_classes > 0:
            intra_loss = intra_loss / num_classes
        
        return intra_loss
    
    def compute_inter_loss(self, proj_features, labels):
        """
        类间分离性: 负的类中心平均距离 (最小化负距离 = 最大化距离)
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
        
        centers_tensor = torch.stack(centers)
        
        # 计算类中心之间的欧氏距离
        distances = torch.cdist(centers_tensor, centers_tensor)
        
        # 提取上三角距离
        mask = ~torch.eye(len(centers), dtype=torch.bool, device=device)
        inter_distances = distances[mask]
        
        # 负距离 (最小化 = 最大化距离)
        inter_loss = -inter_distances.mean()
        
        return inter_loss
    

    
    def __call__(self, proj_source, labels_source, proj_target, labels_target):
        """
        计算总损失: 类内紧凑 + 类间分离
        
        Returns:
            total_loss: 总损失
            loss_dict: 详细损失字典
        """
        device = proj_source.device
        
        # 源域损失
        intra_src = self.compute_intra_loss(proj_source, labels_source)
        inter_src = self.compute_inter_loss(proj_source, labels_source)
        
        # 目标域损失
        intra_tgt = self.compute_intra_loss(proj_target, labels_target)
        inter_tgt = self.compute_inter_loss(proj_target, labels_target)
        
        # 总损失
        total_intra = (intra_src + intra_tgt) / 2.0
        total_inter = (inter_src + inter_tgt) / 2.0
        total_loss = self.intra_weight * total_intra + self.inter_weight * total_inter
        
        # 数值稳定性
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 损失字典
        loss_dict = {
            'intra_src': intra_src.item(),
            'intra_tgt': intra_tgt.item(),
            'inter_src': inter_src.item(),
            'inter_tgt': inter_tgt.item(),
            'total_intra': total_intra.item(),
            'total_inter': total_inter.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


def compute_manifold_loss(proj_source, labels_source, proj_target, labels_target):
    """
    兼容性函数
    """
    loss_fn = ManifoldLoss(intra_weight=1.0, inter_weight=0.5, margin=0.5)
    total_loss, _ = loss_fn(proj_source, labels_source, proj_target, labels_target, verbose=False)
    return total_loss