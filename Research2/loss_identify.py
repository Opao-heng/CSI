import torch


def compute_manifold_loss(proj_source, labels_source, proj_target, labels_target):
    """
    流形紧凑性损失 - 面向OpenMax入侵检测的32维流形空间优化
    
    设计目标:
    1. 类内紧凑性: 同一身份的特征应该聚集在一起（越紧凑，OpenMax建模越准确）
    2. 自监督方式: 不依赖身份分类损失（预训练模型已保证身份识别）
    
    Args:
        proj_source: 源域32维投影特征 [batch, 32]
        labels_source: 源域标签 [batch]
        proj_target: 目标域32维投影特征 [batch, 32]
        labels_target: 目标域标签 [batch]
    
    Returns:
        loss: 流形紧凑性损失（越小表示类内越紧凑）
    """
    device = proj_source.device
    
    # 1. 源域类内紧凑性
    intra_loss_src = 0.0
    unique_labels = torch.unique(labels_source)
    for c in unique_labels:
        mask = (labels_source == c)
        if mask.sum() > 1:
            class_features = proj_source[mask]
            # 计算类中心
            center = class_features.mean(dim=0, keepdim=True)
            # 类内方差（越小越紧凑）
            intra_loss_src += ((class_features - center) ** 2).sum() / mask.sum()
    intra_loss_src = intra_loss_src / len(unique_labels) if len(unique_labels) > 0 else 0.0
    
    # 2. 目标域类内紧凑性
    intra_loss_tgt = 0.0
    unique_labels = torch.unique(labels_target)
    for c in unique_labels:
        mask = (labels_target == c)
        if mask.sum() > 1:
            class_features = proj_target[mask]
            center = class_features.mean(dim=0, keepdim=True)
            intra_loss_tgt += ((class_features - center) ** 2).sum() / mask.sum()
    intra_loss_tgt = intra_loss_tgt / len(unique_labels) if len(unique_labels) > 0 else 0.0
    
    # 总损失: 源域和目标域类内紧凑性的平均
    loss = (intra_loss_src + intra_loss_tgt) / 2.0
    
    # 数值稳定性检查
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=device)
    
    return loss