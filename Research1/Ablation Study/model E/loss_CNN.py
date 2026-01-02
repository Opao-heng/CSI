# loss_CNN.py - Model E: 简化损失函数（移除MMD损失）
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction:
    """
    Model E的简化损失函数：
    只包含分类损失，移除了跨域特征对齐损失（MMD）
    """
    
    def __init__(self, num_classes=10, label_smoothing=0.1, focal_gamma=2.0):
        # 使用标签平滑减少过拟合
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma  # Focal Loss的gamma参数
        
    def focal_loss(self, pred, labels, gamma=2.0):
        """Focal Loss - 处理类别不平衡和难样本"""
        ce_loss = F.cross_entropy(pred, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma) * ce_loss
        return focal_loss.mean()
    
    def classification_loss(self, pred, labels):
        """
        分类损失 - 使用Focal Loss + 标签平滑的组合
        适用于源域和目标域
        """
        focal = self.focal_loss(pred, labels, self.focal_gamma)
        ce = self.ce_loss(pred, labels)
        return 0.7 * focal + 0.3 * ce  # 混合使用
