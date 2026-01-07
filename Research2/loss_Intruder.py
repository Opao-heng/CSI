import torch
import torch.nn.functional as F


class WeightedIntruderLoss:
    """
    加权BCE损失 + L2正则化 (用于训练综合入侵者检测器)
    """
    def __init__(self, pos_weight, l2_weight=1e-4):
        self.pos_weight = pos_weight
        self.l2_weight = l2_weight
        
    def __call__(self, logits, labels, model):
        # 1. 计算带权重的BCE损失
        classification_loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight
        )
        
        # 2. 添加L2正则化
        l2_reg = torch.tensor(0., device=logits.device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
            
        return classification_loss + self.l2_weight * l2_reg
