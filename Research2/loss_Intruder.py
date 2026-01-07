import torch
import torch.nn.functional as F


class IntruderDetectionLoss:
    """
    入侵者检测损失函数 - 开放集学习策略
    
    核心思想：
    训练集只有合法用户，目标是让合法用户的异常分数趋近于0
    推理时，异常分数高的样本被判定为入侵者
    """
    
    def __init__(self):
        """
        初始化损失函数
        """
        pass
    
    def __call__(self, anomaly_scores, is_legal_user=True):
        """
        计算入侵者检测损失
        
        Args:
            anomaly_scores: 模型输出的异常分数 (logits)，shape: (batch_size,)
            is_legal_user: 是否为合法用户，默认为True（训练集只有合法用户）
        
        Returns:
            loss: BCE损失值
        """
        # 合法用户的目标异常分数为0（sigmoid后 < 0.5）
        if is_legal_user:
            target = torch.zeros_like(anomaly_scores)
        else:
            # 入侵者的目标异常分数为1（sigmoid后 > 0.5）
            target = torch.ones_like(anomaly_scores)
        
        # 使用BCE损失：让合法用户的异常分数趋近于0
        loss = F.binary_cross_entropy_with_logits(anomaly_scores, target)
        
        return loss


def compute_intruder_loss(anomaly_scores, is_legal_user=True):
    """
    便捷函数：计算入侵者检测损失
    
    Args:
        anomaly_scores: 模型输出的异常分数 (logits)
        is_legal_user: 是否为合法用户
    
    Returns:
        loss: 损失值
    """
    loss_fn = IntruderDetectionLoss()
    return loss_fn(anomaly_scores, is_legal_user)
