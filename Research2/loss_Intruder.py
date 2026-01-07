import torch
import torch.nn.functional as F


class IntruderDetectionLoss:
    """
    入侵者检测损失函数 - 开放集学习策略 (增强版 - 支持伪入侵者训练)
    
    核心思想 (增强版):
    1. 合法用户：距离小 -> 异常分数应该低
    2. 伪入侵者：随机噪声 -> 异常分数应该高
    通过对比学习，让网络明确学习正常 vs 异常的边界
    
    技术实现:
    1. 合法用户：MSE损失，让异常分数与归一化距离对齐
    2. 伪入侵者：BCE损失，强制输出高分数(标签=1)
    3. 对比约束：拉大合法用户和伪入侵者的分数差距
    """
    
    def __init__(self, distance_weight=0.5, margin=1.0, pseudo_weight=1.0):
        """
        初始化损失函数
        
        Args:
            distance_weight: 距离对齐项的权重
            margin: 距离阈值，超过此值的样本应被判定为异常
            pseudo_weight: 伪入侵者损失权重
        """
        self.distance_weight = distance_weight
        self.margin = margin
        self.pseudo_weight = pseudo_weight
    
    def __call__(self, anomaly_scores, min_distances=None, pseudo_scores=None):
        """
        计算入侵者检测损失 (增强版 - 支持伪入侵者)
        
        Args:
            anomaly_scores: 合法用户的异常分数 (logits), shape: (batch_size,)
            min_distances: 合法用户到最近类中心的距离, shape: (batch_size,)
            pseudo_scores: 伪入侵者的异常分数 (logits), shape: (pseudo_batch_size,) 或 None
        
        Returns:
            loss: 总损失值
        """
        # === 1. 合法用户损失：用距离作为监督信号 ===
        if min_distances is None:
            # 如果没有提供距离，回退到原始BCE损失（合法用户目标=0）
            legal_loss = F.binary_cross_entropy_with_logits(
                anomaly_scores, 
                torch.zeros_like(anomaly_scores)
            )
        else:
            # 归一化距离到[0, 1]区间
            normalized_distances = torch.clamp(min_distances / self.margin, 0, 1)
            
            # 目标：让sigmoid(anomaly_scores) ≈ normalized_distances
            anomaly_probs = torch.sigmoid(anomaly_scores)
            distance_alignment_loss = F.mse_loss(anomaly_probs, normalized_distances)
            
            # 正则化项：鼓励合法用户（小距离）的分数接近0
            legal_mask = (min_distances < self.margin * 0.5)
            if legal_mask.sum() > 0:
                legal_scores = anomaly_scores[legal_mask]
                legal_regularization = F.mse_loss(
                    torch.sigmoid(legal_scores),
                    torch.zeros_like(legal_scores)
                )
            else:
                legal_regularization = torch.tensor(0.0, device=anomaly_scores.device)
            
            legal_loss = distance_alignment_loss + self.distance_weight * legal_regularization
        
        # === 2. 伪入侵者损失：强制输出高分数(标签=1) ===
        if pseudo_scores is not None:
            # BCE损失：伪入侵者的目标标签为1
            pseudo_loss = F.binary_cross_entropy_with_logits(
                pseudo_scores,
                torch.ones_like(pseudo_scores)
            )
            
            # === 3. 对比约束：拉大合法用户和伪入侵者的分数差距 ===
            # 计算平均分数
            legal_mean_score = torch.sigmoid(anomaly_scores).mean()
            pseudo_mean_score = torch.sigmoid(pseudo_scores).mean()
            
            # Hinge loss：确保 pseudo_mean_score > legal_mean_score + margin
            contrastive_margin = 0.3  # 期望伪入侵者比合法用户高0.3
            contrastive_loss = F.relu(contrastive_margin - (pseudo_mean_score - legal_mean_score))
            
            # 总损失
            total_loss = legal_loss + self.pseudo_weight * pseudo_loss + 0.2 * contrastive_loss
        else:
            total_loss = legal_loss
        
        return total_loss


def compute_intruder_loss(anomaly_scores, min_distances=None):
    """
    便捷函数:计算入侵者检测损失 (修复版)
    
    Args:
        anomaly_scores: 模型输出的异常分数 (logits)
        min_distances: 样本到最近类中心的距离
    
    Returns:
        loss: 损失值
    """
    loss_fn = IntruderDetectionLoss()
    return loss_fn(anomaly_scores, min_distances)
