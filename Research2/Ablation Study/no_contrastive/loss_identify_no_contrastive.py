import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class IdentifyDetectionLossNoContrastive(nn.Module):
    def __init__(self, alpha=0.8):  # 适度降低alpha值
        """
        初始化损失函数（无对比损失版本）
        Args:
            alpha: 身份分类损失权重（适度降低以突出对比损失的作用）
        """
        super(IdentifyDetectionLossNoContrastive, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)  # 恢复标签平滑
        
    def identity_classification_loss(self, logits, labels):
        """身份分类损失"""
        # 添加数值稳定性检查
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.ce_loss(logits, labels)
    
    def forward(self, outputs, labels_source, labels_target=None):
        """
        计算总损失（无对比损失版本）
        Args:
            outputs: 模型输出字典
            labels_source: 源域标签
            labels_target: 目标域标签（可选）
        """
        # 检查必要输出是否存在
        if 'logits_source' not in outputs:
            return torch.tensor(0.0, device=labels_source.device), {
                'identity_loss': 0.0,
                'total_loss': 0.0
            }
        
        # 身份分类损失
        identity_loss = self.identity_classification_loss(
            outputs['logits_source'], labels_source)
        
        # 检查是否有NaN或inf值并处理
        if torch.isnan(identity_loss) or torch.isinf(identity_loss):
            identity_loss = torch.tensor(0.0, device=identity_loss.device)
        
        total_loss = self.alpha * identity_loss
            
        # 确保总损失不是NaN或inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=total_loss.device)
            
        # 限制总损失范围以防止梯度爆炸
        total_loss = torch.clamp(total_loss, min=0.0, max=100.0)
            
        return total_loss, {
            'identity_loss': identity_loss.item() if not (torch.isnan(identity_loss) or torch.isinf(identity_loss)) else 0.0,
            'total_loss': total_loss.item() if not (torch.isnan(total_loss) or torch.isinf(total_loss)) else 0.0
        }