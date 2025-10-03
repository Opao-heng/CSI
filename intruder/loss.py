import torch
import torch.nn as nn
import torch.nn.functional as F


class IntruderDetectionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5):
        """
        初始化损失函数
        Args:
            alpha: 身份分类损失权重
            beta: 域适应损失权重  
            gamma: 对比损失权重
        """
        super(IntruderDetectionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
        
    def identity_classification_loss(self, logits, labels):
        """身份分类损失"""
        return self.ce_loss(logits, labels)
    
    def domain_adaptation_loss(self, features_source, features_target):
        """域适应损失 - 最小化源域和目标域特征的MMD距离"""
        # 计算源域和目标域特征的均值
        mean_source = torch.mean(features_source, dim=0)
        mean_target = torch.mean(features_target, dim=0)
        
        # 计算MMD距离（简化版本）
        mmd_loss = torch.mean((mean_source - mean_target) ** 2)
        return mmd_loss
    
    def contrastive_loss(self, features_source, features_target, labels_source, labels_target, temperature=0.1):
        """
        对比损失 - 拉近相同用户在不同域的特征距离
        """
        # 计算所有源域和目标域特征之间的相似度
        similarity_matrix = torch.matmul(features_source, features_target.T) / temperature
        
        # 创建正样本标签矩阵
        labels_source_expanded = labels_source.unsqueeze(1)  # [batch_size, 1]
        labels_target_expanded = labels_target.unsqueeze(0)  # [1, batch_size]
        positive_mask = (labels_source_expanded == labels_target_expanded).float()  # [batch_size, batch_size]
        
        # 计算对比损失
        # 对于每个源域样本，正样本是目标域中相同标签的样本
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        
        # 只考虑正样本
        positive_exp_sim = exp_sim * positive_mask
        sum_positive_exp_sim = torch.sum(positive_exp_sim, dim=1, keepdim=True)
        
        # 避免除零错误
        eps = 1e-8
        loss = -torch.log((sum_positive_exp_sim + eps) / (sum_exp_sim + eps))
        
        return torch.mean(loss)
    
    def forward(self, outputs, labels_source, labels_target=None):
        """
        计算总损失
        Args:
            outputs: 模型输出字典
            labels_source: 源域标签
            labels_target: 目标域标签（可选）
        """
        # 身份分类损失
        identity_loss = self.identity_classification_loss(
            outputs['logits_source'], labels_source)
        
        # 初始化域适应和对比损失
        domain_loss = torch.tensor(0.0, device=identity_loss.device)
        contrastive_loss = torch.tensor(0.0, device=identity_loss.device)
        
        total_loss = self.alpha * identity_loss
        
        # 如果提供了目标域数据，则计算域适应和对比损失
        if labels_target is not None:
            # 域适应损失
            domain_loss = self.domain_adaptation_loss(
                outputs['features_source'], outputs['features_target'])
            
            # 对比损失
            contrastive_loss = self.contrastive_loss(
                outputs['features_source'], outputs['features_target'],
                labels_source, labels_target)
            
            total_loss += self.beta * domain_loss + self.gamma * contrastive_loss
            
        return total_loss, {
            'identity_loss': identity_loss.item(),
            'domain_loss': domain_loss.item() if labels_target is not None else 0,
            'contrastive_loss': contrastive_loss.item() if labels_target is not None else 0,
            'total_loss': total_loss.item()
        }