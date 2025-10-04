import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class IntruderDetectionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05):
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
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
        
    def identity_classification_loss(self, logits, labels):
        """身份分类损失"""
        # 添加数值稳定性检查
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.ce_loss(logits, labels)
    
    def domain_adaptation_loss(self, features_source, features_target):
        """域适应损失 - 改进的MMD距离计算"""
        # 添加数值稳定性检查
        if torch.isnan(features_source).any() or torch.isinf(features_source).any():
            features_source = torch.nan_to_num(features_source, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(features_target).any() or torch.isinf(features_target).any():
            features_target = torch.nan_to_num(features_target, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 对特征进行L2归一化以提高数值稳定性
        features_source = F.normalize(features_source, p=2, dim=1)
        features_target = F.normalize(features_target, p=2, dim=1)
        
        # 计算源域和目标域特征的均值
        mean_source = torch.mean(features_source, dim=0)
        mean_target = torch.mean(features_target, dim=0)
        
        # 计算MMD距离（简化版本）并添加数值稳定性
        mmd_loss = torch.mean((mean_source - mean_target) ** 2)
        
        # 添加额外的分布对齐损失
        # 计算特征的二阶矩（方差）
        var_source = torch.var(features_source, dim=0)
        var_target = torch.var(features_target, dim=0)
        var_loss = torch.mean((var_source - var_target) ** 2)
        
        # 综合一阶和二阶矩损失
        total_mmd_loss = mmd_loss + 0.1 * var_loss
        
        return torch.clamp(total_mmd_loss, min=0.0, max=10.0)  # 限制损失范围
    
    def contrastive_loss(self, features_source, features_target, labels_source, labels_target, temperature=0.1):
        """
        对比损失 - 改进的InfoNCE损失，提高数值稳定性
        """
        batch_size = features_source.size(0)
        
        # 添加数值稳定性检查
        if torch.isnan(features_source).any() or torch.isinf(features_source).any():
            features_source = torch.nan_to_num(features_source, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(features_target).any() or torch.isinf(features_target).any():
            features_target = torch.nan_to_num(features_target, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 对特征进行L2归一化以提高数值稳定性
        features_source = F.normalize(features_source, p=2, dim=1)
        features_target = F.normalize(features_target, p=2, dim=1)
        
        # 计算所有源域和目标域特征之间的相似度 (cosine similarity)
        # 使用矩阵乘法计算相似度矩阵，提高效率
        similarity_matrix = torch.matmul(features_source, features_target.t()) / temperature
        
        # 创建正样本标签矩阵
        labels_source_expanded = labels_source.unsqueeze(1)  # [batch_size, 1]
        labels_target_expanded = labels_target.unsqueeze(0)  # [1, batch_size]
        positive_mask = (labels_source_expanded == labels_target_expanded).float()  # [batch_size, batch_size]
        
        # 数值稳定性的InfoNCE损失实现
        # 对每一行进行数值稳定的log_softmax操作
        # 添加一个小的epsilon值以防止数值问题
        eps = 1e-8
        logits = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0]  # 数值稳定
        logsumexp_logits = torch.logsumexp(logits, dim=1, keepdim=True)
        log_probs = logits - logsumexp_logits
        
        # 只考虑正样本的对数概率
        # 使用掩码选择正样本
        positive_log_probs = log_probs * positive_mask
        # 计算每个样本的正样本数量
        positive_counts = positive_mask.sum(dim=1) + eps  # 添加epsilon防止除零
        
        # 计算平均正样本对数概率
        avg_positive_log_probs = positive_log_probs.sum(dim=1) / positive_counts
        
        # 对比损失是负的平均正样本对数概率
        loss = -torch.mean(avg_positive_log_probs)
        
        # 处理NaN和inf值
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=features_source.device)
        
        # 限制损失范围
        loss = torch.clamp(loss, min=0.0, max=10.0)
        
        return loss
    
    def forward(self, outputs, labels_source, labels_target=None):
        """
        计算总损失
        Args:
            outputs: 模型输出字典
            labels_source: 源域标签
            labels_target: 目标域标签（可选）
        """
        # 检查必要输出是否存在
        if 'logits_source' not in outputs:
            return torch.tensor(0.0, device=labels_source.device), {
                'identity_loss': 0.0,
                'domain_loss': 0.0,
                'contrastive_loss': 0.0,
                'total_loss': 0.0
            }
        
        # 身份分类损失
        identity_loss = self.identity_classification_loss(
            outputs['logits_source'], labels_source)
        
        # 初始化域适应和对比损失
        domain_loss = torch.tensor(0.0, device=identity_loss.device)
        contrastive_loss = torch.tensor(0.0, device=identity_loss.device)
        
        # 检查是否有NaN或inf值并处理
        if torch.isnan(identity_loss) or torch.isinf(identity_loss):
            identity_loss = torch.tensor(0.0, device=identity_loss.device)
        
        total_loss = self.alpha * identity_loss
        
        # 如果提供了目标域数据，则计算域适应和对比损失
        if (labels_target is not None and 
            'features_source' in outputs and 
            'features_target' in outputs and
            outputs['features_source'] is not None and
            outputs['features_target'] is not None):
            
            # 确保特征维度匹配
            if (outputs['features_source'].size(0) == labels_source.size(0) and
                outputs['features_target'].size(0) == labels_target.size(0)):
                
                # 域适应损失
                domain_loss = self.domain_adaptation_loss(
                    outputs['features_source'], outputs['features_target'])
                
                # 对比损失
                contrastive_loss = self.contrastive_loss(
                    outputs['features_source'], outputs['features_target'],
                    labels_source, labels_target)
            
        # 检查是否有NaN或inf值并处理
        if torch.isnan(domain_loss) or torch.isinf(domain_loss):
            domain_loss = torch.tensor(0.0, device=domain_loss.device)
        if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
            contrastive_loss = torch.tensor(0.0, device=contrastive_loss.device)
        
        total_loss += self.beta * domain_loss + self.gamma * contrastive_loss
        
        # 确保总损失不是NaN或inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=total_loss.device)
            
        # 限制总损失范围以防止梯度爆炸
        total_loss = torch.clamp(total_loss, min=0.0, max=100.0)
            
        return total_loss, {
            'identity_loss': identity_loss.item() if not (torch.isnan(identity_loss) or torch.isinf(identity_loss)) else 0.0,
            'domain_loss': domain_loss.item() if not (torch.isnan(domain_loss) or torch.isinf(domain_loss)) else 0.0,
            'contrastive_loss': contrastive_loss.item() if not (torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss)) else 0.0,
            'total_loss': total_loss.item() if not (torch.isnan(total_loss) or torch.isinf(total_loss)) else 0.0
        }