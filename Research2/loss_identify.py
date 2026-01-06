import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentifyDetectionLoss(nn.Module):
    def __init__(self, alpha=1.0, delta=0.1):
        """
        简化的损失函数 - 基于第三章预训练模型
        第三章已学会跨域对齐,第四章只需关注:
        1. 身份分类 (identity_loss)
        2. 流形紧凑性 (manifold_compactness_loss)
        
        Args:
            alpha: 身份分类损失权重
            delta: 流形紧凑性损失权重
        """
        super(IdentifyDetectionLoss, self).__init__()
        self.alpha = alpha
        self.delta = delta
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def identity_classification_loss(self, logits, labels):
        """身份分类损失 - 确保跨域身份识别准确性"""
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.ce_loss(logits, labels)
    
    def manifold_compactness_loss(self, proj_embeddings, labels):
        """
        流形紧凑性损失 - 约束32维流形空间中同一身份的特征更紧凑
        这是为后续入侵检测准备的关键步骤
        """
        if proj_embeddings is None or labels is None:
            return torch.tensor(0.0, device=proj_embeddings.device if proj_embeddings is not None else None)

        # 数值稳定性处理
        if torch.isnan(proj_embeddings).any() or torch.isinf(proj_embeddings).any():
            proj_embeddings = torch.nan_to_num(proj_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)

        unique_labels = torch.unique(labels)
        if unique_labels.numel() == 0:
            return torch.tensor(0.0, device=proj_embeddings.device)

        total_loss = torch.tensor(0.0, device=proj_embeddings.device)
        valid_classes = 0

        for c in unique_labels:
            mask = (labels == c)
            class_embeddings = proj_embeddings[mask]
            if class_embeddings.size(0) < 2:
                continue
            
            # 计算类中心
            center = class_embeddings.mean(dim=0, keepdim=True)
            # L2距离
            class_loss = ((class_embeddings - center) ** 2).sum(dim=1).mean()
            
            if torch.isnan(class_loss) or torch.isinf(class_loss):
                continue
            total_loss = total_loss + class_loss
            valid_classes += 1

        if valid_classes == 0:
            return torch.tensor(0.0, device=proj_embeddings.device)

        total_loss = total_loss / valid_classes
        total_loss = torch.clamp(total_loss, min=0.0, max=10.0)

        return total_loss
    
    def forward(self, outputs, labels_source, labels_target=None):
        """
        计算总损失 - 简化版(基于预训练模型)
        
        Args:
            outputs: 模型输出字典
            labels_source: 源域标签
            labels_target: 目标域标签(可选)
        """
        # 检查必要输出是否存在
        if 'logits_source' not in outputs:
            return torch.tensor(0.0, device=labels_source.device), {
                'identity_loss': 0.0,
                'manifold_loss': 0.0,
                'total_loss': 0.0
            }
        
        # 1. 身份分类损失 (源域)
        identity_loss_src = self.identity_classification_loss(
            outputs['logits_source'], labels_source)
        
        # 2. 目标域身份分类损失 (如果有)
        identity_loss_tgt = torch.tensor(0.0, device=identity_loss_src.device)
        if labels_target is not None and 'logits_target' in outputs:
            identity_loss_tgt = self.identity_classification_loss(
                outputs['logits_target'], labels_target)
        
        # 平均身份损失
        identity_loss = (identity_loss_src + identity_loss_tgt) / 2 if labels_target is not None else identity_loss_src
        
        # 3. 流形紧凑性损失 (32维投影空间)
        manifold_loss = torch.tensor(0.0, device=identity_loss.device)
        
        # 源域流形紧凑性
        if 'proj_source' in outputs and outputs['proj_source'] is not None:
            manifold_loss = manifold_loss + self.manifold_compactness_loss(
                outputs['proj_source'], labels_source)
        
        # 目标域流形紧凑性
        if (labels_target is not None and 
            'proj_target' in outputs and 
            outputs['proj_target'] is not None):
            manifold_loss = manifold_loss + self.manifold_compactness_loss(
                outputs['proj_target'], labels_target)
            manifold_loss = manifold_loss / 2  # 平均
        
        # 数值稳定性检查
        if torch.isnan(identity_loss) or torch.isinf(identity_loss):
            identity_loss = torch.tensor(0.0, device=identity_loss.device)
        if torch.isnan(manifold_loss) or torch.isinf(manifold_loss):
            manifold_loss = torch.tensor(0.0, device=manifold_loss.device)
        
        # 总损失 = 身份分类 + 流形紧凑性
        total_loss = self.alpha * identity_loss + self.delta * manifold_loss
        
        # 限制损失范围
        total_loss = torch.clamp(total_loss, min=0.0, max=100.0)
            
        return total_loss, {
            'identity_loss': identity_loss.item() if not (torch.isnan(identity_loss) or torch.isinf(identity_loss)) else 0.0,
            'manifold_loss': manifold_loss.item() if not (torch.isnan(manifold_loss) or torch.isinf(manifold_loss)) else 0.0,
            'total_loss': total_loss.item() if not (torch.isnan(total_loss) or torch.isinf(total_loss)) else 0.0
        }