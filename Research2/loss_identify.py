import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentifyDetectionLoss(nn.Module):
    def __init__(self, alpha=1.0, delta=0.1, beta=0.05):
        """
        简化的损失函数 - 基于第三章预训练模型
        第三章已学会跨域对齐,第四章关注:
        1. 身份分类 (identity_loss)（源域和目标域）
        2. 流形紧凑性 (manifold_compactness_loss)
        3. Center-aware约束 (center_aware_loss)
        
        Args:
            alpha: 身份分类损失权重
            delta: 流形紧凑性损失权重
            beta: Center-aware损失权重
        """
        super(IdentifyDetectionLoss, self).__init__()
        self.alpha = alpha
        self.delta = delta
        self.beta = beta
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
    
    def center_aware_loss(self, proj_embeddings, labels, class_centers):
        """
        Center-aware损失 - 为OpenMax异常检测优化类中心结构
        
        目标:
        1. 类内紧凑性: 样本靠近其真实类中心
        2. 类间分离性: 不同类中心相互分离
        
        这是第四章的核心创新:显式建模类中心,优化欧式距离度量
        """
        if proj_embeddings is None or labels is None or class_centers is None:
            return torch.tensor(0.0, device=proj_embeddings.device if proj_embeddings is not None else None)
        
        # 1. 类内紧凑性: 样本到其真实类中心的距离
        batch_centers = class_centers[labels]  # [B, 32]
        intra_class_dist = torch.sum((proj_embeddings - batch_centers) ** 2, dim=1).mean()
        
        # 2. 类间分离性: 不同类中心的距离
        center_distances = torch.cdist(class_centers, class_centers)  # [10, 10]
        # 去除对角线(自己到自己的距离)
        mask = ~torch.eye(center_distances.size(0), dtype=bool, device=center_distances.device)
        inter_class_dist = center_distances[mask].mean()
        
        # 最小化类内距离,同时最大化类间距离
        # 使用较小的系数避免过度约束
        loss = intra_class_dist - 0.05 * inter_class_dist
        
        # 数值稳定性
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=proj_embeddings.device)
        
        return torch.clamp(loss, min=0.0, max=10.0)
    
    def forward(self, outputs, labels_source, labels_target=None):
        """
        计算总损失 - 四个独立损失组件
        
        Args:
            outputs: 模型输出字典
            labels_source: 源域标签
            labels_target: 目标域标签(必需)
        
        返回四个损失:
        1. identity_loss_src: 源域身份分类损失
        2. identity_loss_tgt: 目标域身份分类损失
        3. manifold_loss: 流形紧凑性损失(源域+目标域)
        4. center_loss: Center-aware损失(在train中单独计算)
        """
        # 检查必要输出是否存在
        if 'logits_source' not in outputs:
            return torch.tensor(0.0, device=labels_source.device), {
                'identity_loss_src': 0.0,
                'identity_loss_tgt': 0.0,
                'manifold_loss': 0.0,
                'center_loss': 0.0,
                'total_loss': 0.0
            }
        
        # 1. 源域身份分类损失
        identity_loss_src = self.identity_classification_loss(
            outputs['logits_source'], labels_source)
        
        # 2. 目标域身份分类损失
        identity_loss_tgt = torch.tensor(0.0, device=identity_loss_src.device)
        if labels_target is not None and 'logits_target' in outputs:
            identity_loss_tgt = self.identity_classification_loss(
                outputs['logits_target'], labels_target)
        
        # 3. 流形紧凑性损失 (源域+目标域)
        manifold_loss_src = torch.tensor(0.0, device=identity_loss_src.device)
        manifold_loss_tgt = torch.tensor(0.0, device=identity_loss_src.device)
        
        # 源域流形紧凑性
        if 'proj_source' in outputs and outputs['proj_source'] is not None:
            manifold_loss_src = self.manifold_compactness_loss(
                outputs['proj_source'], labels_source)
        
        # 目标域流形紧凑性
        if (labels_target is not None and 
            'proj_target' in outputs and 
            outputs['proj_target'] is not None):
            manifold_loss_tgt = self.manifold_compactness_loss(
                outputs['proj_target'], labels_target)
        
        # 合并流形损失
        manifold_loss = (manifold_loss_src + manifold_loss_tgt) / 2
        
        # 4. Center-aware损失占位(在train_identify.py中单独计算)
        center_loss = torch.tensor(0.0, device=identity_loss_src.device)
        
        # 数值稳定性检查
        if torch.isnan(identity_loss_src) or torch.isinf(identity_loss_src):
            identity_loss_src = torch.tensor(0.0, device=identity_loss_src.device)
        if torch.isnan(identity_loss_tgt) or torch.isinf(identity_loss_tgt):
            identity_loss_tgt = torch.tensor(0.0, device=identity_loss_tgt.device)
        if torch.isnan(manifold_loss) or torch.isinf(manifold_loss):
            manifold_loss = torch.tensor(0.0, device=manifold_loss.device)
        
        # 总损失 = alpha*源域身份 + alpha*目标域身份 + delta*流形 + beta*center(后面加)
        # 注意: 两个身份损失都使用alpha权重,保持对称性
        total_loss = self.alpha * identity_loss_src + self.alpha * identity_loss_tgt + \
                     self.delta * manifold_loss + self.beta * center_loss
        
        # 限制损失范围
        total_loss = torch.clamp(total_loss, min=0.0, max=100.0)
            
        return total_loss, {
            'identity_loss_src': identity_loss_src.item() if not (torch.isnan(identity_loss_src) or torch.isinf(identity_loss_src)) else 0.0,
            'identity_loss_tgt': identity_loss_tgt.item() if not (torch.isnan(identity_loss_tgt) or torch.isinf(identity_loss_tgt)) else 0.0,
            'manifold_loss': manifold_loss.item() if not (torch.isnan(manifold_loss) or torch.isinf(manifold_loss)) else 0.0,
            'center_loss': center_loss.item() if not (torch.isnan(center_loss) or torch.isinf(center_loss)) else 0.0,
            'total_loss': total_loss.item() if not (torch.isnan(total_loss) or torch.isinf(total_loss)) else 0.0
        }