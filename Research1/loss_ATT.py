import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction:
    def __init__(self, num_classes=10, alpha=1.0, beta=1.0, gamma=0.5, delta=0.3):
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        # 损失权重参数
        self.alpha = alpha    # 源域分类损失权重
        self.beta = beta      # 目标域分类损失权重  
        self.gamma = gamma    # 跨域特征一致性损失权重
        self.delta = delta    # 特征一致性损失权重
        
    def source_loss(self, pred_s, labels_s):
        """源域分类损失 L_S"""
        return self.ce_loss(pred_s, labels_s)

    def target_loss(self, pred_t, labels_t):
        """目标域分类损失 L_T"""
        return self.ce_loss(pred_t, labels_t)

    def cross_feature_loss(self, F_s, F_t):
        """改进的跨域特征一致性损失 L_SF"""
        # 使用余弦相似性而不是MSE，避免过度对齐
        F_s_norm = F.normalize(F_s, p=2, dim=1)
        F_t_norm = F.normalize(F_t, p=2, dim=1)
        cosine_sim = F.cosine_similarity(F_s_norm, F_t_norm, dim=1)
        # 将相似性转换为损失（1 - 相似性）
        return torch.mean(1 - cosine_sim)

    def consistency_loss(self, F_s, F_c):
        """特征一致性损失 L_C"""
        # 使用更温和的L1损失替代MSE
        return F.l1_loss(F_s, F_c)
        
    def adversarial_loss(self, F_s, F_t):
        """对抗性损失，促进域不变特征学习"""
        # 简单的域判别损失
        batch_size = F_s.size(0)
        
        # 创建域标签：源域=0，目标域=1
        domain_labels_s = torch.zeros(batch_size, dtype=torch.long, device=F_s.device)
        domain_labels_t = torch.ones(batch_size, dtype=torch.long, device=F_t.device)
        
        # 域分类器（简单的线性层）
        domain_classifier = nn.Linear(F_s.size(1), 2).to(F_s.device)
        
        # 域预测
        domain_pred_s = domain_classifier(F_s)
        domain_pred_t = domain_classifier(F_t)
        
        # 对抗性损失：特征提取器希望混淆域分类器
        domain_loss_s = self.ce_loss(domain_pred_s, domain_labels_t)  # 反向标签
        domain_loss_t = self.ce_loss(domain_pred_t, domain_labels_s)  # 反向标签
        
        return (domain_loss_s + domain_loss_t) / 2

    def total_loss(self, pred_s, pred_t, F_s, F_t, F_c, labels_s, labels_t):
        """改进的总损失函数"""
        ls = self.source_loss(pred_s, labels_s)
        lt = self.target_loss(pred_t, labels_t)
        lsf = self.cross_feature_loss(F_s, F_t)
        lc = self.consistency_loss(F_s, F_c)
        
        # 计算加权总损失
        total = (self.alpha * ls + 
                self.beta * lt + 
                self.gamma * lsf + 
                self.delta * lc)
        
        return total, {
            'source_loss': ls.item(),
            'target_loss': lt.item(), 
            'cross_feature_loss': lsf.item(),
            'consistency_loss': lc.item(),
            'total_loss': total.item()
        }
