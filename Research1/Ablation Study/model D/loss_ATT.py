import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LossFunction:
    def __init__(self, num_classes=10, alpha=1.0, beta=1.0, gamma=0.5, label_smoothing=0.1, focal_gamma=2.0):
        # 使用标签平滑减少过拟合
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.num_classes = num_classes
        # 损失权重参数
        self.alpha = alpha    # 源域分类损失权重
        self.beta = beta      # 目标域分类损失权重  
        self.gamma = gamma    # 跨域特征对齐损失权重（MMD）
        self.focal_gamma = focal_gamma  # Focal Loss的gamma参数
        
    def focal_loss(self, pred, labels, gamma=2.0):
        """Focal Loss - 处理类别不平衡和难样本"""
        ce_loss = F.cross_entropy(pred, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma) * ce_loss
        return focal_loss.mean()
    
    def source_loss(self, pred_s, labels_s):
        """源域分类损失 L_S - 使用Focal Loss"""
        return self.focal_loss(pred_s, labels_s, self.focal_gamma)

    def target_loss(self, pred_t, labels_t):
        """目标域分类损失 L_T - 使用Focal Loss + 标签平滑"""
        focal = self.focal_loss(pred_t, labels_t, self.focal_gamma)
        ce = self.ce_loss(pred_t, labels_t)
        return 0.7 * focal + 0.3 * ce  # 混合使用

    def mmd_loss(self, F_s, F_t, kernel_mul=2.0, kernel_num=5):
        """多核MMD损失 - 更好的域适应特征对齐"""
        batch_size = F_s.size(0)
        
        # 计算核带宽
        total = torch.cat([F_s, F_t], dim=0)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        bandwidth = torch.sum(L2_distance.detach()) / (total.size(0) ** 2 - total.size(0))
        bandwidth = bandwidth / kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        
        # 计算多核MMD
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        kernels = sum(kernel_val)
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        mmd = torch.mean(XX + YY - XY - YX)
        return mmd
    
    def cross_feature_loss(self, F_s, F_t):
        """跨域特征对齐损失 - 使用MMD"""
        return self.mmd_loss(F_s, F_t)
    
    def total_loss(self, pred_s, pred_t, F_s, F_t, labels_s, labels_t):
        """优化后的总损失函数（删除一致性损失）"""
        ls = self.source_loss(pred_s, labels_s)
        lt = self.target_loss(pred_t, labels_t)
        lmmd = self.cross_feature_loss(F_s, F_t)
        
        # 计算加权总损失（不再使用一致性损失）
        total = (self.alpha * ls + 
                self.beta * lt + 
                self.gamma * lmmd)
        
        return total, {
            'source_loss': ls.item(),
            'target_loss': lt.item(), 
            'mmd_loss': lmmd.item(),
            'total_loss': total.item()
        }
