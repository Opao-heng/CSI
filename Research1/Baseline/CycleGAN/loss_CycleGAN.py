import torch
import torch.nn as nn
import torch.nn.functional as F


def adversarial_loss(pred, target_is_real):
    """
    对抗损失（MSE Loss）
    参数:
        pred: 判别器的预测输出
        target_is_real: 是否为真实样本
    """
    if target_is_real:
        target = torch.ones_like(pred)
    else:
        target = torch.zeros_like(pred)
    return F.mse_loss(pred, target)


def cycle_consistency_loss(real, reconstructed):
    """
    循环一致性损失
    参数:
        real: 真实样本
        reconstructed: 循环重构的样本
    返回:
        L1损失
    """
    return F.l1_loss(reconstructed, real)


def identity_loss(real, same):
    """
    身份保持损失
    参数:
        real: 真实样本
        same: 同域映射的输出
    返回:
        L1损失
    """
    return F.l1_loss(same, real)
