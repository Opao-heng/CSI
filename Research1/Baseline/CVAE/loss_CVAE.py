import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE损失函数
    参数:
        recon_x: 重构的样本
        x: 原始样本
        mu: 编码器输出的均值
        logvar: 编码器输出的对数方差
        beta: KL散度的权重系数 (beta-VAE)
    返回:
        总损失、重构损失、KL散度损失
    """
    # 确保重构样本和原始样本维度一致
    if recon_x.shape != x.shape:
        # 如果维度不一致，进行调整
        if recon_x.shape[1] == 3 and x.shape[1] == 56:
            # 从 (N, 3, 56, T) 转换为 (N, 56, 3, T)
            recon_x = recon_x.permute(0, 2, 1, 3)
        elif recon_x.shape[1] == 56 and x.shape[1] == 3:
            # 从 (N, 56, 3, T) 转换为 (N, 3, 56, T)
            x = x.permute(0, 2, 1, 3)
    
    # 重构损失 (MSE)
    B, C, S, T = x.shape
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL散度损失: -0.5 * mean(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def vae_frequency_loss(recon_x, x):
    """
    频域重构损失
    参数:
        recon_x: 重构的样本
        x: 原始样本
    返回:
        频域损失
    """
    # 确保重构样本和原始样本维度一致
    if recon_x.shape != x.shape:
        # 如果维度不一致，进行调整
        if recon_x.shape[1] == 3 and x.shape[1] == 56:
            # 从 (N, 3, 56, T) 转换为 (N, 56, 3, T)
            recon_x = recon_x.permute(0, 2, 1, 3)
        elif recon_x.shape[1] == 56 and x.shape[1] == 3:
            # 从 (N, 56, 3, T) 转换为 (N, 3, 56, T)
            x = x.permute(0, 2, 1, 3)
    
    B, C, S, T = x.shape
    
    # 展平为二维 (使用reshape而不是view以避免内存布局问题)
    x_flat = x.reshape(B, C * S, T)
    recon_flat = recon_x.reshape(B, C * S, T)
    
    # 傅里叶变换
    x_fft = torch.fft.rfft(x_flat, dim=-1)
    recon_fft = torch.fft.rfft(recon_flat, dim=-1)
    
    # 幅度谱损失
    eps = 1e-8
    x_mag = torch.log(x_fft.abs() + eps)
    recon_mag = torch.log(recon_fft.abs() + eps)
    
    freq_loss = F.mse_loss(recon_mag, x_mag)
    
    return freq_loss


def combined_vae_loss(recon_x, x, mu, logvar, beta=1.0, lambda_freq=0.1):
    """
    结合时域和频域的VAE损失
    参数:
        recon_x: 重构的样本
        x: 原始样本
        mu: 编码器输出的均值
        logvar: 编码器输出的对数方差
        beta: KL散度的权重系数
        lambda_freq: 频域损失的权重系数
    返回:
        总损失、重构损失、KL散度损失、频域损失
    """
    # 基础VAE损失
    total_loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar, beta)
    
    # 频域损失
    freq_loss = vae_frequency_loss(recon_x, x)
    
    # 组合损失
    combined_loss = total_loss + lambda_freq * freq_loss
    

    
    return combined_loss, recon_loss, kl_loss, freq_loss