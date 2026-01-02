import torch
import torch.nn.functional as F


def cvae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    CVAE损失函数 = 重建损失 + KL散度
    
    参数:
        recon_x: 重建的数据
        x: 原始数据
        mu: 编码器输出的均值
        logvar: 编码器输出的对数方差
        beta: KL散度的权重系数
    """
    # 重建损失 (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    
    # KL散度损失
    # KL(N(mu, sigma) || N(0, 1))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    
    # 总损失
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def frequency_consistency_loss(real_samples, fake_samples):
    """
    频域一致性损失 - 归一化版本
    """
    B = min(real_samples.size(0), fake_samples.size(0))
    real_samples = real_samples[:B]
    fake_samples = fake_samples[:B]
    
    B, C, S, T = real_samples.shape
    real_flat = real_samples.view(B, C * S, T)
    fake_flat = fake_samples.view(B, C * S, T)
    
    # FFT
    real_fft = torch.fft.rfft(real_flat, dim=-1)
    fake_fft = torch.fft.rfft(fake_flat, dim=-1)
    
    # 幅度谱 - 归一化
    real_mag = real_fft.abs()
    fake_mag = fake_fft.abs()
    
    # 归一化到[0,1]范围
    real_mag_norm = real_mag / (real_mag.max() + 1e-8)
    fake_mag_norm = fake_mag / (fake_mag.max() + 1e-8)
    
    return F.mse_loss(fake_mag_norm, real_mag_norm)


def mmd_loss(real_features, fake_features, sigmas=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """
    MMD损失 - 多尺度高斯核
    """
    n = min(real_features.size(0), fake_features.size(0), 128)
    real_features = real_features[:n]
    fake_features = fake_features[:n]
    
    # 归一化特征以提高稳定性
    real_features = F.normalize(real_features, p=2, dim=1)
    fake_features = F.normalize(fake_features, p=2, dim=1)
    
    def gaussian_kernel_multi(x, y, sigmas):
        x = x.unsqueeze(1)  # (n, 1, d)
        y = y.unsqueeze(0)  # (1, m, d)
        dist = torch.sum((x - y) ** 2, dim=2)  # (n, m)
        
        kernel_sum = None
        for sigma in sigmas:
            kernel = torch.exp(-dist / (2 * sigma ** 2))
            if kernel_sum is None:
                kernel_sum = kernel
            else:
                kernel_sum = kernel_sum + kernel
        return kernel_sum / len(sigmas)
    
    k_xx = gaussian_kernel_multi(real_features, real_features, sigmas)
    k_yy = gaussian_kernel_multi(fake_features, fake_features, sigmas)
    k_xy = gaussian_kernel_multi(real_features, fake_features, sigmas)
    
    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return torch.clamp(mmd, min=0.0)
