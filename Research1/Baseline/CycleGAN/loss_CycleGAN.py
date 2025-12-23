import torch
import torch.nn.functional as F
import torch.autograd as autograd


def cycle_consistency_loss(real_x, reconstructed_x):
    """
    循环一致性损失 - CycleGAN的核心损失
    """
    return F.l1_loss(reconstructed_x, real_x)


def adversarial_loss_lsgan(D_output, target_is_real):
    """
    LSGAN对抗损失 (最小二乘GAN)
    """
    if target_is_real:
        target = torch.ones_like(D_output)
    else:
        target = torch.zeros_like(D_output)
    return F.mse_loss(D_output, target)


def identity_loss(real_x, same_x):
    """
    身份映射损失 - 保持同域映射的一致性
    """
    return F.l1_loss(same_x, real_x)


def frequency_consistency_loss(real_samples, fake_samples):
    """
    频域一致性损失
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
    
    # 幅度谱
    real_mag = real_fft.abs()
    fake_mag = fake_fft.abs()
    
    # 归一化
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
    
    # 归一化特征
    real_features = F.normalize(real_features, p=2, dim=1)
    fake_features = F.normalize(fake_features, p=2, dim=1)
    
    def gaussian_kernel_multi(x, y, sigmas):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        dist = torch.sum((x - y) ** 2, dim=2)
        
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
