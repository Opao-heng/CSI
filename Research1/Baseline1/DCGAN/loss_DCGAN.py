import torch
import torch.nn.functional as F


def dcgan_discriminator_loss(D, real_samples, fake_samples):
    """
    DCGAN判别器损失 - 二元交叉熵
    """
    # 真实样本损失
    pred_real = D(real_samples)
    loss_real = F.binary_cross_entropy(pred_real, torch.ones_like(pred_real))
    
    # 生成样本损失
    pred_fake = D(fake_samples)
    loss_fake = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
    
    # 总损失
    d_loss = (loss_real + loss_fake) * 0.5
    
    return d_loss, pred_real.mean().item(), pred_fake.mean().item()


def dcgan_generator_loss(D, fake_samples):
    """
    DCGAN生成器损失 - 二元交叉熵
    """
    pred_fake = D(fake_samples)
    g_loss = F.binary_cross_entropy(pred_fake, torch.ones_like(pred_fake))
    
    return g_loss


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
