import torch
import torch.nn.functional as F
import torch.autograd as autograd


def wasserstein_discriminator_loss(D, x_t_real, x_hat_t):
    """
    WGAN判别器损失
    """
    fake_score = D(x_hat_t).mean()
    real_score = D(x_t_real).mean()
    d_loss = fake_score - real_score
    return d_loss, (real_score - fake_score).item()


def wasserstein_generator_loss(D, x_hat_t):
    """
    WGAN生成器损失
    """
    return -D(x_hat_t).mean()


def mmd_loss(real_features, fake_features, sigma=1.0):
    """
    MMD损失 - 使用高斯核
    """
    n = min(real_features.size(0), fake_features.size(0), 64)
    real_features = real_features[:n]
    fake_features = fake_features[:n]
    
    # 计算核矩阵
    def gaussian_kernel(x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return torch.exp(-torch.sum((x - y) ** 2, dim=2) / (2 * sigma ** 2))
    
    k_xx = gaussian_kernel(real_features, real_features)
    k_yy = gaussian_kernel(fake_features, fake_features)
    k_xy = gaussian_kernel(real_features, fake_features)
    
    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return torch.clamp(mmd, min=0.0)


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


def reconstruction_loss(x_source, x_generated):
    """
    重建损失 - 保持源域结构
    """
    return F.l1_loss(x_generated, x_source)


def identity_preserving_loss(E, x_source, x_generated):
    """
    身份保持损失
    """
    f_source = E(x_source)
    f_generated = E(x_generated)
    cos_sim = F.cosine_similarity(f_source, f_generated, dim=1)
    return 1.0 - cos_sim.mean()