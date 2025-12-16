import torch
import torch.nn.functional as F
import torch.autograd as autograd


def gradient_penalty(D, x_real, x_fake, device='cuda'):
    """
    WGAN-GP梯度惩罚
    """
    batch_size = min(x_real.size(0), x_fake.size(0))
    x_real = x_real[:batch_size]
    x_fake = x_fake[:batch_size]
    
    # 随机插值
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * x_real + (1 - alpha) * x_fake).requires_grad_(True)
    
    # 计算判别器输出
    d_interpolates = D(interpolates)
    
    # 计算梯度
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 计算梯度惩罚
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    
    return gp


def wasserstein_discriminator_loss(D, x_t_real, x_hat_t, lambda_gp=10.0, device='cuda'):
    """
    WGAN-GP判别器损失
    """
    batch_size = min(x_t_real.size(0), x_hat_t.size(0))
    x_t_real = x_t_real[:batch_size]
    x_hat_t = x_hat_t[:batch_size]
    
    fake_score = D(x_hat_t).mean()
    real_score = D(x_t_real).mean()
    
    # Wasserstein损失
    w_loss = fake_score - real_score
    
    # 梯度惩罚
    gp = gradient_penalty(D, x_t_real, x_hat_t, device)
    
    d_loss = w_loss + lambda_gp * gp
    return d_loss, (real_score - fake_score).item(), gp.item()


def wasserstein_generator_loss(D, x_hat_t):
    """
    WGAN生成器损失
    """
    return -D(x_hat_t).mean()


def mmd_loss(real_features, fake_features, sigmas=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """
    MMD损失 - 多尺度高斯核 (更稳定)
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


def feature_matching_loss(D, x_real, x_fake):
    """
    特征匹配损失 - 让生成器学习判别器的中间特征
    """
    # 获取判别器中间层特征
    B, C, S, T = x_real.shape
    x_real_flat = x_real.view(B, C * S, T)
    x_fake_flat = x_fake.view(B, C * S, T)
    
    # 使用判别器的部分层提取特征
    with torch.no_grad():
        real_feat = D.net[:4](x_real_flat)  # 前几层特征
    fake_feat = D.net[:4](x_fake_flat)
    
    return F.mse_loss(fake_feat, real_feat)


def content_loss(x_source, x_generated):
    """
    内容损失 - 保持时频特性
    """
    # L1损失
    l1 = F.l1_loss(x_generated, x_source)
    
    # 频域损失
    B, C, S, T = x_source.shape
    x_s_flat = x_source.view(B, -1, T)
    x_g_flat = x_generated.view(B, -1, T)
    
    fft_s = torch.fft.rfft(x_s_flat, dim=-1).abs()
    fft_g = torch.fft.rfft(x_g_flat, dim=-1).abs()
    
    # 归一化
    fft_s = fft_s / (fft_s.max() + 1e-8)
    fft_g = fft_g / (fft_g.max() + 1e-8)
    
    freq = F.mse_loss(fft_g, fft_s)
    
    return l1 + 0.1 * freq