import torch
import torch.nn.functional as F
import torch.autograd as autograd


def wasserstein_discriminator_loss(D, x_t_real, x_hat_t, y_real=None, y_fake=None):
    """
    Wasserstein GAN判别器损失(支持按标签均衡)
    判别器要最大化 E[D(real)] - E[D(fake)]，这里返回的是需要“最小化”的损失：
    L_D = -(E[D(real)] - E[D(fake)])
    """
    if y_real is not None and y_fake is not None:
        s_fake = D(x_hat_t).view(-1)
        s_real = D(x_t_real).view(-1)
        unique_labels = torch.unique(y_fake)
        loss_sum = 0.0
        count = 0
        for l in unique_labels:
            mask_fake = (y_fake == l)
            mask_real = (y_real == l)
            if mask_fake.any() and mask_real.any():
                d_fake_mean = s_fake[mask_fake].mean()
                d_real_mean = s_real[mask_real].mean()
                # 判别器期望 real_score - fake_score 越大越好 => 损失为 -(real - fake)
                loss_sum += (-(d_real_mean - d_fake_mean))
                count += 1
        d_loss = loss_sum / max(count, 1)
    else:
        fake_score = D(x_hat_t).mean()
        real_score = D(x_t_real).mean()
        d_loss = -(real_score - fake_score)
    return d_loss


def wasserstein_generator_loss(D, x_hat_t, y_fake=None):
    """
    Wasserstein GAN生成器损失（支持按标签均衡）
    若提供 y_fake，则对每个标签分别求平均后再均衡平均。
    """
    if y_fake is not None:
        s_fake = D(x_hat_t).view(-1)
        unique_labels = torch.unique(y_fake)
        loss_sum = 0.0
        count = 0
        for l in unique_labels:
            mask_fake = (y_fake == l)
            if mask_fake.any():
                loss_sum += (-s_fake[mask_fake].mean())
                count += 1
        return loss_sum / max(count, 1)
    else:
        fake_score = D(x_hat_t).mean()
        return -fake_score


def gaussian_kernel(x, y, sigma=1.0):
    """
    计算高斯核矩阵
    参数:
      x - 特征矩阵1, 形状为 (n, d)
      y - 特征矩阵2, 形状为 (m, d)
      sigma - 核带宽
    返回: 核矩阵, 形状为 (n, m)
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    
    # 计算平方欧氏距离
    x = x.unsqueeze(1)  # (n, 1, d)
    y = y.unsqueeze(0)  # (1, m, d)
    
    # 计算 ||x - y||^2
    distances = torch.sum((x - y) ** 2, dim=2)  # (n, m)
    
    # 应用高斯核
    return torch.exp(-distances / (2 * sigma ** 2))


def mmd_loss(real_features, fake_features, sigmas=None, y_real=None, y_fake=None):
    """
    最大均值差异(MMD)损失 - 用于衡量两个分布之间的差异
    支持按标签均衡：若提供 y_real/y_fake，则对每个标签分别计算MMD后做均衡平均。
    """
    if sigmas is None:
        sigmas = [1.0, 2.0, 4.0]
    if y_real is not None and y_fake is not None:
        unique_labels = torch.unique(y_fake)
        loss_sum = 0.0
        count = 0
        for l in unique_labels:
            mask_real = (y_real == l)
            mask_fake = (y_fake == l)
            if mask_real.any() and mask_fake.any():
                loss_l = 0.0
                for sigma in sigmas:
                    k_rr = gaussian_kernel(real_features[mask_real], real_features[mask_real], sigma)
                    k_ff = gaussian_kernel(fake_features[mask_fake], fake_features[mask_fake], sigma)
                    k_rf = gaussian_kernel(real_features[mask_real], fake_features[mask_fake], sigma)
                    mmd_sq = k_rr.mean() + k_ff.mean() - 2 * k_rf.mean()
                    loss_l += mmd_sq
                loss_l = loss_l / len(sigmas)
                loss_sum += loss_l
                count += 1
        return loss_sum / max(count, 1)
    else:
        loss = 0.0
        for sigma in sigmas:
            k_real_real = gaussian_kernel(real_features, real_features, sigma)
            k_fake_fake = gaussian_kernel(fake_features, fake_features, sigma)
            k_real_fake = gaussian_kernel(real_features, fake_features, sigma)
            mmd_sq = k_real_real.mean() + k_fake_fake.mean() - 2 * k_real_fake.mean()
            loss += mmd_sq
        return loss / len(sigmas)


def frequency_consistency_loss(real_samples, fake_samples, y_real=None, y_fake=None):
    """
    频域一致性损失:只使用幅度谱损失(最重要的部分,速度快3倍)
    支持按标签均衡:若提供 y_real/y_fake,则对每个标签分别计算后均衡平均。
    """
    def freq_loss_impl(real_s, fake_s):
        B, C, S, T = real_s.shape
        real_flat = real_s.view(B, C * S, T)
        fake_flat = fake_s.view(B, C * S, T)
        real_fft = torch.fft.rfft(real_flat, dim=-1)
        fake_fft = torch.fft.rfft(fake_flat, dim=-1)
        eps = 1e-8
        # 只计算幅度谱损失(去掉相位和低频功率计算,大幅提速)
        real_mag = torch.log(real_fft.abs() + eps)
        fake_mag = torch.log(fake_fft.abs() + eps)
        magnitude_loss = F.mse_loss(fake_mag, real_mag)
        return magnitude_loss
    
    if y_real is not None and y_fake is not None:
        unique_labels = torch.unique(y_fake)
        loss_sum = 0.0
        count = 0
        for l in unique_labels:
            mask_real = (y_real == l)
            mask_fake = (y_fake == l)
            if mask_real.any() and mask_fake.any():
                loss_sum += freq_loss_impl(real_samples[mask_real], fake_samples[mask_fake])
                count += 1
        return loss_sum / max(count, 1)
    else:
        return freq_loss_impl(real_samples, fake_samples)