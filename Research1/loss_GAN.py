import torch
import torch.nn.functional as F
import torch.autograd as autograd

def kl_divergence_loss(target_features):
    """
    KL 散度用于衡量目标域数据样本特征分布的内部差异（已弃用，保留用于向后兼容）
    推荐使用 mmd_loss 替代
    """
    eps = 1e-8
    n = target_features.size(0)  # 样本数量

    # 对 target_features 进行归一化
    target_features_normalized = F.softmax(target_features, dim=-1)

    # 计算所有样本对之间的KL散度
    kl_loss = 0.0
    for i in range(n):
        for j in range(n):
            # 计算第i个样本与第j个样本的KL散度
            kl_loss += target_features_normalized[i] * torch.log((target_features_normalized[i] + eps) / (target_features_normalized[j] + eps))

    return kl_loss.sum()


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


def mmd_loss(real_features, fake_features, sigmas=[0.5, 1.0, 2.0]):
    """
    【已优化】最大均值差异(MMD)损失 - 用于衡量两个分布之间的差异
    使用多带宽高斯核，提升分布对齐的鲁棒性
    
    参数:
      real_features - 真实样本特征, 形状为 (n, d)
      fake_features - 生成样本特征, 形状为 (m, d)
      sigmas - 高斯核带宽列表，默认使用多核以提升鲁棒性
    返回: MMD损失值（标量）
    """
    loss = 0.0
    
    for sigma in sigmas:
        # 计算 K(real, real)
        k_real_real = gaussian_kernel(real_features, real_features, sigma)
        # 计算 K(fake, fake)
        k_fake_fake = gaussian_kernel(fake_features, fake_features, sigma)
        # 计算 K(real, fake)
        k_real_fake = gaussian_kernel(real_features, fake_features, sigma)
        
        # MMD² = E[K(real, real)] + E[K(fake, fake)] - 2*E[K(real, fake)]
        mmd_sq = k_real_real.mean() + k_fake_fake.mean() - 2 * k_real_fake.mean()
        loss += mmd_sq
    
    # 对多个核取平均
    return loss / len(sigmas)


def frequency_consistency_loss(real_samples, fake_samples):
    """
    频域一致性损失：约束生成样本与真实样本在频谱上的一致性
    对时间维做 rFFT，比较幅度谱的 MSE
    参数:
      real_samples - 真实样本, 形状为 (B, C, S, T)
      fake_samples - 生成样本, 形状为 (B, C, S, T)
    返回: 频域 MSE（标量）
    """
    # 在时间维度做 FFT
    real_fft = torch.fft.rfft(real_samples, dim=-1)
    fake_fft = torch.fft.rfft(fake_samples, dim=-1)
    real_mag = real_fft.abs()
    fake_mag = fake_fft.abs()
    # 可选：对 C、S 做均值以降低维度差异影响
    # 这里直接在全维度计算 MSE
    return torch.mean((real_mag - fake_mag) ** 2)



def discriminator_loss(D, x_t_real, x_hat_t):
    """
    判别器损失 L_D = -[log(D(x_t_real)) + log(1 - D(x_hat_t))]（已弃用）
    推荐使用 wasserstein_discriminator_loss 替代
    """
    real_pred = D(x_t_real)
    fake_pred = D(x_hat_t)
    real_loss = -torch.mean(torch.log(real_pred + 1e-8))
    fake_loss = -torch.mean(torch.log(1 - fake_pred + 1e-8))
    return real_loss + fake_loss


def compute_gradient_penalty(D, real_samples, fake_samples, device='cuda'):
    """
    计算WGAN-GP的梯度惩罚项
    
    参数:
      D - 判别器网络
      real_samples - 真实样本, 形状为 (B, C, S, T)
      fake_samples - 生成样本, 形状为 (B, C, S, T)
      device - 设备类型
    返回: 梯度惩罚损失（标量）
    """
    batch_size = real_samples.size(0)
    
    # 生成随机插值系数 epsilon ~ Uniform(0, 1)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    epsilon = epsilon.expand_as(real_samples)
    
    # 计算插值样本
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated = interpolated.requires_grad_(True)
    
    # 计算判别器对插值样本的输出
    d_interpolated = D(interpolated)
    
    # 计算梯度
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 将梯度展平
    gradients = gradients.view(batch_size, -1)
    
    # 计算梯度的L2范数
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    
    # 梯度惩罚: (||grad|| - 1)^2
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return gradient_penalty


def wasserstein_discriminator_loss(D, x_t_real, x_hat_t, lambda_gp=10.0, device='cuda'):
    """
    Wasserstein GAN判别器损失（带梯度惩罚）
    L_D = E[D(X_fake)] - E[D(X_real)] + λ_GP * gradient_penalty
    
    参数:
      D - 判别器网络
      x_t_real - 真实目标域样本
      x_hat_t - 生成的虚假样本
      lambda_gp - 梯度惩罚权重，默认10.0
      device - 设备类型
    返回: tuple - (总损失, Wasserstein距离, 梯度惩罚)
    """
    # Wasserstein距离: E[D(fake)] - E[D(real)]
    fake_score = D(x_hat_t).mean()
    real_score = D(x_t_real).mean()
    wasserstein_distance = fake_score - real_score
    
    # 计算梯度惩罚
    gradient_penalty = compute_gradient_penalty(D, x_t_real, x_hat_t, device)
    
    # 总损失
    d_loss = wasserstein_distance + lambda_gp * gradient_penalty
    
    return d_loss, wasserstein_distance, gradient_penalty


def wasserstein_generator_loss(D, x_hat_t):
    """
    Wasserstein GAN生成器损失
    L_G = -E[D(X_fake)]
    
    参数:
      D - 判别器网络
      x_hat_t - 生成的虚假样本
    返回: 生成器损失（标量）
    """
    fake_score = D(x_hat_t).mean()
    return -fake_score


