import torch
import numpy as np
import torch.nn.functional as F


def compute_fid(real_features, fake_features):
    """
    计算Fréchet Inception Distance (FID) - 适配CSI特征
    衡量真实样本和生成样本在特征空间的分布差异
    
    参数:
      real_features - 真实样本特征, 形状为 (N, d)
      fake_features - 生成样本特征, 形状为 (M, d)
    返回: FID分数（越低越好）
    """
    # 计算均值
    mu_real = torch.mean(real_features, dim=0)
    mu_fake = torch.mean(fake_features, dim=0)
    
    # 计算协方差矩阵
    sigma_real = torch.cov(real_features.T)
    sigma_fake = torch.cov(fake_features.T)
    
    # 计算均值差的平方范数
    diff = mu_real - mu_fake
    mean_diff = torch.dot(diff, diff)
    
    # 计算协方差矩阵的迹
    trace_term = torch.trace(sigma_real + sigma_fake)
    
    # 计算 sqrt(sigma_real * sigma_fake)
    # 使用特征值分解的数值稳定方法
    try:
        # 方法1: 使用Cholesky分解
        covmean = torch.mm(sigma_real, sigma_fake)
        # 添加小的正则化项以保证数值稳定性
        covmean = covmean + torch.eye(covmean.size(0), device=covmean.device) * 1e-6
        # 使用特征值分解计算平方根
        eigvals, eigvecs = torch.linalg.eigh(covmean)
        eigvals = torch.clamp(eigvals, min=0)  # 确保非负
        covmean_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
        trace_sqrt = torch.trace(covmean_sqrt)
    except Exception:
        # 降级方案: 使用简化的FID计算
        trace_sqrt = 0.0
    
    # FID = ||mu_real - mu_fake||^2 + Tr(sigma_real + sigma_fake - 2*sqrt(sigma_real*sigma_fake))
    fid = mean_diff + trace_term - 2 * trace_sqrt
    
    return fid.item()


def compute_spectral_fidelity(real_samples, fake_samples):
    """
    计算频谱保真度 - 通过KL散度度量功率谱密度的差异
    
    参数:
      real_samples - 真实样本, 形状为 (N, C, S, T)
      fake_samples - 生成样本, 形状为 (M, C, S, T)
    返回: 频谱保真度分数（越低越好）
    """
    # 确保样本数量一致，取最小值
    min_samples = min(real_samples.size(0), fake_samples.size(0))
    real_samples = real_samples[:min_samples]
    fake_samples = fake_samples[:min_samples]
    
    # 在时间维度进行FFT
    fft_real = torch.fft.rfft(real_samples, dim=-1)
    fft_fake = torch.fft.rfft(fake_samples, dim=-1)
    
    # 计算功率谱密度
    psd_real = torch.abs(fft_real) ** 2
    psd_fake = torch.abs(fft_fake) ** 2
    
    # 归一化为概率分布
    psd_real_norm = psd_real / (psd_real.sum(dim=-1, keepdim=True) + 1e-8)
    psd_fake_norm = psd_fake / (psd_fake.sum(dim=-1, keepdim=True) + 1e-8)
    
    # 计算KL散度
    kl_div = F.kl_div(
        torch.log(psd_fake_norm + 1e-8),
        psd_real_norm,
        reduction='batchmean'
    )
    
    return kl_div.item()



def evaluate_gan_comprehensive(E, G, source_loader, target_loader, device='cuda'):
    """
    综合评估GAN生成质量
    
    参数:
      E - 特征提取器
      G - 生成器
      source_loader - 源域数据加载器
      target_loader - 目标域数据加载器
      device - 设备类型
    返回: dict - 包含所有评估指标的字典
    """
    E.eval()
    G.eval()
    
    # 收集数据
    real_samples_list = []
    fake_samples_list = []
    real_features_list = []
    fake_features_list = []
    real_labels_list = []
    fake_labels_list = []
    
    # 先提取目标域特征
    with torch.no_grad():
        for x_t, labels in target_loader:
            x_t = x_t.to(device)
            real_samples_list.append(x_t.cpu())
            real_features_list.append(E(x_t).cpu())
            real_labels_list.append(labels.cpu())
    
    target_features = torch.cat(real_features_list, dim=0).to(device)
    
    # 生成样本
    with torch.no_grad():
        for x_s, labels in source_loader:
            x_s = x_s.to(device)
            labels = labels.to(device)
            x_fake = G(x_s, target_features)
            fake_samples_list.append(x_fake.cpu())
            fake_features_list.append(E(x_fake).cpu())
            fake_labels_list.append(labels.cpu())
    
    real_samples = torch.cat(real_samples_list, dim=0)
    fake_samples = torch.cat(fake_samples_list, dim=0)
    real_features = torch.cat(real_features_list, dim=0)
    fake_features = torch.cat(fake_features_list, dim=0)
    real_labels = torch.cat(real_labels_list, dim=0)
    fake_labels = torch.cat(fake_labels_list, dim=0)
    
    # 计算评估指标
    metrics = {}
    
    # 1. FID分数
    try:
        metrics['fid'] = compute_fid(real_features, fake_features)
    except Exception as e:
        print(f"FID计算失败: {e}")
        metrics['fid'] = -1
    
    # 2. 频谱保真度（按标签均衡计算）
    try:
        unique_labels = torch.unique(fake_labels)
        sf_sum = 0.0
        count = 0
        for l in unique_labels:
            r_mask = (real_labels == l)
            f_mask = (fake_labels == l)
            if r_mask.any() and f_mask.any():
                sf_l = compute_spectral_fidelity(real_samples[r_mask], fake_samples[f_mask])
                sf_sum += sf_l
                count += 1
        metrics['spectral_fidelity'] = (sf_sum / max(count, 1))
    except Exception as e:
        print(f"频谱保真度计算失败: {e}")
        metrics['spectral_fidelity'] = -1
    
    return metrics
