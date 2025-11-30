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
    # 使用SVD近似
    covmean = torch.mm(sigma_real, sigma_fake)
    covmean_sqrt = torch.linalg.matrix_power(covmean, 0.5)
    trace_sqrt = torch.trace(covmean_sqrt)
    
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


def compute_temporal_correlation(samples, max_lag=100):
    """
    计算时间自相关函数
    
    参数:
      samples - CSI样本, 形状为 (N, C, S, T)
      max_lag - 最大滞后时间步
    返回: 平均自相关系数
    """
    N, C, S, T = samples.shape
    
    # 展平为 (N*C*S, T)
    samples_flat = samples.reshape(-1, T)
    
    # 计算自相关
    autocorr_values = []
    for lag in range(1, min(max_lag, T//2)):
        # 计算相邻时间步的相关性
        x1 = samples_flat[:, :-lag]
        x2 = samples_flat[:, lag:]
        
        # Pearson相关系数
        corr = torch.corrcoef(torch.stack([x1.flatten(), x2.flatten()]))[0, 1]
        autocorr_values.append(corr.item())
    
    return np.mean(autocorr_values)


def compute_channel_smoothness(samples):
    """
    计算信道频率响应平滑度
    评估相邻子载波间的频率响应连续性
    
    参数:
      samples - CSI样本, 形状为 (N, C, S, T)
    返回: 平滑度分数（越低越平滑）
    """
    # 在子载波维度计算一阶导数
    # 形状: (N, C, S-1, T)
    subcarrier_diff = samples[:, :, 1:, :] - samples[:, :, :-1, :]
    
    # 计算导数的标准差（低标准差意味着高平滑性）
    smoothness = torch.std(subcarrier_diff).item()
    
    return smoothness


def compute_diversity_precision_recall(real_features, fake_features, k=3):
    """
    计算多样性-质量权衡指标: Precision和Recall
    
    参数:
      real_features - 真实样本特征, 形状为 (N, d)
      fake_features - 生成样本特征, 形状为 (M, d)
      k - k近邻参数
    返回: tuple - (precision, recall)
      - precision: 生成样本有多少在真实分布内
      - recall: 真实分布被生成样本覆盖的比例
    """
    def compute_nearest_neighbor_distances(x, y, k):
        """计算x中每个点到y的k近邻距离"""
        # x: (N, d), y: (M, d)
        # 计算距离矩阵
        distances = torch.cdist(x, y)  # (N, M)
        # 取k个最近邻
        knn_distances, _ = torch.topk(distances, k, largest=False, dim=1)
        return knn_distances[:, -1]  # 第k个最近邻的距离
    
    # Precision: 生成样本的k近邻在真实样本中
    fake_to_real_dist = compute_nearest_neighbor_distances(fake_features, real_features, k)
    real_to_real_dist = compute_nearest_neighbor_distances(real_features, real_features, k+1)
    
    precision = (fake_to_real_dist < real_to_real_dist.median()).float().mean().item()
    
    # Recall: 真实样本被生成样本覆盖
    real_to_fake_dist = compute_nearest_neighbor_distances(real_features, fake_features, k)
    
    recall = (real_to_fake_dist < real_to_real_dist.median()).float().mean().item()
    
    return precision, recall


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
    
    # 先提取目标域特征
    with torch.no_grad():
        for x_t, _ in target_loader:
            x_t = x_t.to(device)
            real_samples_list.append(x_t.cpu())
            real_features_list.append(E(x_t).cpu())
    
    target_features = torch.cat(real_features_list, dim=0).to(device)
    
    # 生成样本
    with torch.no_grad():
        for x_s, _ in source_loader:
            x_s = x_s.to(device)
            x_fake = G(x_s, target_features)
            fake_samples_list.append(x_fake.cpu())
            fake_features_list.append(E(x_fake).cpu())
    
    real_samples = torch.cat(real_samples_list, dim=0)
    fake_samples = torch.cat(fake_samples_list, dim=0)
    real_features = torch.cat(real_features_list, dim=0)
    fake_features = torch.cat(fake_features_list, dim=0)
    
    # 计算各项指标
    metrics = {}
    
    # 1. FID分数
    try:
        metrics['fid'] = compute_fid(real_features, fake_features)
    except Exception as e:
        print(f"FID计算失败: {e}")
        metrics['fid'] = -1
    
    # 2. 频谱保真度
    try:
        metrics['spectral_fidelity'] = compute_spectral_fidelity(real_samples, fake_samples)
    except Exception as e:
        print(f"频谱保真度计算失败: {e}")
        metrics['spectral_fidelity'] = -1
    
    # 3. 时间相关性
    try:
        metrics['temporal_correlation_real'] = compute_temporal_correlation(real_samples)
        metrics['temporal_correlation_fake'] = compute_temporal_correlation(fake_samples)
    except Exception as e:
        print(f"时间相关性计算失败: {e}")
        metrics['temporal_correlation_real'] = -1
        metrics['temporal_correlation_fake'] = -1
    
    # 4. 信道平滑度
    try:
        metrics['channel_smoothness_real'] = compute_channel_smoothness(real_samples)
        metrics['channel_smoothness_fake'] = compute_channel_smoothness(fake_samples)
    except Exception as e:
        print(f"信道平滑度计算失败: {e}")
        metrics['channel_smoothness_real'] = -1
        metrics['channel_smoothness_fake'] = -1
    
    # 5. Precision & Recall
    try:
        precision, recall = compute_diversity_precision_recall(real_features, fake_features)
        metrics['precision'] = precision
        metrics['recall'] = recall
    except Exception as e:
        print(f"Precision/Recall计算失败: {e}")
        metrics['precision'] = -1
        metrics['recall'] = -1
    
    return metrics
