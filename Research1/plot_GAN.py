import os
from datetime import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


"""
绘制GAN训练过程中的各项指标
参数:
  train_loss_history - 训练损失历史列表，每个元素是包含所有损失的字典
  output_dir - 图形保存目录，默认为'GAN'
返回: 无返回值，直接保存图形文件
"""
def plot_training_metrics(train_loss_history, output_dir='GAN'):
    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤2: 提取各项损失数据（共4项）
    d_losses = [loss['d_loss'] for loss in train_loss_history]  # 判别器损失
    g_adv_losses = [loss['g_adv_loss'] for loss in train_loss_history]  # 对抗损失
    mmd_losses = [loss['mmd_loss'] for loss in train_loss_history]  # MMD损失
    freq_losses = [loss['freq_loss'] for loss in train_loss_history]  # 频域一致性损失
    epochs = range(1, len(train_loss_history) + 1)
    
    # 步骤3: 创建大型图表，包含4个子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 子图1: 判别器损失
    axes[0, 0].plot(epochs, d_losses, 'r-', linewidth=2, label='Discriminator Loss')
    axes[0, 0].set_title('Discriminator Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 子图2: 对抗损失
    axes[0, 1].plot(epochs, g_adv_losses, 'orange', linewidth=2, label='Adversarial Loss')
    axes[0, 1].set_title('Generator Adversarial Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 子图3: MMD损失
    axes[1, 0].plot(epochs, mmd_losses, 'g-', linewidth=2, label='MMD Loss')
    axes[1, 0].set_title('MMD Loss (Distribution Alignment)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 子图4: 频域一致性损失
    axes[1, 1].plot(epochs, freq_losses, 'm-', linewidth=2, label='Frequency Consistency Loss')
    axes[1, 1].set_title('Frequency Consistency Loss', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 步骤4: 调整布局并保存图形
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  训练指标图已保存到: {plot_path}")
    plt.close()
    
    # 步骤5: 绘制所有损失在一张图上的对比
    plt.figure(figsize=(14, 8))
    plt.plot(epochs, d_losses, 'r-', linewidth=2.5, label='Discriminator Loss', alpha=0.8)
    plt.plot(epochs, g_adv_losses, 'orange', linewidth=2.5, label='Adversarial Loss', alpha=0.8)
    plt.plot(epochs, mmd_losses, 'g-', linewidth=2.5, label='MMD Loss', alpha=0.8)
    plt.plot(epochs, freq_losses, 'm-', linewidth=2.5, label='Frequency Loss', alpha=0.8)
    plt.title('All Training Losses Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    all_losses_path = os.path.join(output_dir, 'all_losses_comparison.png')
    plt.savefig(all_losses_path, dpi=300, bbox_inches='tight')
    print(f"  所有损失对比图已保存到: {all_losses_path}")
    plt.close()

"""
将GAN的评估结果保存为JSON文件

作用:
    将综合评估指标和训练损失历史保存为JSON格式，便于后续模型性能分析。

返回:
    无
"""


def save_evaluation_results(comprehensive_metrics, train_losses, output_dir='GAN'):
    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 步骤2: 整理评估指标和训练损失统计
    results = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_metrics': comprehensive_metrics,
        'training_loss_summary': {
            'min_d_loss': min([l['d_loss'] for l in train_losses]) if train_losses else 0,
            'max_d_loss': max([l['d_loss'] for l in train_losses]) if train_losses else 0,
            'final_d_loss': train_losses[-1]['d_loss'] if train_losses else 0,
            'min_g_adv_loss': min([l['g_adv_loss'] for l in train_losses]) if train_losses else 0,
            'max_g_adv_loss': max([l['g_adv_loss'] for l in train_losses]) if train_losses else 0,
            'final_g_adv_loss': train_losses[-1]['g_adv_loss'] if train_losses else 0,
            'min_mmd_loss': min([l['mmd_loss'] for l in train_losses]) if train_losses else 0,
            'max_mmd_loss': max([l['mmd_loss'] for l in train_losses]) if train_losses else 0,
            'final_mmd_loss': train_losses[-1]['mmd_loss'] if train_losses else 0,
            'min_freq_loss': min([l['freq_loss'] for l in train_losses]) if train_losses else 0,
            'max_freq_loss': max([l['freq_loss'] for l in train_losses]) if train_losses else 0,
            'final_freq_loss': train_losses[-1]['freq_loss'] if train_losses else 0,
            'total_epochs': len(train_losses)
        },
        'full_training_history': train_losses
    }

    # 步骤3: 将结果保存为JSON文件
    result_path = os.path.join(output_dir, 'gan_evaluation_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"  评估结果已保存到: {result_path}")


"""
绘制生成样本的幅度图（按天线维度）
参数:
  synthetic_data - 生成的合成样本数据，形状为 (N, C, S, T)
  sample_idx - 要绘制的样本索引，默认为0
  output_dir - 图形保存目录
返回: 无返回值，直接保存图形文件
"""
def plot_synthetic_sample_amplitude(synthetic_data, sample_idx=0, output_dir='GAN'):
    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤2: 提取单个样本数据
    if isinstance(synthetic_data, torch.Tensor):
        sample = synthetic_data[sample_idx].cpu().numpy()  # 形状: (S, C, T) = (56, 3, 6000)
    else:
        sample = synthetic_data[sample_idx]
    
    S, C, T = sample.shape  # S=56(子载波), C=3(天线对), T=6000(时间步)
    
    # 步骤3: 创建子图，每个天线对一个子图
    fig, axes = plt.subplots(C, 1, figsize=(12, 3 * C))
    if C == 1:
        axes = [axes]
    
    # 步骤4: 为每个天线对绘制幅度热图
    for c in range(C):
        amplitude = np.abs(sample[:, c, :])  # 形状: (S, T)
        
        im = axes[c].imshow(amplitude, aspect='auto', cmap='viridis', 
                           interpolation='nearest', origin='lower')
        axes[c].set_title(f'Antenna Pair {c+1} - Amplitude Heatmap', 
                         fontsize=14, fontweight='bold')
        axes[c].set_xlabel('Time Steps', fontsize=12)
        axes[c].set_ylabel('Subcarriers', fontsize=12)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[c])
        cbar.set_label('Amplitude', fontsize=11)
    
    # 步骤5: 调整布局并保存
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'synthetic_sample_{sample_idx}_amplitude.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  生成样本幅度图已保存到: {plot_path}")
    plt.close()


"""
绘制真实样本与生成样本的特征分布二维图（使用t-SNE或PCA降维）
参数:
  real_features - 真实目标域样本的特征，形状为 (N_real, feature_dim)
  fake_features - 生成样本的特征，形状为 (N_fake, feature_dim)
  method - 降维方法，'tsne' 或 'pca'
  output_dir - 图形保存目录
返回: 无返回值，直接保存图形文件
"""
def plot_feature_distribution_2d(real_features, fake_features, method='tsne', output_dir='GAN'):
    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤2: 转换为numpy数组
    if isinstance(real_features, torch.Tensor):
        real_features = real_features.cpu().detach().numpy()
    if isinstance(fake_features, torch.Tensor):
        fake_features = fake_features.cpu().detach().numpy()
    
    # 步骤3: 合并特征并创建标签
    all_features = np.vstack([real_features, fake_features])
    labels = np.array([0] * len(real_features) + [1] * len(fake_features))
    
    # 步骤4: 降维到2D
    print(f"  正在使用 {method.upper()} 进行特征降维...")
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_2d = reducer.fit_transform(all_features)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(all_features)
    else:
        raise ValueError("method必须是'tsne'或'pca'")
    
    # 步骤5: 分离真实和生成样本的2D特征
    real_2d = features_2d[labels == 0]
    fake_2d = features_2d[labels == 1]
    
    # 步骤6: 绘制散点图
    plt.figure(figsize=(12, 10))
    plt.scatter(real_2d[:, 0], real_2d[:, 1], c='blue', alpha=0.5, s=30, 
                label='Real Target Samples', edgecolors='k', linewidth=0.3)
    plt.scatter(fake_2d[:, 0], fake_2d[:, 1], c='red', alpha=0.5, s=30, 
                label='Generated Samples', edgecolors='k', linewidth=0.3)
    
    plt.title(f'Feature Distribution ({method.upper()}) - Real vs Generated', 
              fontsize=16, fontweight='bold')
    plt.xlabel(f'{method.upper()} Component 1', fontsize=13)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=13)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 步骤7: 保存图形
    plot_path = os.path.join(output_dir, f'feature_distribution_{method}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  特征分布图({method.upper()})已保存到: {plot_path}")
    plt.close()


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
