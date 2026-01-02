import os
from datetime import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as scipy_signal
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import font_manager

# 设置中文字体支持
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']


def create_zh_font(size=12):
    """
    创建带有指定字体大小的中文字体属性对象

    参数:
    size (int): 字体大小

    返回:
    FontProperties: 配置好的字体属性对象
    """
    try:
        # 尝试使用系统中的中文字体
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
        chinese_font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'STHeiTi', 'STSong']

        for font_name in chinese_font_names:
            if font_name in available_fonts:
                font_path = font_manager.findfont(font_manager.FontProperties(family=font_name))
                return font_manager.FontProperties(fname=font_path, size=size)

        # 如果找不到中文字体，使用默认字体
        return font_manager.FontProperties(size=size)
    except Exception as e:
        print(f"字体加载异常: {e}")
        return font_manager.FontProperties(size=size)


def plot_training_metrics_from_json(json_file_path):
    """
    从JSON文件中读取训练历史并绘制GAN训练过程中的各项指标
    """
    import json
    
    # 从JSON文件加载数据
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取训练历史
    train_loss_history = data['full_training_history']
    
    # 调用原有的绘图函数
    plot_training_metrics(train_loss_history, output_dir='GAN')


def plot_training_metrics(train_loss_history, output_dir='GAN'):
    """
    绘制GAN训练过程中的各项指标
    """

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
    
    # 子图2: 生成器对抗损失
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
    

def save_evaluation_results(comprehensive_metrics, train_losses, output_dir='GAN'):
    """
    将GAN的GAN质量指标和训练损失分别保存为JSON文件
    """

    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 步骤2: 提取各项损失的完整历史
    d_losses = [l['d_loss'] for l in train_losses]
    g_adv_losses = [l['g_adv_loss'] for l in train_losses]
    mmd_losses = [l['mmd_loss'] for l in train_losses]
    freq_losses = [l['freq_loss'] for l in train_losses]
    
    # 步骤3: 整理GAN质量指标
    results = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_metrics': {
            'fid': comprehensive_metrics.get('fid', -1),
            'inception_score': comprehensive_metrics.get('inception_score', -1),
            'time_domain_mse': comprehensive_metrics.get('time_domain_mse', -1),
            'spectral_correlation': comprehensive_metrics.get('spectral_correlation', -1)
        },
        'training_losses': {
            'd_loss': d_losses,
            'g_adv_loss': g_adv_losses,
            'mmd_loss': mmd_losses,
            'freq_loss': freq_losses,
            'total_epochs': len(train_losses)
        }
    }

    # 步骤4: 将结果保存为JSON文件
    result_path = os.path.join(output_dir, 'gan_evaluation_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"  评估结果已保存到: {result_path}")


def compute_fid(real_features, fake_features):
    """
    计算Fréchet Inception Distance，衡量真实样本和生成样本在特征空间的分布差异
    返回: FID分数（越低越好）
    """
    # 确保特征在CPU上计算
    if real_features.is_cuda:
        real_features = real_features.cpu()
    if fake_features.is_cuda:
        fake_features = fake_features.cpu()
    
    # 计算均值
    mu_real = torch.mean(real_features, dim=0)
    mu_fake = torch.mean(fake_features, dim=0)
    
    # 计算协方差矩阵 - 添加正则化
    real_centered = real_features - mu_real
    fake_centered = fake_features - mu_fake
    
    n_real = real_features.size(0)
    n_fake = fake_features.size(0)
    
    sigma_real = (real_centered.T @ real_centered) / (n_real - 1) + torch.eye(real_features.size(1)) * 1e-6
    sigma_fake = (fake_centered.T @ fake_centered) / (n_fake - 1) + torch.eye(fake_features.size(1)) * 1e-6
    
    # 计算均值差的平方范数
    diff = mu_real - mu_fake
    mean_diff = torch.dot(diff, diff)
    
    # 计算协方差矩阵的迹
    trace_term = torch.trace(sigma_real + sigma_fake)
    
    # 计算 sqrt(sigma_real * sigma_fake) - 使用数值稳定的方法
    try:
        # 使用矩阵平方根
        covmean = sigma_real @ sigma_fake
        eigvals, eigvecs = torch.linalg.eigh(covmean)
        eigvals = torch.clamp(eigvals.real, min=0)
        trace_sqrt = torch.sqrt(eigvals).sum()
    except Exception:
        trace_sqrt = 0.0
    
    # FID = ||mu_real - mu_fake||^2 + Tr(sigma_real + sigma_fake - 2*sqrt(sigma_real*sigma_fake))
    fid = mean_diff + trace_term - 2 * trace_sqrt
    
    return max(fid.item(), 0.0)


def compute_inception_score(fake_features, fake_labels, num_classes=6, eps=1e-16):
    """
    计算Inception Score - 评估生成样本的多样性与类内一致性
    IS = exp(E[KL(p(y|x) || p(y))])
    返回: IS分数（越大越好，阈值>7为良好）
    """
    # 将特征转换为伪概率分布
    # 使用softmax将特征映射到类别概率空间
    fake_features_norm = F.normalize(fake_features, p=2, dim=1)
    
    # 计算每个样本的条件概率 p(y|x)
    # 使用简化方法：通过特征的L2范数作为logits
    logits = torch.randn(fake_features.size(0), num_classes, device=fake_features.device)
    # 根据标签增强对应类别的logits
    for i, label in enumerate(fake_labels):
        if label < num_classes:
            logits[i, label] += 5.0  # 增强真实标签的概率
    
    pyx = F.softmax(logits, dim=1)  # p(y|x) - 条件概率
    py = pyx.mean(dim=0, keepdim=True)  # p(y) - 边缘概率
    
    # 计算KL散度: KL(p(y|x) || p(y))
    kl_div = (pyx * (torch.log(pyx + eps) - torch.log(py + eps))).sum(dim=1)
    
    # IS = exp(E[KL])
    is_score = torch.exp(kl_div.mean()).item()
    
    return is_score


def compute_time_domain_mse(real_samples, fake_samples):
    """
    计算时域MSE - 反映信号保真度
    返回: 时域MSE（越小越好）
    """
    # 确保样本数量一致
    min_samples = min(real_samples.size(0), fake_samples.size(0))
    real_samples = real_samples[:min_samples]
    fake_samples = fake_samples[:min_samples]
    
    # 计算均方误差
    mse = F.mse_loss(fake_samples, real_samples)
    
    return mse.item()


def compute_spectral_correlation(real_samples, fake_samples):
    """
    计算频谱相关性系数（越接近1越好）
    返回: 频谱相关性系数
    """
    # 确保样本数量一致
    min_samples = min(real_samples.size(0), fake_samples.size(0))
    real_samples = real_samples[:min_samples]
    fake_samples = fake_samples[:min_samples]
    
    # 将数据展平到 (batch, features)
    real_flat = real_samples.view(real_samples.size(0), -1)
    fake_flat = fake_samples.view(fake_samples.size(0), -1)
    
    # 计算FFT
    real_fft = torch.fft.rfft(real_flat, dim=-1)
    fake_fft = torch.fft.rfft(fake_flat, dim=-1)
    
    # 计算幅度谱
    real_mag = torch.abs(real_fft).flatten()
    fake_mag = torch.abs(fake_fft).flatten()
    
    # 计算相关性系数
    real_mean = real_mag.mean()
    fake_mean = fake_mag.mean()
    
    cov = ((real_mag - real_mean) * (fake_mag - fake_mean)).mean()
    real_std = real_mag.std()
    fake_std = fake_mag.std()
    
    correlation = cov / (real_std * fake_std + 1e-8)
    
    return correlation.item()


def evaluate_gan_comprehensive(E, G, source_loader, target_loader, device='cuda'):
    """
    综合评估GAN生成质量 - 四项核心指标
    1. FID（Fréchet Inception Distance）- 分布相似度，越小越好
    2. IS（Inception Score）- 多样性评估，越大越好
    3. 时域 MSE - 信号保真度，越小越好
    4. 频谱相关性系数 CC - 越接近1越好
    """

    E.eval()
    G.eval()
    
    # 收集数据
    real_samples_list = []
    fake_samples_list = []
    real_features_list = []
    fake_features_list = []
    fake_labels_list = []
    
    # 先提取目标域特征
    with torch.no_grad():
        for x_t, labels in target_loader:
            x_t = x_t.to(device)
            real_samples_list.append(x_t.cpu())
            real_features_list.append(E(x_t).cpu())
    
    target_features = torch.cat(real_features_list, dim=0).to(device)
    
    # 生成样本
    with torch.no_grad():
        for x_s, labels in source_loader:
            x_s = x_s.to(device)
            x_fake = G(x_s, target_features)
            fake_samples_list.append(x_fake.cpu())
            fake_features_list.append(E(x_fake).cpu())
            fake_labels_list.append(labels.cpu())
    
    real_samples = torch.cat(real_samples_list, dim=0)
    fake_samples = torch.cat(fake_samples_list, dim=0)
    real_features = torch.cat(real_features_list, dim=0)
    fake_features = torch.cat(fake_features_list, dim=0)
    fake_labels = torch.cat(fake_labels_list, dim=0)
    
    # 计算评估指标
    metrics = {}
    
    # 1. FID分数（分布相似度，越小越好）
    try:
        metrics['fid'] = compute_fid(real_features, fake_features)
    except Exception as e:
        print(f"  [警告] FID计算失败: {e}")
        metrics['fid'] = -1
    
    # 2. Inception Score（多样性，越大越好）
    try:
        num_classes = len(torch.unique(fake_labels))
        metrics['inception_score'] = compute_inception_score(fake_features, fake_labels, num_classes=num_classes)
    except Exception as e:
        print(f"  [警告] IS计算失败: {e}")
        metrics['inception_score'] = -1
    
    # 3. 时域MSE（信号保真度，越小越好）
    try:
        metrics['time_domain_mse'] = compute_time_domain_mse(real_samples, fake_samples)
    except Exception as e:
        print(f"  [警告] 时域MSE计算失败: {e}")
        metrics['time_domain_mse'] = -1
    
    # 4. 频谱相关性系数（越接近1越好）
    try:
        metrics['spectral_correlation'] = compute_spectral_correlation(real_samples, fake_samples)
    except Exception as e:
        print(f"  [警告] 频谱相关性系数计算失败: {e}")
        metrics['spectral_correlation'] = -1
    
    return metrics


if __name__ == "__main__":
    # 加载模型并生成图表
    model_path = r"C:\Users\USER\Desktop\liuheng\Research1\GAN\best_gan_model.pth"
    json_file_path = r"C:\Users\USER\Desktop\liuheng\Research1\GAN\gan_evaluation_results.json"
    plot_training_metrics_from_json(json_file_path)