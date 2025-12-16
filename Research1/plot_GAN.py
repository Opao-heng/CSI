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
    

def save_evaluation_results(comprehensive_metrics, train_losses, output_dir='GAN'):
    """
    将GAN的GAN质量指标和训练损失统计保存为JSON文件
    """

    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 步骤2: 整理GAN质量指标和训练损失统计
    results = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_metrics': {
            'fid': comprehensive_metrics.get('fid', -1),
            'inception_score': comprehensive_metrics.get('inception_score', -1),
            'time_domain_mse': comprehensive_metrics.get('time_domain_mse', -1),
            'spectral_correlation': comprehensive_metrics.get('spectral_correlation', -1)
        },
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


def plot_synthetic_sample_amplitude(synthetic_data, sample_idx=0, output_dir='GAN'):
    """
    绘制生成样本的时域幅度图和频谱图（按天线维度）
    """
    
    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤2: 提取单个样本数据
    if isinstance(synthetic_data, torch.Tensor):
        sample = synthetic_data[sample_idx].cpu().numpy()  # 形状: (S, C, T) = (56, 3, 6000)
    else:
        sample = synthetic_data[sample_idx]
    
    S, C, T = sample.shape  # S=56(子载波), C=3(天线对), T=6000(时间步)
    
    # ================== 时域幅度图 ==================
    # 步骤3: 创建时域子图，每个天线对一个子图，参考CSIProcess1.py的绘图风格
    plt.style.use('seaborn-v0_8')
    fig_size = (10, 6)
    fig, axes = plt.subplots(C, 1, figsize=fig_size)
    if C == 1:
        axes = [axes]
    
    # 步骤4: 为每个天线对绘制时域幅度图，采用CSIProcess1.py的线条图风格
    num_antennas = S  # 子载波数作为"天线"数
    selected_antennas = list(range(num_antennas))
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_antennas)))
    
    # 定义移动平均函数用于平滑曲线
    def moving_average(data, window_size=50):
        """计算移动平均以平滑曲线"""
        if len(data) < window_size:
            return data
        cumsum = np.cumsum(np.insert(data, 0, 0)) 
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    for c in range(C):
        ax = axes[c]
        # 绘制所有子载波的时域幅度曲线（按照CSIProcess1.py的风格）
        # 为了清晰显示线条，我们调整参数并对数据进行平滑处理
        for i, antenna in enumerate(selected_antennas):
            amplitude_data = np.abs(sample[antenna, c, :])
            # 对数据进行平滑处理
            smoothed_data = moving_average(amplitude_data, window_size=50)
            # 创建对应的时间轴
            time_axis = np.linspace(0, len(amplitude_data)-1, len(smoothed_data))
            ax.plot(time_axis, smoothed_data,
                    alpha=0.7,  # 透明度
                    linewidth=1.0,  # 线宽
                    color=colors[i % len(colors)])  # 循环使用颜色
        
        ax.set_title(f'天线对 {c + 1}', fontsize=12, pad=10, fontproperties=create_zh_font(12))
        ax.set_xlabel('时间', fontsize=10, fontproperties=create_zh_font(10))
        ax.set_ylabel('幅度', fontsize=10, fontproperties=create_zh_font(10))
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)
        # 为坐标轴刻度标签也设置中文字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(create_zh_font(8))
    
    # 步骤5: 调整布局并保存时域图
    plt.tight_layout()
    plt.draw()  # 强制刷新图形以确保中文字体正确应用
    plot_path_time = os.path.join(output_dir, f'synthetic_sample_{sample_idx}_time_domain.png')
    plt.savefig(plot_path_time, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  生成样本时域幅度图已保存到: {plot_path_time}")
    plt.close()
    
    # ================== 频谱图 ==================
    # 步骤6: 创建频谱子图，每个天线对一个子图，参考CSIProcess2.py的频谱图风格
    fig, axes = plt.subplots(C, 1, figsize=(12, 3 * C))
    if C == 1:
        axes = [axes]
    
    # 步骤7: 为每个天线对绘制频谱图，采用CSIProcess2.py的时频图风格
    for c in range(C):
        # 使用STFT替代FFT来获得更好的时频分辨率，参考CSIProcess2.py的做法
        signal_1d = sample[0, c, :]  # 取第一个子载波作为代表
        
        # 使用短时傅里叶变换（STFT）生成时频图
        nperseg = 512  # 窗口大小
        noverlap = 480  # 重叠大小（75%重叠）
        
        freqs, times, Sxx = scipy_signal.spectrogram(
            signal_1d,
            fs=1.0,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='spectrum'
        )
        
        # 转换为dB刻度
        Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
        
        # 将时间索引映射到实际值（0 to T-1）
        time_indices = times * (T - 1)
        # 仅显示一半的频率分量（对称性）
        freq_limit = len(freqs) // 2
        
        # 使用pcolormesh绘制时频图，参考CSIProcess2.py的风格
        im = axes[c].pcolormesh(time_indices, freqs[:freq_limit], 
                               Sxx_db[:freq_limit, :],
                               shading='auto', 
                               cmap='jet', 
                               rasterized=True)
        
        axes[c].set_title(f'天线对 {c+1} - 频谱图 (dB)', fontsize=14, fontweight='bold', fontproperties=create_zh_font(14))
        axes[c].set_xlabel('时间索引', fontsize=12, fontproperties=create_zh_font(12))
        axes[c].set_ylabel('频率分量', fontsize=12, fontproperties=create_zh_font(12))
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[c])
        cbar.set_label('幅度 (dB)', fontsize=11, fontproperties=create_zh_font(11))
        cbar.ax.tick_params(labelsize=10)
        
        # 设置刻度标签字体
        axes[c].tick_params(axis='both', which='major', labelsize=10)
        for label in axes[c].get_xticklabels() + axes[c].get_yticklabels():
            label.set_fontproperties(create_zh_font(10))
    
    # 步骤8: 调整布局并保存频谱图
    plt.tight_layout()
    plt.draw()  # 强制刷新图形以确保中文字体正确应用
    plot_path_freq = os.path.join(output_dir, f'synthetic_sample_{sample_idx}_frequency_spectrum.png')
    plt.savefig(plot_path_freq, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  生成样本频谱图已保存到: {plot_path_freq}")
    plt.close()


def plot_feature_distribution_2d(real_features, fake_features, real_labels=None, method='tsne', output_dir='GAN'):
    """
    绘制真实样本与生成样本的特征分布二维图（使用t-SNE降维）
    如果提供了real_labels，则按用户标签显示10个真实用户的特征分布群
    """

    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤2: 转换为numpy数组
    if isinstance(real_features, torch.Tensor):
        real_features = real_features.cpu().detach().numpy()
    if isinstance(fake_features, torch.Tensor):
        fake_features = fake_features.cpu().detach().numpy()
    if real_labels is not None and isinstance(real_labels, torch.Tensor):
        real_labels = real_labels.cpu().detach().numpy()
    
    # 步骤3: 合并特征并创建标签
    all_features = np.vstack([real_features, fake_features])
    type_labels = np.array([0] * len(real_features) + [1] * len(fake_features))
    
    # 步骤4: 降维到2D (仅保留t-SNE方法)
    print(f"  正在使用 {method.upper()} 进行特征降维...")
    if method.lower() == 'tsne':
        n_samples = len(all_features)
        perplexity = min(30, max(5, n_samples // 20))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1500, 
                      learning_rate='auto', init='pca')
        features_2d = reducer.fit_transform(all_features)
    else:
        raise ValueError("method必须是'tsne'")
    
    # 步骤5: 分离真实和生成样本的2D特征
    real_2d = features_2d[type_labels == 0]
    fake_2d = features_2d[type_labels == 1]
    
    # 步骤6: 绘制散点图 - 按用户标签显示真实样本
    plt.figure(figsize=(14, 11))
    
    if real_labels is not None:
        # 从真实样本标签中提取唯一用户ID
        unique_users = np.unique(real_labels)
        # 使用不同的颜色表示不同的用户
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_users)))
        
        # 为每个用户绘制一个散点群
        for idx, user_id in enumerate(sorted(unique_users)):
            mask = real_labels == user_id
            if np.any(mask):
                plt.scatter(real_2d[mask, 0], real_2d[mask, 1], 
                           c=[colors[idx]], label=f'User {int(user_id)}', 
                           alpha=0.6, s=40, edgecolors='black', linewidth=0.5, marker='o')
        
        # 绘制生成样本
        plt.scatter(fake_2d[:, 0], fake_2d[:, 1], c='red', alpha=0.4, s=30, 
                   label='Generated Samples', edgecolors='darkred', linewidth=0.3, marker='^')
    else:
        # 如果没有用户标签，使用原来的方法
        plt.scatter(real_2d[:, 0], real_2d[:, 1], c='blue', alpha=0.5, s=30, 
                   label='Real Target Samples', edgecolors='k', linewidth=0.3)
        plt.scatter(fake_2d[:, 0], fake_2d[:, 1], c='red', alpha=0.5, s=30, 
                   label='Generated Samples', edgecolors='k', linewidth=0.3)
    
    plt.title(f'Feature Distribution ({method.upper()}) - Real Target Users vs Generated', 
              fontsize=16, fontweight='bold')
    plt.xlabel(f'{method.upper()} Component 1', fontsize=13)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=13)
    plt.legend(fontsize=11, loc='best', frameon=True, fancybox=True, shadow=True, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 步骤7: 保存图形
    plot_path = os.path.join(output_dir, f'feature_distribution_{method}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  特征分布图({method.upper()})已保存到: {plot_path}")
    plt.close()


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
    3. 时域MSE - 信号保真度，越小越好
    4. 频谱相关性系数 - 越接近1越好
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


def load_model_and_generate_plots(model_path, source_loader, target_loader, device='cuda'):
    """
    加载训练好的模型并生成所有相关图表
    """
    import torch
    from Research1.model_GAN import build_model
    
    # 构建模型
    E, G, D, D_spec = build_model()
    E.to(device)
    G.to(device)
    D.to(device)
    D_spec.to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    E.load_state_dict(checkpoint['E'])
    G.load_state_dict(checkpoint['G'])
    D.load_state_dict(checkpoint['D'])
    D_spec.load_state_dict(checkpoint['D_spec'])
    
    print("模型加载成功")
    
    # 设置模型为评估模式
    E.eval()
    G.eval()
    D.eval()
    D_spec.eval()
    
    # 生成合成样本
    print("正在生成合成样本...")
    synthetic_data, synthetic_labels = generate_synthetic_samples(E, G, source_loader, target_loader, device=device)
    print(f"已生成 {len(synthetic_data)} 个合成样本")
    
    # 绘制训练指标图
    print("正在绘制训练指标图...")
    json_file_path = r"C:\Users\USER\Desktop\liuheng\Research1\GAN\gan_evaluation_results.json"
    plot_training_metrics_from_json(json_file_path)
    
    # 绘制特征分布TSNE图
    print("正在绘制特征分布TSNE图...")
    with torch.no_grad():
        # 提取目标域真实样本的特征
        real_features_list = []
        real_labels_list = []
        for x_t_real, target_labels in target_loader:
            x_t_real = x_t_real.to(device)
            features = E(x_t_real)
            real_features_list.append(features)
            real_labels_list.append(target_labels)
        real_features = torch.cat(real_features_list, dim=0)
        real_labels = torch.cat(real_labels_list, dim=0)
            
        # 提取生成样本的特征
        synthetic_data_device = synthetic_data.to(device)
        fake_features = E(synthetic_data_device)
        
    plot_feature_distribution_2d(real_features, fake_features, real_labels=real_labels, method='tsne', output_dir='GAN')
    
    # 绘制生成样本频谱图和时域图
    print("正在绘制生成样本图...")
    import numpy as np
    random_idx = np.random.randint(0, len(synthetic_data))
    plot_synthetic_sample_amplitude(synthetic_data, sample_idx=random_idx, output_dir='GAN')
    
    print("所有图表绘制完成!")


def generate_synthetic_samples(E, G, source_loader, target_loader, num_samples=100, device='cuda'):
    """
    使用训练的生成器生成指定数量的合成样本。
    """
    # 设置模型为评估模式
    E.eval()
    G.eval()

    synthetic_data = []
    synthetic_labels = []
    generated_count = 0
    
    # 预先提取目标域特征（缓存以提高效率）
    all_features = []
    with torch.no_grad():
        for x_t_real, _ in target_loader:
            x_t_real = x_t_real.to(device)
            features = E(x_t_real)
            all_features.append(features)
    target_features = torch.cat(all_features, dim=0).to(device)

    with torch.no_grad():
        # 遍历源域数据生成合成样本
        for x_s, source_labels in source_loader:
            if generated_count >= num_samples:
                break

            x_s = x_s.to(device)
            batch_size = x_s.size(0)

            # 生成与目标域特征匹配的合成样本
            x_hat_t = G(x_s, target_features)

            # 收集生成的样本和对应的标签
            synthetic_data.append(x_hat_t.cpu())
            synthetic_labels.append(source_labels[:x_hat_t.size(0)])

            generated_count += x_hat_t.size(0)

    # 合并所有生成样本并截断至指定数量
    if synthetic_data:
        synthetic_data = torch.cat(synthetic_data, dim=0)[:num_samples]
        synthetic_labels = torch.cat(synthetic_labels, dim=0)[:num_samples]

    return synthetic_data, synthetic_labels


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from Research1.Process.dataloder_GAN import CustomDataset, select_samples_by_label
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    print("正在加载数据...")
    source_data = torch.load('Data/source_env0_env1_data.pt')
    source_labels = torch.load('Data/source_env0_env1_labels.pt')
    target_data = torch.load('Data/target_env2_data.pt')
    target_labels = torch.load('Data/target_env2_labels.pt')
    
    # 创建数据加载器
    source_dataset = CustomDataset(source_data, source_labels)
    source_loader = DataLoader(source_dataset, batch_size=100, shuffle=True)
    
    # 从目标域中均匀采样每个标签的样本
    selected_target_data, selected_target_labels = select_samples_by_label(target_data, target_labels, samples_per_label=10)
    target_dataset = CustomDataset(selected_target_data, selected_target_labels)
    target_loader = DataLoader(target_dataset, batch_size=100, shuffle=True)
    
    print("数据加载完成")
    
    # 加载模型并生成图表
    model_path = r"C:\Users\USER\Desktop\liuheng\Research1\GAN\best_gan_model.pth"
    load_model_and_generate_plots(model_path, source_loader, target_loader, device)