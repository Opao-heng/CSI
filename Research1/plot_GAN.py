import os
from datetime import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    
    # 步骤2: 提取各项损失数据（共乚5项）
    d_losses = [loss['d_loss'] for loss in train_loss_history]  # 判别器损失
    g_adv_losses = [loss['g_adv_loss'] for loss in train_loss_history]  # 对抗损失
    mmd_losses = [loss['mmd_loss'] for loss in train_loss_history]  # MMD损失
    freq_losses = [loss['freq_loss'] for loss in train_loss_history]  # 频域一致性损失
    g_losses = [loss['g_loss'] for loss in train_loss_history]  # 生成器总损失
    epochs = range(1, len(train_loss_history) + 1)
    
    # 步骤3: 创建大型图表，包含5个子图 (分组5个损失)
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
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
    
    # 子图5: 生成器总损失
    axes[2, 0].plot(epochs, g_losses, 'b-', linewidth=2, label='Generator Total Loss')
    axes[2, 0].set_title('Generator Total Loss (Weighted Sum)', fontsize=14, fontweight='bold')
    axes[2, 0].set_xlabel('Epoch', fontsize=12)
    axes[2, 0].set_ylabel('Loss', fontsize=12)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # 子图6: 所有火输图损失比较 (隐藏)
    axes[2, 1].axis('off')
    
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
    plt.plot(epochs, g_losses, 'b-', linewidth=2.5, label='Generator Total Loss', alpha=0.8)
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
    
    # 步骤6: 保存训练损失历史到JSON文件
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(train_loss_history, f, indent=4, ensure_ascii=False)
    print(f"  训练损失历史已保存到: {history_path}")

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
            'min_g_loss': min([l['g_loss'] for l in train_losses]) if train_losses else 0,
            'max_g_loss': max([l['g_loss'] for l in train_losses]) if train_losses else 0,
            'final_g_loss': train_losses[-1]['g_loss'] if train_losses else 0,
            'min_d_loss': min([l['d_loss'] for l in train_losses]) if train_losses else 0,
            'max_d_loss': max([l['d_loss'] for l in train_losses]) if train_losses else 0,
            'final_d_loss': train_losses[-1]['d_loss'] if train_losses else 0,
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
