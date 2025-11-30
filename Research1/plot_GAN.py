import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


"""
绘制GAN训练过程中的各项指标
参数:
  train_losses - 训练损失列表，长度为epoch数
  evaluation_metrics - 评估指标列表，每个元素是一个包含多种指标的字典
  output_dir - 图形保存目录，默认为'GAN'
返回: 无返回值，直接保存图形文件
"""
def plot_training_metrics(train_losses, output_dir='GAN'):
    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤2: 创建大型图表，包含5个子图
    plt.figure(figsize=(15, 10))
    
    # 步骤3: 绘制训练损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    
    # 步骤8: 调整布局并保存图形
    plt.tight_layout()
    
    # 保存图形到GAN目录
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to {plot_path}")
    
    # 显示图形
    plt.show()
    
    # 步骤9: 生成并保存统计信息
    save_metrics_summary(train_losses, output_dir)


"""
保存训练指标的统计摘要到文本文件
参数:
  train_losses - 训练损失列表
  evaluation_metrics - 评估指标列表
  output_dir - 输出目录
返回: 无返回值，直接保存到文件
"""
def save_metrics_summary(train_losses, evaluation_metrics, output_dir='GAN'):
    # 步骤1: 计算损失相关统计
    if train_losses:
        min_loss = min(train_losses)
        max_loss = max(train_losses)
        avg_loss = np.mean(train_losses)
        final_loss = train_losses[-1]
    else:
        min_loss = max_loss = avg_loss = final_loss = 0
    
    # 步骤2: 计算评估指标的统计
    if evaluation_metrics:
        fake_scores = [m.get('avg_fake_score', 0) for m in evaluation_metrics]
        real_scores = [m.get('avg_real_score', 0) for m in evaluation_metrics]
        mse_values = [m.get('feature_mse', 0) for m in evaluation_metrics]
        cosine_values = [m.get('avg_cosine_similarity', 0) for m in evaluation_metrics]
        diversity_values = [m.get('feature_diversity', 0) for m in evaluation_metrics]
    else:
        fake_scores = real_scores = mse_values = cosine_values = diversity_values = []
    
    # 步骤3: 生成摘要文本
    summary = f"""
========== GAN Training Summary ==========
Training Loss Statistics:
  - Minimum Loss: {min_loss:.6f}
  - Maximum Loss: {max_loss:.6f}
  - Average Loss: {avg_loss:.6f}
  - Final Loss: {final_loss:.6f}

Discriminator Scores:
  - Fake Score (avg): {np.mean(fake_scores):.6f}
  - Real Score (avg): {np.mean(real_scores):.6f}

Feature Matching:
  - MSE (avg): {np.mean(mse_values):.6f}
  - Cosine Similarity (avg): {np.mean(cosine_values):.6f}

Feature Diversity:
  - Variance (avg): {np.mean(diversity_values):.6f}

Total Epochs: {len(evaluation_metrics)}
==========================================
"""
    
    # 步骤4: 保存摘要到文件
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Training summary saved to {summary_path}")


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
            'min_loss': min([l['g_loss'] for l in train_losses]) if train_losses else 0,
            'max_loss': max([l['g_loss'] for l in train_losses]) if train_losses else 0,
            'final_loss': train_losses[-1]['g_loss'] if train_losses else 0,
            'total_epochs': len(train_losses)
        },
        'full_training_history': train_losses
    }

    # 步骤3: 将结果保存为JSON文件
    result_path = os.path.join(output_dir, 'gan_evaluation_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"评估结果已保存到: {result_path}")
