import os
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
def plot_training_metrics(train_losses, evaluation_metrics, output_dir='GAN'):
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
    
    # 步骤4: 绘制判别器评分（真实样本vs生成样本）
    epochs_range = range(1, len(evaluation_metrics)+1)
    
    plt.subplot(2, 3, 2)
    fake_scores = [m.get('avg_fake_score', 0) for m in evaluation_metrics]
    real_scores = [m.get('avg_real_score', 0) for m in evaluation_metrics]
    plt.plot(epochs_range, fake_scores, label='Fake Score', marker='o', markersize=3)
    plt.plot(epochs_range, real_scores, label='Real Score', marker='s', markersize=3)
    plt.title('Discriminator Scores')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 步骤5: 绘制特征匹配误差(MSE)
    plt.subplot(2, 3, 3)
    feature_mse = [m.get('feature_mse', 0) for m in evaluation_metrics]
    plt.plot(epochs_range, feature_mse, color='green', marker='^', markersize=3)
    plt.title('Feature Matching MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)
    
    # 步骤6: 绘制特征余弦相似度
    plt.subplot(2, 3, 4)
    cosine_sim = [m.get('avg_cosine_similarity', 0) for m in evaluation_metrics]
    plt.plot(epochs_range, cosine_sim, color='orange', marker='d', markersize=3)
    plt.title('Average Cosine Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.grid(True, alpha=0.3)
    
    # 步骤7: 绘制特征多样性（方差）
    plt.subplot(2, 3, 5)
    diversity = [m.get('feature_diversity', 0) for m in evaluation_metrics]
    plt.plot(epochs_range, diversity, color='red', marker='*', markersize=8)
    plt.title('Feature Diversity')
    plt.xlabel('Epoch')
    plt.ylabel('Variance')
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
    save_metrics_summary(train_losses, evaluation_metrics, output_dir)


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
