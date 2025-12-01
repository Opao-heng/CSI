import os
import json
import matplotlib.pyplot as plt


"""
绘制CVAE训练过程中的各项指标
参数:
  train_loss_history - 训练损失历史列表，每个元素是包含所有损失的字典
  output_dir - 图形保存目录，默认为'CVAE'
返回: 无返回值，直接保存图形文件
"""
def plot_training_metrics(train_loss_history, output_dir='CVAE'):
    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤2: 提取各项损失数据（共4项）
    total_losses = [loss['total_loss'] for loss in train_loss_history]  # 总损失
    recon_losses = [loss['recon_loss'] for loss in train_loss_history]  # 重构损失
    kl_losses = [loss['kl_loss'] for loss in train_loss_history]  # KL散度损失
    freq_losses = [loss['freq_loss'] for loss in train_loss_history]  # 频域一致性损失
    epochs = range(1, len(train_loss_history) + 1)
    
    # 步骤3: 创建大型图表，包含4个子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 子图1: 总损失
    axes[0, 0].plot(epochs, total_losses, 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 子图2: 重构损失
    axes[0, 1].plot(epochs, recon_losses, 'g-', linewidth=2, label='Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 子图3: KL散度损失
    axes[1, 0].plot(epochs, kl_losses, 'r-', linewidth=2, label='KL Divergence Loss')
    axes[1, 0].set_title('KL Divergence Loss', fontsize=14, fontweight='bold')
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
    
    # 步骤5: 绘制所有损失在一张图上的对比
    plt.figure(figsize=(14, 8))
    plt.plot(epochs, total_losses, 'b-', linewidth=2.5, label='Total Loss', alpha=0.8)
    plt.plot(epochs, recon_losses, 'g-', linewidth=2.5, label='Reconstruction Loss', alpha=0.8)
    plt.plot(epochs, kl_losses, 'r-', linewidth=2.5, label='KL Loss', alpha=0.8)
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
将CVAE的评估结果保存为JSON文件
"""
def save_evaluation_results(comprehensive_metrics, train_losses, output_dir='CVAE'):
    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 步骤2: 整理评估指标和训练损失统计
    results = {
        'timestamp': str(__import__('datetime').datetime.now()),
        'evaluation_metrics': comprehensive_metrics,
        'training_loss_summary': {
            'min_total_loss': min([l['total_loss'] for l in train_losses]) if train_losses else 0,
            'max_total_loss': max([l['total_loss'] for l in train_losses]) if train_losses else 0,
            'final_total_loss': train_losses[-1]['total_loss'] if train_losses else 0,
            'min_recon_loss': min([l['recon_loss'] for l in train_losses]) if train_losses else 0,
            'max_recon_loss': max([l['recon_loss'] for l in train_losses]) if train_losses else 0,
            'final_recon_loss': train_losses[-1]['recon_loss'] if train_losses else 0,
            'min_kl_loss': min([l['kl_loss'] for l in train_losses]) if train_losses else 0,
            'max_kl_loss': max([l['kl_loss'] for l in train_losses]) if train_losses else 0,
            'final_kl_loss': train_losses[-1]['kl_loss'] if train_losses else 0,
            'min_freq_loss': min([l['freq_loss'] for l in train_losses]) if train_losses else 0,
            'max_freq_loss': max([l['freq_loss'] for l in train_losses]) if train_losses else 0,
            'final_freq_loss': train_losses[-1]['freq_loss'] if train_losses else 0,
            'total_epochs': len(train_losses)
        },
        'full_training_history': train_losses
    }

    # 步骤3: 将结果保存为JSON文件
    result_path = os.path.join(output_dir, 'cvae_evaluation_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"  评估结果已保存到: {result_path}")