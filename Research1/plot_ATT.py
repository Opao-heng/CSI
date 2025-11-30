"""
交叉注意力模型训练结果可视化模块 (plot_ATT.py)

该模块负责可视化交叉注意力模型的训练过程，包括：
1. 训练损失与测试损失曲线
2. 测试准确率曲线
3. 学习率调度变化曲线
"""

import matplotlib.pyplot as plt
import os
import json


"""
函数: plot_training_curves

功能: 绘制训练过程中的损失、准确率和学习率曲线，并保存到Attention目录

参数:
  train_losses: list - 每个epoch的训练损失列表
  test_losses: list - 每个epoch的测试损失列表
  test_accuracies: list - 每个epoch的测试准确率列表
  optimizer: torch.optim.Optimizer - 优化器对象（用于获取最终学习率）
  save_dir: str - 保存图像的目录，默认为 'Attention'

返回值: 无

工作流程:
  1. 创建3个子图（损失曲线、准确率曲线、学习率曲线）
  2. 绘制训练/测试损失对比
  3. 绘制测试准确率变化
  4. 绘制学习率调度过程
  5. 保存高质量图像到Attention目录
"""
def plot_training_curves(train_losses, test_losses, test_accuracies, optimizer, save_dir='Attention'):
    # 步骤1: 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 步骤2: 创建图形窗口
    plt.figure(figsize=(15, 5))

    # 步骤3: 子图1 - 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(test_losses, label='Test Loss', color='red', linewidth=2)
    plt.title('Loss Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 步骤4: 子图2 - 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(test_accuracies, label='Test Accuracy', color='green', linewidth=2)
    plt.title('Accuracy Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 步骤5: 子图3 - 绘制学习率变化曲线
    plt.subplot(1, 3, 3)
    lrs = [group['lr'] for group in optimizer.param_groups]
    plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='purple', linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # 步骤6: 保存图像到Attention目录
    save_path = os.path.join(save_dir, 'Attention_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到 {save_path}")
    
    plt.show()


"""
函数: save_training_history

功能: 将训练历史数据保存为JSON文件，便于后续分析和恢复

参数:
  train_losses: list - 训练损失列表
  test_losses: list - 测试损失列表
  test_accuracies: list - 测试准确率列表
  best_accuracy: float - 最佳准确率
  total_epochs: int - 实际训练轮数
  save_dir: str - 保存文件的目录，默认为 'Attention'

返回值: 无
"""
def save_training_history(train_losses, test_losses, test_accuracies, best_accuracy, total_epochs, save_dir='Attention'):
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 构建历史数据字典
    history = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'best_accuracy': best_accuracy,
        'total_epochs': total_epochs
    }
    
    # 保存为JSON格式
    save_path = os.path.join(save_dir, 'training_history.json')
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"训练历史已保存到 {save_path}")
