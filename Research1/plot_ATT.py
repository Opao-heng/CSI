import matplotlib.pyplot as plt
import os
import json

# 设置中文字体和美化参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True


"""
从JSON文件加载训练历史数据
"""
def load_training_history(history_path):
    if not os.path.exists(history_path):
        print(f"训练历史文件不存在: {history_path}")
        return None
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        return history
    except Exception as e:
        print(f"加载训练历史文件失败: {e}")
        return None


"""
绘制训练损失曲线图(总损失、源域损失、目标域损失、跨域特征损失、一致性损失)
"""
def plot_training_loss_curves(history_file_path, save_path):
    # 加载历史数据
    history = load_training_history(history_file_path)
    if history is None:
        return
    
    # 提取数据
    train_losses = history.get('train_losses', [])
    loss_components_history = history.get('loss_components_history', [])
    
    if not train_losses or not loss_components_history:
        print("训练历史中缺少必要的数据")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(14, 8))
    
    # 绘制总损失曲线
    plt.plot(epochs, train_losses, 'b-', label='总损失 (Total Loss)', linewidth=2.5, marker='o', markersize=4)
    
    # 绘制各项损失曲线
    source_losses = [comp['source'] for comp in loss_components_history]
    target_losses = [comp['target'] for comp in loss_components_history]
    cross_feature_losses = [comp['cross_feature'] for comp in loss_components_history]
    consistency_losses = [comp['consistency'] for comp in loss_components_history]
    
    plt.plot(epochs, source_losses, label='源域损失 (Source Loss)', linewidth=2.5, marker='s', markersize=4)
    plt.plot(epochs, target_losses, label='目标域损失 (Target Loss)', linewidth=2.5, marker='^', markersize=4)
    plt.plot(epochs, cross_feature_losses, label='跨域特征损失 (Cross-Feature Loss)', linewidth=2.5, marker='d', markersize=4)
    plt.plot(epochs, consistency_losses, label='一致性损失 (Consistency Loss)', linewidth=2.5, marker='*', markersize=6)
    
    plt.title('交叉注意力模型训练损失变化曲线', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('训练轮数 (Epoch)', fontsize=14, fontweight='bold')
    plt.ylabel('损失值 (Loss)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练损失曲线已保存到 {save_path}")


"""
绘制准确率曲线图(训练准确率、验证准确率)
"""
def plot_accuracy_curves(history_file_path, save_path):
    # 加载历史数据
    history = load_training_history(history_file_path)
    if history is None:
        return
    
    # 提取准确率数据
    train_accuracies = history.get('train_accuracies', [])
    val_accuracies = history.get('val_accuracies', [])
    
    if not train_accuracies and not val_accuracies:
        print("训练历史中缺少准确率数据")
        return
    
    epochs = range(1, len(train_accuracies) + 1 if train_accuracies else 
                  len(val_accuracies) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # 绘制训练和验证准确率曲线(带标记点)
    if train_accuracies:
        plt.plot(epochs, train_accuracies, 'b-', label='训练准确率 (Train Accuracy)', linewidth=2.5, marker='o', markersize=4)
    if val_accuracies:
        plt.plot(epochs, val_accuracies, 'g-', label='验证准确率 (Validation Accuracy)', linewidth=2.5, marker='s', markersize=4)
    
    plt.title('训练与验证准确率变化曲线', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('训练轮数 (Epoch)', fontsize=14, fontweight='bold')
    plt.ylabel('准确率 (%)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.set_ylim([0, 100])  # 设置y轴范围为0-100%
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练与验证准确率曲线已保存到 {save_path}")


"""
绘制测试准确率曲线图(源域测试准确率、目标域测试准确率)
"""
def plot_test_accuracy_curves(history_file_path, save_path):
    # 加载历史数据
    history = load_training_history(history_file_path)
    if history is None:
        return
    
    # 提取测试准确率数据
    test_results_history = history.get('test_results_history', [])
    
    # 从test_results_history中提取源域和目标域的测试准确率
    src_test_accuracies = []
    tgt_test_accuracies = []
    
    if test_results_history:
        for result in test_results_history:
            src_acc = result.get('src_test_accuracy', 0)
            tgt_acc = result.get('tgt_test_accuracy', 0)
            src_test_accuracies.append(src_acc)
            tgt_test_accuracies.append(tgt_acc)
    
    if not src_test_accuracies and not tgt_test_accuracies:
        print("训练历史中缺少测试准确率数据")
        return
    
    epochs = range(1, len(src_test_accuracies) + 1 if src_test_accuracies else 
                  len(tgt_test_accuracies) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # 绘制源域和目标域测试准确率曲线(带标记点)
    if src_test_accuracies:
        plt.plot(epochs, src_test_accuracies, 'r-', label='源域测试准确率 (Source Test Accuracy)', linewidth=2.5, marker='^', markersize=4)
    if tgt_test_accuracies:
        plt.plot(epochs, tgt_test_accuracies, 'm-', label='目标域测试准确率 (Target Test Accuracy)', linewidth=2.5, marker='d', markersize=4)
    
    plt.title('源域与目标域测试准确率变化曲线', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('训练轮数 (Epoch)', fontsize=14, fontweight='bold')
    plt.ylabel('准确率 (%)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.set_ylim([0, 100])  # 设置y轴范围为0-100%
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"源域与目标域测试准确率曲线已保存到 {save_path}")


"""
 一次性绘制所有训练结果图表
"""
def plot_all_training_results(history_file_path, save_dir='Attention'):
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("\n=== 开始生成所有训练可视化图表 ===")
    
    # 1. 绘制训练损失曲线(包含所有损失组件)
    loss_curve_path = os.path.join(save_dir, 'attention_loss_curves.png')
    plot_training_loss_curves(history_file_path, loss_curve_path)
    
    # 2. 绘制训练与验证准确率曲线
    accuracy_curve_path = os.path.join(save_dir, 'attention_train_val_accuracy_curves.png')
    plot_accuracy_curves(history_file_path, accuracy_curve_path)
    
    # 3. 绘制源域与目标域测试准确率曲线
    test_accuracy_curve_path = os.path.join(save_dir, 'attention_test_accuracy_curves.png')
    plot_test_accuracy_curves(history_file_path, test_accuracy_curve_path)
    
    print("=== 所有训练可视化图表生成完成 ===")


if __name__ == "__main__":
    import sys
    
    # 检查是否提供了训练历史文件路径
    if len(sys.argv) > 1:
        history_file_path = sys.argv[1]
    else:
        history_file_path = 'Attention/training_history.json'
    
    # 检查文件是否存在
    if not os.path.exists(history_file_path):
        print(f"错误: 训练历史文件不存在 - {history_file_path}")
        print("请先运行训练脚本 train_ATT.py 生成训练历史数据")
        sys.exit(1)
    
    # 绘制所有训练结果
    plot_all_training_results(history_file_path, save_dir='Attention')
    print("\n交叉注意力模型可视化完成,所有图表已生成!")
