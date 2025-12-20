import json
import matplotlib.pyplot as plt
import os
import sys
import torch
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_identify import IdentifyDetectionSystem
from model_intruder import LearnableComprehensiveIntruderDetector
from Research2.Process.dataloader_intruder import load_intruder_data, create_intruder_data_loaders

# 设置中文字体和美化参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True

def load_training_history(history_path):
    """
    加载训练历史数据
    """
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

def plot_training_loss(train_losses, save_path):
    """
    绘制训练损失曲线
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2.5, marker='o', markersize=4)
    plt.title('训练损失变化曲线', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('训练轮数', fontsize=14, fontweight='bold')
    plt.ylabel('损失值', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
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

def plot_accuracy(val_accuracies, test_accuracies, save_path):
    """
    绘制准确率曲线（验证集、测试集）
    """
    epochs = range(1, len(val_accuracies) + 1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, val_accuracies, 'g-', label='验证集准确率', linewidth=2.5, marker='s', markersize=4)
    plt.plot(epochs, test_accuracies, 'r-', label='测试集准确率', linewidth=2.5, marker='^', markersize=4)
    plt.title('验证集与测试集准确率变化曲线', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('训练轮数', fontsize=14, fontweight='bold')
    plt.ylabel('准确率', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"准确率曲线已保存到 {save_path}")

def plot_f1_score(val_f1_scores, test_f1_scores, save_path):
    """
    绘制F1分数曲线
    """
    epochs = range(1, len(val_f1_scores) + 1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, val_f1_scores, 'g-', label='验证集F1分数', linewidth=2.5, marker='s', markersize=4)
    plt.plot(epochs, test_f1_scores, 'r-', label='测试集F1分数', linewidth=2.5, marker='^', markersize=4)
    plt.title('验证集与测试集F1分数变化曲线', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('训练轮数', fontsize=14, fontweight='bold')
    plt.ylabel('F1分数', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"F1分数曲线已保存到 {save_path}")

def plot_precision(val_precisions, test_precisions, save_path):
    """
    绘制精确率曲线
    """
    epochs = range(1, len(val_precisions) + 1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, val_precisions, 'g-', label='验证集精确率', linewidth=2.5, marker='s', markersize=4)
    plt.plot(epochs, test_precisions, 'r-', label='测试集精确率', linewidth=2.5, marker='^', markersize=4)
    plt.title('验证集与测试集精确率变化曲线', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('训练轮数', fontsize=14, fontweight='bold')
    plt.ylabel('精确率', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"精确率曲线已保存到 {save_path}")

def plot_recall(val_recalls, test_recalls, save_path):
    """
    绘制召回率曲线
    """
    epochs = range(1, len(val_recalls) + 1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, val_recalls, 'g-', label='验证集召回率', linewidth=2.5, marker='s', markersize=4)
    plt.plot(epochs, test_recalls, 'r-', label='测试集召回率', linewidth=2.5, marker='^', markersize=4)
    plt.title('验证集与测试集召回率变化曲线', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('训练轮数', fontsize=14, fontweight='bold')
    plt.ylabel('召回率', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"召回率曲线已保存到 {save_path}")

def plot_known_vs_intruder_distribution(scores, labels, save_path):
    """
    绘制已知用户和入侵者在入侵者模型决策后的分布图
    """
    # 分离已知用户和入侵者的分数
    known_user_scores = scores[labels == 0]
    intruder_scores = scores[labels == 1]
    
    plt.figure(figsize=(12, 8))
    
    # 绘制直方图
    plt.hist(known_user_scores, bins=50, alpha=0.7, label='已知用户', color='#1f77b4', edgecolor='black', linewidth=0.5)
    plt.hist(intruder_scores, bins=50, alpha=0.7, label='入侵者', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    
    # 添加统计信息
    known_mean = np.mean(known_user_scores)
    intruder_mean = np.mean(intruder_scores)
    
    plt.axvline(known_mean, color='#1f77b4', linestyle='--', linewidth=2, 
                label=f'已知用户均值: {known_mean:.3f}')
    plt.axvline(intruder_mean, color='#ff7f0e', linestyle='--', linewidth=2, 
                label=f'入侵者均值: {intruder_mean:.3f}')
    
    plt.title('已知用户与入侵者决策分数分布对比', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('决策分数', fontsize=14, fontweight='bold')
    plt.ylabel('频次', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已知用户与入侵者分布图已保存到 {save_path}")

def extract_scores_and_labels(device):
    """
    从测试数据中提取模型的决策分数和真实标签
    """
    print("=== 提取入侵者检测模型决策分数 ===")
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 初始化身份识别模型
    print("加载身份识别模型...")
    identity_model_path = os.path.join(current_dir, "identify", "best_identify_model.pth")
    if not os.path.exists(identity_model_path):
        print(f"未找到身份识别模型: {identity_model_path}")
        return None, None
    
    identity_model = IdentifyDetectionSystem(num_classes=10, feature_dim=128, projection_dim=32).to(device)
    checkpoint = torch.load(identity_model_path, map_location=device)
    identity_model.load_state_dict(checkpoint['model_state_dict'])
    identity_model.eval()
    print(f"身份识别模型加载完成")
    
    # 初始化入侵者检测模型
    print("加载入侵者检测模型...")
    intruder_model_path = os.path.join(current_dir, "intruder", "best_intruder_detector.pth")
    if not os.path.exists(intruder_model_path):
        print(f"未找到入侵者检测模型: {intruder_model_path}")
        return None, None
    
    intruder_model = LearnableComprehensiveIntruderDetector(num_known_users=10, feature_dim=128).to(device)
    checkpoint = torch.load(intruder_model_path, map_location=device)
    intruder_model.load_state_dict(checkpoint['model_state_dict'])
    intruder_model.eval()
    print(f"入侵者检测模型加载完成")
    
    # 加载入侵者检测数据
    print("加载入侵者检测数据...")
    datasets = load_intruder_data()
    data_loaders = create_intruder_data_loaders(datasets, batch_size=32)
    
    # 提取测试集的预测结果
    print("提取测试集预测结果...")
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loaders['intruder_test']):
            # 处理不同格式的batch数据
            if len(batch) == 3:
                data, labels, identity_labels = batch
            else:
                data, labels = batch
                identity_labels = None  # 如果没有identity_labels，则设为None
            
            # 移动到设备
            data = data.to(device)
            labels = labels.to(device)
            
            # 使用身份识别模型提取特征
            identity_outputs = identity_model(data)
            features = identity_outputs['features']
            logits = identity_outputs['logits']
            
            # 检查特征和logits的维度，确保至少是2D
            if features.dim() == 1:
                features = features.unsqueeze(0)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            # 确保batch维度一致
            batch_size = data.size(0)
            if features.size(0) != batch_size:
                features = features[:batch_size] if features.size(0) > batch_size else features
            if logits.size(0) != batch_size:
                logits = logits[:batch_size] if logits.size(0) > batch_size else logits
            
            # 使用综合入侵者检测器
            if identity_labels is not None:
                detector_outputs = intruder_model(features, logits, identity_labels)
            else:
                # 如果没有identity_labels，创建一个默认的标签数组
                # 假设所有样本都是合法用户（标签为0）
                default_identity_labels = torch.zeros(batch_size, dtype=torch.long, device=features.device)
                detector_outputs = intruder_model(features, logits, default_identity_labels)
            
            scores = detector_outputs['probabilities']  # 获取概率分数
            
            # 确保分数维度一致
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            
            # 收集预测结果和真实标签
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if len(all_scores) == 0:
        print("未能提取到预测结果")
        return None, None
    
    print(f"提取到 {len(all_scores)} 个样本的预测结果")
    return np.array(all_scores), np.array(all_labels)

def plot_all_training_curves(history_path, picture_dir):
    """
    根据训练历史绘制所有训练曲线
    """
    history = load_training_history(history_path)
    if history is None:
        return
    
    # 创建保存目录
    os.makedirs(picture_dir, exist_ok=True)
    
    # 提取训练数据
    train_losses = history.get('train_losses', [])
    val_metrics = history.get('val_metrics', [])  # (accuracy, f1, precision, recall)
    test_metrics = history.get('test_metrics', [])  # (accuracy, f1, precision, recall)
    
    if not train_losses or not val_metrics or not test_metrics:
        print("训练历史中缺少必要数据")
        return
    
    # 提取各项指标
    val_accuracies = [m[0] for m in val_metrics]
    val_f1_scores = [m[1] for m in val_metrics]
    val_precisions = [m[2] for m in val_metrics]
    val_recalls = [m[3] for m in val_metrics]
    
    test_accuracies = [m[0] for m in test_metrics]
    test_f1_scores = [m[1] for m in test_metrics]
    test_precisions = [m[2] for m in test_metrics]
    test_recalls = [m[3] for m in test_metrics]
    
    # 绘制训练损失曲线
    plot_training_loss(train_losses, os.path.join(picture_dir, 'training_loss.png'))
    
    # 绘制准确率曲线（验证集和测试集）
    plot_accuracy(val_accuracies, test_accuracies, os.path.join(picture_dir, 'accuracy.png'))
    
    # 绘制F1分数曲线
    plot_f1_score(val_f1_scores, test_f1_scores, os.path.join(picture_dir, 'f1_score.png'))
    
    # 绘制精确率曲线
    plot_precision(val_precisions, test_precisions, os.path.join(picture_dir, 'precision.png'))
    
    # 绘制召回率曲线
    plot_recall(val_recalls, test_recalls, os.path.join(picture_dir, 'recall.png'))

def main():
    """
    主函数 - 用于直接运行入侵者检测可视化功能
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建图片保存目录
    picture_dir = 'intruder'
    os.makedirs(picture_dir, exist_ok=True)

    # 绘制训练曲线（如果存在训练历史文件）
    intruder_history_path = 'intruder/training_history.json'
    if os.path.exists(intruder_history_path):
        plot_all_training_curves(intruder_history_path, picture_dir)

    # 绘制已知用户和入侵者分布图
    scores, labels = extract_scores_and_labels(device)
    if scores is not None and labels is not None:
        plot_known_vs_intruder_distribution(scores, labels, os.path.join(picture_dir, 'known_vs_intruder_distribution.png'))

    print("入侵者检测可视化完成!")


if __name__ == "__main__":
    main()