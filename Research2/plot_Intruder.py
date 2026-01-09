import json
import matplotlib.pyplot as plt
import os
import sys
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from matplotlib import font_manager

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_Identify import IdentifyDetectionSystem
from model_Intruder import LearnableComprehensiveIntruderDetector
from Research2.DataProcess.dataloader_intruder import load_intruder_data, create_intruder_data_loaders

# 设置中文字体支持 - 中文宋体,英文数字Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 设置全局字体配置:使用font fallback机制实现中英文分离
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# 科研绘图风格配置
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.15
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5

print("已设置字体: 中文-宋体(SimSun), 英文/数字-Times New Roman")


def get_chinese_font_properties(size=20):
    """
    获取中文字体属性(宋体)
    """
    try:
        return font_manager.FontProperties(family='SimSun', size=size)
    except:
        return font_manager.FontProperties(family='sans-serif', size=size)


def get_english_font_properties(size=20):
    """
    获取英文/数字字体属性(Times New Roman)
    """
    try:
        return font_manager.FontProperties(family='Times New Roman', size=size)
    except:
        return font_manager.FontProperties(family='serif', size=size)


def get_mixed_font_properties(size=20):
    """
    获取混合字体属性(中文宋体+英文Times New Roman)
    通过设置fallback实现中英文分离
    """
    try:
        prop = font_manager.FontProperties(size=size)
        prop.set_family(['Times New Roman', 'SimSun'])
        return prop
    except:
        return font_manager.FontProperties(family='sans-serif', size=size)


def create_mixed_text_with_fonts(ax, text, fontsize=20, **kwargs):
    """
    创建支持中英文分离字体的文本对象
    中文使用宋体,英文和数字使用Times New Roman
    """
    import re
    
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
    has_english_or_digit = bool(re.search(r'[a-zA-Z0-9]', text))
    
    if has_chinese and has_english_or_digit:
        prop = font_manager.FontProperties(size=fontsize)
        prop.set_family(['Times New Roman', 'SimSun'])
        return text, prop
    elif has_chinese:
        return text, get_chinese_font_properties(fontsize)
    else:
        return text, get_english_font_properties(fontsize)


# 创建默认字体属性 - 统一字体大小为24
zh_font = get_chinese_font_properties(24)
en_font = get_english_font_properties(24)
mixed_font = get_mixed_font_properties(24)


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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 使用更专业的配色和样式
    ax.plot(epochs, train_losses, color='#2E86AB', linestyle='-', linewidth=2.5,
            marker='o', markersize=5, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor='#2E86AB', label='训练损失', alpha=0.9)
    
    # 根据规范，不显示标题及横纵坐标，统一字体大小为24
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # 优化图例样式
    legend = ax.legend(prop=zh_font, fontsize=24, loc='upper right', 
                      frameon=True, shadow=True, fancybox=True, 
                      framealpha=0.95, edgecolor='#CCCCCC')
    legend.get_frame().set_linewidth(1.2)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, color='gray')
    
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=24, width=1.5, length=6, colors='#333333')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练损失曲线已保存到 {save_path}")


def plot_auroc(val_aurocs, test_aurocs, save_path):
    """
    绘制 AUROC 曲线（验证集、测试集）
    """
    epochs = range(1, len(val_aurocs) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用更专业的配色方案
    ax.plot(epochs, val_aurocs, color='#18A558', linestyle='-', linewidth=2.5,
            marker='s', markersize=5, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor='#18A558', label='验证集', alpha=0.9)
    ax.plot(epochs, test_aurocs, color='#F18F01', linestyle='-', linewidth=2.5,
            marker='^', markersize=5, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor='#F18F01', label='测试集', alpha=0.9)

    # 根据规范，不显示标题及横纵坐标，统一字体大小为24
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 优化图例样式
    legend = ax.legend(prop=zh_font, fontsize=24, loc='lower right',
                      frameon=True, shadow=True, fancybox=True,
                      framealpha=0.95, edgecolor='#CCCCCC')
    legend.get_frame().set_linewidth(1.2)
        
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, color='gray')
        
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
        
    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=24, width=1.5, length=6, colors='#333333')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUROC 曲线已保存到 {save_path}")


def plot_confusion_matrix_from_scores(scores, labels, save_path, threshold=None):
    """
    根据分数和标签绘制混淆矩阵
    - threshold 为决策阈值；若为 None，则基于 ROC 曲线自动选择 Youden 指数 (TPR - FPR) 最大的最佳阈值
    """
    scores = np.array(scores).squeeze()
    labels = np.array(labels).astype(int)

    # 若未指定阈值，基于 ROC 曲线自动选择最佳阈值
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        threshold = thresholds[best_idx]

    preds = (scores >= threshold).astype(int)

    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 使用更专业的配色方案
    im = ax.imshow(cm, interpolation='nearest', cmap='YlOrRd', alpha=0.85)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label('样本数量', fontproperties=zh_font, fontsize=24, rotation=270, labelpad=20)

    classes = ['合法用户', '入侵者']
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontproperties=zh_font, fontsize=24)
    ax.set_yticklabels(classes, fontproperties=zh_font, fontsize=24)

    # 根据规范，不显示标题及横纵坐标，统一字体大小为24
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('')

    # 在每个格子中写上数字和百分比
    thresh = cm.max() / 2.
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / total * 100
            text_content = f'{cm[i, j]}\n({percentage:.1f}%)'
            ax.text(j, i, text_content,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "#333333",
                    fontproperties=en_font, fontsize=24, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵图已保存到 {save_path} (阈值 = {threshold:.3f})")


def extract_scores_and_labels(device):
    """
    从测试数据中提取模型的决策分数和真实标签
    """
    print("=== 提取入侵者检测模型决策分数 ===")
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 初始化身份识别模型
    print("加载身份识别模型...")
    identity_model_path = os.path.join(current_dir, "R_Identify", "best_identify_model.pth")
    identity_model = IdentifyDetectionSystem(num_classes=10, feature_dim=512, projection_dim=32).to(device)
    checkpoint = torch.load(identity_model_path, map_location=device)
    identity_model.load_state_dict(checkpoint['model_state_dict'])
    identity_model.eval()
    
    # 初始化入侵者检测模型
    print("加载入侵者检测模型...")
    intruder_model_path = os.path.join(current_dir, "R_Intruder", "best_intruder_detector.pth")
    intruder_model = LearnableComprehensiveIntruderDetector(num_known_users=10, feature_dim=512).to(device)
    checkpoint = torch.load(intruder_model_path, map_location=device)
    intruder_model.load_state_dict(checkpoint['model_state_dict'])
    intruder_model.eval()
    
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


def plot_roc_curve(scores, labels, save_path):
    """
    使用测试集分数绘制 ROC 曲线
    """
    scores = np.array(scores).squeeze()
    labels = np.array(labels).astype(int)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制ROC曲线 - 使用更专业的样式
    label_text = f'ROC曲线 (AUC = {roc_auc:.3f})'
    ax.plot(fpr, tpr, color='#C73E1D', lw=3.0, linestyle='-',
            label=label_text, alpha=0.9)
    
    # 绘制对角线参考线
    ax.plot([0, 1], [0, 1], color='#666666', lw=2.0, linestyle='--', 
            label='随机分类器 (AUC = 0.500)', alpha=0.6)
    
    # 填充ROC曲线下方区域
    ax.fill_between(fpr, tpr, alpha=0.15, color='#C73E1D')
    
    # 设置坐标轴范围
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    # 根据规范，不显示标题及横纵坐标，统一字体大小为24
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    
    # 优化图例样式
    legend = ax.legend(prop=zh_font, loc="lower right", fontsize=24, 
                      frameon=True, shadow=True, fancybox=True,
                      framealpha=0.95, edgecolor='#CCCCCC')
    legend.get_frame().set_linewidth(1.2)
    
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.tick_params(labelsize=24, width=1.5, length=6, colors='#333333')
    
    # 设置刻度标签字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)
    
    # 添加网格线
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC 曲线已保存到 {save_path}")


def plot_combined_metrics(test_accuracies, test_f1_scores, test_precisions, test_recalls, save_path):
    """
    将准确率、F1、Precision、Recall 四条曲线合并到一个2x2子图中绘制（仅测试集）
    """
    epochs = range(1, len(test_accuracies) + 1)
    
    # 增大图片尺寸，采用合理比例
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 专业配色方案
    colors = ['#A23B72', '#C73E1D', '#2E86AB', '#18A558']
    # 子图标识符
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    subplot_titles = [
        '准确率',
        'F1分数',
        '精确率',
        '召回率'
    ]
    ylabel_texts = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    
    for idx in range(4):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        color = colors[idx]
        
        # 选择对应的数据
        if idx == 0:
            data = test_accuracies
        elif idx == 1:
            data = test_f1_scores
        elif idx == 2:
            data = test_precisions
        else:
            data = test_recalls
        
        # 绘制曲线
        ax.plot(epochs, data, color=color, linestyle='-', linewidth=2.5,
                marker='so^d'[idx], markersize=5, markerfacecolor='white', 
                markeredgewidth=2, markeredgecolor=color, alpha=0.9)
        
        # 根据规范，不显示横纵坐标及标题，统一字体大小为24
        # 子图标识符采用混合字体(英文Times New Roman, 中文宋体)
        ax.set_ylabel('')
        ax.set_xlabel(subplot_labels[idx] + ' ' + subplot_titles[idx], 
                     fontproperties=mixed_font, fontsize=24, labelpad=15)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, color='gray')
        
        # 美化坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
        
        # 设置刻度 - 统一字体大小为24
        ax.tick_params(axis='both', which='major', labelsize=24, width=1.5, length=6, colors='#333333')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(en_font)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, hspace=0.5, wspace=0.3)  # 进一步调整子图间距和底部空间
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"四合一综合指标图已保存到 {save_path}")


def plot_all_training_curves(history_path, picture_dir):
    """
    根据训练历史绘制所有训练曲线
    """
    history = load_training_history(history_path)
    if history is None:
        return
    
    # 创建保存目录
    os.makedirs(picture_dir, exist_ok=True)
    
    # 提取训练数据 - 适配新格式
    train_losses = history.get('train_losses', [])
    test_accuracies = history.get('test_accuracies', [])
    test_f1_scores = history.get('test_f1s', [])
    test_precisions = history.get('test_precisions', [])
    test_recalls = history.get('test_recalls', [])
    
    if not train_losses:
        print("训练历史中缺少训练损失数据")
        return
    
    # 绘制训练损失曲线
    plot_training_loss(train_losses, os.path.join(picture_dir, 'training_loss.png'))
    
    # 绘制四合一综合指标图（准确率 + F1 + 精确率 + 召回率）
    if test_accuracies and test_f1_scores and test_precisions and test_recalls:
        plot_combined_metrics(
            test_accuracies,
            test_f1_scores,
            test_precisions,
            test_recalls,
            os.path.join(picture_dir, 'combined_metrics.png')
        )


def main():
    """
    主函数 - 用于直接运行入侵者检测可视化功能
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建图片保存目录
    picture_dir = 'R_Intruder'
    os.makedirs(picture_dir, exist_ok=True)

    # 绘制训练曲线（如果存在训练历史文件）
    intruder_history_path = 'R_Intruder/training_history.json'
    if os.path.exists(intruder_history_path):
        plot_all_training_curves(intruder_history_path, picture_dir)

    # 绘制 ROC 曲线 + 混淆矩阵
    scores, labels = extract_scores_and_labels(device)
    if scores is not None and labels is not None:
        plot_roc_curve(scores, labels, os.path.join(picture_dir, 'roc_curve.png'))
        plot_confusion_matrix_from_scores(scores, labels, os.path.join(picture_dir, 'confusion_matrix.png'))

    print("入侵者检测可视化完成!")


if __name__ == "__main__":
    main()