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

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True

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

# 创建默认字体属性
zh_font = get_chinese_font_properties(20)
en_font = get_english_font_properties(20)
mixed_font = get_mixed_font_properties(20)

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
    plt.plot(epochs, train_losses, color='#0072BD', linestyle='-', linewidth=2.0,
             marker='^', markersize=6, label='训练损失')
    
    title_text, title_font = create_mixed_text_with_fonts(None, '训练损失变化曲线', 20)
    plt.title(title_text, fontproperties=title_font, fontweight='bold', pad=20)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '训练轮数', 20)
    plt.xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '损失值', 20)
    plt.ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
    
    plt.legend(prop=zh_font, fontsize=20, loc='upper right')
    plt.tight_layout()
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练损失曲线已保存到 {save_path}")

def plot_accuracy(val_accuracies, test_accuracies, save_path):
    """
    绘制准确率曲线（验证集、测试集）
    """
    epochs = range(1, len(val_accuracies) + 1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, val_accuracies, color='#0072BD', linestyle='-', linewidth=2.0,
             marker='^', markersize=6, label='验证集准确率')
    plt.plot(epochs, test_accuracies, color='#D95319', linestyle='-', linewidth=2.0,
             marker='v', markersize=6, label='测试集准确率')
    
    title_text, title_font = create_mixed_text_with_fonts(None, '验证集与测试集准确率变化曲线', 20)
    plt.title(title_text, fontproperties=title_font, fontweight='bold', pad=20)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '训练轮数', 20)
    plt.xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '准确率', 20)
    plt.ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
    
    plt.legend(prop=zh_font, fontsize=20, loc='lower right')
    plt.tight_layout()
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"准确率曲线已保存到 {save_path}")





def plot_auroc(val_aurocs, test_aurocs, save_path):
    """
    绘制 AUROC 曲线（验证集、测试集）
    """
    epochs = range(1, len(val_aurocs) + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, val_aurocs, 'g-', label='验证集', linewidth=2.5, marker='s', markersize=4)
    plt.plot(epochs, test_aurocs, 'r-', label='测试集', linewidth=2.5, marker='^', markersize=4)
    
    title_text, title_font = create_mixed_text_with_fonts(None, '验证集与测试集AUROC变化曲线', 20)
    plt.title(title_text, fontproperties=title_font, fontweight='bold', pad=20)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '训练轮数', 20)
    plt.xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, 'AUROC', 20)
    plt.ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
    
    plt.legend(prop=zh_font, fontsize=20, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUROC 曲线已保存到 {save_path}")


def plot_all_metrics_panel(val_f1_scores, test_f1_scores,
                           val_precisions, test_precisions,
                           val_recalls, test_recalls,
                           val_aurocs, test_aurocs,
                           save_path):
    """
    将 F1、Precision、Recall、AUROC 四条曲线合并到一个 2x2 子图中绘制
    上排：F1（左）、Precision（右）
    下排：Recall（左）、AUROC（右）
    """
    epochs = range(1, len(val_f1_scores) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    # 1. F1 分数
    ax = axes[0]
    ax.plot(epochs, val_f1_scores, color='#0072BD', linestyle='-', linewidth=2.0,
            marker='^', markersize=4, label='验证集')
    ax.plot(epochs, test_f1_scores, color='#D95319', linestyle='-', linewidth=2.0,
            marker='v', markersize=4, label='测试集')
    
    title_text, title_font = create_mixed_text_with_fonts(None, 'F1分数', 20)
    ax.set_title(title_text, fontproperties=title_font, fontweight='bold', pad=10)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '训练轮数', 20)
    ax.set_xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, 'F1', 20)
    ax.set_ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
    
    ax.legend(prop=zh_font, fontsize=20, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)

    # 2. Precision
    ax = axes[1]
    ax.plot(epochs, val_precisions, color='#0072BD', linestyle='-', linewidth=2.0,
            marker='^', markersize=4, label='验证集')
    ax.plot(epochs, test_precisions, color='#D95319', linestyle='-', linewidth=2.0,
            marker='v', markersize=4, label='测试集')
    
    title_text, title_font = create_mixed_text_with_fonts(None, '精确率', 20)
    ax.set_title(title_text, fontproperties=title_font, fontweight='bold', pad=10)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '训练轮数', 20)
    ax.set_xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '精确率', 20)
    ax.set_ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
    
    ax.legend(prop=zh_font, fontsize=20, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)

    # 3. Recall
    ax = axes[2]
    ax.plot(epochs, val_recalls, color='#0072BD', linestyle='-', linewidth=2.0,
            marker='^', markersize=4, label='验证集')
    ax.plot(epochs, test_recalls, color='#D95319', linestyle='-', linewidth=2.0,
            marker='v', markersize=4, label='测试集')
    
    title_text, title_font = create_mixed_text_with_fonts(None, '召回率', 20)
    ax.set_title(title_text, fontproperties=title_font, fontweight='bold', pad=10)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '训练轮数', 20)
    ax.set_xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '召回率', 20)
    ax.set_ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
    
    ax.legend(prop=zh_font, fontsize=20, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)

    # 4. AUROC
    ax = axes[3]
    ax.plot(epochs, val_aurocs, color='#0072BD', linestyle='-', linewidth=2.0,
            marker='^', markersize=4, label='验证集')
    ax.plot(epochs, test_aurocs, color='#D95319', linestyle='-', linewidth=2.0,
            marker='v', markersize=4, label='测试集')
    
    title_text, title_font = create_mixed_text_with_fonts(None, 'AUROC', 20)
    ax.set_title(title_text, fontproperties=title_font, fontweight='bold', pad=10)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '训练轮数', 20)
    ax.set_xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, 'AUROC', 20)
    ax.set_ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
    
    ax.legend(prop=zh_font, fontsize=20, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"F1 / Precision / Recall / AUROC 综合曲线图已保存到 {save_path}")



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

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    classes = ['合法用户', '入侵者']
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontproperties=zh_font, fontsize=20)
    ax.set_yticklabels(classes, fontproperties=zh_font, fontsize=20)

    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '真实标签', 20)
    ax.set_ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '预测标签', 20)
    ax.set_xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    # 标题中包含阈值数字，需要混合字体
    title_text = f'入侵者检测混淆矩阵'
    title_text_full, title_font = create_mixed_text_with_fonts(None, title_text, 20)
    plt.title(title_text_full, fontproperties=title_font, fontweight='bold', pad=20)

    # 在每个格子中写上数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontproperties=en_font, fontsize=20)

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
    intruder_model_path = os.path.join(current_dir, "R_Intruder", "best_intruder_detector.pth")
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


def plot_roc_curve(scores, labels, save_path):
    """
    使用测试集分数绘制 ROC 曲线
    """
    scores = np.array(scores).squeeze()
    labels = np.array(labels).astype(int)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(9, 7))
    
    # 绘制ROC曲线 - 去掉marker，使用平滑曲线
    label_text = f'ROC曲线'
    plt.plot(fpr, tpr, color='#0072BD', lw=2.5, linestyle='-',
             label=label_text)
    
    # 绘制对角线参考线
    plt.plot([0, 1], [0, 1], color='#888888', lw=2.0, linestyle='--', 
             label='随机分类器', alpha=0.7)
    
    # 设置坐标轴范围
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    # 设置坐标轴标签
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '假正率', 20)
    plt.xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '真正率', 20)
    plt.ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
    
    title_text, title_font = create_mixed_text_with_fonts(None, '入侵者检测ROC曲线', 20)
    plt.title(title_text, fontproperties=title_font, fontweight='bold', pad=20)
    
    # 美化图例
    plt.legend(prop=zh_font, loc="lower right", fontsize=20, frameon=True, shadow=False, 
              fancybox=False, framealpha=0.95)
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(labelsize=20, width=1.2)
    
    # 设置刻度标签字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)
    
    # 添加网格线，增加可读性
    plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC 曲线已保存到 {save_path}")

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
    val_metrics = history.get('val_metrics', [])  # (accuracy, f1, precision, recall, auroc)
    test_metrics = history.get('test_metrics', [])  # (accuracy, f1, precision, recall, auroc)
    
    if not train_losses or not val_metrics or not test_metrics:
        print("训练历史中缺少必要数据")
        return
    
    # 提取各项指标
    val_accuracies = [m[0] for m in val_metrics]
    val_f1_scores = [m[1] for m in val_metrics]
    val_precisions = [m[2] for m in val_metrics]
    val_recalls = [m[3] for m in val_metrics]
    val_aurocs = [m[4] for m in val_metrics]
    
    test_accuracies = [m[0] for m in test_metrics]
    test_f1_scores = [m[1] for m in test_metrics]
    test_precisions = [m[2] for m in test_metrics]
    test_recalls = [m[3] for m in test_metrics]
    test_aurocs = [m[4] for m in test_metrics]
    
    # 绘制训练损失曲线
    plot_training_loss(train_losses, os.path.join(picture_dir, 'training_loss.png'))
    
    # 绘制准确率曲线（验证集和测试集）
    plot_accuracy(val_accuracies, test_accuracies, os.path.join(picture_dir, 'accuracy.png'))
    
    # 将 F1 / Precision / Recall / AUROC 四个指标合并到一张图中
    plot_all_metrics_panel(
        val_f1_scores, test_f1_scores,
        val_precisions, test_precisions,
        val_recalls, test_recalls,
        val_aurocs, test_aurocs,
        os.path.join(picture_dir, 'metrics_panel.png')
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