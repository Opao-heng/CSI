import matplotlib.pyplot as plt
import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import font_manager

# 设置中文字体支持 - 中文宋体，英文数字Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 设置全局字体配置：使用font fallback机制实现中英文分离
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True

print("已设置字体: 中文-宋体(SimSun), 英文/数字-Times New Roman")


def get_chinese_font_properties(size=20):
    """
    获取中文字体属性（宋体）
    """
    try:
        return font_manager.FontProperties(family='SimSun', size=size)
    except:
        return font_manager.FontProperties(family='sans-serif', size=size)

def get_english_font_properties(size=20):
    """
    获取英文/数字字体属性（Times New Roman）
    """
    try:
        return font_manager.FontProperties(family='Times New Roman', size=size)
    except:
        return font_manager.FontProperties(family='serif', size=size)

def get_mixed_font_properties(size=20):
    """
    获取混合字体属性（中文宋体+英文Times New Roman）
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
    中文使用宋体，英文和数字使用Times New Roman
    
    Returns:
        formatted_text: 格式化后的文本
        font_properties: 字体属性
    """
    import re
    
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
    has_english_or_digit = bool(re.search(r'[a-zA-Z0-9]', text))
    
    if has_chinese and has_english_or_digit:
        # 混合文本：使用fallback机制
        prop = font_manager.FontProperties(size=fontsize)
        prop.set_family(['Times New Roman', 'SimSun'])
        return text, prop
    elif has_chinese:
        # 纯中文
        return text, get_chinese_font_properties(fontsize)
    else:
        # 纯英文/数字
        return text, get_english_font_properties(fontsize)

# 创建默认字体属性
zh_font = get_chinese_font_properties(20)
en_font = get_english_font_properties(20)
mixed_font = get_mixed_font_properties(20)

def load_training_history(history_path):
    """
    从JSON文件加载训练历史数据
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


def plot_training_loss_curves(history_file_path, save_path):
    """
    绘制训练损失曲线图(总损失、源域损失、目标域损失、MMD损失)
    """

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
    plt.plot(epochs, train_losses, 'b-', label='总损失', linewidth=2.5, marker='o', markersize=4)
    
    # 绘制各项损失曲线（支持新旧两种格式）
    source_losses = [comp['source'] for comp in loss_components_history]
    target_losses = [comp['target'] for comp in loss_components_history]
    
    # 兼容旧格式（cross_feature, consistency）和新格式（mmd）
    if 'mmd' in loss_components_history[0]:
        mmd_losses = [comp['mmd'] for comp in loss_components_history]
        plt.plot(epochs, source_losses, label='源域损失', linewidth=2.5, marker='s', markersize=4)
        plt.plot(epochs, target_losses, label='目标域损失', linewidth=2.5, marker='^', markersize=4)
        plt.plot(epochs, mmd_losses, label='MMD损失', linewidth=2.5, marker='d', markersize=4)
    else:
        # 旧格式兼容
        cross_feature_losses = [comp.get('cross_feature', 0) for comp in loss_components_history]
        consistency_losses = [comp.get('consistency', 0) for comp in loss_components_history]
        plt.plot(epochs, source_losses, label='源域损失', linewidth=2.5, marker='s', markersize=4)
        plt.plot(epochs, target_losses, label='目标域损失', linewidth=2.5, marker='^', markersize=4)
        if any(cross_feature_losses):
            plt.plot(epochs, cross_feature_losses, label='跨域特征损失', linewidth=2.5, marker='d', markersize=4)
        if any(consistency_losses):
            plt.plot(epochs, consistency_losses, label='一致性损失', linewidth=2.5, marker='*', markersize=6)
    
    # 设置标题和标签（中英文分离字体）
    title_text, title_font = create_mixed_text_with_fonts(None, '交叉注意力模型训练损失变化曲线', 20)
    plt.title(title_text, fontproperties=title_font, fontweight='bold', pad=20)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '训练轮数', 20)
    plt.xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '损失值', 20)
    plt.ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(prop=zh_font, fontsize=20, loc='upper right')
    plt.tight_layout()
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # 设置刻度标签字体为Times New Roman
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)
        label.set_fontsize(20)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练损失曲线已保存到 {save_path}")


def plot_accuracy_curves(history_file_path, save_path):
    """
    绘制准确率曲线图(训练准确率、验证准确率)
    """

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
        plt.plot(epochs, train_accuracies, 'b-', label='训练准确率', linewidth=2.5, marker='o', markersize=4)
    if val_accuracies:
        plt.plot(epochs, val_accuracies, 'g-', label='验证准确率', linewidth=2.5, marker='s', markersize=4)
    
    # 设置标题和标签（中英文分离字体）
    title_text, title_font = create_mixed_text_with_fonts(None, '训练与验证准确率变化曲线', 20)
    plt.title(title_text, fontproperties=title_font, fontweight='bold', pad=20)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '训练轮数', 20)
    plt.xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '准确率 (%)', 20)
    plt.ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(prop=zh_font, fontsize=20, loc='lower right')
    plt.tight_layout()
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.set_ylim([0, 100])  # 设置y轴范围为0-100%
    
    # 设置刻度标签字体为Times New Roman
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)
        label.set_fontsize(20)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练与验证准确率曲线已保存到 {save_path}")


def plot_test_accuracy_curves(history_file_path, save_path):
    """
    绘制测试准确率曲线图(源域测试准确率、目标域测试准确率)
    """
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
        plt.plot(epochs, src_test_accuracies, 'r-', label='源域测试准确率', linewidth=2.5, marker='^', markersize=4)
    if tgt_test_accuracies:
        plt.plot(epochs, tgt_test_accuracies, 'm-', label='目标域测试准确率', linewidth=2.5, marker='d', markersize=4)
    
    # 设置标题和标签（中英文分离字体）
    title_text, title_font = create_mixed_text_with_fonts(None, '源域与目标域测试准确率变化曲线', 20)
    plt.title(title_text, fontproperties=title_font, fontweight='bold', pad=20)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '训练轮数', 20)
    plt.xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '准确率 (%)', 20)
    plt.ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(prop=zh_font, fontsize=20, loc='lower right')
    plt.tight_layout()
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.set_ylim([0, 100])  # 设置y轴范围为0-100%
    
    # 设置刻度标签字体为Times New Roman
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)
        label.set_fontsize(20)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"源域与目标域测试准确率曲线已保存到 {save_path}")


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path, title='混淆矩阵'):
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        num_classes: 类别总数
        save_path: 图表保存路径
        title: 图表标题
    """
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # 绘制混淆矩阵热力图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 使用seaborn绘制热力图，但不显示注释
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=range(num_classes),
                yticklabels=range(num_classes),
                cbar_kws={'shrink': 0.8},
                linewidths=0.5,
                linecolor='gray',
                ax=ax)
    
    # 手动添加文本注释，使用Times New Roman字体
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j + 0.5, i + 0.5, str(cm[i, j]),
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=20,
                          fontproperties=en_font)
    
    # 设置标题和标签（中英文分离字体）
    title_text, title_font = create_mixed_text_with_fonts(None, title, 20)
    ax.set_title(title_text, fontproperties=title_font, fontweight='bold', pad=20)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '预测标签', 20)
    ax.set_xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '真实标签', 20)
    ax.set_ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
    
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # 设置刻度标签字体为Times New Roman
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)
        label.set_fontsize(20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到 {save_path}")


def plot_all_training_results(history_file_path, save_dir='Attention'):
    """
     一次性绘制所有训练结果图表
    """

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
        print("请先运行训练脚本 train_CNN.py 生成训练历史数据")
        sys.exit(1)
    
    # 绘制所有训练结果
    plot_all_training_results(history_file_path, save_dir='Attention')
    print("\n交叉注意力模型可视化完成,所有图表已生成!")
