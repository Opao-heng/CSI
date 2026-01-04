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
    优化版本：适用于论文发表，采用四个子图布局
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
    
    # 提取各项损失数据
    source_losses = [comp['source'] for comp in loss_components_history]
    target_losses = [comp['target'] for comp in loss_components_history]
    
    # 学术期刊标准配色方案
    colors = {
        'total': '#2E86AB',      # 深蓝色
        'source': '#A23B72',     # 深紫红
        'target': '#F18F01',     # 橙色
        'mmd': '#06A77D'         # 青绿色
    }
    
    print(f"开始绘制训练指标，共 {len(train_losses)} 个epoch")
    print(f"总损失范围: [{min(train_losses):.4f}, {max(train_losses):.4f}]")
    print(f"源域损失范围: [{min(source_losses):.4f}, {max(source_losses):.4f}]")
    print(f"目标域损失范围: [{min(target_losses):.4f}, {max(target_losses):.4f}]")
    
    # 创建大型图表，包含4个子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # 子图1: 总损失
    axes[0, 0].plot(epochs, train_losses, color=colors['total'], linewidth=2, label='总损失')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=20)
    for label in axes[0, 0].get_xticklabels() + axes[0, 0].get_yticklabels():
        label.set_fontproperties(en_font)
    title_text, title_font = create_mixed_text_with_fonts(axes[0, 0], '(a) 总损失', 20)
    axes[0, 0].set_xlabel(title_text, fontproperties=title_font, fontsize=20)
    
    # 子图2: 源域损失
    axes[0, 1].plot(epochs, source_losses, color=colors['source'], linewidth=2, label='源域损失')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=20)
    for label in axes[0, 1].get_xticklabels() + axes[0, 1].get_yticklabels():
        label.set_fontproperties(en_font)
    title_text, title_font = create_mixed_text_with_fonts(axes[0, 1], '(b) 源域损失', 20)
    axes[0, 1].set_xlabel(title_text, fontproperties=title_font, fontsize=20)
    
    # 子图3: 目标域损失
    axes[1, 0].plot(epochs, target_losses, color=colors['target'], linewidth=2, label='目标域损失')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=20)
    for label in axes[1, 0].get_xticklabels() + axes[1, 0].get_yticklabels():
        label.set_fontproperties(en_font)
    title_text, title_font = create_mixed_text_with_fonts(axes[1, 0], '(c) 目标域损失', 20)
    axes[1, 0].set_xlabel(title_text, fontproperties=title_font, fontsize=20)
    
    # 子图4: MMD损失（兼容旧格式）
    if 'mmd' in loss_components_history[0]:
        mmd_losses = [comp['mmd'] for comp in loss_components_history]
        print(f"MMD损失范围: [{min(mmd_losses):.4f}, {max(mmd_losses):.4f}]")
        axes[1, 1].plot(epochs, mmd_losses, color=colors['mmd'], linewidth=2, label='MMD损失')
        title_text, title_font = create_mixed_text_with_fonts(axes[1, 1], '(d) MMD损失', 20)
    else:
        # 旧格式兼容：显示跨域特征损失或一致性损失
        cross_feature_losses = [comp.get('cross_feature', 0) for comp in loss_components_history]
        consistency_losses = [comp.get('consistency', 0) for comp in loss_components_history]
        
        if any(cross_feature_losses):
            axes[1, 1].plot(epochs, cross_feature_losses, color=colors['mmd'], linewidth=2, label='跨域特征损失')
            title_text, title_font = create_mixed_text_with_fonts(axes[1, 1], '(d) 跨域特征损失', 20)
        elif any(consistency_losses):
            axes[1, 1].plot(epochs, consistency_losses, color='#C73E1D', linewidth=2, label='一致性损失')
            title_text, title_font = create_mixed_text_with_fonts(axes[1, 1], '(d) 一致性损失', 20)
        else:
            title_text, title_font = create_mixed_text_with_fonts(axes[1, 1], '(d) 其他损失', 20)
    
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=20)
    for label in axes[1, 1].get_xticklabels() + axes[1, 1].get_yticklabels():
        label.set_fontproperties(en_font)
    axes[1, 1].set_xlabel(title_text, fontproperties=title_font, fontsize=20)
    
    # 调整布局并保存图形
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
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
    优化版：更适合科研论文发表
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
    
    # 创建图形，使用更适合论文的尺寸比例
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 定义科研论文常用的配色方案（红色和紫色，高对比度）
    colors = {
        'source': '#d62728',     # 红色
        'target': '#9467bd'      # 紫色
    }
    
    # 绘制源域和目标域测试准确率曲线 - 移除标记点，优化线型
    if src_test_accuracies:
        ax.plot(epochs, src_test_accuracies, color=colors['source'], 
                label='源域测试准确率', linewidth=2.5, linestyle='-', alpha=0.9)
    if tgt_test_accuracies:
        ax.plot(epochs, tgt_test_accuracies, color=colors['target'], 
                label='目标域测试准确率', linewidth=2.5, linestyle='-', alpha=0.9)
    
    # 设置标题和标签（中英文分离字体）
    #title_text, title_font = create_mixed_text_with_fonts(None, '源域与目标域测试准确率变化曲线', 18)
    #ax.set_title(title_text, fontproperties=title_font, fontweight='bold', pad=15)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '训练轮数', 18)
    ax.set_xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold', labelpad=10)
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '准确率 (%)', 18)
    ax.set_ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold', labelpad=10)
    
    # 优化网格样式 - 更细腻的网格
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color='gray')
    ax.set_axisbelow(True)  # 网格在图形下方
    
    # 优化图例样式
    legend_font = get_chinese_font_properties(size=18)
    legend = ax.legend(prop=legend_font, loc='lower right', 
                      frameon=True, fancybox=False, shadow=False,
                      framealpha=0.9, edgecolor='black', facecolor='white')
    legend.get_frame().set_linewidth(1.0)
    
    # 美化坐标轴 - 只保留左侧和底部边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # 设置y轴范围为0-100%，并设置主次刻度
    ax.set_ylim([0, 105])  # 稍微超出100，留出边距
    ax.set_yticks(np.arange(0, 101, 20))  # 主刻度：0, 20, 40, 60, 80, 100
    ax.set_yticks(np.arange(0, 101, 10), minor=True)  # 次刻度：每10
    
    # 优化刻度样式
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1.0, length=3)
    
    # 设置刻度标签字体为Times New Roman
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)
        label.set_fontsize(16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"源域与目标域测试准确率曲线已保存到 {save_path}")


def get_predictions_and_labels(model, dataloader, device, domain_type='target'):
    """
    获取模型在指定域上的所有预测标签和真实标签，用于生成混淆矩阵。
    
    参数:
        model: 待评估的模型
        dataloader: 数据加载器
        device: 计算设备
        domain_type: 域类型('source' 或 'target')
    
    返回:
        all_predictions: 所有预测标签列表
        all_labels: 所有真实标签列表
    """
    import torch
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            
            # 根据域类型选择不同的前向传播方式
            if domain_type == 'target':
                # 目标域：源域输入为零张量
                _, pred, _, _, _ = model(torch.zeros_like(data).to(device), data)
            else:
                # 源域：目标域输入为零张量
                pred, _, _, _, _ = model(data, torch.zeros_like(data).to(device))
            
            # 获取预测标签
            _, predicted = torch.max(pred, 1)
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    return all_predictions, all_labels


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
                cbar_kws={'shrink': 0.8, 'label': ''},
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
    #title_text, title_font = create_mixed_text_with_fonts(None, title, 20)
    #ax.set_title(title_text, fontproperties=title_font, fontweight='bold', pad=20)
    
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
    
    # 设置colorbar字体大小
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.tick_params(labelsize=20)
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(en_font)
            label.set_fontsize(20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到 {save_path}")


def plot_confusion_matrices_from_model(model_path, data_dir='Data', save_dir='R_CAL', device=None):
    """
    从保存的模型加载并生成混淆矩阵
    
    参数:
        model_path: 模型权重文件路径
        data_dir: 数据文件目录
        save_dir: 图表保存目录
        device: 计算设备
    """
    import torch
    from torch.utils.data import DataLoader
    from model_ATT import CrossAttentionModel
    from Research1.DataProcess.dataloder_ATT import CustomDataset
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n=== 开始生成混淆矩阵 ===")
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = CrossAttentionModel(num_classes=10).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载数据
    print("加载数据文件...")
    source_data = torch.load(os.path.join(data_dir, 'source_env0_env1_data.pt'))
    source_labels = torch.load(os.path.join(data_dir, 'source_env0_env1_labels.pt'))
    target_data = torch.load(os.path.join(data_dir, 'target_env2_gan_data.pt'))
    target_labels = torch.load(os.path.join(data_dir, 'target_env2_gan_labels.pt'))
    
    # 创建数据加载器
    source_dataset = CustomDataset(source_data, source_labels)
    target_dataset = CustomDataset(target_data, target_labels)
    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=False)
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 生成源域混淆矩阵
    print("生成源域混淆矩阵...")
    src_predictions, src_labels_list = get_predictions_and_labels(model, source_loader, device, domain_type='source')
    src_confusion_matrix_path = os.path.join(save_dir, 'source_confusion_matrix.png')
    plot_confusion_matrix(src_labels_list, src_predictions, 10, src_confusion_matrix_path, 
                         title='源域混淆矩阵')
    
    # 生成目标域混淆矩阵
    print("生成目标域混淆矩阵...")
    tgt_predictions, tgt_labels_list = get_predictions_and_labels(model, target_loader, device, domain_type='target')
    tgt_confusion_matrix_path = os.path.join(save_dir, 'target_confusion_matrix.png')
    plot_confusion_matrix(tgt_labels_list, tgt_predictions, 10, tgt_confusion_matrix_path,
                         title='目标域混淆矩阵')
    
    print("=== 混淆矩阵生成完成 ===")


if __name__ == "__main__":
    # 默认路径配置
    history_file_path = 'R_CAL/training_history.json'
    model_path = 'R_CAL/best_attention_model.pth'
    data_dir = 'Data'
    save_dir = 'R_CAL'

    # 1. 绘制训练损失曲线(包含所有损失组件)
    loss_curve_path = os.path.join(save_dir, 'attention_loss_curves.png')
    plot_training_loss_curves(history_file_path, loss_curve_path)

    # 2. 绘制训练与验证准确率曲线
    accuracy_curve_path = os.path.join(save_dir, 'attention_train_val_accuracy_curves.png')
    plot_accuracy_curves(history_file_path, accuracy_curve_path)

    # 3. 绘制源域与目标域测试准确率曲线
    test_accuracy_curve_path = os.path.join(save_dir, 'attention_test_accuracy_curves.png')
    plot_test_accuracy_curves(history_file_path, test_accuracy_curve_path)

    # 4. 绘制混淆矩阵
    plot_confusion_matrices_from_model(model_path, data_dir=data_dir, save_dir=save_dir)
    
    print("\n=== 交叉注意力模型可视化完成,所有图表已生成! ===")
