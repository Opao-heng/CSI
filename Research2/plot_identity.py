import json
import matplotlib.pyplot as plt
import os
import sys
import torch
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import font_manager
from model_identify import IdentifyDetectionSystem

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

def plot_training_loss_curves(history_file_path, save_path):
    """
    从历史文件中读取数据并绘制训练损失曲线图（总损失、身份损失、对比损失）
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
    
    plt.figure(figsize=(12, 8))
    
    # 绘制总损失曲线
    plt.plot(epochs, train_losses, 'b-', label='总损失', linewidth=2.5, marker='o', markersize=4)
    
    # 绘制各项损失曲线
    identity_losses = [comp['identity'] for comp in loss_components_history]
    contrastive_losses = [comp['contrastive'] for comp in loss_components_history]
    
    plt.plot(epochs, identity_losses, label='身份损失', linewidth=2.5, marker='s', markersize=4)
    plt.plot(epochs, contrastive_losses, label='对比损失', linewidth=2.5, marker='^', markersize=4)
    
    # 设置标题和标签 - 中文用宋体，英文数字用Times New Roman
    title_text, title_font = create_mixed_text_with_fonts(None, '训练损失变化曲线', 20)
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
    
    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练损失曲线已保存到 {save_path}")

def plot_accuracy_curves(history_file_path, save_path):
    """
    从历史文件中读取数据并绘制准确率曲线图（训练准确率、验证准确率）
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
    
    # 绘制训练和验证准确率曲线（带标记点）
    if train_accuracies:
        plt.plot(epochs, train_accuracies, 'b-', label='训练准确率', linewidth=2.5, marker='o', markersize=4)
    if val_accuracies:
        plt.plot(epochs, val_accuracies, 'g-', label='验证准确率', linewidth=2.5, marker='s', markersize=4)
    
    # 设置标题和标签
    title_text, title_font = create_mixed_text_with_fonts(None, '训练与验证准确率变化曲线', 20)
    plt.title(title_text, fontproperties=title_font, fontweight='bold', pad=20)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '训练轮数', 20)
    plt.xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '准确率', 20)
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
    ax.set_ylim([0, 100])
    
    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练与验证准确率曲线已保存到 {save_path}")

def plot_test_accuracy_curves(history_file_path, save_path):
    """
    从历史文件中读取数据并绘制测试准确率曲线图（源域测试准确率、目标域测试准确率）
    """
    # 加载历史数据
    history = load_training_history(history_file_path)
    if history is None:
        return
    
    # 提取测试准确率数据
    test_accuracies = history.get('test_accuracies', [])
    
    # 从test_accuracies中提取源域和目标域的测试准确率
    src_test_accuracies = []
    tgt_test_accuracies = []
    
    if test_accuracies:
        for result in test_accuracies:
            src_acc = result.get('src_identity_test', 0)
            tgt_acc = result.get('tgt_identity_test', 0)
            src_test_accuracies.append(src_acc)
            tgt_test_accuracies.append(tgt_acc)
    
    if not src_test_accuracies and not tgt_test_accuracies:
        print("训练历史中缺少测试准确率数据")
        return
    
    epochs = range(1, len(src_test_accuracies) + 1 if src_test_accuracies else 
                  len(tgt_test_accuracies) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # 绘制源域和目标域测试准确率曲线（带标记点）
    if src_test_accuracies:
        plt.plot(epochs, src_test_accuracies, 'r-', label='源域测试准确率', linewidth=2.5, marker='^', markersize=4)
    if tgt_test_accuracies:
        plt.plot(epochs, tgt_test_accuracies, 'm-', label='目标域测试准确率', linewidth=2.5, marker='d', markersize=4)
    
    # 设置标题和标签
    title_text, title_font = create_mixed_text_with_fonts(None, '源域与目标域测试准确率变化曲线', 20)
    plt.title(title_text, fontproperties=title_font, fontweight='bold', pad=20)
    
    xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '训练轮数', 20)
    plt.xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
    
    ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '准确率', 20)
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
    ax.set_ylim([0, 100])
    
    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(en_font)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"源域与目标域测试准确率曲线已保存到 {save_path}")

def visualize_identity_features(device):
    """
    可视化身份识别模型的特征分布
    """
    print("=== 身份识别模型特征可视化 ===")
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建保存目录
    picture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'R_Identify')
    os.makedirs(picture_dir, exist_ok=True)
    
    # 绘制准确率变化曲线（如果存在训练历史文件）
    identity_history_path = os.path.join(current_dir, "R_Identify", "training_history.json")
    if os.path.exists(identity_history_path):
        # 绘制训练与验证准确率曲线
        save_path_train_val_accuracy = os.path.join(picture_dir, 'identity_train_val_accuracy_curves.png')
        plot_accuracy_curves(identity_history_path, save_path_train_val_accuracy)
        
        # 绘制源域与目标域测试准确率曲线
        save_path_test_accuracy = os.path.join(picture_dir, 'identity_test_accuracy_curves.png')
        plot_test_accuracy_curves(identity_history_path, save_path_test_accuracy)


def plot_identity_confusion_matrix(device, save_path_source, save_path_target):
    """
    生成并绘制身份识别模型的源域和目标域混淆矩阵
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 初始化身份识别模型
    print("加载身份识别模型用于混淆矩阵生成...")
    model = IdentifyDetectionSystem(num_classes=10, feature_dim=128, projection_dim=32).to(device)
    model.eval()
    
    # 加载模型权重（如果存在）
    identity_model_path = os.path.join(current_dir, "R_Identify", "best_identify_model.pth")
    if not os.path.exists(identity_model_path):
        print(f"未找到身份识别模型: {identity_model_path}")
        return
    
    checkpoint = torch.load(identity_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"身份识别模型加载完成")
    
    # === 1. 处理源域数据 ===
    print("\n处理源域数据...")
    source_data_path = os.path.join(current_dir, 'Data', 'source_env0_env1_data.pt')
    source_labels_path = os.path.join(current_dir, 'Data', 'source_env0_env1_labels.pt')
    
    if os.path.exists(source_data_path) and os.path.exists(source_labels_path):
        source_complete_data = torch.load(source_data_path)
        source_complete_labels = torch.load(source_labels_path)
        print(f"源域数据形状: {source_complete_data.shape}")
        print(f"源域标签形状: {source_complete_labels.shape}")
        
        # 获取源域预测结果
        y_true_source = []
        y_pred_source = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(source_complete_data), batch_size):
                batch_data = source_complete_data[i:i+batch_size].to(device)
                batch_labels = source_complete_labels[i:i+batch_size]
                
                # 获取模型预测
                outputs = model(batch_data)
                logits = outputs['logits']
                _, predicted = torch.max(logits, 1)
                
                # 收集真实标签和预测标签
                y_true_source.extend(batch_labels.numpy())
                y_pred_source.extend(predicted.cpu().numpy())
        
        # 绘制源域混淆矩阵
        if y_true_source and y_pred_source:
            cm_source = confusion_matrix(y_true_source, y_pred_source)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_source, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=[f'用户{i}' for i in range(10)],
                        yticklabels=[f'用户{i}' for i in range(10)],
                        cbar_kws={'shrink': 0.8},
                        linewidths=0.1,
                        annot_kws={'fontproperties': en_font, 'fontsize': 16})
            
            #title_text, title_font = create_mixed_text_with_fonts(None, '身份识别模型源域混淆矩阵', 16)
            #plt.title(title_text, fontproperties=title_font, fontweight='bold', pad=20)
            
            xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '预测标签', 16)
            plt.xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
            
            ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '真实标签', 16)
            plt.ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
            
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            # 设置刻度标签字体
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(mixed_font)
            
            plt.tight_layout()
            plt.savefig(save_path_source, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"源域混淆矩阵已保存到 {save_path_source}")
        else:
            print("未能生成源域混淆矩阵数据")
    else:
        print("未找到源域数据或标签文件")
    
    # === 2. 处理目标域数据 ===
    print("\n处理目标域数据...")
    target_data_path = os.path.join(current_dir, 'Data', 'target_env2_data.pt')
    target_labels_path = os.path.join(current_dir, 'Data', 'target_env2_labels.pt')
    
    if os.path.exists(target_data_path) and os.path.exists(target_labels_path):
        target_complete_data = torch.load(target_data_path)
        target_complete_labels = torch.load(target_labels_path)
        print(f"目标域数据形状: {target_complete_data.shape}")
        print(f"目标域标签形状: {target_complete_labels.shape}")
        
        # 获取目标域预测结果
        y_true_target = []
        y_pred_target = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(target_complete_data), batch_size):
                batch_data = target_complete_data[i:i+batch_size].to(device)
                batch_labels = target_complete_labels[i:i+batch_size]
                
                # 获取模型预测
                outputs = model(batch_data)
                logits = outputs['logits']
                _, predicted = torch.max(logits, 1)
                
                # 收集真实标签和预测标签
                y_true_target.extend(batch_labels.numpy())
                y_pred_target.extend(predicted.cpu().numpy())
        
        # 绘制目标域混淆矩阵
        if y_true_target and y_pred_target:
            cm_target = confusion_matrix(y_true_target, y_pred_target)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_target, annot=True, fmt='d', cmap='Oranges', 
                        xticklabels=[f'用户{i}' for i in range(10)],
                        yticklabels=[f'用户{i}' for i in range(10)],
                        cbar_kws={'shrink': 0.8},
                        linewidths=0.1,
                        annot_kws={'fontproperties': en_font, 'fontsize': 16})
            
            #title_text, title_font = create_mixed_text_with_fonts(None, '身份识别模型目标域混淆矩阵', 16)
            #plt.title(title_text, fontproperties=title_font, fontweight='bold', pad=20)
            
            xlabel_text, xlabel_font = create_mixed_text_with_fonts(None, '预测标签', 16)
            plt.xlabel(xlabel_text, fontproperties=xlabel_font, fontweight='bold')
            
            ylabel_text, ylabel_font = create_mixed_text_with_fonts(None, '真实标签', 16)
            plt.ylabel(ylabel_text, fontproperties=ylabel_font, fontweight='bold')
            
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            # 设置刻度标签字体
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(mixed_font)
            
            plt.tight_layout()
            plt.savefig(save_path_target, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"目标域混淆矩阵已保存到 {save_path_target}")
        else:
            print("未能生成目标域混淆矩阵数据")
    else:
        print("未找到目标域数据或标签文件")

def main():
    """
    主函数 - 用于直接运行身份识别可视化功能，调用所有画图和保存功能
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建图片保存目录
    picture_dir = 'R_Identify'
    os.makedirs(picture_dir, exist_ok=True)
    
    # 可视化身份识别模型特征（包括准确率曲线等）
    visualize_identity_features(device)
    
    # 绘制身份识别模型混淆矩阵（源域和目标域）
    confusion_matrix_source_path = os.path.join(picture_dir, 'identity_source_confusion_matrix.png')
    confusion_matrix_target_path = os.path.join(picture_dir, 'identity_target_confusion_matrix.png')
    plot_identity_confusion_matrix(device, confusion_matrix_source_path, confusion_matrix_target_path)
    
    # 绘制训练损失曲线
    identity_history_path = 'R_Identify/training_history.json'
    if os.path.exists(identity_history_path):
        loss_curve_path = os.path.join(picture_dir, 'identity_loss_curves.png')
        plot_training_loss_curves(identity_history_path, loss_curve_path)
    
    print("身份识别可视化完成，所有图表已生成!")

if __name__ == "__main__":
    main()
