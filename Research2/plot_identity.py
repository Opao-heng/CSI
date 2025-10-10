import json
import matplotlib.pyplot as plt
import os
import sys
import torch
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model_identify import IdentifyDetectionSystem

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
    plt.plot(epochs, train_losses, 'b-', label='总损失 (Total Loss)', linewidth=2.5, marker='o', markersize=4)
    
    # 绘制各项损失曲线
    identity_losses = [comp['identity'] for comp in loss_components_history]
    contrastive_losses = [comp['contrastive'] for comp in loss_components_history]
    
    plt.plot(epochs, identity_losses, label='身份损失 (Identity Loss)', linewidth=2.5, marker='s', markersize=4)
    plt.plot(epochs, contrastive_losses, label='对比损失 (Contrastive Loss)', linewidth=2.5, marker='^', markersize=4)
    
    plt.title('训练损失变化曲线', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('训练轮数 (Epoch)', fontsize=14, fontweight='bold')
    plt.ylabel('损失值 (Loss)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
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

def plot_feature_distribution_identity(source_features, target_features, save_path):
    """
    绘制源域和目标域特征分布直方图
    """
    # 计算特征的L2范数
    source_norms = np.linalg.norm(source_features, axis=1)
    target_norms = np.linalg.norm(target_features, axis=1)
    
    plt.figure(figsize=(12, 8))
    
    # 使用更美观的颜色和透明度
    plt.hist(source_norms, bins=50, alpha=0.7, label='源域特征', color='#1f77b4', edgecolor='black', linewidth=0.5)
    plt.hist(target_norms, bins=50, alpha=0.7, label='目标域特征', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    
    plt.title('身份识别模型特征分布', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('特征L2范数', fontsize=14, fontweight='bold')
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
    print(f"特征分布图已保存到 {save_path}")

def extract_features_identity_by_dataset(device):
    """
    根据数据集原始数据分别提取源域和目标域特征
    
    直接从数据加载器加载的原始数据中提取特征：
    - 源域数据: torch.load('../data/SourceData/source_data.pt') - 2007个样本
    - 目标域数据: torch.load('../data/TargetData/target_data.pt') - 1032个样本
    """
    # 直接加载原始数据
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_data_path = os.path.join(current_dir, '..', 'data', 'SourceData', 'source_data.pt')
    source_labels_path = os.path.join(current_dir, '..', 'data', 'SourceData', 'source_labels.pt')
    target_data_path = os.path.join(current_dir, '..', 'data', 'TargetData', 'target_data.pt')
    target_labels_path = os.path.join(current_dir, '..', 'data', 'TargetData', 'target_labels.pt')
    
    # 加载源域数据（2007个样本）
    print("直接加载源域数据...")
    source_complete_data = torch.load(source_data_path)
    source_complete_labels = torch.load(source_labels_path)
    print(f"源域数据形状: {source_complete_data.shape}")
    print(f"源域标签形状: {source_complete_labels.shape}")
    
    # 加载目标域数据（1032个样本）
    print("直接加载目标域数据...")
    target_complete_data = torch.load(target_data_path)
    target_complete_labels = torch.load(target_labels_path)
    print(f"目标域数据形状: {target_complete_data.shape}")
    print(f"目标域标签形状: {target_complete_labels.shape}")
    
    source_features = []
    source_labels_list = []
    target_features = []
    target_labels_list = []
    
    batch_size = 32
    
    # 初始化模型用于特征提取
    model = IdentifyDetectionSystem(num_classes=10, feature_dim=128, projection_dim=32).to(device)
    model.eval()
    
    # 加载模型权重（如果存在）
    identity_model_path = os.path.join(current_dir, "identify", "best_identify_model.pth")
    if os.path.exists(identity_model_path):
        checkpoint = torch.load(identity_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载身份识别模型: {identity_model_path}")
    
    with torch.no_grad():
        # 处理源域数据（2007个样本）
        print("提取源域数据特征...")
        for i in range(0, len(source_complete_data), batch_size):
            batch_data = source_complete_data[i:i+batch_size].to(device)
            
            # 提取特征 - 注意模型的返回格式
            outputs = model(batch_data)
            if 'features' in outputs:
                features = outputs['features'].cpu().numpy()
            else:
                features = outputs['features_source'].cpu().numpy() if 'features_source' in outputs else outputs.cpu().numpy()
            
            source_features.append(features)
            source_labels_list.append(source_complete_labels[i:i+batch_size].numpy())
        
        # 处理目标域数据（1032个样本）
        print("提取目标域数据特征...")
        for i in range(0, len(target_complete_data), batch_size):
            batch_data = target_complete_data[i:i+batch_size].to(device)
            
            # 提取特征 - 注意模型的返回格式
            outputs = model(batch_data)
            if 'features' in outputs:
                features = outputs['features'].cpu().numpy()
            else:
                features = outputs['features_source'].cpu().numpy() if 'features_source' in outputs else outputs.cpu().numpy()
            
            target_features.append(features)
            target_labels_list.append(target_complete_labels[i:i+batch_size].numpy())
    
    source_features_all = np.vstack(source_features) if source_features else np.array([])
    source_labels_all = np.hstack(source_labels_list) if source_labels_list else np.array([])
    target_features_all = np.vstack(target_features) if target_features else np.array([])
    target_labels_all = np.hstack(target_labels_list) if target_labels_list else np.array([])
    
    return (source_features_all, source_labels_all), (target_features_all, target_labels_all)

def plot_tsne_identity_separate(source_data, target_data, save_path_source, save_path_target):
    """
    分别绘制源域和目标域的身份识别模型T-SNE图
    """
    source_features, source_labels = source_data
    target_features, target_labels = target_data
    
    if len(source_features) == 0 or len(target_features) == 0:
        print("特征数据为空，无法绘制图形")
        return
    
    # 合并所有特征数据进行统一的T-SNE降维
    all_features = np.vstack([source_features, target_features])
    
    print("正在进行T-SNE降维...")
    # 调整T-SNE参数以适应不同数量的样本
    n_samples = len(all_features)
    # perplexity应该小于样本数的一半，且通常在5-50之间
    perplexity = min(30, max(5, n_samples // 20))
    n_iter = max(1000, min(3000, n_samples * 2))
    
    # 对于大量样本，使用pca初始化可以提高速度和稳定性
    init_method = 'pca' if n_samples > 1000 else 'random'
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter, learning_rate='auto', 
                init=init_method, early_exaggeration=12.0, metric='euclidean')
    all_features_2d = tsne.fit_transform(all_features)
    
    # 分离降维后的源域和目标域数据
    source_features_2d = all_features_2d[:len(source_features)]
    target_features_2d = all_features_2d[len(source_features):]
    
    # 为每个用户绘制散点图
    unique_labels = np.unique(np.hstack([source_labels, target_labels]))
    num_classes = len(unique_labels)
    
    # 使用更现代的颜色映射
    if num_classes <= 10:
        colors = plt.cm.get_cmap('tab10', num_classes)
    elif num_classes <= 20:
        colors = plt.cm.get_cmap('tab20', num_classes)
    else:
        colors = plt.cm.get_cmap('viridis', num_classes)
    
    # 根据样本数量调整点的大小和透明度，使图形更清晰
    source_n_samples = len(source_features)
    target_n_samples = len(target_features)
    
    # 源域图形参数 - 样本越多，点越小，透明度越高
    source_alpha = max(0.5, min(0.8, 1500 / source_n_samples))
    source_size = max(20, min(60, 1500 / source_n_samples))
    
    # 目标域图形参数
    target_alpha = max(0.5, min(0.8, 1500 / target_n_samples))
    target_size = max(20, min(60, 1500 / target_n_samples))
    
    # 绘制源域T-SNE图
    print("绘制源域T-SNE图...")
    plt.figure(figsize=(12, 10))
    
    for i, label in enumerate(unique_labels):
        mask = source_labels == label
        if np.any(mask):  # 只绘制存在数据的类别
            plt.scatter(source_features_2d[mask, 0], source_features_2d[mask, 1], 
                       c=[colors(i)], label=f'用户 {label}', alpha=source_alpha, s=source_size, 
                       edgecolors='black', linewidth=0.5, marker='o')
    
    plt.title(f'身份识别模型 - 源域用户特征分布 (T-SNE)\n(样本数: {source_n_samples})', 
              fontsize=18, pad=20, fontweight='bold')
    plt.xlabel('T-SNE维度1', fontsize=14, fontweight='bold')
    plt.ylabel('T-SNE维度2', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11, markerscale=2.0, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    plt.tight_layout()
    plt.savefig(save_path_source, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"源域T-SNE图已保存到 {save_path_source}")
    
    # 绘制目标域T-SNE图
    print("绘制目标域T-SNE图...")
    plt.figure(figsize=(12, 10))
    
    for i, label in enumerate(unique_labels):
        mask = target_labels == label
        if np.any(mask):  # 只绘制存在数据的类别
            plt.scatter(target_features_2d[mask, 0], target_features_2d[mask, 1], 
                       c=[colors(i)], label=f'用户 {label}', alpha=target_alpha, s=target_size, 
                       edgecolors='black', linewidth=0.5, marker='o')
    
    plt.title(f'身份识别模型 - 目标域用户特征分布 (T-SNE)\n(样本数: {target_n_samples})', 
              fontsize=18, pad=20, fontweight='bold')
    plt.xlabel('T-SNE维度1', fontsize=14, fontweight='bold')
    plt.ylabel('T-SNE维度2', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11, markerscale=2.0, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # 美化坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    plt.tight_layout()
    plt.savefig(save_path_target, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"目标域T-SNE图已保存到 {save_path_target}")

def visualize_identity_features(device):
    """
    可视化身份识别模型的特征分布
    """
    print("=== 身份识别模型特征可视化 ===")
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 初始化身份识别模型
    print("加载身份识别模型...")
    identity_model_path = os.path.join(current_dir, "identify", "best_identify_model.pth")

    # 提取源域和目标域数据特征
    print("提取身份识别源域和目标域数据特征...")
    # 直接加载原始数据并提取特征
    source_data, target_data = extract_features_identity_by_dataset(device)
    
    source_features, source_labels = source_data
    target_features, target_labels = target_data
    print(f"源域提取到 {len(source_features)} 个样本的特征，特征维度: {source_features.shape[1]}")
    print(f"目标域提取到 {len(target_features)} 个样本的特征，特征维度: {target_features.shape[1]}")

    # 创建保存目录
    picture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'identify')
    os.makedirs(picture_dir, exist_ok=True)
    
    # 分别绘制源域和目标域的T-SNE图
    save_path_source = os.path.join(picture_dir, 'identity_source_tsne.png')
    save_path_target = os.path.join(picture_dir, 'identity_target_tsne.png')
    plot_tsne_identity_separate((source_features, source_labels), (target_features, target_labels), 
                               save_path_source, save_path_target)
    
    # 绘制特征分布直方图
    save_path_distribution = os.path.join(picture_dir, 'identity_feature_distribution.png')
    plot_feature_distribution_identity(source_features, target_features, save_path_distribution)
    
    # 绘制准确率变化曲线（如果存在训练历史文件）
    identity_history_path = os.path.join(current_dir, "identify", "training_history.json")
    if os.path.exists(identity_history_path):
        # 绘制训练与验证准确率曲线
        save_path_train_val_accuracy = os.path.join(picture_dir, 'identity_train_val_accuracy_curves.png')
        plot_accuracy_curves(identity_history_path, save_path_train_val_accuracy)
        
        # 绘制源域与目标域测试准确率曲线
        save_path_test_accuracy = os.path.join(picture_dir, 'identity_test_accuracy_curves.png')
        plot_test_accuracy_curves(identity_history_path, save_path_test_accuracy)

def plot_identity_confusion_matrix(device, save_path):
    """
    生成并绘制身份识别模型的混淆矩阵
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 初始化身份识别模型
    print("加载身份识别模型用于混淆矩阵生成...")
    model = IdentifyDetectionSystem(num_classes=10, feature_dim=128, projection_dim=32).to(device)
    model.eval()
    
    # 加载模型权重（如果存在）
    identity_model_path = os.path.join(current_dir, "identify", "best_identify_model.pth")
    if not os.path.exists(identity_model_path):
        print(f"未找到身份识别模型: {identity_model_path}")
        return
    
    checkpoint = torch.load(identity_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"身份识别模型加载完成")
    
    # 直接加载源域数据和标签
    source_data_path = os.path.join(current_dir, '..', 'data', 'SourceData', 'source_data.pt')
    source_labels_path = os.path.join(current_dir, '..', 'data', 'SourceData', 'source_labels.pt')
    
    if not os.path.exists(source_data_path) or not os.path.exists(source_labels_path):
        print("未找到源域数据或标签文件")
        return
    
    source_complete_data = torch.load(source_data_path)
    source_complete_labels = torch.load(source_labels_path)
    print(f"源域数据形状: {source_complete_data.shape}")
    print(f"源域标签形状: {source_complete_labels.shape}")
    
    # 获取预测结果
    y_true = []
    y_pred = []
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
            y_true.extend(batch_labels.numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # 绘制混淆矩阵
    if y_true and y_pred:
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        
        # 使用更美观的配色方案
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[f'用户{i}' for i in range(10)],
                    yticklabels=[f'用户{i}' for i in range(10)],
                    cbar_kws={'shrink': 0.8},
                    linewidths=0.1)
        
        plt.title("身份识别模型混淆矩阵", fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('预测标签', fontsize=14, fontweight='bold')
        plt.ylabel('真实标签', fontsize=14, fontweight='bold')
        
        # 美化坐标轴
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混淆矩阵已保存到 {save_path}")
    else:
        print("未能生成混淆矩阵数据")

def main():
    """
    主函数 - 用于直接运行身份识别可视化功能，调用所有画图和保存功能
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建图片保存目录
    picture_dir = 'identify'
    os.makedirs(picture_dir, exist_ok=True)
    
    # 可视化身份识别模型特征（包括T-SNE图、特征分布图等）
    visualize_identity_features(device)
    
    # 绘制身份识别模型混淆矩阵
    confusion_matrix_path = os.path.join(picture_dir, 'identity_confusion_matrix.png')
    plot_identity_confusion_matrix(device, confusion_matrix_path)
    
    # 绘制训练损失曲线
    identity_history_path = 'identify/training_history.json'
    if os.path.exists(identity_history_path):
        loss_curve_path = os.path.join(picture_dir, 'identity_loss_curves.png')
        plot_training_loss_curves(identity_history_path, loss_curve_path)
        
        # 绘制训练与验证准确率曲线
        accuracy_curve_path = os.path.join(picture_dir, 'identity_train_val_accuracy_curves.png')
        plot_accuracy_curves(identity_history_path, accuracy_curve_path)
        
        # 绘制源域与目标域测试准确率曲线
        test_accuracy_curve_path = os.path.join(picture_dir, 'identity_test_accuracy_curves.png')
        plot_test_accuracy_curves(identity_history_path, test_accuracy_curve_path)
    
    print("身份识别可视化完成，所有图表已生成!")

if __name__ == "__main__":
    main()
