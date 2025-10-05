import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model_identify import IdentifyDetectionSystem
from model_intruder import ComprehensiveIntruderDetector, TraditionalOpenMax
from intruder_data_loader import load_intruder_data, create_intruder_data_loaders
from sklearn.metrics import f1_score, precision_score, recall_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


def extract_features(model, data_loader, device):
    """
    从数据加载器中提取特征和标签
    """
    model.eval()
    all_features = []
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            try:
                # 移动到设备
                data = data.to(device)
                labels = labels.numpy()
                
                # 提取特征和预测
                outputs = model(data)
                features = outputs['features'].cpu().numpy()
                logits = outputs['logits'].cpu().numpy()
                
                all_features.append(features)
                all_logits.append(logits)
                all_labels.append(labels)
            except Exception as e:
                print(f"提取特征时出现错误 (batch {batch_idx}): {e}")
                continue
    
    if len(all_features) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return np.vstack(all_features), np.vstack(all_logits), np.hstack(all_labels)


def validate_intruder_detector(model, identity_model, data_loader, device):
    """
    在验证集上测试入侵者检测器性能
    """
    model.eval()
    identity_model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            try:
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
                detector_outputs = model(features, logits)
                predictions = detector_outputs['predictions']
                
                # 确保预测结果和标签维度一致
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                
                # 收集预测结果和真实标签
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"验证过程中出现错误 (batch {batch_idx}): {e}")
                continue
    
    # 计算评估指标
    if len(all_predictions) == 0 or len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 处理空数组情况
    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    accuracy = np.mean(all_predictions == all_labels) if len(all_labels) > 0 else 0.0
    
    # 使用zero_division=0避免警告
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    
    return accuracy, f1, precision, recall


def test_intruder_detector(model, identity_model, data_loader, device):
    """
    在测试集上测试入侵者检测器性能
    """
    model.eval()
    identity_model.eval()
    
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            try:
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
                detector_outputs = model(features, logits)
                predictions = detector_outputs['predictions']
                scores = detector_outputs['fusion_prob']
                
                # 确保预测结果和分数维度一致
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                if scores.dim() == 0:
                    scores = scores.unsqueeze(0)
                
                # 收集预测结果和真实标签
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
            except Exception as e:
                print(f"测试过程中出现错误 (batch {batch_idx}): {e}")
                continue
    
    # 计算评估指标
    if len(all_predictions) == 0 or len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, np.array([])
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # 处理空数组情况
    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, np.array([])
    
    accuracy = np.mean(all_predictions == all_labels) if len(all_labels) > 0 else 0.0
    
    # 使用zero_division=0避免警告
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    
    return accuracy, f1, precision, recall, all_scores


def plot_training_curves(train_losses, val_metrics, test_metrics):
    """
    绘制训练曲线
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # 绘制训练损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失', marker='o')
    plt.title('训练损失变化')
    plt.xlabel('训练轮数')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制验证集指标曲线
    val_accuracies = [m[0] for m in val_metrics]
    val_f1_scores = [m[1] for m in val_metrics]
    val_precisions = [m[2] for m in val_metrics]
    val_recalls = [m[3] for m in val_metrics]
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='验证集准确率', marker='s')
    plt.plot(epochs, val_f1_scores, 'r-', label='验证集F1分数', marker='^')
    plt.title('验证集性能指标')
    plt.xlabel('训练轮数')
    plt.ylabel('指标值')
    plt.legend()
    plt.grid(True)
    
    # 绘制测试集指标曲线
    test_accuracies = [m[0] for m in test_metrics]
    test_f1_scores = [m[1] for m in test_metrics]
    test_precisions = [m[2] for m in test_metrics]
    test_recalls = [m[3] for m in test_metrics]
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs, test_accuracies, 'g--', label='测试集准确率', marker='s')
    plt.plot(epochs, test_f1_scores, 'r--', label='测试集F1分数', marker='^')
    plt.title('测试集性能指标')
    plt.xlabel('训练轮数')
    plt.ylabel('指标值')
    plt.legend()
    plt.grid(True)
    
    # 分别绘制各项指标
    plt.subplot(2, 3, 4)
    plt.plot(epochs, val_precisions, 'b-', label='验证集精确率', marker='o')
    plt.plot(epochs, test_precisions, 'b--', label='测试集精确率', marker='s')
    plt.title('精确率变化')
    plt.xlabel('训练轮数')
    plt.ylabel('精确率')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(epochs, val_recalls, 'm-', label='验证集召回率', marker='o')
    plt.plot(epochs, test_recalls, 'm--', label='测试集召回率', marker='s')
    plt.title('召回率变化')
    plt.xlabel('训练轮数')
    plt.ylabel('召回率')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs, val_f1_scores, 'c-', label='验证集F1分数', marker='o')
    plt.plot(epochs, test_f1_scores, 'c--', label='测试集F1分数', marker='s')
    plt.title('F1分数变化')
    plt.xlabel('训练轮数')
    plt.ylabel('F1分数')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('intruder/training_curves.png')
    plt.close()


def plot_score_distribution(scores, labels):
    """
    绘制决策分数分布图
    """
    plt.figure(figsize=(12, 8))
    
    # 分离合法用户和入侵者的分数
    legal_scores = scores[labels == 0]
    intruder_scores = scores[labels == 1]
    
    # 绘制合法用户分数分布
    plt.subplot(2, 2, 1)
    plt.hist(legal_scores, bins=50, alpha=0.7, label='合法用户', color='blue')
    plt.xlabel('决策分数')
    plt.ylabel('频次')
    plt.title('合法用户决策分数分布')
    plt.legend()
    plt.grid(True)
    
    # 绘制入侵者分数分布
    plt.subplot(2, 2, 2)
    plt.hist(intruder_scores, bins=50, alpha=0.7, label='入侵者', color='red')
    plt.xlabel('决策分数')
    plt.ylabel('频次')
    plt.title('入侵者决策分数分布')
    plt.legend()
    plt.grid(True)
    
    # 绘制对比图
    plt.subplot(2, 2, 3)
    plt.hist(legal_scores, bins=50, alpha=0.7, label='合法用户', color='blue')
    plt.hist(intruder_scores, bins=50, alpha=0.7, label='入侵者', color='red')
    plt.xlabel('决策分数')
    plt.ylabel('频次')
    plt.title('决策分数对比')
    plt.legend()
    plt.grid(True)
    
    # 绘制箱线图
    plt.subplot(2, 2, 4)
    data = [legal_scores, intruder_scores]
    plt.boxplot(data, labels=['合法用户', '入侵者'])
    plt.ylabel('决策分数')
    plt.title('决策分数箱线图')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('intruder/score_distribution.png')
    plt.close()


def compute_threshold(features, labels, method='percentile'):
    """
    基于已知用户分布计算最优阈值
    """
    # 分离合法用户和入侵者特征
    legal_features = features[labels == 0]
    intruder_features = features[labels == 1]
    
    if len(legal_features) == 0 or len(intruder_features) == 0:
        return 0.5  # 默认阈值
    
    # 计算合法用户的特征均值和标准差
    legal_mean = np.mean(legal_features, axis=0)
    legal_std = np.std(legal_features, axis=0) + 1e-8  # 避免除零
    
    # 计算每个样本到合法用户分布中心的距离
    legal_distances = np.linalg.norm(legal_features - legal_mean, axis=1)
    intruder_distances = np.linalg.norm(intruder_features - legal_mean, axis=1)
    
    # 根据方法选择阈值
    if method == 'percentile':
        # 使用百分位数方法
        threshold = np.percentile(legal_distances, 95)  # 95%的合法用户在这个阈值内
    elif method == 'minimax':
        # 最小化最大错误率
        all_distances = np.concatenate([legal_distances, intruder_distances])
        min_error = float('inf')
        best_threshold = 0.5
        
        for t in np.linspace(np.min(all_distances), np.max(all_distances), 100):
            # 计算错误率
            false_acceptance = np.sum(legal_distances > t) / len(legal_distances) if len(legal_distances) > 0 else 0  # 合法用户被拒绝
            false_rejection = np.sum(intruder_distances < t) / len(intruder_distances) if len(intruder_distances) > 0 else 0  # 入侵者被接受
            error_rate = (false_acceptance + false_rejection) / 2
            
            if error_rate < min_error:
                min_error = error_rate
                best_threshold = t
                
        threshold = best_threshold
    else:
        threshold = 0.5  # 默认阈值
    
    return threshold


def train_intruder_detector(model_path, output_path, device):
    """
    训练综合入侵者检测器（用于二分类：合法用户 vs 入侵者）
    """

    # 初始化身份识别模型
    print("加载身份识别模型...")
    identity_model = IdentifyDetectionSystem(num_classes=10, feature_dim=128, projection_dim=32).to(device)
    
    # 加载模型权重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        identity_model.load_state_dict(checkpoint['model_state_dict'])
        identity_model.eval()
        print(f"身份识别模型加载完成")
    except Exception as e:
        print(f"加载身份识别模型失败: {e}")
        return
    
    # 加载入侵者检测专用数据
    print("加载入侵者检测数据...")
    try:
        datasets = load_intruder_data()
        data_loaders = create_intruder_data_loaders(datasets, batch_size=32)
    except Exception as e:
        print(f"加载入侵者检测数据失败: {e}")
        return
    
    # 提取训练数据的特征用于传统OpenMax模型训练
    print("提取训练数据特征用于传统OpenMax模型训练...")
    try:
        train_features, train_logits, train_labels = extract_features(
            identity_model, data_loaders['intruder_train'], device)
        
        # 训练传统OpenMax模型（只使用合法用户数据）
        traditional_openmax = TraditionalOpenMax(num_known_users=10)
        traditional_openmax.fit(train_features, train_labels)
        print("传统OpenMax模型训练完成")
        
        # 计算最优阈值
        optimal_threshold = compute_threshold(train_features, train_labels, method='minimax')
        print(f"计算得到的最优阈值: {optimal_threshold:.4f}")
    except Exception as e:
        print(f"训练传统OpenMax模型失败: {e}")
        traditional_openmax = TraditionalOpenMax(num_known_users=10)
        optimal_threshold = 0.5  # 使用默认阈值
    
    # 初始化综合入侵者检测器（二分类模型）
    comprehensive_detector = ComprehensiveIntruderDetector(
        num_known_users=10, feature_dim=128).to(device)
    
    # 将训练好的传统OpenMax模型设置到综合检测器中
    comprehensive_detector.traditional_openmax = traditional_openmax
    
    # 设置优化器，使用余弦退火学习率调度器
    optimizer = torch.optim.Adam(comprehensive_detector.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # 损失函数 - 使用BCELoss进行二分类
    bce_criterion = torch.nn.BCELoss()
    
    print("开始训练综合入侵者检测器...")
    comprehensive_detector.train()
    
    num_epochs = 100
    best_f1_score = 0.0  # 基于F1分数进行早停
    early_stop_counter = 0
    patience = 20  # 早停耐心值
    
    # 记录训练历史
    train_losses = []
    val_metrics = []  # (accuracy, f1, precision, recall)
    test_metrics = []  # (accuracy, f1, precision, recall)
    
    # 确保保存模型的目录存在
    os.makedirs('intruder', exist_ok=True)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        batch_count = 0
        
        # 训练循环
        for batch_idx, (data, labels) in enumerate(data_loaders['intruder_train']):
            try:
                # 移动到设备
                data = data.to(device)
                labels = labels.to(device).float()  # 转换为float用于BCE损失
                
                # 前向传播
                optimizer.zero_grad()
                
                # 使用身份识别模型提取特征
                with torch.no_grad():
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
                
                # 使用综合入侵者检测器进行检测
                detector_outputs = comprehensive_detector(features, logits)
                fusion_prob = detector_outputs['fusion_prob']
                
                # 确保概率是正确的形状
                if fusion_prob.dim() == 0:
                    fusion_prob = fusion_prob.unsqueeze(0)
                
                # 确保标签和预测概率维度一致
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                if labels.size(0) != fusion_prob.size(0):
                    labels = labels[:fusion_prob.size(0)] if labels.size(0) > fusion_prob.size(0) else labels
                
                # 计算损失
                classification_loss = bce_criterion(fusion_prob, labels)
                
                # 反向传播和优化
                classification_loss.backward()
                torch.nn.utils.clip_grad_norm_(comprehensive_detector.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += classification_loss.item()
                batch_count += 1
                
                # 统计准确率
                predicted_labels = (fusion_prob > 0.5).float()  # 使用float类型进行比较
                correct += (predicted_labels == labels).sum().item()
                total += labels.size(0)
                
            except Exception as e:
                print(f"训练过程中出现错误 (batch {batch_idx}): {e}")
                # 跳过这个batch继续训练
                continue
        
        # 计算训练准确率
        train_accuracy = correct / total if total > 0 else 0.0
        train_losses.append(total_loss / batch_count if batch_count > 0 else 0)
        
        # 在每个epoch后测试入侵者检测器性能
        try:
            val_accuracy, val_f1, val_precision, val_recall = validate_intruder_detector(
                comprehensive_detector, identity_model, data_loaders['intruder_validation'], device)
            
            test_accuracy, test_f1, test_precision, test_recall = validate_intruder_detector(
                comprehensive_detector, identity_model, data_loaders['intruder_test'], device)
            
            val_metrics.append((val_accuracy, val_f1, val_precision, val_recall))
            test_metrics.append((test_accuracy, test_f1, test_precision, test_recall))
            
            print(f'Epoch [{epoch+1}/{num_epochs}], 训练准确率: {train_accuracy:.4f}')
            print(f'  验证集 - 准确率: {val_accuracy:.4f}, F1: {val_f1:.4f}, 精确率: {val_precision:.4f}, 召回率: {val_recall:.4f}')
            print(f'  测试集 - 准确率: {test_accuracy:.4f}, F1: {test_f1:.4f}, 精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}')
            print(f'  训练损失: {total_loss/batch_count:.4f}, 当前学习率: {scheduler.get_last_lr()[0]:.6f}')
        except Exception as e:
            print(f"评估模型性能时出现错误: {e}")
            continue
        
        # 更新学习率
        try:
            scheduler.step()
        except Exception as e:
            print(f"更新学习率时出现错误: {e}")
        
        # 保存最佳模型（基于验证集F1分数）
        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            early_stop_counter = 0
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': comprehensive_detector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f1_score': best_f1_score,
                    'val_metrics': (val_accuracy, val_f1, val_precision, val_recall),
                    'test_metrics': (test_accuracy, test_f1, test_precision, test_recall),
                    'optimal_threshold': optimal_threshold
                }, output_path)
                print(f'  保存最佳模型 (验证集F1分数: {best_f1_score:.4f})')
            except Exception as e:
                print(f"保存模型时出现错误: {e}")
        else:
            early_stop_counter += 1
            print(f'  早停计数器: {early_stop_counter}/{patience}')
            
        # 早停检查
        if early_stop_counter >= patience:
            print(f'  验证集F1分数在 {patience} 个epoch内未提升，提前停止训练')
            break
    
    print(f"训练完成! 最佳验证集F1分数: {best_f1_score:.4f}")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_metrics, test_metrics)
    print("训练曲线已保存到 intruder/training_curves.png")

    # 在测试集上进行最终评估并绘制分数分布图
    test_accuracy, test_f1, test_precision, test_recall, test_scores = test_intruder_detector(
        comprehensive_detector, identity_model, data_loaders['intruder_test'], device)

    print(f"最终测试结果 - 准确率: {test_accuracy:.4f}, F1: {test_f1:.4f}, 精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}")

    # 绘制决策分数分布图
    _, _, test_labels = extract_features(identity_model, data_loaders['intruder_test'], device)
    plot_score_distribution(test_scores, test_labels)
    print("决策分数分布图已保存到 intruder/score_distribution.png")

    # 保存最终模型
    torch.save({
        'model_state_dict': comprehensive_detector.state_dict(),
        'best_f1_score': best_f1_score,
        'optimal_threshold': optimal_threshold
    }, 'intruder/final_comprehensive_intruder_detector.pth')
    print("最终模型已保存到 intruder/final_comprehensive_intruder_detector.pth")


def main():
    """
    主训练函数
    """
    print("开始训练入侵者检测器...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 如果使用CUDA，设置一些优化选项以减少警告
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # 模型路径
    model_path = "identify/best_identify_model.pth"  # 身份识别训练好的模型
    output_path = "intruder/comprehensive_intruder_detector.pth"

    # 训练入侵者检测器
    train_intruder_detector(model_path, output_path, device)


if __name__ == "__main__":
    main()