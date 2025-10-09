import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model_identify import IdentifyDetectionSystem
from model_intruder import LearnableComprehensiveIntruderDetector
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
    all_identity_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data, labels, identity_labels = batch

            # 移动到设备
            data = data.to(device)
            labels = labels.numpy()
            identity_labels = identity_labels.numpy()

            # 提取特征和预测
            outputs = model(data)
            features = outputs['features'].cpu().numpy()
            logits = outputs['logits'].cpu().numpy()

            all_features.append(features)
            all_logits.append(logits)
            all_labels.append(labels)
            all_identity_labels.append(identity_labels)

    if len(all_features) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    identity_labels_result = np.hstack(all_identity_labels) if all_identity_labels else np.array([])
    return np.vstack(all_features), np.vstack(all_logits), np.hstack(all_labels), identity_labels_result


def validate_intruder_detector(model, identity_model, data_loader, device):
    """
    在验证集上测试入侵者检测器性能
    """
    model.eval()
    identity_model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data, labels, identity_labels = batch

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
            detector_outputs = model(features, logits, identity_labels)
            predictions = detector_outputs['predictions']
            probabilities = detector_outputs['probabilities']

            # 确保预测结果和标签维度一致
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            if probabilities.dim() == 0:
                probabilities = probabilities.unsqueeze(0)

            # 收集预测结果和真实标签
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    if len(all_predictions) == 0 or len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 处理空数组情况
    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    accuracy = np.mean(all_predictions == all_labels) if len(all_labels) > 0 else 0.0
    
    # 使用zero_division='warn'避免警告
    f1 = f1_score(all_labels, all_predictions, zero_division='warn')
    precision = precision_score(all_labels, all_predictions, zero_division='warn')
    recall = recall_score(all_labels, all_predictions, zero_division='warn')
    
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
        for batch_idx, batch in enumerate(data_loader):
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
                detector_outputs = model(features, logits, identity_labels)
            else:
                # 如果没有identity_labels，创建一个默认的标签数组
                # 假设所有样本都是合法用户（标签为0）
                default_identity_labels = torch.zeros(batch_size, dtype=torch.long, device=features.device)
                detector_outputs = model(features, logits, default_identity_labels)
            predictions = detector_outputs['predictions']
            scores = detector_outputs['probabilities']  # 修改这里，使用正确的键
            
            # 确保预测结果和分数维度一致
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            
            # 收集预测结果和真实标签
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
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
    
    # 使用zero_division='warn'避免警告
    f1 = f1_score(all_labels, all_predictions, zero_division='warn')
    precision = precision_score(all_labels, all_predictions, zero_division='warn')
    recall = recall_score(all_labels, all_predictions, zero_division='warn')
    
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


def train_intruder_detector(model_path, output_path, device):
    """
    训练综合入侵者检测器（用于二分类：合法用户 vs 入侵者）
    """

    # 初始化身份识别模型
    print("加载身份识别模型...")
    identity_model = IdentifyDetectionSystem(num_classes=10, feature_dim=128, projection_dim=32).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    identity_model.load_state_dict(checkpoint['model_state_dict'])
    identity_model.eval()
    print(f"身份识别模型加载完成")
    
    # 加载入侵者检测专用数据
    print("加载入侵者检测数据...")
    datasets = load_intruder_data()
    data_loaders = create_intruder_data_loaders(datasets, batch_size=32)

    # 初始化综合入侵者检测器（二分类模型）
    comprehensive_detector = LearnableComprehensiveIntruderDetector(num_known_users=10, feature_dim=128).to(device)
    
    # 设置优化器，使用更稳定的学习率和权重衰减
    optimizer = torch.optim.AdamW(comprehensive_detector.parameters(), lr=1e-3, weight_decay=1e-4)  # 调整学习率和权重衰减
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 使用StepLR替代ReduceLROnPlateau
    
    # 损失函数 - 使用带权重的BCEWithLogitsLoss处理类别不平衡问题
    # 计算正负样本权重
    train_dataset = datasets['intruder_train']
    total_samples = len(train_dataset)
    positive_samples = sum(1 for _, label, _ in train_dataset if label == 1)
    negative_samples = total_samples - positive_samples

    pos_weight = torch.tensor([negative_samples / positive_samples], device=device)
    bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"数据集统计: 总样本数={total_samples}, 正样本数={positive_samples}, 负样本数={negative_samples}, 正负样本权重={pos_weight.item():.2f}")

    comprehensive_detector.train()
    
    num_epochs = 100
    best_f1_score = 0.0  # 基于F1分数进行早停
    early_stop_counter = 0
    patience = 40  # 增加早停耐心值
    
    # 记录训练历史
    train_losses = []
    val_metrics = []  # (accuracy, f1, precision, recall)
    test_metrics = []  # (accuracy, f1, precision, recall)

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        positive_predictions = 0  # 记录正类预测数量
        positive_labels = 0  # 记录实际正类数量
        
        batch_count = 0
        
        # 训练循环
        for batch_idx, batch in enumerate(data_loaders['intruder_train']):
            # 处理不同格式的batch数据
            data, labels, identity_labels = batch
            
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

            # 使用综合入侵者检测器进行检测
            detector_outputs = comprehensive_detector(features, logits, identity_labels)
            output_logits = detector_outputs['logits']  # 使用logits而不是probabilities

            # 计算损失
            classification_loss = bce_criterion(output_logits, labels)
            
            # 添加L2正则化
            l2_reg = torch.tensor(0., device=device)
            for param in comprehensive_detector.parameters():
                l2_reg += torch.norm(param)
            total_loss_with_reg = classification_loss + 1e-4 * l2_reg  # 添加正则化项
            
            # 反向传播和优化
            total_loss_with_reg.backward()
            
            # 使用梯度裁剪
            torch.nn.utils.clip_grad_norm_(comprehensive_detector.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += classification_loss.item()
            batch_count += 1
            
            # 统计准确率和正类预测数量
            with torch.no_grad():
                probabilities = torch.sigmoid(output_logits)
                predicted_labels = (probabilities > 0.5).float()  # 使用float类型进行比较
                correct += (predicted_labels == labels).sum().item()
                positive_predictions += predicted_labels.sum().item()
                positive_labels += labels.sum().item()
                total += labels.size(0)
        
        # 计算训练准确率和正类预测比例
        train_accuracy = correct / total if total > 0 else 0.0
        positive_prediction_ratio = positive_predictions / total if total > 0 else 0.0
        positive_label_ratio = positive_labels / total if total > 0 else 0.0
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_loss)
        
        # 更新学习率
        scheduler.step()
        
        # 在每个epoch后测试入侵者检测器性能
        val_accuracy, val_f1, val_precision, val_recall = validate_intruder_detector(
            comprehensive_detector, identity_model, data_loaders['intruder_validation'], device)

        test_accuracy, test_f1, test_precision, test_recall = validate_intruder_detector(
            comprehensive_detector, identity_model, data_loaders['intruder_test'], device)

        val_metrics.append((val_accuracy, val_f1, val_precision, val_recall))
        test_metrics.append((test_accuracy, test_f1, test_precision, test_recall))

        # 简化输出信息，只显示损失、学习率、验证集和测试集的准确率
        print(f'Epoch [{epoch+1}/{num_epochs}], 损失: {avg_loss:.4f}, 学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  验证集准确率: {val_accuracy:.4f}, 测试集准确率: {test_accuracy:.4f}')

        # 保存最佳模型（基于验证集F1分数）
        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': comprehensive_detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1_score': best_f1_score,
                'val_metrics': (val_accuracy, val_f1, val_precision, val_recall),
                'test_metrics': (test_accuracy, test_f1, test_precision, test_recall),
            }, output_path)
            print(f'  保存最佳模型 (验证集F1分数: {best_f1_score:.4f})')
        else:
            early_stop_counter += 1
            if early_stop_counter % 5 == 0:  # 每5个epoch显示一次早停计数器
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
    _, _, test_labels, _ = extract_features(identity_model, data_loaders['intruder_test'], device)
    plot_score_distribution(test_scores, test_labels)
    print("决策分数分布图已保存到 intruder/score_distribution.png")

    # 保存最终模型
    torch.save({
        'model_state_dict': comprehensive_detector.state_dict(),
        'best_f1_score': best_f1_score,
    }, 'intruder/final_intruder_detector.pth')
    print("最终模型已保存到 intruder/final_intruder_detector.pth")


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
    output_path = "intruder/best_intruder_detector.pth"

    # 训练入侵者检测器
    train_intruder_detector(model_path, output_path, device)


if __name__ == "__main__":
    main()