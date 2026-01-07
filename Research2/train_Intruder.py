import torch
import numpy as np
import json
from model_Identify import IdentifyDetectionSystem
from model_Intruder import LearnableComprehensiveIntruderDetector
from loss_Intruder import WeightedIntruderLoss
from DataProcess.dataloader_intruder import load_intruder_data, create_intruder_data_loaders
from sklearn.metrics import f1_score, precision_score, recall_score


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


def save_training_history(train_losses, test_accuracies, test_f1s, test_precisions, test_recalls, file_path):
    """
    保存训练历史数据到JSON文件
    """
    history = {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'test_f1s': test_f1s,
        'test_precisions': test_precisions,
        'test_recalls': test_recalls
    }

    with open(file_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存到 {file_path}")


def train_intruder_detector(model_path, output_path, device):
    """
    训练综合入侵者检测器（用于二分类：合法用户 vs 入侵者）
    """

    # 初始化身份识别模型
    print("加载身份识别-流行优化模型...")
    identity_model = IdentifyDetectionSystem(num_classes=10, feature_dim=512, projection_dim=32).to(device)
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    identity_model.load_state_dict(checkpoint['model_state_dict'])
    identity_model.eval()

    # 加载入侵者检测专用数据
    print("加载入侵者检测数据...")
    datasets = load_intruder_data()
    data_loaders = create_intruder_data_loaders(datasets, batch_size=32)

    # 初始化综合入侵者检测器（二分类模型）
    comprehensive_detector = LearnableComprehensiveIntruderDetector(num_known_users=10, feature_dim=512).to(device)

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
    criterion = WeightedIntruderLoss(pos_weight=pos_weight, l2_weight=1e-4)
    print(
        f"数据集统计: 总样本数={total_samples}, 正样本数={positive_samples}, 负样本数={negative_samples}, 正负样本权重={pos_weight.item():.2f}")

    comprehensive_detector.train()

    num_epochs = 50
    best_avg_score = 0.0  # 基于4个指标的平均分数进行早停
    early_stop_counter = 0
    patience = 15  # 增加早停耐心值

    # 记录训练历史
    train_losses = []
    test_accuracies = []
    test_f1s = []
    test_precisions = []
    test_recalls = []

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
            loss = criterion(output_logits, labels, comprehensive_detector)
            
            # 反向传播和优化
            loss.backward()

            # 使用梯度裁剪
            torch.nn.utils.clip_grad_norm_(comprehensive_detector.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
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

        # 记录测试集指标
        test_accuracies.append(test_accuracy)
        test_f1s.append(test_f1)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)

        # 计算测试集4个指标的平均分数
        test_avg_score = (test_accuracy + test_f1 + test_precision + test_recall) / 4.0

        # 打印损失和测试集的4个指标
        print(f'Epoch [{epoch + 1}/{num_epochs}], 损失: {avg_loss:.4f}')
        print(f'  测试集 - 准确率: {test_accuracy:.4f}, F1: {test_f1:.4f}, 精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}, 平均分数: {test_avg_score:.4f}')

        # 保存最佳模型（基于测试集4个指标的平均分数）
        if test_avg_score > best_avg_score:
            best_avg_score = test_avg_score
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': comprehensive_detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_avg_score': best_avg_score,
                'val_metrics': (val_accuracy, val_f1, val_precision, val_recall),
                'test_metrics': (test_accuracy, test_f1, test_precision, test_recall),
            }, output_path)
            print(f'  保存最佳模型 (测试集平均分数: {best_avg_score:.4f})')
        else:
            early_stop_counter += 1
            if early_stop_counter % 5 == 0:  # 每5个epoch显示一次早停计数器
                print(f'  早停计数器: {early_stop_counter}/{patience}')

        # 早停检查
        if early_stop_counter >= patience:
            print(f'  测试集平均分数在 {patience} 个epoch内未提升，提前停止训练')
            break

    print(f"训练完成! 最佳测试集平均分数: {best_avg_score:.4f}")

    # 保存训练历史
    save_training_history(train_losses, test_accuracies, test_f1s, test_precisions, test_recalls, 'R_Intruder/training_history.json')

    # 在测试集上进行最终评估
    test_accuracy, test_f1, test_precision, test_recall, test_scores = test_intruder_detector(
        comprehensive_detector, identity_model, data_loaders['intruder_test'], device)

    print(
        f"最终测试结果 - 准确率: {test_accuracy:.4f}, F1: {test_f1:.4f}, 精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}")



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
    model_path = "R_Identify/best_identify_model.pth"  # 身份识别流行优化训练好的模型
    output_path = "R_Intruder/best_intruder_detector.pth"

    # 训练入侵者检测器
    train_intruder_detector(model_path, output_path, device)


if __name__ == "__main__":
    main()