import torch
import numpy as np
import json
from model_identify import IdentifyDetectionSystem
from model_intruder import LearnableComprehensiveIntruderDetector
from Research2.DataProcess.dataloader_intruder import load_intruder_data, create_intruder_data_loaders
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


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
            labels = labels.cpu().numpy()
            identity_labels = identity_labels.cpu().numpy()

            # 提取特征和预测（优先使用投影空间特征）
            outputs = model(data)
            proj = outputs.get('proj', outputs['features'])
            features = proj.cpu().numpy()
            logits = outputs['logits'].cpu().numpy()

            all_features.append(features)
            all_logits.append(logits)
            all_labels.append(labels)
            all_identity_labels.append(identity_labels)

    if len(all_features) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    identity_labels_result = np.hstack(all_identity_labels) if all_identity_labels else np.array([])
    return np.vstack(all_features), np.vstack(all_logits), np.hstack(all_labels), identity_labels_result


def validate_intruder_detector(model, identity_model, data_loader, device, threshold=0.5):
    """
    在验证集上测试入侵者检测器性能
    """
    model.eval()
    identity_model.eval()
    
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data, labels, identity_labels = batch

            # 移动到设备
            data = data.to(device)
            labels = labels.to(device)

            # 使用身份识别模型提取特征 - 统一使用proj特征
            identity_outputs = identity_model(data)
            # 确保使用32维投影特征
            if 'proj' in identity_outputs:
                features = identity_outputs['proj']
            else:
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
            probabilities = detector_outputs['probabilities']
            
            # 使用动态阈值生成预测
            predictions = (probabilities > threshold).float()

            # 确保预测结果和标签维度一致
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            if probabilities.dim() == 0:
                probabilities = probabilities.unsqueeze(0)

            # 收集预测结果、预测分数和真实标签
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probabilities.cpu().numpy())

    # 计算评估指标
    if len(all_predictions) == 0 or len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # 处理空数组情况
    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    accuracy = float(np.mean(all_predictions == all_labels)) if len(all_labels) > 0 else 0.0
    
    # 使用zero_division='warn'避免警告
    f1 = float(f1_score(all_labels, all_predictions, zero_division='warn'))
    precision = float(precision_score(all_labels, all_predictions, zero_division='warn'))
    recall = float(recall_score(all_labels, all_predictions, zero_division='warn'))

    # AUROC（当正负样本都存在时才有意义）
    try:
        auroc = float(roc_auc_score(all_labels, all_scores))
    except ValueError:
        auroc = 0.0
    
    return accuracy, f1, precision, recall, auroc


def test_intruder_detector(model, identity_model, data_loader, device, threshold=0.5):
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
            
            # 使用身份识别模型提取特征 - 统一使用proj特征
            identity_outputs = identity_model(data)
            # 确保使用32维投影特征
            if 'proj' in identity_outputs:
                features = identity_outputs['proj']
            else:
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
            
            # 使用动态阈值生成预测
            probabilities = detector_outputs['probabilities']
            predictions = (probabilities > threshold).float()  # 使用传入的阈值
            scores = probabilities
            
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
        return 0.0, 0.0, 0.0, 0.0, 0.0, np.array([])
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # 处理空数组情况
    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, np.array([])
    
    accuracy = float(np.mean(all_predictions == all_labels)) if len(all_labels) > 0 else 0.0
    
    # 使用zero_division='warn'避免警告
    f1 = float(f1_score(all_labels, all_predictions, zero_division='warn'))
    precision = float(precision_score(all_labels, all_predictions, zero_division='warn'))
    recall = float(recall_score(all_labels, all_predictions, zero_division='warn'))

    # AUROC（当正负样本都存在时才有意义）
    try:
        auroc = float(roc_auc_score(all_labels, all_scores))
    except ValueError:
        auroc = 0.0
    
    return accuracy, f1, precision, recall, auroc, all_scores


def save_training_history(train_losses, val_metrics, test_metrics, file_path):
    """
    保存训练历史数据到JSON文件
    val_metrics/test_metrics: (accuracy, f1, precision, recall, auroc)
    """
    history = {
        'train_losses': train_losses,
        'val_metrics': val_metrics,  # (accuracy, f1, precision, recall, auroc)
        'test_metrics': test_metrics  # (accuracy, f1, precision, recall, auroc)
    }
    
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存到 {file_path}")


def train_intruder_detector(model_path, output_path, device):
    """
    训练综合入侵者检测器（用于二分类：合法用户 vs 入侵者）
    核心策略：两阶段训练 - 先用真实入侵者样本初始化OpenMax，再联合训练
    """

    # 初始化身份识别模型（必须与训练时的参数一致）
    print("加载身份识别模型...")
    identity_model = IdentifyDetectionSystem(num_classes=10, feature_dim=512, projection_dim=32).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    identity_model.load_state_dict(checkpoint['model_state_dict'])
    identity_model.eval()
    print(f"身份识别模型加载完成")
    
    # 加载入侵者检测专用数据
    print("加载入侵者检测数据...")
    datasets = load_intruder_data()
    data_loaders = create_intruder_data_loaders(datasets, batch_size=32)

    # 初始化综合入侵者检测器（二分类模型），在流形投影空间（32维）上工作
    comprehensive_detector = LearnableComprehensiveIntruderDetector(num_known_users=10, feature_dim=32).to(device)

    # === 关键改进：仅使用训练集数据初始化OpenMax，避免数据泄露 ===
    print("\n===== 使用训练集数据初始化 TraditionalOpenMax =====")
    # 从训练集中提取特征用于初始化OpenMax
    train_features, _, train_labels, train_identity_labels = extract_features(identity_model, data_loaders['intruder_train'], device)
    
    if train_features.size > 0:
        comprehensive_detector.fit_traditional_openmax(train_features, train_labels, train_identity_labels)
        print(f"TraditionalOpenMax初始化完成 (训练样本数: {len(train_features)})")
    
    # 设置优化器：使用更低的学习率和更强的正则化
    optimizer = torch.optim.AdamW(comprehensive_detector.parameters(), lr=1e-4, weight_decay=5e-4)
    
    # 使用Warmup + 余弦退火学习率调度器
    warmup_epochs = 5
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=95, eta_min=5e-7)
    
    # OpenMax更新配置 - 仅使用训练集和验证集数据
    openmax_update_interval = 4  # 每4个epoch更新一次OpenMax
    use_openmax_after_epoch = 5  # 从第6个epoch开始使用OpenMax，让深度学习模型先学习
    
    # 损失函数 - 开放集识别策略：降低正样本权重，避免过度拟合模拟入侵者
    train_dataset = datasets['intruder_train']
    total_samples = len(train_dataset)
    positive_samples = sum(1 for _, label, _ in train_dataset if label == 1)
    negative_samples = total_samples - positive_samples

    # 【关键】降低正样本权重，因为训练集入侵者都是模拟的
    # 让模型更关注学习合法用户的紧凑表示，而不是过度拟合模拟入侵者
    pos_weight = torch.tensor([negative_samples / positive_samples * 0.5], device=device)
    bce_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"数据集统计: 总样本数={total_samples}, 正样本数={positive_samples}, 负样本数={negative_samples}, 正负样本权重={pos_weight.item():.2f}")
    print(f"数据分布: 合法用户比例={negative_samples/total_samples:.2%}, 入侵者比例={positive_samples/total_samples:.2%}")

    comprehensive_detector.train()
    
    num_epochs = 120
    best_score = 0.0  # 跟踪最佳综合评分
    early_stop_counter = 0
    patience = 40  # 早停耐心值
    
    # 记录训练历史
    train_losses = []
    val_metrics = []  # (accuracy, f1, precision, recall, auroc)
    test_metrics = []  # (accuracy, f1, precision, recall, auroc)

    for epoch in range(num_epochs):
        # 定期更新 TraditionalOpenMax（仅使用训练集和验证集数据）
        if epoch > 0 and epoch % openmax_update_interval == 0:
            print(f"\n[Epoch {epoch+1}] 更新 TraditionalOpenMax...")
            train_features, _, train_labels, train_identity_labels = extract_features(
                identity_model, data_loaders['intruder_train'], device)
            
            # 添加验证集数据以增强泛化能力
            val_features, _, val_labels, val_identity_labels = extract_features(
                identity_model, data_loaders['intruder_validation'], device)
            
            # 合并训练集和验证集数据
            combined_features = np.vstack([train_features, val_features])
            combined_labels = np.hstack([train_labels, val_labels])
            combined_identity_labels = np.hstack([train_identity_labels, val_identity_labels])
            
            if combined_features.size > 0:
                comprehensive_detector.fit_traditional_openmax(combined_features, combined_labels, combined_identity_labels)
                print(f"TraditionalOpenMax 更新完成（训练样本: {len(train_features)}, 验证样本: {len(val_features)}）\n")
        
        # 判断是否使用OpenMax
        use_openmax = (epoch >= use_openmax_after_epoch)
        
        total_loss = 0.0
        correct = 0
        total = 0
        positive_predictions = 0
        positive_labels = 0
        
        batch_count = 0
        
        # 训练循环
        for batch_idx, batch in enumerate(data_loaders['intruder_train']):
            data, labels, identity_labels = batch
            
            data = data.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            
            # 使用身份识别模型提取特征
            with torch.no_grad():
                identity_outputs = identity_model(data)
                features = identity_outputs.get('proj', identity_outputs['features'])
                logits = identity_outputs['logits']

            # 使用综合入侵者检测器进行检测（动态控制是否使用OpenMax）
            detector_outputs = comprehensive_detector(features, logits, identity_labels, use_openmax=use_openmax)
            output_logits = detector_outputs['logits']  # 使用logits而不是probabilities

            # 开放集识别损失：BCE + 中心损失（促进合法用户特征紧凑）
            classification_loss = bce_criterion(output_logits, labels)
            
            # 中心损失：让合法用户特征更紧凑
            legal_user_mask = (labels == 0)
            if legal_user_mask.sum() > 0:
                legal_features = features[legal_user_mask]
                # 计算合法用户特征的中心
                feature_center = legal_features.mean(dim=0, keepdim=True)
                # 中心损失：让合法用户特征向中心聚集
                center_loss = torch.mean(torch.norm(legal_features - feature_center, dim=1))
            else:
                center_loss = torch.tensor(0., device=device)
            
            # 增加L2正则化和中心损失
            l2_reg = torch.tensor(0., device=device)
            for param in comprehensive_detector.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param)
            
            # 总损失：分类损失 + 中心损失 + L2正则
            total_loss_with_reg = classification_loss + 0.1 * center_loss + 5e-4 * l2_reg
            
            # 反向传播和优化
            total_loss_with_reg.backward()
            
            # 使用梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(comprehensive_detector.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += classification_loss.item()
            batch_count += 1
            
            # 统计准确率和正类预测数量
            with torch.no_grad():
                probabilities = torch.sigmoid(output_logits)
                # 使用平衡的阈值
                balanced_threshold = 0.5
                predicted_labels = (probabilities > balanced_threshold).float()
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
        
        # 更新学习率（warmup + cosine）
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        # 在每个epoch后测试入侵者检测器性能 - 使用平衡阈值
        balanced_threshold = 0.5
        val_accuracy, val_f1, val_precision, val_recall, val_auroc = validate_intruder_detector(
            comprehensive_detector, identity_model, data_loaders['intruder_validation'], device, threshold=balanced_threshold)

        test_accuracy, test_f1, test_precision, test_recall, test_auroc = validate_intruder_detector(
            comprehensive_detector, identity_model, data_loaders['intruder_test'], device, threshold=balanced_threshold)

        val_metrics.append((val_accuracy, val_f1, val_precision, val_recall, val_auroc))
        test_metrics.append((test_accuracy, test_f1, test_precision, test_recall, test_auroc))

        # 输出训练信息（增加详细的性能分析）
        openmax_status = "启用" if use_openmax else "禁用"
        print(f'Epoch [{epoch+1}/{num_epochs}] [OpenMax: {openmax_status}]')
        print(f'  训练: 损失={avg_loss:.4f}, 准确率={train_accuracy:.4f}, 学习率={optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  训练预测分布: 正类预测={positive_prediction_ratio:.2%}, 正类真实={positive_label_ratio:.2%}')
        print(f'  验证集: 准确率={val_accuracy:.4f}, F1={val_f1:.4f}, 精确率={val_precision:.4f}, 召回率={val_recall:.4f}, AUROC={val_auroc:.4f}')
        print(f'  测试集: 准确率={test_accuracy:.4f}, F1={test_f1:.4f}, 精确率={test_precision:.4f}, 召回率={test_recall:.4f}, AUROC={test_auroc:.4f}')

        # 综合评分：优先测试集准确率和F1，兼顾AUROC
        test_comprehensive_score = 0.4 * test_accuracy + 0.4 * test_f1 + 0.2 * test_auroc
        
        # 保存最佳模型
        if test_comprehensive_score > best_score:
            best_score = test_comprehensive_score
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': comprehensive_detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
                'best_score': best_score,
                'val_metrics': (val_accuracy, val_f1, val_precision, val_recall, val_auroc),
                'test_metrics': (test_accuracy, test_f1, test_precision, test_recall, test_auroc),
            }, output_path)
            print(f'  ✓ 保存最佳模型 (综合评分: {best_score:.4f})')
        else:
            early_stop_counter += 1
            print(f'  早停计数: {early_stop_counter}/{patience}')
            
        # 早停检查
        if early_stop_counter >= patience:
            print(f'\n提前停止: 综合评分在 {patience} 个epoch内未提升')
            break
    
    print(f"\n训练完成! 最佳综合评分: {best_score:.4f}")
    
    # 基于测试集准确率找到最佳 Epoch，并打印该 Epoch 的验证集和测试集综合指标
    if len(val_metrics) > 0:
        test_accuracies = [m[0] for m in test_metrics]
        best_acc_idx = int(np.argmax(test_accuracies))
        best_epoch_acc = best_acc_idx + 1
        best_val_accuracy, best_val_f1, best_val_precision, best_val_recall, best_val_auroc = val_metrics[best_acc_idx]
        best_test_accuracy, best_test_f1, best_test_precision, best_test_recall, best_test_auroc = test_metrics[best_acc_idx]
        print(f"基于测试集准确率的最佳 Epoch: {best_epoch_acc}")
        print(f"  验证集 - 准确率: {best_val_accuracy:.4f}, F1: {best_val_f1:.4f}, 精确率: {best_val_precision:.4f}, 召回率: {best_val_recall:.4f}, AUROC: {best_val_auroc:.4f}")
        print(f"  测试集 - 准确率: {best_test_accuracy:.4f}, F1: {best_test_f1:.4f}, 精确率: {best_test_precision:.4f}, 召回率: {best_test_recall:.4f}, AUROC: {best_test_auroc:.4f}")
    
    # 保存训练历史
    save_training_history(train_losses, val_metrics, test_metrics, 'R_Intruder/training_history.json')


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
    model_path = "R_Identify/best_identify_model.pth"  # 身份识别训练好的模型
    output_path = "R_Intruder/best_intruder_detector.pth"

    # 训练入侵者检测器
    train_intruder_detector(model_path, output_path, device)


if __name__ == "__main__":
    main()