import torch
import numpy as np
import json
from model_Identify import IdentifyDetectionSystem
from model_Intruder import LearnableComprehensiveIntruderDetector
from loss_Intruder import IntruderDetectionLoss
from Research2.DataProcess.dataloader_intruder import load_intruder_data, create_intruder_data_loaders
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def generate_pseudo_intruders(batch_size, feature_dim, device, noise_type='gaussian'):
    """
    生成伪入侵者样本（随机噪声）
    
    Args:
        batch_size: 生成的样本数量
        feature_dim: 特征维度（32维流形空间）
        device: 设备
        noise_type: 噪声类型，'gaussian' 或 'uniform'
    
    Returns:
        pseudo_features: 伪入侵者特征 (batch_size, feature_dim)
    """
    if noise_type == 'gaussian':
        # 高斯噪声：均值0，标准差1
        pseudo_features = torch.randn(batch_size, feature_dim, device=device)
    elif noise_type == 'uniform':
        # 均匀分布噪声：[-1, 1]
        pseudo_features = torch.rand(batch_size, feature_dim, device=device) * 2 - 1
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    
    # L2归一化，保持与真实特征相同的尺度
    pseudo_features = torch.nn.functional.normalize(pseudo_features, p=2, dim=1)
    
    return pseudo_features


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


def validate_intruder_detector(model, identity_model, data_loader, device, threshold=0.5, pos_label=1, return_scores=False):
    """
    在验证集或测试集上测试入侵者检测器性能
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
            # 确保使用32维投彡特征
            if 'proj' in identity_outputs:
                features = identity_outputs['proj']
            else:
                features = identity_outputs['features']
            logits = identity_outputs['logits']

            # 检查特征和logits的维度，确保至少晈2D
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
        if return_scores:
            return 0.0, 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([])
        else:
            return 0.0, 0.0, 0.0, 0.0, 0.0
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # 处理空数组情况
    if len(all_labels) == 0:
        if return_scores:
            return 0.0, 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([])
        else:
            return 0.0, 0.0, 0.0, 0.0, 0.0
    
    accuracy = float(np.mean(all_predictions == all_labels)) if len(all_labels) > 0 else 0.0
    
    # 使用zero_division=0避免警告，并允许指定正类标签
    f1 = float(f1_score(all_labels, all_predictions, pos_label=pos_label, zero_division=0))
    precision = float(precision_score(all_labels, all_predictions, pos_label=pos_label, zero_division=0))
    recall = float(recall_score(all_labels, all_predictions, pos_label=pos_label, zero_division=0))

    # AUROC（当正负样本都存在时才有意义）
    try:
        auroc = float(roc_auc_score(all_labels, all_scores))
    except ValueError:
        auroc = 0.0
    
    if return_scores:
        return accuracy, f1, precision, recall, auroc, all_scores, all_labels
    else:
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
    训练综合入侵者检测器 - 开放集学习策略
    核心策略：只用合法用户训练，学习紧凑流形表示
    """

    # 初始化身份识别模型（必须与训练时的参数一致）
    print("加载身份识别-流形优化模型...")
    identity_model = IdentifyDetectionSystem(num_classes=10, feature_dim=512, projection_dim=32).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    identity_model.load_state_dict(checkpoint['model_state_dict'])
    identity_model.eval()
    
    # 加载入侵者检测专用数据
    print("加载入侵者检测数据...")
    datasets = load_intruder_data()
    data_loaders = create_intruder_data_loaders(datasets, batch_size=32)

    # 初始化综合入侵者检测器，在流形投影空间（32维）上工作
    comprehensive_detector = LearnableComprehensiveIntruderDetector(num_known_users=10, feature_dim=32).to(device)

    # === 初始化类中心（只执行一次） ===
    print("\n===== 从训练集初始化类中心 =====")
    # 从训练集中提取特征用于初始化类中心
    train_features, _, train_labels, train_identity_labels = extract_features(identity_model, data_loaders['intruder_train'], device)
    
    if train_features.size > 0:
        # 转换为tensor
        train_features_tensor = torch.from_numpy(train_features).float().to(device)
        train_identity_tensor = torch.from_numpy(train_identity_labels).long().to(device)
        
        # 初始化OneClassDetector的类中心
        comprehensive_detector.one_class_detector.initialize_centers(train_features_tensor, train_identity_tensor)
        
        # 初始化TraditionalOpenMax
        comprehensive_detector.fit_traditional_openmax(train_features, train_labels, train_identity_labels)
        print(f"TraditionalOpenMax初始化完成 (训练样本数: {len(train_features)})")
    
    # 初始化损失函数
    print("\n初始化损失函数...")
    loss_fn = IntruderDetectionLoss()
    print("损失函数: IntruderDetectionLoss (合法用户目标异常分数=0)")
    
    # 设置优化器：训练融合网络和distance_to_score MLP
    # 修复Bug: 必须同时优化distance_to_score,否则它一直是随机初始化状态
    trainable_params = list(comprehensive_detector.fusion_network.parameters()) + \
                       list(comprehensive_detector.one_class_detector.distance_to_score.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    
    # 使用余弦退火学习率调度器
    warmup_epochs = 3
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=47, eta_min=1e-6)
    
    # OpenMax更新配置
    openmax_update_interval = 10  # 每10个epoch更新一次OpenMax
    use_openmax_after_epoch = 5   # 从第6个epoch开始使用OpenMax
    
    comprehensive_detector.train()
    
    num_epochs = 50
    best_score = 0.0  # 跟踪最佳综合评分
    early_stop_counter = 0
    patience = 20  # 早停耐心值
    
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
        batch_count = 0
        
        # ====== 开放集学习训练循环：合法用户 + 伪入侵者 ======
        for batch_idx, batch in enumerate(data_loaders['intruder_train']):
            data, labels, identity_labels = batch
            
            data = data.to(device)
            labels = labels.to(device).float()  # 训练集只有合法用户（标签均为0）
            
            optimizer.zero_grad()
            
            # 使用身份识别模型提取特征
            with torch.no_grad():
                identity_outputs = identity_model(data)
                features = identity_outputs.get('proj', identity_outputs['features'])  # 32维流形特征
                logits = identity_outputs['logits']

            # === 核心修改 1：生成伪入侵者 ===
            # 动态生成一批随机噪声作为伪入侵者
            batch_size = features.size(0)
            pseudo_batch_size = batch_size // 2  # 伪入侵者数量为合法用户的一半
            pseudo_features = generate_pseudo_intruders(
                pseudo_batch_size, 
                feature_dim=32,  # 32维流形空间
                device=device,
                noise_type='gaussian'
            )
            
            # 为伪入侵者生成虚拟logits（全零，因为不属于任何已知类）
            pseudo_logits = torch.zeros(pseudo_batch_size, logits.size(1), device=device)
            # 为伪入侵者生成虚拟身份标签（-1表示未知）
            pseudo_identity_labels = torch.full((pseudo_batch_size,), -1, dtype=torch.long, device=device)

            # === 核心修改 2：分别处理合法用户和伪入侵者 ===
            # 1. 处理合法用户
            legal_outputs = comprehensive_detector(features, logits, identity_labels, use_openmax=use_openmax)
            legal_anomaly_scores = legal_outputs['logits']  # 合法用户的异常分数
            legal_min_distances = legal_outputs['min_distance']  # 到最近类中心的距离
            
            # 2. 处理伪入侵者
            pseudo_outputs = comprehensive_detector(pseudo_features, pseudo_logits, pseudo_identity_labels, use_openmax=use_openmax)
            pseudo_anomaly_scores = pseudo_outputs['logits']  # 伪入侵者的异常分数
            
            # === 核心修改 3：使用增强版损失函数 ===
            # 计算损失：同时监督合法用户和伪入侵者
            loss = loss_fn(
                anomaly_scores=legal_anomaly_scores,  # 合法用户分数
                min_distances=legal_min_distances,    # 合法用户距离
                pseudo_scores=pseudo_anomaly_scores   # 伪入侵者分数
            )
            
            # 反向传播和优化
            loss.backward()
            # 修复Bug: 梯度裁剪需要包含所有训练的参数
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        # 计算平均损失
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_loss)
        
        # 更新学习率（warmup + cosine）
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        # 在每个epoch后，根据验证集合法用户分数自适应选择阈值
        # 先在验证集上收集分数（验证集只包含合法用户，标签为0）
        _, _, _, _, _, val_scores, val_labels = validate_intruder_detector(
            comprehensive_detector,
            identity_model,
            data_loaders['intruder_validation'],
            device,
            threshold=0.5,  # 阈值对分数本身无影响
            pos_label=0,
            return_scores=True
        )

        val_scores = np.array(val_scores)
        val_labels = np.array(val_labels)

        if len(val_scores) > 0:
            # 选取合法用户分数的高分位数作为入侵者判定阈值（例如95%分位）
            dynamic_threshold = float(np.quantile(val_scores, 0.95))
        else:
            dynamic_threshold = 0.5

        # 使用动态阈值在验证集上计算针对合法用户(0类)的指标
        if len(val_labels) == 0:
            val_accuracy = val_f1 = val_precision = val_recall = val_auroc = 0.0
        else:
            val_predictions = (val_scores > dynamic_threshold).astype(int)
            val_accuracy = float(np.mean(val_predictions == val_labels))
            val_f1 = float(f1_score(val_labels, val_predictions, pos_label=0, zero_division=0))
            val_precision = float(precision_score(val_labels, val_predictions, pos_label=0, zero_division=0))
            val_recall = float(recall_score(val_labels, val_predictions, pos_label=0, zero_division=0))
            try:
                val_auroc = float(roc_auc_score(val_labels, val_scores))
            except ValueError:
                val_auroc = 0.0

        # 使用相同阈值在测试集上评估（入侵者=1 为正类）
        test_accuracy, test_f1, test_precision, test_recall, test_auroc = validate_intruder_detector(
            comprehensive_detector,
            identity_model,
            data_loaders['intruder_test'],
            device,
            threshold=dynamic_threshold,
            pos_label=1
        )

        val_metrics.append((val_accuracy, val_f1, val_precision, val_recall, val_auroc))
        test_metrics.append((test_accuracy, test_f1, test_precision, test_recall, test_auroc))

        # 输出训练信息
        openmax_status = "启用" if use_openmax else "禁用"
        print(f'Epoch [{epoch+1}/{num_epochs}] [OpenMax: {openmax_status}]')
        print(f'  训练: 损失={avg_loss:.4f}, 学习率={optimizer.param_groups[0]["lr"]:.6f}')
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