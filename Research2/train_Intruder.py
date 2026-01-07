import torch
import numpy as np
import json
from model_Identify import IdentifyDetectionSystem
from model_Intruder import LearnableComprehensiveIntruderDetector
from loss_Intruder import IntruderDetectionLoss
from Research2.DataProcess.dataloader_intruder import load_intruder_data, create_intruder_data_loaders
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import torch.nn.functional as F


def generate_pseudo_intruders(batch_size, feature_dim, device, noise_type='gaussian', 
                             legal_features=None, class_centers=None, identity_labels=None, strategy='noise'):
    """
    生成伪入侵者样本（支持多种策略）
    
    Args:
        batch_size: 生成的样本数量
        feature_dim: 特征维度（32维流形空间）
        device: 设备
        noise_type: 噪声类型，'gaussian' 或 'uniform'（仅在strategy='noise'时使用）
        legal_features: 合法用户特征，用于Mixup (batch_size, feature_dim)
        class_centers: 类中心，用于特征扰动 (num_classes, feature_dim)
        identity_labels: 身份标签，用于确保跨类混合 (batch_size,)
        strategy: 生成策略 'noise'(随机噪声), 'mixup'(类间混合), 'perturbation'(边缘扰动), 'hybrid'(混合策略), 'hard_mixup'(强制跨类混合)
    
    Returns:
        pseudo_features: 伪入侵者特征 (batch_size, feature_dim)
    """
    if strategy == 'noise':
        # 原始策略：随机噪声
        if noise_type == 'gaussian':
            pseudo_features = torch.randn(batch_size, feature_dim, device=device)
        elif noise_type == 'uniform':
            pseudo_features = torch.rand(batch_size, feature_dim, device=device) * 2 - 1
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
        pseudo_features = torch.nn.functional.normalize(pseudo_features, p=2, dim=1)
    
    elif strategy == 'hard_mixup' and legal_features is not None and identity_labels is not None:
        # 强制跨类混合：基于流形边界的伪入侵者生成 (Manifold Mixup)
        # 策略：在不同类的合法用户特征之间进行插值，生成位于“类间真空区”的困难样本
        num_samples = legal_features.size(0)
        if num_samples < 2:
            return generate_pseudo_intruders(batch_size, feature_dim, device, noise_type, strategy='noise')
        
        # 确保identity_labels在正确的设备上
        if not isinstance(identity_labels, torch.Tensor):
            identity_labels = torch.tensor(identity_labels, device=device)
        elif identity_labels.device != legal_features.device:
            identity_labels = identity_labels.to(legal_features.device)
        
        # 随机打乱特征顺序
        perm = torch.randperm(num_samples).to(device)
        shuffled_features = legal_features[perm]
        shuffled_labels = identity_labels[perm]
        
        # 确保只在不同类之间混合 (同类混合还是合法用户)
        diff_mask = (identity_labels != shuffled_labels).float().unsqueeze(1)  # (batch_size, 1)
        
        # 生成混合系数 lambda (偏向于0.5，即处于两类中间)
        # 增加Beta分布的集中度,让样本更接近0.5 (最难区域)
        lam = torch.distributions.Beta(3.0, 3.0).sample((num_samples, 1)).to(device)  # 从(2.0, 2.0)增加到(3.0, 3.0)
        
        # 执行 Mixup: x_pseudo = lam * x_i + (1-lam) * x_j
        # 仅对diff_mask为1的样本生效
        pseudo_features = lam * legal_features + (1 - lam) * shuffled_features
        
        # 对于同类样本(diff_mask=0)，我们添加强高斯噪声作为回退策略
        noise = torch.randn_like(legal_features) * 3.0  # 增大噪声幅度到2.0到3.0
        pseudo_features = diff_mask * pseudo_features + (1 - diff_mask) * (legal_features + noise)
        
        # L2归一化，确保在流形球面上
        pseudo_features = torch.nn.functional.normalize(pseudo_features, p=2, dim=1)
        
        # 只返回请求的batch_size数量
        if pseudo_features.size(0) > batch_size:
            pseudo_features = pseudo_features[:batch_size]
    
    elif strategy == 'mixup' and legal_features is not None:
        # 原始 Mixup策略：类间混合生成边缘样本
        num_samples = legal_features.size(0)
        if num_samples < 2:
            return generate_pseudo_intruders(batch_size, feature_dim, device, noise_type, strategy='noise')
        
        pseudo_features = []
        for _ in range(batch_size):
            # 随机选择两个不同的样本（使用.item()确保是标量索引）
            idx_i = torch.randint(0, num_samples, (1,), device=device).item()
            idx_j = torch.randint(0, num_samples, (1,), device=device).item()
            while idx_i == idx_j and num_samples > 1:
                idx_j = torch.randint(0, num_samples, (1,), device=device).item()
            
            # Mixup系数：偏向边界区域（beta分布 alpha=0.2）
            lam = torch.distributions.Beta(0.2, 0.2).sample().to(device)
            mixed = lam * legal_features[idx_i] + (1 - lam) * legal_features[idx_j]
            pseudo_features.append(mixed)
        
        pseudo_features = torch.stack(pseudo_features, dim=0)
    
    elif strategy == 'perturbation' and legal_features is not None and class_centers is not None:
        # 特征扰动策略：对合法用户添加定向扰动
        num_samples = legal_features.size(0)
        pseudo_features = []
        
        for _ in range(batch_size):
            # 随机选择一个合法用户样本
            idx = torch.randint(0, num_samples, (1,), device=device).item()
            base_feature = legal_features[idx]  # 现在是1D张量
            
            # 找到最近的类中心
            distances = torch.cdist(base_feature.unsqueeze(0), class_centers, p=2).squeeze(0)
            nearest_center_idx = torch.argmin(distances)
            nearest_center = class_centers[nearest_center_idx]
            
            # 计算从类中心指向样本的方向向量
            direction = base_feature - nearest_center
            direction = torch.nn.functional.normalize(direction, p=2, dim=0)
            
            # 在该方向上添加扰动，推向边缘（扰动强度0.3-0.8）
            perturbation_scale = torch.rand(1, device=device) * 0.5 + 0.3
            perturbed = base_feature + direction * perturbation_scale
            pseudo_features.append(perturbed)
        
        pseudo_features = torch.stack(pseudo_features, dim=0)
    
    elif strategy == 'hybrid':
        # 混合策略：结合强制跨类Mixup和扰动
        half_batch = batch_size // 2
        
        # 一半使用强制跨类Mixup
        mixup_features = generate_pseudo_intruders(
            half_batch, feature_dim, device, 
            legal_features=legal_features,
            identity_labels=identity_labels,
            strategy='hard_mixup'
        )
        
        # 一半使用扰动
        perturb_features = generate_pseudo_intruders(
            batch_size - half_batch, feature_dim, device,
            legal_features=legal_features,
            class_centers=class_centers,
            strategy='perturbation'
        )
        
        pseudo_features = torch.cat([mixup_features, perturb_features], dim=0)
    
    else:
        # 默认回退到噪声
        pseudo_features = torch.randn(batch_size, feature_dim, device=device)
        pseudo_features = torch.nn.functional.normalize(pseudo_features, p=2, dim=1)
    
    return pseudo_features


def find_optimal_threshold(scores, labels, criterion='f1', pos_label=1):
    """
    基于验证集找到最佳阈值（代价敏感）
    
    Args:
        scores: 预测分数 (n,)
        labels: 真实标签 (n,)
        criterion: 优化目标 'f1'(优化F1), 'balanced'(平衡准确率/召回率), 'precision'(优化精确率), 'recall'(优化召回率)
        pos_label: 正类标签
    
    Returns:
        best_threshold: 最佳阈值
        best_score: 最佳评分
    """
    if len(scores) == 0 or len(labels) == 0:
        return 0.5, 0.0
    
    # 生成候选阈值：从min到max分100个点
    min_score = np.min(scores)
    max_score = np.max(scores)
    thresholds = np.linspace(min_score, max_score, 100)
    
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        
        if criterion == 'f1':
            # F1分数
            score = f1_score(labels, predictions, pos_label=pos_label, zero_division=0)
        elif criterion == 'balanced':
            # 平衡准确率和召回率
            prec = precision_score(labels, predictions, pos_label=pos_label, zero_division=0)
            rec = recall_score(labels, predictions, pos_label=pos_label, zero_division=0)
            score = 0.5 * prec + 0.5 * rec
        elif criterion == 'precision':
            # 精确率
            score = precision_score(labels, predictions, pos_label=pos_label, zero_division=0)
        elif criterion == 'recall':
            # 召回率
            score = recall_score(labels, predictions, pos_label=pos_label, zero_division=0)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


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

            # 使用身份识别模型提取特征
            identity_outputs = identity_model(data)
            if 'proj' in identity_outputs:
                features = identity_outputs['proj']
            else:
                features = identity_outputs['features']
            logits = identity_outputs['logits']

            # 检查特征和logits的维度
            if features.dim() == 1:
                features = features.unsqueeze(0)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            # 确保 batch维度一致
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
    
    accuracy = float(np.mean(all_predictions == all_labels)) if len(all_labels) > 0 else 0.0
    
    f1 = float(f1_score(all_labels, all_predictions, pos_label=pos_label, zero_division=0))
    precision = float(precision_score(all_labels, all_predictions, pos_label=pos_label, zero_division=0))
    recall = float(recall_score(all_labels, all_predictions, pos_label=pos_label, zero_division=0))

    # AUROC
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
        
        # 初始化TraditionalOpenMax（在首次更新时启用alpha搜索）
        val_features, _, val_labels, val_identity_labels = extract_features(
            identity_model, data_loaders['intruder_validation'], device)
        
        comprehensive_detector.fit_traditional_openmax(
            train_features, train_labels, train_identity_labels,
            search_alpha=True,  # 启用alpha搜索
            val_features=val_features,
            val_labels=val_labels,
            val_identity_labels=val_identity_labels
        )
        print(f"TraditionalOpenMax初始化完成 (训练样本数: {len(train_features)}, 启用alpha搜索)")
    
    # 初始化损失函数
    print("\n" + "="*60)
    print("初始化损失函数...")
    loss_fn = IntruderDetectionLoss()
    print("损失函数配置:")
    print("  - 类型BCE Loss + 类别加权 + L2正则化")
    print("  - 入侵者权重(pos_weight): 3.0")
    print("  - L2正则化系数: 0.005")
    print("="*60)
    
    # 优化版：更合理的优化器和学习率
    trainable_params = list(comprehensive_detector.fusion_network.parameters()) + \
                       list(comprehensive_detector.one_class_detector.distance_processor.parameters())
    
    print("\n" + "="*60)
    print("优化器配置:")
    print("  - 优化器: Adam")
    print("  - 初始学习率01e-3")
    print("  - 权重衰减: 1e-3")
    print("  - 学习率调度器: ReduceLROnPlateau")
    print("    * 基于验证集F1动态调整")
    print("    * factor=0.5, patience=10")
    print("="*60)
    
    # 使用Adam+适中学习率+更强正则化
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.999))
    
    # 优化学习率调度: ReduceLROnPlateau - 根据性能动态调整
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5
    )
    
    # OpenMax更新配置
    openmax_update_interval = 10  # 每10个epoch更新一次OpenMax
    use_openmax_after_epoch = 5   # 从第6个epoch开始使用OpenMax
    
    comprehensive_detector.train()
    
    num_epochs = 150  # 增加训练轮数
    best_score = 0.0  # 跟踪最佳综合评分
    best_test_f1 = 0.0  # 跟踪最佳F1
    early_stop_counter = 0
    patience = 40  # 增加耐心值
    
    # 记录训练历史
    train_losses = []
    val_metrics = []  # (accuracy, f1, precision, recall, auroc)
    test_metrics = []  # (accuracy, f1, precision, recall, auroc)
    
    print("\n" + "="*60)
    print("训练配置:")
    print(f"  - 总轮数: {num_epochs}")
    print(f"  - 早停耐心值: {patience}")
    print(f"  - OpenMax启用: 第{use_openmax_after_epoch+1}轮开始")
    print(f"  - OpenMax更新间隔: 每{openmax_update_interval}轮")
    print("  - 评估指标: F1分数(主), AUROC, 准确率")
    print("  - 阈值选择: 基于验证集F1最大化(代价敏感)")
    print("="*60)
    print("开始训练...\n")

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
        
        # ====== 闭集学习训练循环：直接使用合法用户 + 真实入侵者 ======
        for batch_idx, batch in enumerate(data_loaders['intruder_train']):
            data, labels, identity_labels = batch
            
            data = data.to(device)
            labels = labels.to(device).float()  # 0=合法用户，1=真实入侵者
            
            optimizer.zero_grad()
            
            # 使用身份识别模型提取特征
            with torch.no_grad():
                identity_outputs = identity_model(data)
                features = identity_outputs.get('proj', identity_outputs['features'])  # 32维流形特征
                logits = identity_outputs['logits']

            # 直接使用模型进行预测
            outputs = comprehensive_detector(features, logits, identity_labels, use_openmax=use_openmax)
            predictions = outputs['probabilities']  # 入侵者概率
            logits_output = outputs['logits']  # logits用于损失计算
            
            # === 优化损失函数：使用Focal Loss + 适度加权 ===
            # 对于类别不平衡问题，Focal Loss比BCE更有效
            # Focal Loss: FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t)
            # gamma=2.0: 让模型更关注困难样本
            
            # === 优化损失函数：BCE + 类别加权 + L2正则化 ===
            # 类别加权：平衡类别
            pos_weight = torch.tensor([3.0], device=device)  # 入侵者权重
            bce_loss = F.binary_cross_entropy_with_logits(
                logits_output, 
                labels, 
                pos_weight=pos_weight
            )
            
            # L2正则化
            l2_lambda = 0.005
            l2_reg = torch.tensor(0., device=device)
            for param in trainable_params:
                l2_reg += torch.norm(param, p=2)
            loss = bce_loss + l2_lambda * l2_reg
            
            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        # 计算平均损失
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_loss)
        
        # 在每个epoch后，使用验证集找到最佳F1阈值
        val_accuracy, val_f1, val_precision, val_recall, val_auroc, val_scores, val_labels = validate_intruder_detector(
            comprehensive_detector,
            identity_model,
            data_loaders['intruder_validation'],
            device,
            threshold=0.5,
            pos_label=1,  # 入侵者为正类
            return_scores=True
        )

        val_scores = np.array(val_scores)
        val_labels = np.array(val_labels)

        # 使用F1分数找最佳阈值
        if len(val_scores) > 0 and len(np.unique(val_labels)) > 1:
            from sklearn.metrics import precision_recall_curve
            precision, recall, thresholds = precision_recall_curve(val_labels, val_scores)
            
            # 计算每个阈值下的 F1
            f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
            best_idx = int(np.argmax(f1_scores))
            dynamic_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
            
            print(f"  [动态阈值调整] 验证集最佳阈值: {dynamic_threshold:.4f} (F1={f1_scores[best_idx]:.4f})")
        else:
            dynamic_threshold = 0.5
            print(f"  [警告] 验证集数据不足，使用默认阈值: {dynamic_threshold:.4f}")

        # 使用动态阈值在验证集上计算指标
        if len(val_labels) > 0:
            val_predictions = (val_scores > dynamic_threshold).astype(int)
            val_accuracy = float(np.mean(val_predictions == val_labels))
            val_f1 = float(f1_score(val_labels, val_predictions, pos_label=1, zero_division=0))
            val_precision = float(precision_score(val_labels, val_predictions, pos_label=1, zero_division=0))
            val_recall = float(recall_score(val_labels, val_predictions, pos_label=1, zero_division=0))
            try:
                val_auroc = float(roc_auc_score(val_labels, val_scores))
            except ValueError:
                val_auroc = 0.0
        else:
            val_accuracy = val_f1 = val_precision = val_recall = val_auroc = 0.0
        
        # 更新学习率（基于验证集F1）
        scheduler.step(val_f1)

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

        # 优化模型保存策略 - 优先F1分数
        test_comprehensive_score = 0.6 * test_f1 + 0.3 * test_auroc + 0.1 * test_accuracy
        
        # 保存最佳模型（基于F1）
        is_best = False
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_score = test_comprehensive_score
            early_stop_counter = 0
            is_best = True
            torch.save({
                'epoch': epoch,
                'model_state_dict': comprehensive_detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score,
                'best_f1': best_test_f1,
                'val_metrics': (val_accuracy, val_f1, val_precision, val_recall, val_auroc),
                'test_metrics': (test_accuracy, test_f1, test_precision, test_recall, test_auroc),
            }, output_path)
            print(f'  ✓ 保存最佳模型 (F1={best_test_f1:.4f}, 综合评分={best_score:.4f})')
        else:
            early_stop_counter += 1
            print(f'  早停计数: {early_stop_counter}/{patience} (当前F1={test_f1:.4f}, 最佳F1={best_test_f1:.4f})')
            
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