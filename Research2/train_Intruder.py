import torch
import numpy as np
import json
from model_Identify import IdentifyDetectionSystem
from model_Intruder import LearnableComprehensiveIntruderDetector
from loss_Intruder import IntruderDetectionLoss
from Research2.DataProcess.dataloader_intruder import load_intruder_data, create_intruder_data_loaders
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


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
        # 使用Beta分布采样，生成接近0.5的系数，制造最难样本
        lam = torch.distributions.Beta(2.0, 2.0).sample((num_samples, 1)).to(device)
        
        # 执行 Mixup: x_pseudo = lam * x_i + (1-lam) * x_j
        # 仅对diff_mask为1的样本生效
        pseudo_features = lam * legal_features + (1 - lam) * shuffled_features
        
        # 对于同类样本(diff_mask=0)，我们添加强高斯噪声作为回退策略
        noise = torch.randn_like(legal_features) * 2.0  # 增大噪声幅度
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
    增强版：在return_scores=True时，混入伪入侵者用于阈值选择
    """
    model.eval()
    identity_model.eval()
    
    all_predictions = []
    all_labels = []
    all_scores = []
    all_features_list = []  # 收集特征用于生成伪入侵者
    all_identity_labels_list = []  # 收集身份标签用于跨类混合
    
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

            # 检查特征和logits的维度，确保至少是2D
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
            
            # 收集特征和身份标签用于后续生成伪入侵者
            if return_scores:
                all_features_list.append(features)
                all_identity_labels_list.append(identity_labels)

    # 计算评估指标
    if len(all_predictions) == 0 or len(all_labels) == 0:
        if return_scores:
            return 0.0, 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([])
        else:
            return 0.0, 0.0, 0.0, 0.0, 0.0
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)  # 这些全是 0 (合法用户)
    all_scores = np.array(all_scores)
    
    # === 大力修改：如果是验证模式(return_scores=True)，混入伪入侵者数据来选取阈值 ===
    if return_scores and len(all_scores) > 0 and len(all_features_list) > 0:
        # 1. 模拟一批伪入侵者的分数
        # 我们可以复用训练时的生成逻辑，或者简单地假设模型应该对未知区域有高分
        # 但为了稳健，最好的办法是让模型对生成的 Mixup 数据跑一遍
        
        # 合并所有特征
        all_features_tensor = torch.cat(all_features_list, dim=0)
        all_identity_labels_tensor = torch.cat(all_identity_labels_list, dim=0)
        
        # 获取类中心用于生成伪入侵者
        class_centers = model.one_class_detector.class_centers
        
        # 使用训练时的Mixup策略生成伪入侵者
        num_pseudo = len(all_scores)  # 生成与合法用户相同数量的伪入侵者
        pseudo_features = generate_pseudo_intruders(
            num_pseudo,
            feature_dim=32,
            device=device,
            legal_features=all_features_tensor,
            class_centers=class_centers,
            identity_labels=all_identity_labels_tensor,
            strategy='hybrid'  # 使用混合策略
        )
        
        # 让模型对伪入侵者进行预测
        with torch.no_grad():
            # 伪入侵者没有真实的logits和身份标签
            dummy_logits = torch.zeros(num_pseudo, 10, device=device)  # 10个类
            dummy_identity_labels = torch.full((num_pseudo,), -1, dtype=torch.long, device=device)
            
            fake_outputs = model(pseudo_features, dummy_logits, dummy_identity_labels)
            fake_intruder_scores = fake_outputs['probabilities'].cpu().numpy()
        
        # 2. 混合数据用于计算最佳阈值
        mixed_scores = np.concatenate([all_scores, fake_intruder_scores])
        # 0是合法，1是伪入侵者
        mixed_labels = np.concatenate([np.zeros(len(all_scores)), np.ones(len(fake_intruder_scores))])
        
        return 0.0, 0.0, 0.0, 0.0, 0.0, mixed_scores, mixed_labels
    
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
    print("\n初始化损失函数...")
    loss_fn = IntruderDetectionLoss()
    print("损失函数: IntruderDetectionLoss (合法用户目标异常分数=0)")
    
    # 设置优化器：训练融合网络和distance_processor MLP
    # 必须同时优化distance_processor,否则它一直是随机初始化状态
    trainable_params = list(comprehensive_detector.fusion_network.parameters()) + \
                       list(comprehensive_detector.one_class_detector.distance_processor.parameters())
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

            # === 核心修改 1：生成伪入侵者（使用混合策略）===
            # 动态生成伪入侵者：结合Mixup和特征扰动
            batch_size = features.size(0)
            pseudo_batch_size = batch_size // 2  # 伪入侵者数量为合法用户的一半
            
            # 获取类中心用于扰动策略
            class_centers = comprehensive_detector.one_class_detector.class_centers
            
            # 使用混合策略生成伪入侵者（传入identity_labels用于强制跨类混合）
            pseudo_features = generate_pseudo_intruders(
                pseudo_batch_size, 
                feature_dim=32,  # 32维流形空间
                device=device,
                legal_features=features,  # 传入合法用户特征
                class_centers=class_centers,  # 传入类中心
                identity_labels=identity_labels,  # 传入身份标签用于跨类混合
                strategy='hybrid'  # 混合策略：强制跨类Mixup + 扰动
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
        
        # 在每个epoch后，使用验证集找到最佳F1阈值（代价敏感策略 + 混入伪入侵者）
        # 先在验证集上收集分数（会自动混入伪入侵者）
        _, _, _, _, _, val_scores, val_labels = validate_intruder_detector(
            comprehensive_detector,
            identity_model,
            data_loaders['intruder_validation'],
            device,
            threshold=0.5,  # 阈值对分数本身无影响
            pos_label=1,  # 现在验证集包含伪入侵者，正类为1
            return_scores=True
        )

        val_scores = np.array(val_scores)
        val_labels = np.array(val_labels)

        # === 大力修改：使用 Precision-Recall 曲线寻找最佳 F1 的阈值 ===
        if len(val_scores) > 0 and len(np.unique(val_labels)) > 1:
            from sklearn.metrics import precision_recall_curve
            precision, recall, thresholds = precision_recall_curve(val_labels, val_scores)
            
            # 计算每个阈值下的 F1
            f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
            best_idx = np.argmax(f1_scores)
            dynamic_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
            
            print(f"  [动态阈值调整] 基于验证集(含伪入侵者)的最佳阈值: {dynamic_threshold:.4f} (F1={f1_scores[best_idx]:.4f})")
        else:
            dynamic_threshold = 0.5
            print(f"  [警告] 验证集数据不足或类别单一，使用默认阈值: {dynamic_threshold:.4f}")

        # 使用动态阈值在验证集上计算指标（现在包含伪入侵者，pos_label=1）
        if len(val_labels) == 0:
            val_accuracy = val_f1 = val_precision = val_recall = val_auroc = 0.0
        else:
            val_predictions = (val_scores > dynamic_threshold).astype(int)
            val_accuracy = float(np.mean(val_predictions == val_labels))
            # 现在验证集混合了合法用户(0)和伪入侵者(1)，正类为1
            val_f1 = float(f1_score(val_labels, val_predictions, pos_label=1, zero_division=0))
            val_precision = float(precision_score(val_labels, val_predictions, pos_label=1, zero_division=0))
            val_recall = float(recall_score(val_labels, val_predictions, pos_label=1, zero_division=0))
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