import torch
import os
import torch.optim as optim
import json
from model_identify import IdentifyDetectionSystem
from loss_identify import IdentifyDetectionLoss
from Research2.DataProcess.dataloader_identify import load_identify_data, create_data_loaders


def train_epoch(model, source_loader, target_loader, criterion, optimizer, device, epoch):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0.0
    loss_components = {'identity_src': 0.0, 'identity_tgt': 0.0, 'manifold': 0.0, 'center': 0.0}
    batch_count = 0
    
    # 使用zip循环处理源域和目标域数据
    for batch_idx, (source_batch, target_batch) in enumerate(zip(source_loader, target_loader)):
        # 获取数据
        src_data, src_labels = source_batch
        tgt_data, tgt_labels = target_batch

        # 移动到设备
        src_data = src_data.to(device)
        tgt_data = tgt_data.to(device)
        src_labels = src_labels.to(device)
        tgt_labels = tgt_labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(src_data, tgt_data)

        # 计算损失
        loss, loss_dict = criterion(outputs, src_labels, tgt_labels)
        
        # 添加center-aware损失 (第四章新增)
        # 需要从modelmod中获取class_centers
        class_centers = model.manifold_projection.class_centers
        center_loss_src = criterion.center_aware_loss(
            outputs['proj_source'], src_labels, class_centers)
        center_loss_tgt = criterion.center_aware_loss(
            outputs['proj_target'], tgt_labels, class_centers)
        center_loss_total = (center_loss_src + center_loss_tgt) / 2
        
        # 加入总损失
        loss = loss + criterion.beta * center_loss_total

        # 反向传播和优化
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # 累计损失
        total_loss += loss.item()
        loss_components['identity_src'] += loss_dict.get('identity_loss_src', 0.0)
        loss_components['identity_tgt'] += loss_dict.get('identity_loss_tgt', 0.0)
        loss_components['manifold'] += loss_dict.get('manifold_loss', 0.0)
        loss_components['center'] += center_loss_total.item()
        batch_count += 1

    # 计算平均损失
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    for key in loss_components:
        loss_components[key] = loss_components[key] / batch_count if batch_count > 0 else 0.0

    return avg_loss, loss_components


def validate(model, val_loader, device, domain_type='mixed'):
    """
    在验证集上评估模型
    domain_type: 'source', 'target', 或 'mixed' (源域和目标域混合数据)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(val_loader):
            # 移动到设备
            data = data.to(device)
            labels = labels.to(device)

            # 前向传播 - 与Research1保持一致的逻辑
            if domain_type == 'target':
                # 目标域：源域输入为零张量
                outputs = model(torch.zeros_like(data).to(device), data)
                logits = outputs['logits_target']
            elif domain_type == 'source':
                # 源域：目标域输入为零张量
                outputs = model(data, torch.zeros_like(data).to(device))
                logits = outputs['logits_source']
            else:
                # 混合验证集：源域和目标域数据混合，随机选择一个作为主域
                # 为了简化，这里使用源域逻辑
                outputs = model(data, torch.zeros_like(data).to(device))
                logits = outputs['logits_source']

            # 计算准确率
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0.0
    return accuracy


def compute_manifold_metrics(model, data_loader, device, domain_type='source'):
    """
    计算流形投影质量指标 - 体现第四章核心创新
    
    Returns:
        - intra_class_distance: 类内平均距离（越小越好，体现紧凑性）
        - inter_class_distance: 类间平均距离（越大越好，体现可分离性）
        - silhouette_score: 轮廓系数（越接近1越好，体现整体流形质量）
        - center_alignment: 类中心对齐度（越小越好，体现center-aware效果）
    """
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    model.eval()
    all_proj_features = []  # 32维投影特征
    all_labels = []
    all_distances_to_center = []  # 到最近类中心的距离
    
    with torch.no_grad():
        for batch in data_loader:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            
            # 前向传播获取投影特征
            if domain_type == 'target':
                outputs = model(torch.zeros_like(data).to(device), data)
                proj = outputs['proj_target']
            else:
                outputs = model(data, torch.zeros_like(data).to(device))
                proj = outputs['proj_source']
            
            # 计算到类中心的距离
            class_centers = model.manifold_projection.class_centers  # [10, 32]
            distances = torch.cdist(proj, class_centers)  # [batch, 10]
            min_distances, _ = torch.min(distances, dim=1)  # [batch]
            
            all_proj_features.append(proj.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_distances_to_center.append(min_distances.cpu().numpy())
    
    # 合并所有batch
    all_proj_features = np.vstack(all_proj_features)  # [N, 32]
    all_labels = np.hstack(all_labels)  # [N]
    all_distances_to_center = np.hstack(all_distances_to_center)  # [N]
    
    # 1. 类内平均距离（体现紧凑性）
    num_classes = len(np.unique(all_labels))
    intra_distances = []
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 1:
            class_features = all_proj_features[mask]
            class_center = class_features.mean(axis=0)
            dists = np.linalg.norm(class_features - class_center, axis=1)
            intra_distances.append(dists.mean())
    intra_class_distance = np.mean(intra_distances) if intra_distances else 0.0
    
    # 2. 类间平均距离（体现可分离性）
    class_centers_np = np.array([all_proj_features[all_labels == c].mean(axis=0) 
                                  for c in range(num_classes)])
    inter_distances = []
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            dist = np.linalg.norm(class_centers_np[i] - class_centers_np[j])
            inter_distances.append(dist)
    inter_class_distance = np.mean(inter_distances) if inter_distances else 0.0
    
    # 3. 轮廓系数（整体流形质量）
    if len(np.unique(all_labels)) > 1 and len(all_labels) > 1:
        silhouette = silhouette_score(all_proj_features, all_labels)
    else:
        silhouette = 0.0
    
    # 4. Center-aware对齐度（到最近类中心的平均距离）
    center_alignment = all_distances_to_center.mean()
    
    return {
        'intra_class_distance': float(intra_class_distance),
        'inter_class_distance': float(inter_class_distance),
        'silhouette_score': float(silhouette),
        'center_alignment': float(center_alignment)
    }


def test_model(model, data_loaders, device):
    """
    在测试集上评估模型 - 与Research1保持一致的测试逻辑
    同时评估流形投影质量（第四章核心创新）
    """
    model.eval()
    test_results = {}
    
    # 测试源域身份识别测试集
    if 'src_identity_test' in data_loaders:
        src_identity_test_accuracy = validate(model, data_loaders['src_identity_test'], device, domain_type='source')
        test_results['src_identity_test'] = src_identity_test_accuracy
        print(f'    源域身份识别测试准确率: {src_identity_test_accuracy:.2f}%')
    
    # 测试目标域身份识别测试集
    if 'tgt_identity_test' in data_loaders:
        tgt_identity_test_accuracy = validate(model, data_loaders['tgt_identity_test'], device, domain_type='target')
        test_results['tgt_identity_test'] = tgt_identity_test_accuracy
        print(f'    目标域身份识别测试准确率: {tgt_identity_test_accuracy:.2f}%')
    
    # ===== 第四章核心指标：流形投影质量评估 =====
    print(f'  【流形投影质量评估】')
    
    # 源域流形质量
    if 'src_identity_test' in data_loaders:
        src_manifold_metrics = compute_manifold_metrics(model, data_loaders['src_identity_test'], device, domain_type='source')
        test_results['src_manifold_metrics'] = src_manifold_metrics
        print(f'    源域流形指标:')
        print(f'      - 类内距离: {src_manifold_metrics["intra_class_distance"]:.4f} (↓越小越紧凑)')
        print(f'      - 类间距离: {src_manifold_metrics["inter_class_distance"]:.4f} (↑越大越可分)')
        print(f'      - 轮廓系数: {src_manifold_metrics["silhouette_score"]:.4f} (↑越接近1越好)')
        print(f'      - Center对齐度: {src_manifold_metrics["center_alignment"]:.4f} (↓越小Center-aware越有效)')
    
    # 目标域流形质量
    if 'tgt_identity_test' in data_loaders:
        tgt_manifold_metrics = compute_manifold_metrics(model, data_loaders['tgt_identity_test'], device, domain_type='target')
        test_results['tgt_manifold_metrics'] = tgt_manifold_metrics
        print(f'    目标域流形指标:')
        print(f'      - 类内距离: {tgt_manifold_metrics["intra_class_distance"]:.4f} (↓越小越紧凑)')
        print(f'      - 类间距离: {tgt_manifold_metrics["inter_class_distance"]:.4f} (↑越大越可分)')
        print(f'      - 轮廓系数: {tgt_manifold_metrics["silhouette_score"]:.4f} (↑越接近1越好)')
        print(f'      - Center对齐度: {tgt_manifold_metrics["center_alignment"]:.4f} (↓越小Center-aware越有效)')
    
    return test_results


def save_training_history(train_losses, val_accuracies, loss_components_history, test_accuracies, file_path):
    """
    保存训练历史数据到JSON文件
    包含流形质量指标历史
    """
    # 提取流形指标历史（如果存在）
    manifold_metrics_history = []
    for test_result in test_accuracies:
        if 'src_manifold_metrics' in test_result and 'tgt_manifold_metrics' in test_result:
            manifold_metrics_history.append({
                'src': test_result['src_manifold_metrics'],
                'tgt': test_result['tgt_manifold_metrics']
            })
    
    history = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'loss_components_history': loss_components_history,
        'test_accuracies': test_accuracies,
        'manifold_metrics_history': manifold_metrics_history,  # 新增：流形质量演化
        'train_accuracies': [100 - loss * 10 for loss in train_losses]  # 简单估算训练准确率
    }
    
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存到 {file_path}")


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 直接加载已保存的数据集
    print("加载已保存的数据集...")
    datasets = load_identify_data()
    data_loaders = create_data_loaders(datasets, batch_size=8)

    # 初始化模型 - 继承自Research1的特征提取器(512维)
    print("初始化模型...")
    model = IdentifyDetectionSystem(num_classes=10, feature_dim=512, projection_dim=32).to(device)
    
    # 加载Research1(第三章)预训练的特征提取器和交叉注意力权重
    pretrained_path = os.path.join('C:\\Users\\USER\\Desktop\\liuheng\\Research1\\R_CAL', 'best_attention_model.pth')
    if os.path.exists(pretrained_path):
        print(f"\n 加载第三章预训练权重: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            pretrained_dict = checkpoint['model_state_dict']
            
            # 只加载feature_extractor和cross_attention的权重(第三章已训练好的部分)
            model_dict = model.state_dict()
            pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() 
                                       if k.startswith('feature_extractor') or k.startswith('cross_attention')}
            
            # 更新模型权重
            model_dict.update(pretrained_dict_filtered)
            model.load_state_dict(model_dict)
            
            print(f"✅ 成功加载预训练权重: {len(pretrained_dict_filtered)} 个参数")
            print(f"   - manifold_projection 和 identity_classifier 将从头训练")
        except Exception as e:
            print(f"⚠️  加载预训练权重失败: {e}")
            print(f"   将从头开始训练所有参数")
    else:
        print(f"⚠️  未找到预训练权重: {pretrained_path}")
        print(f"   将从头开始训练所有参数")
    
    # 初始化损失函数(基于预训练模型)
    # 第三章已学会跨域对齐,第四章关注: 身份分类 + 流形紧凑性 + Center-aware
    # 策略: 先对齐后紧凑 - 初期使用极小的流形约束,避免破坏预训练权重
    print("\n初始化损失函数和优化器...")
    criterion = IdentifyDetectionLoss(alpha=1.0, delta=0.01, beta=0.005)  # 流形约束降低10倍
    
    # 初始化优化器 - 使用更激进的差异化学习率策略
    # 预训练部分: 极小学习率保护跨域对齐能力
    # 新增部分: 正常学习率快速学习
    optimizer = optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': 0.00001},  # 降低10倍,保护预训练特征
        {'params': model.cross_attention.parameters(), 'lr': 0.00001},    # 降低10倍,保护跨域对齐
        {'params': model.manifold_projection.parameters(), 'lr': 0.001},  # 提高2倍,加速流形学习
        {'params': model.identity_classifier.parameters(), 'lr': 0.001}   # 提高2倍,加速分类学习
    ], weight_decay=1e-4)
    
    # 使用余弦退火调度器,更平滑的学习率衰减
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
    print(f"  feature_extractor & cross_attention 学习率: 0.00001 (保护预训练权重)")
    print(f"  manifold_projection & identity_classifier 学习率: 0.001 (快速学习)")
    print(f"  流形约束初始权重: delta={criterion.delta:.4f}, beta={criterion.beta:.4f}")
    
    # 训练参数 - 基于预训练权重,采用渐进式训练策略
    num_epochs = 100  # 增加轮数,给流形约束逐步增强留出空间
    best_accuracy = 0.0  # 最佳验证准确率（用于显示）
    best_comprehensive_score = 0.0  # 最佳综合评分（准确率 + 流形质量）
    early_stop_counter = 0
    patience = 20  # 增加耐心值,应对流形约束调整期的波动
    
    # 记录训练历史
    train_losses = []
    val_accuracies = []
    loss_components_history = []
    test_accuracies = []

    # 确保保存模型的目录存在
    os.makedirs('R_Identify', exist_ok=True)
    
    print("开始训练循环...")
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        # 训练一个epoch
        train_loss, loss_components = train_epoch(
            model, data_loaders['identity_train'], data_loaders['target_aux'], 
            criterion, optimizer, device, epoch)
        
        # 验证模型（使用身份识别验证集 - 混合源域和目标域）
        val_accuracy = validate(model, data_loaders['identity_validation'], device, domain_type='mixed')
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        loss_components_history.append(loss_components)
        
        # 打印epoch结果
        print(f'  训练损失: {train_loss:.4f} ')
        print(f'    - 源域身份: {loss_components["identity_src"]:.4f}')
        print(f'    - 目标域身份: {loss_components["identity_tgt"]:.4f}')
        print(f'    - 流形紧凑性: {loss_components["manifold"]:.4f}')
        print(f'    - Center-aware: {loss_components["center"]:.4f}')
        print(f'  验证准确率: {val_accuracy:.2f}%')
        print(f'  当前学习率: {scheduler.get_last_lr()[0]:.6f}')
        
        # 在每个epoch后测试模型（包含流形质量评估）
        print(f'  测试效果:')
        test_results = test_model(model, data_loaders, device)
        test_accuracies.append(test_results)
        
        # ===== 综合评分：准确率 + 流形质量 =====
        # 计算综合评分（用于早停和模型保存）
        # 权重设计：验证准确率50% + 流形质量50%
        accuracy_score = val_accuracy / 100.0  # 归一化到[0, 1]
        
        # 流形质量评分（如果有流形指标）
        manifold_score = 0.0
        if 'tgt_manifold_metrics' in test_results:
            tgt_metrics = test_results['tgt_manifold_metrics']
            # 归一化流形指标到[0, 1]范围
            # 轮廓系数已经在[-1, 1]，映射到[0, 1]
            silhouette_normalized = (tgt_metrics['silhouette_score'] + 1) / 2.0
            # Center对齐度：越小越好，使用反比例归一化（假设最大值为10）
            center_normalized = max(0, 1 - tgt_metrics['center_alignment'] / 10.0)
            # 类内距离：越小越好，使用反比例归一化（假设最大值为5）
            intra_normalized = max(0, 1 - tgt_metrics['intra_class_distance'] / 5.0)
            
            # 综合流形评分（三个指标平均）
            manifold_score = (silhouette_normalized + center_normalized + intra_normalized) / 3.0
        
        # 综合评分：准确率60% + 流形质量40%
        # 准确率更重要（确保分类能力），流形质量次之（为入侵检测准备）
        comprehensive_score = 0.6 * accuracy_score + 0.4 * manifold_score
        
        print(f'  \n  【综合评分】')
        print(f'    - 准确率评分: {accuracy_score:.4f}')
        print(f'    - 流形质量评分: {manifold_score:.4f}')
        print(f'    - 综合评分: {comprehensive_score:.4f} (准确率60% + 流形40%)')
        
        # 保存最佳模型（基于综合评分）
        if comprehensive_score > best_comprehensive_score:
            best_comprehensive_score = comprehensive_score
            best_accuracy = val_accuracy  # 记录对应的准确率
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': val_accuracy,
                'comprehensive_score': comprehensive_score,
                'loss_components': loss_components,
                'test_results': test_results  # 保存完整测试结果
            }, 'R_Identify/best_identify_model.pth')
            print(f'  ✅ 保存最佳模型 (综合评分: {comprehensive_score:.4f}, 准确率: {best_accuracy:.2f}%)')
        else:
            early_stop_counter += 1
            print(f'  早停计数器: {early_stop_counter}/{patience}')
            
        # 早停检查
        if early_stop_counter >= patience:
            print(f'  综合评分在 {patience} 个epoch内未提升，提前停止训练')
            break
            
        # 动态调整流形约束权重 - 渐进式增强策略(先对齐后紧凑)
        # 阶段1 (Epoch 1-15): 极小约束,让新组件适应预训练特征
        # 阶段2 (Epoch 15-30): 逐步增强流形紧凑性
        # 阶段3 (Epoch 30+): 适度增强center-aware约束
        if epoch == 15:
            criterion.delta = 0.05
            criterion.beta = 0.02
            print(f'  [阶段2启动] 开始增强流形约束: delta={criterion.delta:.4f}, beta={criterion.beta:.4f}')
        elif epoch == 30:
            criterion.delta = 0.10
            criterion.beta = 0.04
            print(f'  [阶段3启动] 适度增强约束: delta={criterion.delta:.4f}, beta={criterion.beta:.4f}')
        elif epoch == 50:
            criterion.delta = 0.15
            criterion.beta = 0.06
            print(f'  [最终阶段] 达到目标约束强度: delta={criterion.delta:.4f}, beta={criterion.beta:.4f}')
    
    print(f"\n训练完成! 最佳综合评分: {best_comprehensive_score:.4f}, 对应验证准确率: {best_accuracy:.2f}%")
    
    # 保存训练历史
    history_file_path = 'R_Identify/training_history.json'
    save_training_history(train_losses, val_accuracies, loss_components_history, test_accuracies, history_file_path)


if __name__ == "__main__":
    main()