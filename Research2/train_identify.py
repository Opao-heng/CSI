import torch
import os
import torch.optim as optim
import json
from model_identify import IdentifyDetectionSystem
from loss_identify import IdentifyDetectionLoss
from Research2.DataProcess.dataloader_identify import load_identify_data, create_data_loaders


def train_epoch(model, source_loader, target_loader, optimizer, device, epoch):
    """
    训练一个epoch - 纯流形投影优化版
    第三章预训练已学会身份识别，第四章只需优化32维流形投影质量
    """
    model.train()
    
    # 冻结预训练部分（feature_extractor + cross_attention）
    # 只训练manifold_projection，加速收敛且避免破坏预训练权重
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    for param in model.cross_attention.parameters():
        param.requires_grad = False
    for param in model.identity_classifier.parameters():
        param.requires_grad = False  # 分类器也冻结，不需要了
    
    total_loss = 0.0
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

        # 计算流形质量损失（自监督方式）
        # 策略：让网络学习紧凑的流形，不依赖外部损失
        # 这里只做前向传播，让梯度通过软分配机制自然优化
        proj_source = outputs['proj_source']
        proj_target = outputs['proj_target']
        
        # 计算流形紧凑性（类内聚集）
        manifold_loss = compute_manifold_loss(proj_source, src_labels, proj_target, tgt_labels)
        
        # 反向传播（只更新manifold_projection）
        manifold_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.manifold_projection.parameters(), max_norm=1.0)
        
        optimizer.step()

        # 累计损失
        total_loss += manifold_loss.item()
        batch_count += 1

    # 计算平均损失
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0

    return avg_loss


def compute_manifold_loss(proj_source, labels_source, proj_target, labels_target):
    """
    流形质量自监督损失
    目标：类内紧凑 + 特征归一化
    """
    device = proj_source.device
    
    # 1. 类内紧凑性（源域）
    intra_loss_src = 0.0
    unique_labels = torch.unique(labels_source)
    for c in unique_labels:
        mask = (labels_source == c)
        if mask.sum() > 1:
            class_features = proj_source[mask]
            # 计算类内方差（越小越紧凑）
            center = class_features.mean(dim=0, keepdim=True)
            intra_loss_src += ((class_features - center) ** 2).sum() / mask.sum()
    intra_loss_src = intra_loss_src / len(unique_labels) if len(unique_labels) > 0 else 0.0
    
    # 2. 类内紧凑性（目标域）
    intra_loss_tgt = 0.0
    unique_labels = torch.unique(labels_target)
    for c in unique_labels:
        mask = (labels_target == c)
        if mask.sum() > 1:
            class_features = proj_target[mask]
            center = class_features.mean(dim=0, keepdim=True)
            intra_loss_tgt += ((class_features - center) ** 2).sum() / mask.sum()
    intra_loss_tgt = intra_loss_tgt / len(unique_labels) if len(unique_labels) > 0 else 0.0
    
    # 总损失：类内紧凑性
    loss = (intra_loss_src + intra_loss_tgt) / 2.0
    
    return loss if not torch.isnan(loss) else torch.tensor(0.0, device=device)


def compute_manifold_metrics(model, data_loader, device, domain_type='source'):
    """
    计算流形投影质量指标 - 面向 Train_intruder.py 入侵检测的特征优化评估
    
    核心指标设计思路:
    1. intra_class_distance: 类内紧凑度（越小越好） - OpenMax需要紧凑的正常类分布
    2. inter_class_distance: 类间分离度（越大越好） - 避免不同身份混淆
    3. silhouette_score: 流形整体质量（越接近1越好） - 评估聚类质量
    4. center_alignment: 类中心对齐度（越小越好） - Center-aware机制效果
    5. compactness_ratio: 类内密度比（新增，越大越好） - 类间距离/类内距离
    6. davies_bouldin_index: DB指数（新增，越小越好） - 聚类分离度
    7. calinski_harabasz_score: CH指数（新增，越大越好） - 聚类方差比
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    import numpy as np
    
    model.eval()
    all_proj_features = []  # 32维投影特征
    all_labels = []
    all_distances_to_center = []  # 到最近类中心的距离
    all_distances_to_assigned_center = []  # 到所属类中心的距离
    
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
            class_centers_normalized = torch.nn.functional.normalize(class_centers, p=2, dim=1)
            distances = torch.cdist(proj, class_centers_normalized)  # [batch, 10]
            min_distances, _ = torch.min(distances, dim=1)  # [batch]
            
            # 计算到所属类中心的距离
            assigned_distances = distances[torch.arange(len(labels)), labels]
            
            all_proj_features.append(proj.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_distances_to_center.append(min_distances.cpu().numpy())
            all_distances_to_assigned_center.append(assigned_distances.cpu().numpy())
    
    # 合并所有batch
    all_proj_features = np.vstack(all_proj_features)  # [N, 32]
    all_labels = np.hstack(all_labels)  # [N]
    all_distances_to_center = np.hstack(all_distances_to_center)  # [N]
    all_distances_to_assigned_center = np.hstack(all_distances_to_assigned_center)  # [N]
    
    # 1. 类内平均距离（体现紧凑性，越小越好）
    num_classes = len(np.unique(all_labels))
    intra_distances = []
    intra_stds = []  # 类内标准差
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 1:
            class_features = all_proj_features[mask]
            class_center = class_features.mean(axis=0)
            dists = np.linalg.norm(class_features - class_center, axis=1)
            intra_distances.append(dists.mean())
            intra_stds.append(dists.std())
    intra_class_distance = np.mean(intra_distances) if intra_distances else 0.0
    intra_class_std = np.mean(intra_stds) if intra_stds else 0.0
    
    # 2. 类间平均距离（体现可分离性，越大越好）
    class_centers_np = np.array([all_proj_features[all_labels == c].mean(axis=0) 
                                  for c in range(num_classes)])
    inter_distances = []
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            dist = np.linalg.norm(class_centers_np[i] - class_centers_np[j])
            inter_distances.append(dist)
    inter_class_distance = np.mean(inter_distances) if inter_distances else 0.0
    
    # 3. 轮廓系数（整体流形质量，-1到1，越接近1越好）
    if len(np.unique(all_labels)) > 1 and len(all_labels) > 1:
        silhouette = silhouette_score(all_proj_features, all_labels)
    else:
        silhouette = 0.0
    
    # 4. Center-aware对齐度（到最近类中心的平均距离，越小越好）
    center_alignment = all_distances_to_center.mean()
    
    # 5. 类内密度比（越大越好，体现紧凑度相对于分离度）
    compactness_ratio = (inter_class_distance / (intra_class_distance + 1e-8)) if intra_class_distance > 0 else 0.0
    
    # 6. Davies-Bouldin指数（越小越好，评估聚类分离度）
    if len(np.unique(all_labels)) > 1 and len(all_labels) > 1:
        db_index = davies_bouldin_score(all_proj_features, all_labels)
    else:
        db_index = 0.0
    
    # 7. Calinski-Harabasz指数（越大越好，评估聚类方差比）
    if len(np.unique(all_labels)) > 1 and len(all_labels) > 1:
        ch_score = calinski_harabasz_score(all_proj_features, all_labels)
    else:
        ch_score = 0.0
    
    # 8. 到所属类中心的平均距离（越小越好，体现Center-aware的直接效果）
    assigned_center_distance = all_distances_to_assigned_center.mean()
    
    return {
        'intra_class_distance': float(intra_class_distance),
        'intra_class_std': float(intra_class_std),
        'inter_class_distance': float(inter_class_distance),
        'silhouette_score': float(silhouette),
        'center_alignment': float(center_alignment),
        'compactness_ratio': float(compactness_ratio),
        'davies_bouldin_index': float(db_index),
        'calinski_harabasz_score': float(ch_score),
        'assigned_center_distance': float(assigned_center_distance)
    }


def test_model(model, data_loaders, device):
    """
    评估流形投影质量 - 专注于入侵检测特征优化（第四章核心）
    不再关注身份识别准确率（已由第三章预训练模型保证）
    """
    model.eval()
    test_results = {}
    
    # ===== 第四章核心指标：流形投影质量评估 =====
    print(f'  【流形投影质量评估 - 入侵检测特征优化】')
    
    # 源域流形质量
    if 'src_identity_test' in data_loaders:
        src_manifold_metrics = compute_manifold_metrics(model, data_loaders['src_identity_test'], device, domain_type='source')
        test_results['src_manifold_metrics'] = src_manifold_metrics
        print(f'    源域32维流形空间:')
        print(f'      - 类内紧凑度: {src_manifold_metrics["intra_class_distance"]:.4f} (↓越小越好，利于OpenMax建模)')
        print(f'      - 类间分离度: {src_manifold_metrics["inter_class_distance"]:.4f} (↑越大越好，减少误判)')
        print(f'      - 整体流形质量: {src_manifold_metrics["silhouette_score"]:.4f} (↑越接近1越好)')
        print(f'      - 类中心对齐: {src_manifold_metrics["center_alignment"]:.4f} (↓越小越好，Center-aware生效)')
        print(f'      - 类内密度比: {src_manifold_metrics.get("compactness_ratio", 0):.4f} (↑越大越紧凑)')
    
    # 目标域流形质量
    if 'tgt_identity_test' in data_loaders:
        tgt_manifold_metrics = compute_manifold_metrics(model, data_loaders['tgt_identity_test'], device, domain_type='target')
        test_results['tgt_manifold_metrics'] = tgt_manifold_metrics
        print(f'    目标域32维流形空间:')
        print(f'      - 类内紧凑度: {tgt_manifold_metrics["intra_class_distance"]:.4f} (↓越小越好，利于OpenMax建模)')
        print(f'      - 类间分离度: {tgt_manifold_metrics["inter_class_distance"]:.4f} (↑越大越好，减少误判)')
        print(f'      - 整体流形质量: {tgt_manifold_metrics["silhouette_score"]:.4f} (↑越接近1越好)')
        print(f'      - 类中心对齐: {tgt_manifold_metrics["center_alignment"]:.4f} (↓越小越好，Center-aware生效)')
        print(f'      - 类内密度比: {tgt_manifold_metrics.get("compactness_ratio", 0):.4f} (↑越大越紧凑)')
    
    return test_results


def save_training_history(train_losses, manifold_metrics_history, manifold_scores_history, file_path):
    """
    保存训练历史数据到JSON文件（面向入侵检测的流形优化记录）
    """
    history = {
        'train_losses': train_losses,
        'manifold_metrics_history': manifold_metrics_history,
        'manifold_scores_history': manifold_scores_history,  # 新增综合评分历史
        'training_purpose': '32维流形投影优化，面向OpenMax入侵检测',
        'key_metrics': {
            'intra_class_distance': '类内紧凑度（越小越好）',
            'inter_class_distance': '类间分离度（越大越好）',
            'silhouette_score': '轮廓系数（越接近1越好）',
            'center_alignment': 'Center对齐度（越小越好）',
            'compactness_ratio': '紧凑度比（越大越好）',
            'davies_bouldin_index': 'DB指数（越小越好）',
            'calinski_harabasz_score': 'CH指数（越大越好）'
        }
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
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

    # 第三章预训练已学会身份识别，第四章只需优化流形投影
    print("\n初始化优化器...")
    
    # 只优化manifold_projection，其他部分冻结
    optimizer = optim.AdamW(
        model.manifold_projection.parameters(),  # 只优化流形投影
        lr=0.003,  # 更高的学习率，因为只优化一个小模块
        weight_decay=1e-4
    )
    
    # 使用余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)
    print(f"  [训练策略]")
    print(f"    - 冻结预训练部分: feature_extractor + cross_attention + identity_classifier")
    print(f"    - 只训练: manifold_projection (lr=0.003)")
    print(f"    - 损失函数: 自监督流形紧凑性损失（类内方差最小化）")
    
    # 训练参数
    num_epochs = 60  # 减少轮数，流形投影收敛快
    best_manifold_score = 0.0  # 最佳流形质量评分
    early_stop_counter = 0
    patience = 12  # 降低耐心值
    
    # 记录训练历史（专注流形质量）
    train_losses = []
    manifold_metrics_history = []
    manifold_scores_history = []  # 新增：记录综合评分

    # 确保保存模型的目录存在
    os.makedirs('R_Identify', exist_ok=True)
    
    print("开始训练循环...")
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        # 训练一个epoch
        train_loss = train_epoch(
            model, data_loaders['identity_train'], data_loaders['target_aux'], 
            optimizer, device, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        train_losses.append(train_loss)
        
        # 打印epoch结果 - 聚焦流形质量
        print(f'  流形紧凑性损失: {train_loss:.4f}')
        print(f'  当前学习率: {scheduler.get_last_lr()[0]:.6f}')
        
        # 在每个epoch后测试流形质量
        print(f'  【流形投影质量评估】')
        test_results = test_model(model, data_loaders, device)
        
        # 提取流形指标
        if 'tgt_manifold_metrics' in test_results:
            tgt_metrics = test_results['tgt_manifold_metrics']
            manifold_metrics_history.append(tgt_metrics)
            
            # 流形质量评分（面向OpenMax入侵检测优化）
            # 1. 轮廓系数（-1到1）归一化到[0, 1]
            silhouette_normalized = (tgt_metrics['silhouette_score'] + 1) / 2.0
            
            # 2. Center对齐度（越小越好）归一化到[0, 1]
            center_normalized = max(0, 1 - tgt_metrics['center_alignment'] / 2.0)
            
            # 3. 类内紧凑度（越小越好）归一化到[0, 1]
            intra_normalized = max(0, 1 - tgt_metrics['intra_class_distance'] / 0.5)
            
            # 4. 类间分离度（越大越好）归一化到[0, 1]
            # 假设最大类间距离为4.0（根据32维L2归一化特征）
            inter_normalized = min(1.0, tgt_metrics['inter_class_distance'] / 4.0)
            
            # 5. 紧凑度比（越大越好）归一化到[0, 1]
            # 假设最大紧凑度比为10
            compactness_normalized = min(1.0, tgt_metrics['compactness_ratio'] / 10.0)
            
            # 6. DB指数（越小越好）归一化到[0, 1]
            # 假设最大DB指数为3.0
            db_normalized = max(0, 1 - tgt_metrics['davies_bouldin_index'] / 3.0)
            
            # 7. CH指数（越大越好）归一化到[0, 1]
            # 假设最大CH指数为1000
            ch_normalized = min(1.0, tgt_metrics['calinski_harabasz_score'] / 1000.0)
            
            # ===== 综合流形评分（面向OpenMax优化的加权方案）=====
            # 权重设计理念:
            # - 类内紧凑性最重要（30%）: OpenMax需要紧凑的正常类分布
            # - 轮廓系数次之（25%）: 整体流形质量
            # - 紧凑度比重要（20%）: 类间分离度/类内距离
            # - Center对齐（15%）: Center-aware机制效果
            # - 类间分离度（10%）: 避免误判
            manifold_score = (intra_normalized * 0.30 +       # 类内紧凑性
                            silhouette_normalized * 0.25 +    # 整体流形质量
                            compactness_normalized * 0.20 +   # 紧凑度比
                            center_normalized * 0.15 +        # Center对齐
                            inter_normalized * 0.10)          # 类间分离
            
            print(f'\n  【面向OpenMax的流形质量评分】')
            print(f'    - 类内紧凑性(30%): {intra_normalized:.4f} (原始={tgt_metrics["intra_class_distance"]:.4f})')
            print(f'    - 轮廓系数(25%): {silhouette_normalized:.4f} (原始={tgt_metrics["silhouette_score"]:.4f})')
            print(f'    - 紧凑度比(20%): {compactness_normalized:.4f} (原始={tgt_metrics["compactness_ratio"]:.4f})')
            print(f'    - Center对齐(15%): {center_normalized:.4f} (原始={tgt_metrics["center_alignment"]:.4f})')
            print(f'    - 类间分离(10%): {inter_normalized:.4f} (原始={tgt_metrics["inter_class_distance"]:.4f})')
            print(f'    - 综合流形评分: {manifold_score:.4f} ★核心指标★')
            
            # 保存最佳模型（基于流形质量）
            if manifold_score > best_manifold_score:
                best_manifold_score = manifold_score
                early_stop_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'manifold_score': manifold_score,
                    'test_results': test_results
                }, 'R_Identify/best_identify_model.pth')
                print(f'  ✅ 保存最佳模型 (流形评分: {manifold_score:.4f})')
            else:
                early_stop_counter += 1
                print(f'  早停计数器: {early_stop_counter}/{patience}')
            
            # 记录综合评分
            manifold_scores_history.append({
                'epoch': epoch,
                'manifold_score': float(manifold_score),
                'components': {
                    'intra_normalized': float(intra_normalized),
                    'silhouette_normalized': float(silhouette_normalized),
                    'compactness_normalized': float(compactness_normalized),
                    'center_normalized': float(center_normalized),
                    'inter_normalized': float(inter_normalized)
                }
            })
        else:
            early_stop_counter += 1
            
        # 早停检查
        if early_stop_counter >= patience:
            print(f'  流形评分在 {patience} 个epoch内未提升，提前停止训练')
            break
    
    print(f"\n训练完成! 最佳流形评分: {best_manifold_score:.4f}")
    
    # 保存训练历史
    history_file_path = 'R_Identify/training_history.json'
    save_training_history(train_losses, manifold_metrics_history, manifold_scores_history, history_file_path)


if __name__ == "__main__":
    main()