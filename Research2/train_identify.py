import torch
import os
import torch.optim as optim
import json
from model_identify import IdentifyDetectionSystem
from loss_identify import compute_manifold_loss, ManifoldLoss
from Research2.DataProcess.dataloader_identify import load_identify_data, create_data_loaders


def train_epoch(model, source_loader, target_loader, optimizer, loss_fn, device, epoch):
    """
    训练一个epoch - 纯流形投影优化版
    """
    model.train()
    
    # 冻结预训练部分
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    for param in model.cross_attention.parameters():
        param.requires_grad = False
    for param in model.identity_classifier.parameters():
        param.requires_grad = False
    
    total_loss = 0.0
    batch_count = 0
    
    # 使用zip循环处理源域和目标域数据
    for batch_idx, (source_batch, target_batch) in enumerate(zip(source_loader, target_loader)):
        src_data, src_labels = source_batch
        tgt_data, tgt_labels = target_batch

        src_data = src_data.to(device)
        tgt_data = tgt_data.to(device)
        src_labels = src_labels.to(device)
        tgt_labels = tgt_labels.to(device)

        optimizer.zero_grad()
        outputs = model(src_data, tgt_data)

        proj_source = outputs['proj_source']
        proj_target = outputs['proj_target']
        
        # 使用ManifoldLoss计算损失
        manifold_loss, loss_dict = loss_fn(proj_source, src_labels, proj_target, tgt_labels, 
                                           verbose=(batch_idx == 0 and epoch % 5 == 0))  # 每5个epoch打印第一个batch
        
        manifold_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.manifold_projection.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += manifold_loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return avg_loss


def compute_manifold_loss(proj_source, labels_source, proj_target, labels_target):
    """
    流形质量自监督损失 (优化版)
    目标：类内紧凑 + 类间分离
    """
    device = proj_source.device
    
    # 1. 源域类内紧凑性
    intra_loss_src = 0.0
    unique_labels = torch.unique(labels_source)
    for c in unique_labels:
        mask = (labels_source == c)
        if mask.sum() > 1:
            class_features = proj_source[mask]
            center = class_features.mean(dim=0, keepdim=True)
            intra_loss_src += ((class_features - center) ** 2).sum() / mask.sum()
    intra_loss_src = intra_loss_src / len(unique_labels) if len(unique_labels) > 0 else 0.0
    
    # 2. 目标域类内紧凑性
    intra_loss_tgt = 0.0
    unique_labels = torch.unique(labels_target)
    for c in unique_labels:
        mask = (labels_target == c)
        if mask.sum() > 1:
            class_features = proj_target[mask]
            center = class_features.mean(dim=0, keepdim=True)
            intra_loss_tgt += ((class_features - center) ** 2).sum() / mask.sum()
    intra_loss_tgt = intra_loss_tgt / len(unique_labels) if len(unique_labels) > 0 else 0.0
    
    # 3. 类间分离性 - 鼓励不同类的中心相互远离
    # 计算所有类中心
    centers_src = []
    unique_labels_src = torch.unique(labels_source)
    for c in unique_labels_src:
        mask = (labels_source == c)
        if mask.sum() > 0:
            centers_src.append(proj_source[mask].mean(dim=0))
    
    centers_tgt = []
    unique_labels_tgt = torch.unique(labels_target)
    for c in unique_labels_tgt:
        mask = (labels_target == c)
        if mask.sum() > 0:
            centers_tgt.append(proj_target[mask].mean(dim=0))
    
    # 类间分离损失：最大化类中心之间的距离
    inter_loss = 0.0
    if len(centers_src) > 1:
        centers_src_tensor = torch.stack(centers_src)
        distances = torch.cdist(centers_src_tensor, centers_src_tensor)
        mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=device)
        inter_distances = distances[mask]
        inter_loss += -inter_distances.mean()  # 负号：最小化负距离 = 最大化距离
    
    if len(centers_tgt) > 1:
        centers_tgt_tensor = torch.stack(centers_tgt)
        distances = torch.cdist(centers_tgt_tensor, centers_tgt_tensor)
        mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=device)
        inter_distances = distances[mask]
        inter_loss += -inter_distances.mean()
    
    num_domains = (len(centers_src) > 1) + (len(centers_tgt) > 1)
    inter_loss = inter_loss / num_domains if num_domains > 0 else 0.0
    
    # 总损失：类内紧凑性(1.0) + 类间分离性(0.3)
    intra_loss = (intra_loss_src + intra_loss_tgt) / 2.0
    loss = intra_loss + 0.3 * inter_loss
    
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=device)
    
    return loss


def compute_manifold_metrics(model, data_loader, device, domain_type='source'):
    """
    计算流形投影质量指标（简化版）
    只计算类内距离和类间距离，移除silhouette_score以提高速度
    """
    import numpy as np
    
    model.eval()
    all_proj_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            
            if domain_type == 'target':
                outputs = model(torch.zeros_like(data).to(device), data)
                proj = outputs['proj_target']
            else:
                outputs = model(data, torch.zeros_like(data).to(device))
                proj = outputs['proj_source']
            
            all_proj_features.append(proj.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_proj_features = np.vstack(all_proj_features)
    all_labels = np.hstack(all_labels)
    
    # 类内平均距离
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
    
    # 类间平均距离
    class_centers = np.array([all_proj_features[all_labels == c].mean(axis=0) 
                              for c in range(num_classes)])
    inter_distances = []
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            dist = np.linalg.norm(class_centers[i] - class_centers[j])
            inter_distances.append(dist)
    inter_class_distance = np.mean(inter_distances) if inter_distances else 0.0
    
    return {
        'intra_class_distance': float(intra_class_distance),
        'inter_class_distance': float(inter_class_distance)
    }


def test_model(model, data_loaders, device):
    """
    评估流形投影质量（简化版）
    """
    model.eval()
    test_results = {}
    
    # 只评估目标域流形质量
    if 'tgt_identity_test' in data_loaders:
        tgt_metrics = compute_manifold_metrics(model, data_loaders['tgt_identity_test'], device, domain_type='target')
        test_results['tgt_manifold_metrics'] = tgt_metrics
        print(f'  [流形] 类内:{tgt_metrics["intra_class_distance"]:.4f} 类间:{tgt_metrics["inter_class_distance"]:.4f}')
    
    return test_results


def save_training_history(train_losses, manifold_metrics_history, 
                         detailed_loss_history, file_path):
    """
    保存训练历史数据到JSON文件
    """
    history = {
        'train_losses': train_losses,
        'manifold_metrics_history': manifold_metrics_history,
        'detailed_loss_history': detailed_loss_history,
        'training_purpose': '32维流形投影优化，面向OpenMax入侵检测',
        'key_metrics': {
            'intra_class_distance': '类内紧凑度（越小越好）',
            'inter_class_distance': '类间分离度（越大越好）',
            'silhouette_score': '轮廓系数（越接近1越好）'
        },
        'loss_components': {
            'intra_loss_src': '源域类内紧凑性损失',
            'intra_loss_tgt': '目标域类内紧凑性损失',
            'inter_loss_src': '源域类间分离性损失',
            'inter_loss_tgt': '目标域类间分离性损失',
            'total_intra_loss': '总类内损失',
            'total_inter_loss': '总类间损失'
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
    
    # 初始化损失函数（简化版）
    loss_fn = ManifoldLoss(
        intra_weight=1.0,      # 类内紧凑性权重
        inter_weight=0.1,      # 类间分离性权重（降低，避免过度分离）
        margin=0.3,            # 类中心分离的margin（降低）
        variance_weight=0.1,   # 类内方差约束（降低）
        triplet_weight=0.2     # Triplet损失权重（降低）
    )
    print(f"  [损失函数 - 简化优化版]")
    print(f"    - 类内紧凑: {loss_fn.intra_weight}, 类间分离: {loss_fn.inter_weight}")
    print(f"    - Margin: {loss_fn.margin}, 方差: {loss_fn.variance_weight}, Triplet: {loss_fn.triplet_weight}")
    
    # 只优化manifold_projection
    optimizer = optim.Adam(
        model.manifold_projection.parameters(),
        lr=0.0005,  # 降低学习率
        weight_decay=1e-5
    )
    
    # 简化的学习率调度
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    print(f"  [训练策略]")
    print(f"    - 冻结: feature_extractor + cross_attention + identity_classifier")
    print(f"    - 训练: manifold_projection (lr=0.0005)")
    print(f"    - 学习率衰减: 每30轮减半")
    
    # 训练参数
    num_epochs = 50
    
    # 记录训练历史
    train_losses = []
    manifold_metrics_history = []
    detailed_loss_history = []

    # 确保保存模型的目录存在
    os.makedirs('R_Identify', exist_ok=True)
    
    print("开始训练...")
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        # 训练
        train_loss = train_epoch(
            model, data_loaders['identity_train'], data_loaders['target_aux'], 
            optimizer, loss_fn, device, epoch)
        
        scheduler.step()
        train_losses.append(train_loss)
        
        # 记录详细损失
        epoch_loss_history = loss_fn.get_loss_history()
        detailed_loss_history.append({
            'epoch': epoch,
            'total_loss': train_loss
        })
        
        # 打印结果
        print(f'  损失: {train_loss:.6f} | 学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 每5个epoch评估一次
        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_results = test_model(model, data_loaders, device)
            if 'tgt_manifold_metrics' in test_results:
                manifold_metrics_history.append(test_results['tgt_manifold_metrics'])
    
    print(f"\n训练完成!")
    
    # 最终评估
    print(f"\n最终评估...")
    test_results = test_model(model, data_loaders, device)
    
    # 保存模型
    print(f"\n保存模型...")
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'test_results': test_results
    }, 'R_Identify/best_identify_model.pth')
    print(f"✅ 模型已保存: R_Identify/best_identify_model.pth")
    
    # 保存训练历史
    history_file_path = 'R_Identify/training_history.json'
    save_training_history(train_losses, manifold_metrics_history, 
                         detailed_loss_history, history_file_path)


if __name__ == "__main__":
    main()