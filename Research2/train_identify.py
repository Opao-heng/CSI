import torch
import os
import torch.optim as optim
import json
from model_identify import IdentifyDetectionSystem
from loss_identify import ManifoldLoss
from DataProcess.dataloader_identify import load_identify_data, create_data_loaders


def train_epoch(model, source_loader, target_loader, optimizer, loss_fn, device):
    """
    训练一个epoch - 流形投影优化
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
    loss_details = {'intra_src': 0.0, 'intra_tgt': 0.0, 'inter_src': 0.0, 'inter_tgt': 0.0}
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
        
        # 计算损失
        manifold_loss, loss_dict = loss_fn(proj_source, src_labels, proj_target, tgt_labels)
        
        manifold_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.manifold_projection.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += manifold_loss.item()
        for key in loss_details:
            loss_details[key] += loss_dict[key]
        batch_count += 1

    # 计算平均损失
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    for key in loss_details:
        loss_details[key] /= batch_count if batch_count > 0 else 1.0
    
    return avg_loss, loss_details


def compute_manifold_metrics(model, data_loader, device, domain_type='source'):
    """
    计算流形投影质量指标：类内距离和类间距离
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
    
    # 类内平均距离（越小越好）
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
    
    # 类间平均距离（越大越好）
    class_centers = np.array([all_proj_features[all_labels == c].mean(axis=0) 
                              for c in range(num_classes)])
    inter_distances = []
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            dist = np.linalg.norm(class_centers[i] - class_centers[j])
            inter_distances.append(dist)
    inter_class_distance = np.mean(inter_distances) if inter_distances else 0.0
    
    return {
        'intra_distance': float(intra_class_distance),
        'inter_distance': float(inter_class_distance),
        'separation_ratio': float(inter_class_distance / (intra_class_distance + 1e-8))
    }


def test_model(model, data_loaders, device):
    """
    评估流形投影质量
    """
    model.eval()
    
    # 评估目标域流形质量
    if 'tgt_identity_test' in data_loaders:
        tgt_metrics = compute_manifold_metrics(model, data_loaders['tgt_identity_test'], device, domain_type='target')
        return tgt_metrics
    
    return {}


def save_training_history(history, file_path):
    """
    保存训练历史数据到JSON文件
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"✅ 训练历史已保存: {file_path}")


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据集
    print("\n" + "="*60)
    print("加载数据集...")
    datasets = load_identify_data()
    data_loaders = create_data_loaders(datasets, batch_size=8)
    print(f"✅ 数据集加载完成")

    # 初始化模型
    print("\n" + "="*60)
    print("初始化模型...")
    model = IdentifyDetectionSystem(num_classes=10, feature_dim=512, projection_dim=32).to(device)
    
    # 加载预训练权重
    pretrained_path = os.path.join('C:\\Users\\USER\\Desktop\\liuheng\\Research1\\R_CAL', 'best_attention_model.pth')
    if os.path.exists(pretrained_path):
        print(f"加载预训练权重: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            pretrained_dict = checkpoint['model_state_dict']
            
            # 只加载特征提取器和交叉注意力的权重
            model_dict = model.state_dict()
            pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() 
                                       if k.startswith('feature_extractor') or k.startswith('cross_attention')}
            
            model_dict.update(pretrained_dict_filtered)
            model.load_state_dict(model_dict)
            
            print(f"✅ 成功加载 {len(pretrained_dict_filtered)} 个预训练参数")
        except Exception as e:
            print(f"⚠️  加载预训练权重失败: {e}")
    else:
        print(f"⚠️  未找到预训练权重")

    # 初始化损失函数 - 优化权重配比
    print("\n" + "="*60)
    print("初始化损失函数...")
    loss_fn = ManifoldLoss(intra_weight=1.0, inter_weight=0.5)  # 提升类间权重
    print(f"损失函数配置: 类内权重={loss_fn.intra_weight}, 类间权重={loss_fn.inter_weight}")
    
    # 只优化流形投影层
    optimizer = optim.Adam(
        model.manifold_projection.parameters(),
        lr=0.002,  # 提升初始学习率
        weight_decay=1e-4  # 增强正则化
    )
    # 使用余弦退火调度器，更平滑的学习率衰减
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-5
    )
    
    print(f"优化器配置: lr=0.002 (余弦退火), weight_decay=1e-4")
    print(f"冻结层: feature_extractor, cross_attention, identity_classifier")
    print(f"训练层: manifold_projection")
    
    # 训练参数
    num_epochs = 80  # 增加训练轮数，利用余弦退火的重启机制
    print(f"训练轮数: {num_epochs}")
    
    # 训练历史记录
    history = {
        'config': {
            'num_epochs': num_epochs,
            'batch_size': 8,
            'learning_rate': 0.002,
            'projection_dim': 32,
            'loss_weights': {'intra': 1.0, 'inter': 0.5},
            'scheduler': 'CosineAnnealingWarmRestarts',
            'weight_decay': 1e-4
        },
        'epochs': []
    }

    # 确保保存目录存在
    os.makedirs('R_Identify', exist_ok=True)
    
    # 训练循环
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)
    
    best_ratio = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        # 训练
        train_loss, loss_details = train_epoch(
            model, data_loaders['identity_train'], data_loaders['target_aux'], 
            optimizer, loss_fn, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印训练损失
        print(f"[训练] 总损失: {train_loss:.6f} | 学习率: {current_lr:.6f}")
        print(f"  └─ 类内损失: 源域={loss_details['intra_src']:.4f}, 目标域={loss_details['intra_tgt']:.4f}")
        print(f"  └─ 类间损失: 源域={loss_details['inter_src']:.4f}, 目标域={loss_details['inter_tgt']:.4f}")
        
        # 记录epoch数据
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'loss_details': loss_details,
            'learning_rate': current_lr
        }
        
        # 每5个epoch评估一次
        if (epoch + 1) % 5 == 0 or epoch == 0:
            metrics = test_model(model, data_loaders, device)
            if metrics:
                print(f"[评估] 类内距离: {metrics['intra_distance']:.4f} (↓越小越好)")
                print(f"  └─ 类间距离: {metrics['inter_distance']:.4f} (↑越大越好)")
                print(f"  └─ 分离比率: {metrics['separation_ratio']:.4f} (↑越大越好)")
                
                epoch_data['metrics'] = metrics
                
                # 保存最佳模型 - 综合评估类内距离和分离比率
                combined_score = metrics['separation_ratio'] - 0.5 * metrics['intra_distance']
                if epoch == 0 or combined_score > history.get('best_combined_score', 0):
                    history['best_combined_score'] = combined_score
                    best_ratio = metrics['separation_ratio']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': metrics,
                        'train_loss': train_loss,
                        'combined_score': combined_score,
                        'loss_config': {
                            'intra_weight': loss_fn.intra_weight,
                            'inter_weight': loss_fn.inter_weight
                        },
                        'training_config': history['config']
                    }, 'R_Identify/best_identify_model.pth')
                    print(f"  💾 保存最佳模型 (分离比率: {best_ratio:.4f}, 综合分数: {combined_score:.4f})")
        
        history['epochs'].append(epoch_data)
    
    print("\n" + "="*60)
    print(f"训练完成!")
    print(f"\n最佳模型信息:")
    print(f"  保存路径: R_Identify/best_identify_model.pth")
    print(f"  最佳分离比率: {best_ratio:.4f}")
    
    # 保存训练历史
    history_file_path = 'R_Identify/training_history.json'
    save_training_history(history, history_file_path)
    
    print("\n" + "="*60)
    print("所有文件已保存:")
    print("  ✅ R_Identify/best_identify_model.pth")
    print("  ✅ R_Identify/training_history.json")
    print("="*60)


if __name__ == "__main__":
    main()