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
        proj_source = outputs['proj_source']
        proj_target = outputs['proj_target']
        
        # 使用ManifoldLoss类计算损失（带打印）
        manifold_loss, loss_dict = loss_fn(proj_source, src_labels, proj_target, tgt_labels, 
                                           verbose=(batch_idx == 0))  # 只打印第一个batch
        
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
    计算流形投影质量指标 - 面向OpenMax入侵检测的核心指标
    
    精简后的3个核心指标:
    1. intra_class_distance: 类内紧凑度（越小越好） - OpenMax需要紧凑的正常类分布
    2. inter_class_distance: 类间分离度（越大越好） - 避免不同身份混淆
    3. silhouette_score: 流形整体质量（-1到1，越接近1越好） - 综合评估聚类质量
    """
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    model.eval()
    all_proj_features = []  # 32维投影特征
    all_labels = []
    
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
            
            all_proj_features.append(proj.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 合并所有batch
    all_proj_features = np.vstack(all_proj_features)  # [N, 32]
    all_labels = np.hstack(all_labels)  # [N]
    
    # 1. 类内平均距离（体现紧凑性，越小越好）
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
    
    return {
        'intra_class_distance': float(intra_class_distance),
        'inter_class_distance': float(inter_class_distance),
        'silhouette_score': float(silhouette)
    }


def test_model(model, data_loaders, device):
    """
    评估流形投影质量 - 专注于入侵检测特征优化（第四章核心）
    不再关注身份识别准确率（已由第三章预训练模型保证）
    """
    model.eval()
    test_results = {}
    
    # ===== 第四章核心指标：流形投影质量评估 =====
    print(f'  [流形质量]', end=' ')
    
    # 源域流形质量
    if 'src_identity_test' in data_loaders:
        src_manifold_metrics = compute_manifold_metrics(model, data_loaders['src_identity_test'], device, domain_type='source')
        test_results['src_manifold_metrics'] = src_manifold_metrics
        print(f'源域[紧凑:{src_manifold_metrics["intra_class_distance"]:.4f} 分离:{src_manifold_metrics["inter_class_distance"]:.4f} 轮廓:{src_manifold_metrics["silhouette_score"]:.3f}]', end=' ')
    
    # 目标域流形质量
    if 'tgt_identity_test' in data_loaders:
        tgt_manifold_metrics = compute_manifold_metrics(model, data_loaders['tgt_identity_test'], device, domain_type='target')
        test_results['tgt_manifold_metrics'] = tgt_manifold_metrics
        print(f'目标域[紧凑:{tgt_manifold_metrics["intra_class_distance"]:.4f} 分离:{tgt_manifold_metrics["inter_class_distance"]:.4f} 轮廓:{tgt_manifold_metrics["silhouette_score"]:.3f}]')
    
    return test_results


def save_training_history(train_losses, manifold_metrics_history, manifold_scores_history, 
                         detailed_loss_history, file_path):
    """
    保存训练历史数据到JSON文件（面向入侵检测的流形优化记录）
    """
    history = {
        'train_losses': train_losses,
        'manifold_metrics_history': manifold_metrics_history,
        'manifold_scores_history': manifold_scores_history,  # 新增综合评分历史
        'detailed_loss_history': detailed_loss_history,  # 新增详细损失历史
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
    
    # 初始化损失函数
    loss_fn = ManifoldLoss(intra_weight=1.0, inter_weight=0.3)
    print(f"  [损失函数]")
    print(f"    - 类内紧凑性权重: {loss_fn.intra_weight}")
    print(f"    - 类间分离性权重: {loss_fn.inter_weight}")
    
    # 只优化manifold_projection，其他部分冻结
    optimizer = optim.AdamW(
        model.manifold_projection.parameters(),  # 只优化流形投影
        lr=0.001,  # 降低学习率，更稳定的优化
        weight_decay=1e-3  # 增加权重衰减，防止过拟合
    )
    
    # 使用余弦退火调度器 + Warmup
    warmup_epochs = 5
    total_epochs = 100  # 增加训练轮数
    
    def warmup_cosine_schedule(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    
    print(f"  [训练策略 - 优化版]")
    print(f"    - 冻结预训练部分: feature_extractor + cross_attention + identity_classifier")
    print(f"    - 只训练: manifold_projection (初始lr=0.001, warmup 5轮)")
    print(f"    - 损失函数: 类内紧凑性 + 类间分离性（权重0.3）")
    print(f"    - 训练轮数: {total_epochs} (增加以获得更好的流形)")
    
    # 训练参数
    num_epochs = 100  # 增加训练轮数
    best_manifold_score = -float('inf')  # 最佳流形质量评分（不再使用早停）
    
    # 记录训练历史（专注流形质量）
    train_losses = []
    manifold_metrics_history = []
    manifold_scores_history = []  # 新增：记录综合评分
    detailed_loss_history = []  # 新增：记录详细损失

    # 确保保存模型的目录存在
    os.makedirs('R_Identify', exist_ok=True)
    
    print("开始训练循环...")
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        # 训练一个epoch
        train_loss = train_epoch(
            model, data_loaders['identity_train'], data_loaders['target_aux'], 
            optimizer, loss_fn, device, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        train_losses.append(train_loss)
        
        # 记录详细损失
        epoch_loss_history = loss_fn.get_loss_history()
        detailed_loss_history.append({
            'epoch': epoch,
            'loss_components': {
                'intra_loss_src': epoch_loss_history['intra_loss_src'][-1] if epoch_loss_history['intra_loss_src'] else 0,
                'intra_loss_tgt': epoch_loss_history['intra_loss_tgt'][-1] if epoch_loss_history['intra_loss_tgt'] else 0,
                'inter_loss_src': epoch_loss_history['inter_loss_src'][-1] if epoch_loss_history['inter_loss_src'] else 0,
                'inter_loss_tgt': epoch_loss_history['inter_loss_tgt'][-1] if epoch_loss_history['inter_loss_tgt'] else 0,
                'total_intra_loss': epoch_loss_history['total_intra_loss'][-1] if epoch_loss_history['total_intra_loss'] else 0,
                'total_inter_loss': epoch_loss_history['total_inter_loss'][-1] if epoch_loss_history['total_inter_loss'] else 0
            }
        })
        
        # 打印epoch结果 - 聚焦流形质量
        print(f'  损失: {train_loss:.6f} | 学习率: {scheduler.get_last_lr()[0]:.6f}')
        
        # 在每个epoch后测试流形质量
        test_results = test_model(model, data_loaders, device)
        
        # 提取流形指标
        if 'tgt_manifold_metrics' in test_results:
            tgt_metrics = test_results['tgt_manifold_metrics']
            manifold_metrics_history.append(tgt_metrics)
            
            # 直接使用原始指标值，不进行归一化处理
            intra_raw = tgt_metrics['intra_class_distance']  # 类内紧凑度（原始值）
            silhouette_raw = tgt_metrics['silhouette_score']  # 轮廓系数（原始值）
            inter_raw = tgt_metrics['inter_class_distance']  # 类间分离度（原始值）
            
            # 综合评分：直接加权求和原始值（不归一化）
            # 注意：类内距离越小越好，所以用负值；类间距离越大越好，轮廓系数越大越好
            manifold_score = (-intra_raw * 0.40 +     # 类内紧凑性（负值，因为越小越好）
                            silhouette_raw * 0.35 +   # 整体流形质量
                            inter_raw * 0.25)         # 类间分离
            
            # 打印原始指标值
            print(f'  [评分] 紧凑:{intra_raw:.4f}(40%) 轮廓:{silhouette_raw:.4f}(35%) 分离:{inter_raw:.4f}(25%) → 综合:{manifold_score:.4f} ★')
            
            # 保存最佳模型（基于流形质量，不使用早停）
            if manifold_score > best_manifold_score:
                best_manifold_score = manifold_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'manifold_score': manifold_score,
                    'test_results': test_results
                }, 'R_Identify/best_identify_model.pth')
                print(f'  ✅ 最佳模型 (评分:{manifold_score:.4f})')
            
            # 记录综合评分（原始值）
            manifold_scores_history.append({
                'epoch': epoch,
                'manifold_score': float(manifold_score),
                'raw_components': {
                    'intra_class_distance': float(intra_raw),
                    'silhouette_score': float(silhouette_raw),
                    'inter_class_distance': float(inter_raw)
                }
            })
    
    print(f"\n训练完成! 最佳流形评分: {best_manifold_score:.4f}")
    
    # 保存最终模型（第100轮）
    print(f"\n保存最终模型...")
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_manifold_score': manifold_score if 'tgt_manifold_metrics' in test_results else 0.0,
        'best_manifold_score': best_manifold_score,
        'test_results': test_results
    }, 'R_Identify/final_identify_model.pth')
    print(f"✅ 最终模型已保存: R_Identify/final_identify_model.pth")
    print(f"✅ 最佳模型已保存: R_Identify/best_identify_model.pth (评分: {best_manifold_score:.4f})")
    
    # 保存训练历史
    history_file_path = 'R_Identify/training_history.json'
    save_training_history(train_losses, manifold_metrics_history, manifold_scores_history, 
                         detailed_loss_history, history_file_path)


if __name__ == "__main__":
    main()