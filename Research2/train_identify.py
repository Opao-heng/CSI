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


def validate(model, val_loader, device):
    """
    在验证集上评估模型
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(val_loader):
            # 移动到设备
            data = data.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(data)
            logits = outputs['logits']

            # 计算准确率
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0.0
    return accuracy


def test_model(model, data_loaders, device):
    """
    在测试集上评估模型
    """
    model.eval()
    test_results = {}
    
    # 测试源域身份识别测试集
    if 'src_identity_test' in data_loaders:
        src_identity_test_accuracy = validate(model, data_loaders['src_identity_test'], device)
        test_results['src_identity_test'] = src_identity_test_accuracy
        print(f'    源域身份识别测试准确率: {src_identity_test_accuracy:.2f}%')
    
    # 测试目标域身份识别测试集
    if 'tgt_identity_test' in data_loaders:
        tgt_identity_test_accuracy = validate(model, data_loaders['tgt_identity_test'], device)
        test_results['tgt_identity_test'] = tgt_identity_test_accuracy
        print(f'    目标域身份识别测试准确率: {tgt_identity_test_accuracy:.2f}%')
    
    return test_results


def save_training_history(train_losses, val_accuracies, loss_components_history, test_accuracies, file_path):
    """
    保存训练历史数据到JSON文件
    """
    history = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'loss_components_history': loss_components_history,
        'test_accuracies': test_accuracies,
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
    pretrained_path = os.path.join('Research1', 'R_CAL', 'best_attention_model.pth')
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
    print("\n初始化损失函数和优化器...")
    criterion = IdentifyDetectionLoss(alpha=1.0, delta=0.15, beta=0.08)
    
    # 初始化优化器 - 使用较小学习率微调预训练的特征提取器
    # 对新增的manifold_projection和identity_classifier使用正常学习率
    optimizer = optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': 0.0001},  # 预训练部分用小学习率微调
        {'params': model.cross_attention.parameters(), 'lr': 0.0001},    # 预训练部分用小学习率微调
        {'params': model.manifold_projection.parameters(), 'lr': 0.0005}, # 新增部分用大学习率
        {'params': model.identity_classifier.parameters(), 'lr': 0.0005}  # 新增部分用大学习率
    ], weight_decay=1e-4)
    
    # 使用余弦退火调度器,更平滑的学习率衰减
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    print(f"  feature_extractor & cross_attention 学习率: 0.0001 (微调预训练权重)")
    print(f"  manifold_projection & identity_classifier 学习率: 0.0005 (从头训练)")
    
    # 训练参数 - 基于预训练权重,可能收敛更快
    num_epochs = 80  # 使用预训练权重,减少训练轮数
    best_accuracy = 0.0
    early_stop_counter = 0
    patience = 15  # 预训练模型收敛快,适度减少patience
    
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
        
        # 验证模型（使用身份识别验证集）
        val_accuracy = validate(model, data_loaders['identity_validation'], device)
        
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
        
        # 在每个epoch后测试模型
        print(f'  测试效果:')
        test_results = test_model(model, data_loaders, device)
        test_accuracies.append(test_results)
        
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': val_accuracy,
                'loss_components': loss_components
            }, 'R_Identify/best_identify_model.pth')
            print(f'  保存最佳模型 (准确率: {best_accuracy:.2f}%)')
        else:
            early_stop_counter += 1
            print(f'  早停计数器: {early_stop_counter}/{patience}')
            
        # 早停检查
        if early_stop_counter >= patience:
            print(f'  验证准确率在 {patience} 个epoch内未提升，提前停止训练')
            break
            
        # 动态调整流形约束权重 - 逐渐增强Center-aware约束
        if epoch == 20:
            criterion.delta = 0.2
            criterion.beta = 0.12
            print(f'  [Epoch 20] 增强流形约束: delta={criterion.delta:.4f}, beta={criterion.beta:.4f}')
        elif epoch == 40:
            criterion.delta = 0.25
            criterion.beta = 0.15
            print(f'  [Epoch 40] 继续增强流形约束: delta={criterion.delta:.4f}, beta={criterion.beta:.4f}')
    
    print(f"\n训练完成! 最佳验证准确率: {best_accuracy:.2f}%")
    
    # 保存训练历史
    history_file_path = 'R_Identify/training_history.json'
    save_training_history(train_losses, val_accuracies, loss_components_history, test_accuracies, history_file_path)


if __name__ == "__main__":
    main()