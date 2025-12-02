import torch
import torch.optim as optim
from model_ATT import CrossAttentionModel
from loss_ATT import LossFunction
from torch.utils.data import DataLoader
from Research1.Process.dataloder_ATT import CustomDataset
import os


def train_epoch(model, dataloader_source, dataloader_target, criterion, optimizer, scheduler=None):
    """
    执行一个完整的训练周期，对模型进行源域和目标域的联合训练。
    """

    model.train()
    total_loss = 0.0
    loss_components = {'source': 0.0, 'target': 0.0, 'cross_feature': 0.0, 'consistency': 0.0}
    
    # 初始化数据加载器迭代器
    max_batches = max(len(dataloader_source), len(dataloader_target))
    source_iter = iter(dataloader_source)
    target_iter = iter(dataloader_target)
    
    for batch_idx in range(max_batches):
        # 获取源域数据（支持循环迭代）
        try:
            src_data, src_labels = next(source_iter)
        except StopIteration:
            source_iter = iter(dataloader_source)
            src_data, src_labels = next(source_iter)
            
        # 获取目标域数据（支持循环迭代）
        try:
            tgt_data, tgt_labels = next(target_iter)
        except StopIteration:
            target_iter = iter(dataloader_target)
            tgt_data, tgt_labels = next(target_iter)
            
        # 处理批次大小不一致，取较小的大小
        min_batch_size = min(src_data.size(0), tgt_data.size(0))
        src_data, src_labels = src_data[:min_batch_size], src_labels[:min_batch_size]
        tgt_data, tgt_labels = tgt_data[:min_batch_size], tgt_labels[:min_batch_size]
        
        # 将数据移到指定设备
        src_data, src_labels = src_data.to(device), src_labels.to(device)
        tgt_data, tgt_labels = tgt_data.to(device), tgt_labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        pred_s, pred_t, F_s, F_t, F_c = model(src_data, tgt_data)

        # 计算各项损失
        ls = criterion.source_loss(pred_s, src_labels)
        lt = criterion.target_loss(pred_t, tgt_labels)
        lsf = criterion.cross_feature_loss(F_s, F_t)
        lc = criterion.consistency_loss(F_s, F_c)

        # 加权组合损失
        loss = (criterion.alpha * ls + 
               criterion.beta * lt + 
               criterion.gamma * lsf + 
               criterion.delta * lc)
               
        # 反向传播
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        
        # 学习率调度更新
        if scheduler:
            scheduler.step()

        # 损失累积
        total_loss += loss.item()
        loss_components['source'] += ls.item()
        loss_components['target'] += lt.item()
        loss_components['cross_feature'] += lsf.item()
        loss_components['consistency'] += lc.item()

    # 计算平均损失
    avg_loss = total_loss / max_batches
    for key in loss_components:
        loss_components[key] = loss_components[key] / max_batches
        
    return avg_loss, loss_components


def validate_on_domain(model, dataloader, device, domain_type='target'):
    """
    在指定域的数据集上评估模型性能，domain_type: 'source' 或 'target'，计算损失和准确率。
    """

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            
            # 根据域类型选择不同的前向传播方式
            if domain_type == 'target':
                # 目标域：源域输入为零张量
                _, pred, _, _, _ = model(torch.zeros_like(data).to(device), data)
            else:
                # 源域：目标域输入为零张量
                pred, _, _, _, _ = model(data, torch.zeros_like(data).to(device))
            
            # 计算准确率
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0.0
    return accuracy


def test_model(model, dataloader_source, dataloader_target, criterion):
    """
    在测试集上评估模型
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for tgt_batch in dataloader_target:
            tgt_data, tgt_labels = tgt_batch
            tgt_data, tgt_labels = tgt_data.to(device), tgt_labels.to(device)

            # 模型前向传播（源域输入为零张量）
            _, pred_t, _, _, _ = model(torch.zeros_like(tgt_data).to(device), tgt_data)

            # 计算目标域损失
            loss = criterion.target_loss(pred_t, tgt_labels)
            total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(pred_t, 1)
            total += tgt_labels.size(0)
            correct += (predicted == tgt_labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader_target)

    return avg_loss, accuracy


def save_best_model(model, optimizer, scheduler, path, accuracy, epoch, loss_components):
    """
    保存最佳模型权重和训练信息到指定路径。
    """
    # 创建保存路径目录（若不存在）
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    # 保存模型状态字典和元信息
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': accuracy,
        'loss_components': loss_components
    }, path)
    print(f"  保存最佳模型 (准确率: {accuracy:.2f}%)")


def save_training_history(train_losses, val_accuracies, loss_components_history, test_results_history, save_dir='Attention'):
    """
    保存训练历史数据到JSON文件
    """
    import json
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 构建历史数据字典
    history = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'loss_components_history': loss_components_history,
        'test_results_history': test_results_history,
        'train_accuracies': [100 - loss * 10 for loss in train_losses]  # 简单估算
    }
    
    # 保存为JSON格式
    save_path = os.path.join(save_dir, 'training_history.json')
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"训练历史已保存到 {save_path}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据文件
    print("加载 Attention 数据文件...")
    source_data = torch.load('Data/source_env0_env1_data.pt')
    source_labels = torch.load('Data/source_env0_env1_labels.pt')
    target_data = torch.load('Data/target_env2_gan_data.pt')
    target_labels = torch.load('Data/target_env2_gan_labels.pt')

    # 查看数据形状
    print(f"Source data shape: {source_data.shape}")
    print(f"Source labels shape: {source_labels.shape}")
    print(f"Target data shape: {target_data.shape}")
    print(f"Target labels shape: {target_labels.shape}")

    # 创建数据集数据加载器
    source_dataset = CustomDataset(source_data, source_labels)
    target_dataset = CustomDataset(target_data, target_labels)
    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)

    # 模型初始化
    model = CrossAttentionModel(num_classes=10).to(device)

    # 损失函数、优化器、学习率调度器配置
    criterion = LossFunction(
        num_classes=10,
        alpha=1.0,      # 源域分类损失权重
        beta=1.0,       # 目标域分类损失权重
        gamma=0.3,      # 跨域特征对齐损失权重
        delta=0.2       # 一致性损失权重
    )
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # 训练超参数设置
    num_epochs = 10
    best_accuracy = 0.0
    best_model_path = "Attention/best_attention_model.pth"
    patience = 15  # 早停耐心值
    early_stop_counter = 0

    train_losses = []  # 训练损失记录
    val_accuracies = []  # 验证准确率记录
    loss_components_history = []  # 损失组件历史
    test_results_history = []  # 测试结果历史

    # 主训练循环
    print("开始训练循环...")
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        # 执行一个训练周期
        train_loss, loss_components = train_epoch(model, source_loader, target_loader, criterion, optimizer, scheduler)
        
        # 验证模型（使用目标域数据）
        val_accuracy = validate_on_domain(model, target_loader, device, domain_type='target')
        
        # 记录历史数据
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        loss_components_history.append(loss_components)
        
        # 打印训练进度
        print(f'  训练损失: {train_loss:.4f} '
              f'(源域: {loss_components["source"]:.4f}, '
              f'目标: {loss_components["target"]:.4f}, '
              f'跨域: {loss_components["cross_feature"]:.4f}, '
              f'一致: {loss_components["consistency"]:.4f})')
        print(f'  验证准确率: {val_accuracy:.2f}%')
        print(f'  当前学习率: {scheduler.get_last_lr()[0]:.6f}')
        
        # 在每个epoch后测试模型
        print(f'  测试效果:')
        src_test_accuracy = validate_on_domain(model, source_loader, device, domain_type='source')
        tgt_test_accuracy = validate_on_domain(model, target_loader, device, domain_type='target')
        test_results = {
            'src_test_accuracy': src_test_accuracy,
            'tgt_test_accuracy': tgt_test_accuracy
        }
        test_results_history.append(test_results)
        print(f'    源域测试准确率: {src_test_accuracy:.2f}%')
        print(f'    目标域测试准确率: {tgt_test_accuracy:.2f}%')
        
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            early_stop_counter = 0
            save_best_model(model, optimizer, scheduler, best_model_path, val_accuracy, epoch, loss_components)
        else:
            early_stop_counter += 1
            print(f'  早停计数器: {early_stop_counter}/{patience}')
            
        # 早停检查：若无改进，提前终止训练
        if early_stop_counter >= patience:
            print(f'  验证准确率在 {patience} 个epoch内未提升，提前停止训练')
            break
            
        # 动态调整损失权重：随训练进展逐步减少领域适应的重要性
        if epoch > 20:
            criterion.gamma = float(max(0.1, criterion.gamma * 0.98))
            criterion.delta = float(max(0.05, criterion.delta * 0.98))
            print(f'  调整损失权重: gamma={criterion.gamma:.4f}, delta={criterion.delta:.4f}')

    print(f"\n训练完成! 最佳验证准确率: {best_accuracy:.2f}%")
    
    # 保存训练历史
    save_training_history(train_losses, val_accuracies, loss_components_history, test_results_history, save_dir='Attention')
    
    # 训练结果可视化
    from plot_ATT import plot_all_training_results
    plot_all_training_results('Attention/training_history.json', save_dir='Attention')
    
    # 保存最终模型
    final_model_path = 'Attention/final_attention_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_accuracy': best_accuracy,
    }, final_model_path)
    print(f"最终模型已保存到 {final_model_path}")
