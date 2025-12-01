import torch
import torch.optim as optim
from model_ATT import CrossAttentionModel
from loss_ATT import LossFunction
from torch.utils.data import DataLoader
from Research1.Process.dataloder_ATT import CustomDataset
from plot_ATT import plot_training_curves, save_training_history
import os

"""
执行一个完整的训练周期，对模型进行源域和目标域的联合训练。

参数:
    model (CrossAttentionModel): 跨域注意力模型
    dataloader_source (DataLoader): 源域数据加载器
    dataloader_target (DataLoader): 目标域数据加载器
    criterion (LossFunction): 损失函数对象
    optimizer (torch.optim.Optimizer): 优化器
    scheduler (torch.optim.lr_scheduler, optional): 学习率调度器，默认为None

返回:
    tuple: (avg_loss, loss_components)
        - avg_loss (float): 平均总损失
        - loss_components (dict): 各类损失组成部分的字典
"""
def train_epoch(model, dataloader_source, dataloader_target, criterion, optimizer, scheduler=None):
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


"""
在目标域数据集上评估模型性能，计算损失和准确率。

参数:
    model (CrossAttentionModel): 跨域注意力模型
    dataloader_source (DataLoader): 源域数据加载器（此函数中未使用）
    dataloader_target (DataLoader): 目标域数据加载器
    criterion (LossFunction): 损失函数对象

返回:
    tuple: (avg_loss, accuracy)
        - avg_loss (float): 目标域平均损失
        - accuracy (float): 目标域准确率 (0-1之间)
"""
def test_model(model, dataloader_source, dataloader_target, criterion):
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
            pred_t_prob = torch.softmax(pred_t, dim=1)
            _, predicted = torch.max(pred_t_prob, 1)
            total += tgt_labels.size(0)
            correct += (predicted == tgt_labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader_target)

    return avg_loss, accuracy


"""
保存最佳模型权重和训练信息到指定路径。

参数:
    model (torch.nn.Module): 待保存的模型
    path (str): 模型保存路径
    accuracy (float): 当前模型的准确率
    epoch (int): 当前训练轮数

返回:
    无返回值
"""
def save_best_model(model, path, accuracy, epoch):
    # 创建保存路径目录（若不存在）
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    # 保存模型状态字典和元信息
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
    }, path)
    print(f"Best model saved at {path} with accuracy: {accuracy:.4f}")


"""
主训练函数，完整的模型初始化、训练、验证和评估流程。

工作流程:
    1. 加载源域和目标域数据
    2. 模型初始化和结构配置
    3. 损失函数、优化器、学习率调度器配置
    4. 数据加载器创建
    5. 训练循环：前向传播 -> 损失计算 -> 反向传播 -> 参数更新
    6. 性能评估和最佳模型保存
    7. 训练曲线可视化和数据保存
"""
if __name__ == "__main__":

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
    test_losses = []  # 测试损失记录
    test_accuracies = []  # 测试准确率记录

    # 主训练循环
    for epoch in range(num_epochs):
        # 执行一个训练周期
        train_loss, loss_components = train_epoch(model, source_loader, target_loader, criterion, optimizer, scheduler)
        # 在目标域上评估模型
        test_loss, accuracy = test_model(model, source_loader, target_loader, criterion)

        # 记录历史数据
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        # 输出训练进度
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} (S:{loss_components['source']:.3f}, T:{loss_components['target']:.3f}, SF:{loss_components['cross_feature']:.3f}, C:{loss_components['consistency']:.3f})")
        print(f"  Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            early_stop_counter = 0
            save_best_model(model, best_model_path, accuracy, epoch)
        else:
            early_stop_counter += 1
            
        # 早停检查：若无改进，提前终止训练
        if early_stop_counter >= patience:
            print(f"早停在第 {epoch+1} 轮，最佳准确率: {best_accuracy:.4f}")
            break
            
        # 动态调整损失权重：随训练进展逐步减少领域适应的重要性
        if epoch > 20:
            criterion.gamma = float(max(0.1, criterion.gamma * 0.98))
            criterion.delta = float(max(0.05, criterion.delta * 0.98))

    # 训练结果可视化与数据保存
    plot_training_curves(train_losses, test_losses, test_accuracies, optimizer, save_dir='Attention')
    save_training_history(train_losses, test_losses, test_accuracies, best_accuracy, epoch + 1, save_dir='Attention')

    print(f"训练完成，最终训练损失: {train_loss:.4f}")
    print(f"训练完成，最佳准确率: {best_accuracy:.4f}")
