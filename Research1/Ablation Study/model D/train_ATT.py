import torch
import torch.optim as optim
import math
import json
import os
from model_ATT import CrossAttentionModel
from loss_ATT import LossFunction
from torch.utils.data import DataLoader
import sys
sys.path.append(r'c:/Users/USER/Desktop/liuheng/Research1')
from DataProcess.dataloder_GAN import CustomDataset, select_samples_by_label


def train_epoch(model, dataloader_source, dataloader_target, criterion, optimizer, scheduler=None):
    """
    执行一个完整的训练周期，对模型进行源域和目标域的联合训练。
    """

    model.train()
    total_loss = 0.0
    loss_components = {'source': 0.0, 'target': 0.0, 'mmd': 0.0}
    
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

        # 前向传播（不再使用F_c）
        pred_s, pred_t, F_s, F_t, _ = model(src_data, tgt_data)

        # 计算各项损失（删除一致性损失）
        ls = criterion.source_loss(pred_s, src_labels)
        lt = criterion.target_loss(pred_t, tgt_labels)
        lmmd = criterion.cross_feature_loss(F_s, F_t)

        # 加权组合损失
        loss = (criterion.alpha * ls + 
               criterion.beta * lt + 
               criterion.gamma * lmmd)
               
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
        loss_components['mmd'] += lmmd.item()

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


def save_best_model(model, path, src_accuracy, tgt_accuracy, epoch):
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
        'source_accuracy': src_accuracy,
        'target_accuracy': tgt_accuracy
    }, path)
    print(f"  保存最佳模型 (源域准确率: {src_accuracy:.2f}%, 目标域准确率: {tgt_accuracy:.2f}%)")


def save_training_results(src_accuracy, tgt_accuracy, save_dir='model_D'):
    """
    保存训练结果到JSON文件（只保存关键指标）
    """
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 构建结果字典
    results = {
        'source_accuracy': src_accuracy,
        'target_accuracy': tgt_accuracy
    }
    
    # 保存为JSON格式
    save_path = os.path.join(save_dir, 'attention_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"训练结果已保存到 {save_path}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据文件（Model D消融实验：使用原始目标域数据，不使用GAN增强数据）
    print("加载 R_CAL 数据文件（Model D: 无数据生成模块）...")
    source_data = torch.load('Data/source_env0_env1_data.pt')
    source_labels = torch.load('Data/source_env0_env1_labels.pt')
    
    # 关键修改：使用原始目标域数据，不使用GAN增强数据
    target_data_full = torch.load('Data/target_env2_data.pt')
    target_labels_full = torch.load('Data/target_env2_labels.pt')
    
    # 从目标域中均匀采样每个标签的20个样本
    print("从目标域采样每个标签20个样本（无数据增强）...")
    target_data, target_labels = select_samples_by_label(target_data_full, target_labels_full, samples_per_label=20)

    # 查看数据形状
    print(f"Source Raw data shape: {source_data.shape}")
    print(f"Source labels shape: {source_labels.shape}")
    print(f"Target Raw data shape (sampled): {target_data.shape}")
    print(f"Target labels shape (sampled): {target_labels.shape}")

    # 创建数据集数据加载器
    source_dataset = CustomDataset(source_data, source_labels)
    target_dataset = CustomDataset(target_data, target_labels)
    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)

    # 模型初始化
    model = CrossAttentionModel(num_classes=10).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== 模型参数统计 ===")
    print(f"总参数量: {total_params:,}")
    print(f"="*30 + "\n")

    # 训练超参数设置
    num_epochs = 150
    best_avg_accuracy = 0.0
    best_model_path = "model_D/best_attention_model.pth"
    patience = 15
    early_stop_counter = 0

    # 损失函数、优化器、学习率调度器配置
    criterion = LossFunction(
        num_classes=10,
        alpha=1.0,
        beta=1.5,
        gamma=0.5,
        label_smoothing=0.1,
        focal_gamma=2.0
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 使用Warmup + 余弦退火学习率策略
    warmup_epochs = 10
    total_steps = num_epochs * max(len(source_loader), len(target_loader))
    warmup_steps = warmup_epochs * max(len(source_loader), len(target_loader))
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_src_accuracy = 0.0
    best_tgt_accuracy = 0.0

    # 主训练循环
    print("开始训练循环...")
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        # 执行一个训练周期
        train_loss, loss_components = train_epoch(model, source_loader, target_loader, criterion, optimizer, scheduler)
        
        # 打印训练进度
        print(f'  训练损失: {train_loss:.4f} '
              f'(源域: {loss_components["source"]:.4f}, '
              f'目标: {loss_components["target"]:.4f}, '
              f'MMD: {loss_components["mmd"]:.4f})')
        
        # 测试模型
        print(f'  测试效果:')
        src_test_accuracy = validate_on_domain(model, source_loader, device, domain_type='source')
        tgt_test_accuracy = validate_on_domain(model, target_loader, device, domain_type='target')
        print(f'    源域测试准确率: {src_test_accuracy:.2f}%')
        print(f'    目标域测试准确率: {tgt_test_accuracy:.2f}%')
        
        # 计算源域和目标域测试准确率的平均值
        avg_test_accuracy = (src_test_accuracy + tgt_test_accuracy) / 2
        
        # 保存最佳模型（基于平均测试准确率）
        if avg_test_accuracy > best_avg_accuracy:
            best_avg_accuracy = avg_test_accuracy
            best_src_accuracy = src_test_accuracy
            best_tgt_accuracy = tgt_test_accuracy
            early_stop_counter = 0
            save_best_model(model, best_model_path, src_test_accuracy, tgt_test_accuracy, epoch)
        else:
            early_stop_counter += 1
            print(f'  早停计数器: {early_stop_counter}/{patience}')
            
        # 早停检查
        if early_stop_counter >= patience:
            print(f'  验证准确率在 {patience} 个epoch内未提升，提前停止训练')
            break
            
        # 动态调整损失权重
        if epoch > 30 and epoch % 10 == 0:
            criterion.beta = min(2.0, criterion.beta * 1.05)
            criterion.gamma = max(0.2, criterion.gamma * 0.95)
            print(f'  调整损失权重: beta={criterion.beta:.4f}, gamma={criterion.gamma:.4f}')

    print(f"\n训练完成! 最佳源域准确率: {best_src_accuracy:.2f}%, 最佳目标域准确率: {best_tgt_accuracy:.2f}%")
    
    # 保存训练结果
    save_training_results(best_src_accuracy, best_tgt_accuracy, save_dir='model_D')
