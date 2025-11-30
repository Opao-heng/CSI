import torch
import torch.optim as optim
from model_ATT import CrossAttentionModel
from loss_ATT import LossFunction
from Research1.dataloder_ATT import source_loader, target_loader
import matplotlib.pyplot as plt
import os

"""
设备配置：根据系统可用资源自动选择GPU或CPU
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

"""
全局变量初始化：用于记录训练过程中的损失和准确率数据
"""
train_losses = []  # 训练损失记录
test_losses = []   # 测试损失记录
test_accuracies = []  # 测试准确率记录

"""
函数: train_epoch

功能: 执行一个完整的训练周期，对模型进行源域和目标域的联合训练

参数:
  model: CrossAttentionModel - 跨域注意力模型
  dataloader_source: DataLoader - 源域数据加载器
  dataloader_target: DataLoader - 目标域数据加载器
  criterion: LossFunction - 损失函数对象
  optimizer: torch.optim.Optimizer - 优化器
  scheduler: torch.optim.lr_scheduler (可选) - 学习率调度器，默认为None

返回值:
  avg_loss: float - 平均总损失
  loss_components: dict - 各类损失组成部分的字典
"""
def train_epoch(model, dataloader_source, dataloader_target, criterion, optimizer, scheduler=None):
    # 设置模型为训练模式
    model.train()
    total_loss = 0.0
    loss_components = {'source': 0.0, 'target': 0.0, 'cross_feature': 0.0, 'consistency': 0.0}
    
    # 步骤1: 初始化数据加载器迭代器
    max_batches = max(len(dataloader_source), len(dataloader_target))
    source_iter = iter(dataloader_source)
    target_iter = iter(dataloader_target)
    
    # 步骤2: 遍历所有批次
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
            
        # 处理批次大小不一致：取两个批次中较小的大小
        min_batch_size = min(src_data.size(0), tgt_data.size(0))
        src_data, src_labels = src_data[:min_batch_size], src_labels[:min_batch_size]
        tgt_data, tgt_labels = tgt_data[:min_batch_size], tgt_labels[:min_batch_size]
        
        # 将数据移到指定设备
        src_data, src_labels = src_data.to(device), src_labels.to(device)
        tgt_data, tgt_labels = tgt_data.to(device), tgt_labels.to(device)

        # 步骤3: 梯度清零
        optimizer.zero_grad()

        # 步骤4: 前向传播（包含错误处理）
        try:
            pred_s, pred_t, F_s, F_t, F_c = model(src_data, tgt_data)
        except Exception as e:
            print(f"\n模型前向传播错误 (batch {batch_idx}):")
            print(f"  源域数据形状: {src_data.shape}")
            print(f"  目标域数据形状: {tgt_data.shape}")
            print(f"  错误信息: {e}")
            raise e

        # 步骤5: 计算各项损失
        ls = criterion.source_loss(pred_s, src_labels)  # 源域分类损失
        lt = criterion.target_loss(pred_t, tgt_labels)  # 目标域分类损失
        lsf = criterion.cross_feature_loss(F_s, F_t)   # 跨域特征对齐损失
        lc = criterion.consistency_loss(F_s, F_c)      # 一致性损失

        # 步骤6: 加权组合损失
        loss = (criterion.alpha * ls + 
               criterion.beta * lt + 
               criterion.gamma * lsf + 
               criterion.delta * lc)
               
        # 步骤7: 反向传播
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 步骤8: 参数更新
        optimizer.step()
        
        # 学习率调度更新
        if scheduler:
            scheduler.step()

        # 步骤9: 损失累积
        total_loss += loss.item()
        loss_components['source'] += ls.item()
        loss_components['target'] += lt.item()
        loss_components['cross_feature'] += lsf.item()
        loss_components['consistency'] += lc.item()

    # 步骤10: 计算平均损失
    avg_loss = total_loss / max_batches
    for key in loss_components:
        loss_components[key] = loss_components[key] / max_batches
        
    return avg_loss, loss_components


"""
函数: test_model

功能: 在目标域数据集上评估模型性能，计算损失和准确率

参数:
  model: CrossAttentionModel - 跨域注意力模型
  dataloader_source: DataLoader - 源域数据加载器（此函数中未使用）
  dataloader_target: DataLoader - 目标域数据加载器
  criterion: LossFunction - 损失函数对象

返回值:
  avg_loss: float - 目标域平均损失
  accuracy: float - 目标域准确率 (0-1之间)
"""
def test_model(model, dataloader_source, dataloader_target, criterion):
    # 设置模型为评估模式（禁用dropout等）
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # 禁用梯度计算提高效率
    with torch.no_grad():
        # 遍历目标域数据
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

    # 计算平均指标
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader_target)

    return avg_loss, accuracy


"""
函数: save_best_model

功能: 保存最佳模型权重和训练信息到指定路径

参数:
  model: torch.nn.Module - 待保存的模型
  path: str - 模型保存路径
  accuracy: float - 当前模型的准确率
  epoch: int - 当前训练轮数

返回值: 无
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
函数: main

功能: 主训练函数，组织完整的模型初始化、训练、验证和评估流程

参数: 无

返回值: 无

工作流程:
  1. 模型初始化和结构测试
  2. 损失函数、优化器、学习率调度器配置
  3. 数据加载器有效性验证
  4. 训练循环：前向传播 -> 损失计算 -> 反向传播 -> 参数更新
  5. 性能评估和最佳模型保存
  6. 训练曲线可视化和绘制
"""
def main():
    # 阶段1: 模型初始化
    model = CrossAttentionModel(num_classes=10).to(device)
    
    # 模型结构验证：检查输入输出维度是否正确
    print("\n=== 模型结构测试 ===")
    try:
        test_src = torch.randn(2, 56, 3, 6000).to(device)
        test_tgt = torch.randn(2, 56, 3, 6000).to(device)
        
        print(f"测试输入形状: src={test_src.shape}, tgt={test_tgt.shape}")
        
        with torch.no_grad():
            pred_s, pred_t, F_s, F_t, F_c = model(test_src, test_tgt)
            print(f"模型输出形状:")
            print(f"  pred_s: {pred_s.shape}")
            print(f"  pred_t: {pred_t.shape}")
            print(f"  F_s: {F_s.shape}")
            print(f"  F_t: {F_t.shape}")
            print(f"  F_c: {F_c.shape}")
            print("模型结构测试通过！")
            
    except Exception as e:
        print(f"模型测试失败: {e}")
        print("请检查模型结构...")
        return
    
    # 阶段2: 损失函数配置
    criterion = LossFunction(
        num_classes=10,
        alpha=1.0,      # 源域分类损失权重
        beta=1.0,       # 目标域分类损失权重
        gamma=0.3,      # 跨域特征对齐损失权重
        delta=0.2       # 一致性损失权重
    )
    
    # 阶段3: 优化器与学习率调度器配置
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # 阶段4: 训练超参数设置
    num_epochs = 100
    best_accuracy = 0.0
    best_model_path = "Attention/best_attention_model.pth"
    patience = 15  # 早停耐心值
    early_stop_counter = 0

    print("\n开始改进的训练和测试...")
    
    # 阶段5: 数据加载器验证
    print("\n=== 数据加载器测试 ===")
    try:
        for i, (src_batch, tgt_batch) in enumerate(zip(source_loader, target_loader)):
            src_data, src_labels = src_batch
            tgt_data, tgt_labels = tgt_batch
            print(f"第{i+1}个批次 - 源域: {src_data.shape}, 目标域: {tgt_data.shape}")
            print(f"  源域标签范围: [{src_labels.min()}, {src_labels.max()}]")
            print(f"  目标域标签范围: [{tgt_labels.min()}, {tgt_labels.max()}]")
            if i >= 2:  # 只检查前3个批次
                break
    except Exception as e:
        print(f"数据加载器测试失败: {e}")
        return
    
    # 阶段6: 主训练循环
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

    # 阶段7: 训练结果可视化
    plt.figure(figsize=(15, 5))

    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(test_accuracies, label='Test Accuracy', color='green')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制学习率变化曲线
    plt.subplot(1, 3, 3)
    lrs = [group['lr'] for group in optimizer.param_groups]
    plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='purple')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # 确保输出目录存在
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # 保存和显示训练曲线
    plt.savefig("Attention/Attention_training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"训练完成。最佳准确率: {best_accuracy:.4f}")
    print(f"曲线已保存到 Attention/Attention_training_curves.png")


"""
脚本入口点：当该模块被直接运行时执行主函数
"""
if __name__ == "__main__":
    main()
