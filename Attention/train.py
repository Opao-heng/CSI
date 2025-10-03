# train_identify.py
import torch
import torch.optim as optim
from attention import CrossAttentionModel
from loss import LossFunction
from pre_process.dataloder_ATT import source_loader, target_loader
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化记录列表
train_losses = []
test_losses = []
test_accuracies = []

def train_epoch(model, dataloader_source, dataloader_target, criterion, optimizer, scheduler=None):
    """
    改进的训练函数：
    1. 动态调整损失权重
    2. 添加学习率调度
    3. 优化数据匹配策略
    """
    model.train()
    total_loss = 0.0
    loss_components = {'source': 0.0, 'target': 0.0, 'cross_feature': 0.0, 'consistency': 0.0}
    
    # 使用较大的数据集作为基准
    max_batches = max(len(dataloader_source), len(dataloader_target))
    
    source_iter = iter(dataloader_source)
    target_iter = iter(dataloader_target)
    
    for batch_idx in range(max_batches):
        # 获取源域数据
        try:
            src_data, src_labels = next(source_iter)
        except StopIteration:
            source_iter = iter(dataloader_source)
            src_data, src_labels = next(source_iter)
            
        # 获取目标域数据
        try:
            tgt_data, tgt_labels = next(target_iter)
        except StopIteration:
            target_iter = iter(dataloader_target)
            tgt_data, tgt_labels = next(target_iter)
            
        # 处理批次大小不一致的情况
        min_batch_size = min(src_data.size(0), tgt_data.size(0))
        src_data, src_labels = src_data[:min_batch_size], src_labels[:min_batch_size]
        tgt_data, tgt_labels = tgt_data[:min_batch_size], tgt_labels[:min_batch_size]
        
        src_data, src_labels = src_data.to(device), src_labels.to(device)
        tgt_data, tgt_labels = tgt_data.to(device), tgt_labels.to(device)

        optimizer.zero_grad()

        try:
            pred_s, pred_t, F_s, F_t, F_c = model(src_data, tgt_data)
        except Exception as e:
            print(f"\n模型前向传播错误 (batch {batch_idx}):")
            print(f"  源域数据形状: {src_data.shape}")
            print(f"  目标域数据形状: {tgt_data.shape}")
            print(f"  错误信息: {e}")
            raise e

        # 计算各项损失
        ls = criterion.source_loss(pred_s, src_labels)
        lt = criterion.target_loss(pred_t, tgt_labels)
        lsf = criterion.cross_feature_loss(F_s, F_t)
        lc = criterion.consistency_loss(F_s, F_c)

        # 加权总损失
        loss = (criterion.alpha * ls + 
               criterion.beta * lt + 
               criterion.gamma * lsf + 
               criterion.delta * lc)
               
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler:
            scheduler.step()

        # 统计损失
        total_loss += loss.item()
        loss_components['source'] += ls.item()
        loss_components['target'] += lt.item()
        loss_components['cross_feature'] += lsf.item()
        loss_components['consistency'] += lc.item()

    # 返回平均损失
    avg_loss = total_loss / max_batches
    for key in loss_components:
        loss_components[key] = loss_components[key] / max_batches
        
    return avg_loss, loss_components


def test_model(model, dataloader_source, dataloader_target, criterion):
    """
    测试模型在目标域上的性能：计算目标域的准确率和交叉熵损失
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for tgt_batch in dataloader_target:
            tgt_data, tgt_labels = tgt_batch
            tgt_data, tgt_labels = tgt_data.to(device), tgt_labels.to(device)

            _, pred_t, _, _, _ = model(torch.zeros_like(tgt_data).to(device), tgt_data)

            loss = criterion.target_loss(pred_t, tgt_labels)
            total_loss += loss.item()

            pred_t_prob = torch.softmax(pred_t, dim=1)
            _, predicted = torch.max(pred_t_prob, 1)
            total += tgt_labels.size(0)
            correct += (predicted == tgt_labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader_target)

    return avg_loss, accuracy


def save_best_model(model, path, accuracy, epoch):
    """
    保存当前最佳模型
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
    }, path)
    print(f"Best model saved at {path} with accuracy: {accuracy:.4f}")


def main():
    # 使用改进的损失函数参数
    model = CrossAttentionModel(num_classes=10).to(device)
    
    # 添加模型测试，确保维度正确
    print("\n=== 模型结构测试 ===")
    try:
        # 创建测试数据
        test_src = torch.randn(2, 56, 3, 6000).to(device)
        test_tgt = torch.randn(2, 56, 3, 6000).to(device)
        
        print(f"测试输入形状: src={test_src.shape}, tgt={test_tgt.shape}")
        
        # 测试前向传播
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
    
    criterion = LossFunction(
        num_classes=10,
        alpha=1.0,      # 源域分类损失
        beta=1.0,       # 目标域分类损失
        gamma=0.3,      # 跨域特征损失（降低权重）
        delta=0.2       # 一致性损失（降低权重）
    )
    
    # 使用AdamW优化器和余弦退火调度器
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    num_epochs = 100  # 增加训练轮数
    best_accuracy = 0.0
    best_model_path = "models/best_cross_attention_model.pth"
    
    # 早停参数
    patience = 15
    early_stop_counter = 0

    print("\n开始改进的训练和测试...")
    
    # 数据加载器测试
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
    for epoch in range(num_epochs):
        train_loss, loss_components = train_epoch(model, source_loader, target_loader, criterion, optimizer, scheduler)
        test_loss, accuracy = test_model(model, source_loader, target_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} (S:{loss_components['source']:.3f}, T:{loss_components['target']:.3f}, SF:{loss_components['cross_feature']:.3f}, C:{loss_components['consistency']:.3f})")
        print(f"  Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型和早停
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            early_stop_counter = 0
            save_best_model(model, best_model_path, accuracy, epoch)
        else:
            early_stop_counter += 1
            
        # 早停检查
        if early_stop_counter >= patience:
            print(f"早停在第 {epoch+1} 轮，最佳准确率: {best_accuracy:.4f}")
            break
            
        # 动态调整损失权重（随训练进展逐渐减少领域适应损失）
        if epoch > 20:
            criterion.gamma = float(max(0.1, criterion.gamma * 0.98))
            criterion.delta = float(max(0.05, criterion.delta * 0.98))

    # 绘制曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(test_accuracies, label='Test Accuracy', color='green')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 新增：学习率曲线
    plt.subplot(1, 3, 3)
    lrs = [group['lr'] for group in optimizer.param_groups]
    plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='purple')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # 确保目录存在
    import os
    if not os.path.exists("results"):
        os.makedirs("results")
    
    plt.savefig("results/improved_training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"训练完成。最佳准确率: {best_accuracy:.4f}")
    print(f"曲线已保存到 results/improved_training_curves.png")


if __name__ == "__main__":
    main()
