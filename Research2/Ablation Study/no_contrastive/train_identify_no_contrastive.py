import torch
import os
import sys
import torch.optim as optim
import json

# 添加上级目录到sys.path以解决导入问题
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 保存当前工作目录
original_cwd = os.getcwd()
# 切换到dataloader_identify.py所在的目录
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

from model_identify_no_contrastive import IdentifyDetectionSystemNoContrastive
from loss_identify_no_contrastive import IdentifyDetectionLossNoContrastive
from dataloader_identify import load_identify_data, create_data_loaders

def train_epoch(model, source_loader, target_loader, criterion, optimizer, device, epoch):
    """
    训练一个epoch（无对比损失版本）
    """
    model.train()
    total_loss = 0.0
    loss_components = {'identity': 0.0}
    batch_count = 0
    
    # 使用zip循环处理源域和目标域数据
    for batch_idx, (source_batch, target_batch) in enumerate(zip(source_loader, target_loader)):
        # 限制每个epoch的批次数以加快训练（消融实验调整）
        if batch_idx >= 100:  # 增加批次数限制到100
            break
            
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

        # 反向传播和优化
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # 累计损失
        total_loss += loss.item()
        loss_components['identity'] += loss_dict['identity_loss']
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
    
    # 消融实验标识：无对比损失版本
    print("=" * 60)
    print("消融实验：身份识别模型（无对比损失版本）")
    print("模型配置：适度降低复杂度以展示对比损失的作用")
    print("预期效果：性能应低于完整模型但高于随机猜测")
    print("=" * 60)
    
    # 直接加载已保存的数据集
    print("加载已保存的数据集...")
    try:
        datasets = load_identify_data()
    except FileNotFoundError as e:
        print(f"数据文件未找到: {e}")
        print("请确保数据文件存在于正确的位置，或者创建模拟数据以进行测试。")
        # 恢复原始工作目录
        os.chdir(original_cwd)
        return
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        # 恢复原始工作目录
        os.chdir(original_cwd)
        return
        
    data_loaders = create_data_loaders(datasets, batch_size=8)
    
    # 恢复原始工作目录
    os.chdir(original_cwd)

    # 初始化模型（无对比损失版本）
    print("初始化模型...")
    model = IdentifyDetectionSystemNoContrastive(num_classes=10, feature_dim=128, projection_dim=32).to(device)
    
    # 初始化损失函数 (无对比损失版本)
    criterion = IdentifyDetectionLossNoContrastive(alpha=1.0)
    
    # 初始化优化器 (使用AdamW优化器，适度学习率)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # 恢复学习率和权重衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 训练参数
    num_epochs = 100  # 恢复训练轮数
    best_accuracy = 0.0
    early_stop_counter = 0
    patience = 15  # 恢复耐心值
    
    # 记录训练历史
    train_losses = []
    val_accuracies = []
    loss_components_history = []
    test_accuracies = []

    # 确保保存模型的目录存在（使用相对于当前工作目录的路径）
    save_dir = '.'  # 保存在当前目录（与py文件同级）
    os.makedirs(save_dir, exist_ok=True)
    
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
        print(f'  训练损失: {train_loss:.4f} '
              f'(身份: {loss_components["identity"]:.4f})')
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
            }, os.path.join(save_dir, 'best_identify_model_no_contrastive.pth'))
            print(f'  保存最佳模型 (准确率: {best_accuracy:.2f}%)')
        else:
            early_stop_counter += 1
            print(f'  早停计数器: {early_stop_counter}/{patience}')
            
        # 早停检查
        if early_stop_counter >= patience:
            print(f'  验证准确率在 {patience} 个epoch内未提升，提前停止训练')
            break
    
    print(f"\n训练完成! 最佳验证准确率: {best_accuracy:.2f}%")
    
    # 保存训练历史
    history_file_path = os.path.join(save_dir, 'training_history.json')
    save_training_history(train_losses, val_accuracies, loss_components_history, test_accuracies, 
                         history_file_path)

    # 最终测试
    print("进行最终测试...")
    final_test_results = test_model(model, data_loaders, device)
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': final_test_results.get('identity_test', 0.0),
    }, os.path.join(save_dir, 'final_identify_model_no_contrastive.pth'))
    print(f"最终模型已保存到 {os.path.join(save_dir, 'final_identify_model_no_contrastive.pth')}")


if __name__ == "__main__":
    main()