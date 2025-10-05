import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model import IdentifyDetectionSystem, LearnableComprehensiveIntruderDetector
from intruder_data_loader import load_intruder_data, create_intruder_data_loaders


def extract_features(model, data_loader, device):
    """
    从数据加载器中提取特征和标签
    """
    model.eval()
    all_features = []
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            # 移动到设备
            data = data.to(device)
            labels = labels.numpy()
            
            # 提取特征和预测
            outputs = model(data)
            features = outputs['features'].cpu().numpy()
            logits = outputs['logits'].cpu().numpy()
            
            all_features.append(features)
            all_logits.append(logits)
            all_labels.append(labels)
    
    return np.vstack(all_features), np.vstack(all_logits), np.hstack(all_labels)

def initialize_learnable_detector(learnable_detector, device):
    """
    初始化可学习检测器
    """
    print("初始化可学习入侵者检测器...")
    # 使用随机初始化
    return

def test_intruder_detector(model, identity_model, data_loader, device, detector_path=None):
    """
    测试入侵者检测器性能（只计算准确率）
    """
    model.eval()
    identity_model.eval()
    
    correct = 0
    total = 0
    
    # 加载可学习的入侵者检测器
    if detector_path and os.path.exists(detector_path):
        checkpoint = torch.load(detector_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
    
    with torch.no_grad():
        for data, labels in data_loader:
            # 移动到设备
            data = data.to(device)
            labels = labels.to(device)
            
            # 使用身份识别模型提取特征
            identity_outputs = identity_model(data)
            features = identity_outputs['features']
            logits = identity_outputs['logits']
            
            # 使用可学习的入侵者检测器
            detector_outputs = model(features, features, logits, logits)
            predictions = detector_outputs['predictions']
            
            # 计算准确率
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    # 计算准确率
    detection_accuracy = correct / total if total > 0 else 0.0
    
    return {
        'detection_accuracy': detection_accuracy
    }

def plot_training_curves(train_losses, val_accuracies, loss_components_history):
    """
    绘制训练曲线
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 绘制训练损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 绘制验证准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # 绘制各项损失曲线
    plt.subplot(1, 3, 3)
    classification_losses = [comp['classification'] for comp in loss_components_history]
    energy_losses = [comp['energy'] for comp in loss_components_history]
    fusion_losses = [comp['fusion'] for comp in loss_components_history]
    
    plt.plot(epochs, classification_losses, label='Classification Loss')
    plt.plot(epochs, energy_losses, label='Energy Loss')
    plt.plot(epochs, fusion_losses, label='Fusion Loss')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('intruder/training_curves.png')
    plt.close()


def plot_detection_metrics(detection_metrics_history):
    """
    绘制入侵者检测准确率变化曲线
    """
    epochs = range(1, len(detection_metrics_history) + 1)
    
    detection_accuracies = [metrics['detection_accuracy'] for metrics in detection_metrics_history]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制准确率曲线
    plt.plot(epochs, detection_accuracies, 'b-', label='Detection Accuracy', marker='o')
    plt.title('Intruder Detection Accuracy Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('intruder/detection_accuracy.png')
    plt.close()


def train_intruder_detector(model_path, output_path, device):
    """
    训练可学习的入侵者检测器（使用源域和目标域身份特征）
    """
    print("开始训练可学习的入侵者检测器...")


    # 初始化身份识别模型
    print("加载身份识别模型...")
    identity_model = IdentifyDetectionSystem(num_classes=10, feature_dim=128, projection_dim=32).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    identity_model.load_state_dict(checkpoint['model_state_dict'])
    identity_model.eval()
    
    print(f"身份识别模型加载完成")
    
    # 加载入侵者检测专用数据
    print("加载入侵者检测数据...")
    datasets = load_intruder_data()
    
    data_loaders = create_intruder_data_loaders(datasets, batch_size=32)
    
    # 初始化可学习的综合入侵者检测器
    print("初始化可学习的综合入侵者检测器...")
    learnable_detector = LearnableComprehensiveIntruderDetector(num_classes=10, feature_dim=128, identity_classes=10).to(device)
    
    # 初始化可学习检测器
    initialize_learnable_detector(learnable_detector, device)
    
    # 设置优化器，只优化入侵者检测器的参数
    optimizer = torch.optim.Adam(learnable_detector.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    # 损失函数
    classification_criterion = torch.nn.CrossEntropyLoss()
    mse_criterion = torch.nn.MSELoss()
    
    print("开始训练可学习的入侵者检测器...")
    learnable_detector.train()
    
    num_epochs = 30
    best_accuracy = 0.0
    early_stop_counter = 0
    patience = 10
    
    # 记录训练历史
    train_losses = []
    val_accuracies = []
    loss_components_history = []
    detection_metrics_history = []
    
    # 确保保存模型的目录存在
    os.makedirs('intruder', exist_ok=True)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 损失组件统计
        total_classification_loss = 0.0
        total_energy_loss = 0.0
        total_fusion_loss = 0.0
        batch_count = 0
        
        # 训练循环
        for batch_idx, (data, labels) in enumerate(data_loaders['intruder_train']):
            # 移动到设备
            data = data.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            
            # 使用身份识别模型提取特征
            with torch.no_grad():
                identity_outputs = identity_model(data)
                features = identity_outputs['features']
                logits = identity_outputs['logits']
            
            # 使用可学习的入侵者检测器进行检测
            detector_outputs = learnable_detector(features, features, logits, logits)
            predictions = detector_outputs['predictions']
            openmax_probs = detector_outputs['openmax_probabilities']
            energy_scores = detector_outputs['energy_scores']
            
            # 创建目标标签（二分类：0-合法用户，1-入侵者）
            classification_targets = torch.zeros(data.size(0), 2, device=device)
            classification_targets.scatter_(1, labels.unsqueeze(1), 1)
            
            # 计算损失
            # 1. 分类损失
            classification_loss = mse_criterion(openmax_probs[:, :2], classification_targets)
            
            # 2. 能量损失（希望合法用户的能量分数低，入侵者的能量分数高）
            legal_mask = (labels == 0)
            intruder_mask = (labels == 1)
            
            energy_loss = 0.0
            if legal_mask.sum() > 0:
                legal_energy = energy_scores[legal_mask]
                energy_loss += torch.mean(legal_energy)  # 希望合法用户的能量分数尽可能低
            
            if intruder_mask.sum() > 0:
                intruder_energy = energy_scores[intruder_mask]
                energy_loss += torch.mean(torch.relu(2.0 - intruder_energy))  # 希望入侵者的能量分数尽可能高(>2.0)
            
            # 3. 融合权重损失（鼓励模型做出明确的决策）
            fusion_weights = detector_outputs['fusion_weights']
            fusion_loss = torch.mean(torch.abs(fusion_weights - 0.5))  # 鼓励权重远离0.5
            
            # 总损失
            total_loss_batch = classification_loss + 0.1 * energy_loss + 0.05 * fusion_loss
            
            # 反向传播和优化
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(learnable_detector.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_classification_loss += classification_loss.item()
            total_energy_loss += energy_loss.item() if isinstance(energy_loss, torch.Tensor) else energy_loss
            total_fusion_loss += fusion_loss.item()
            batch_count += 1
            
            # 统计准确率
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # 每20个batch打印一次进度（减少打印频率以避免警告）
            if (batch_idx + 1) % 20 == 0:
                pass  # 不再打印每个batch的损失信息
        
        # 计算平均损失和准确率
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        # 记录训练历史
        train_losses.append(avg_loss)
        val_accuracies.append(accuracy)
        loss_components_history.append({
            'classification': total_classification_loss / batch_count if batch_count > 0 else 0.0,
            'energy': total_energy_loss / batch_count if batch_count > 0 else 0.0,
            'fusion': total_fusion_loss / batch_count if batch_count > 0 else 0.0
        })
        
        print(f'Epoch [{epoch+1}/{num_epochs}], 平均损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}')
        
        # 在每个epoch后测试入侵者检测器性能
        print(f'  测试效果:')
        test_results = test_intruder_detector(
            learnable_detector, identity_model, data_loaders['intruder_test'], device, 
            detector_path=None)
        print(f'    入侵者检测准确率: {test_results["detection_accuracy"]:.4f}')
        
        # 记录检测指标历史
        detection_metrics_history.append(test_results)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if test_results["detection_accuracy"] > best_accuracy:
            best_accuracy = test_results["detection_accuracy"]
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': learnable_detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'detection_metrics': test_results
            }, output_path)
            print(f'  保存最佳模型 (准确率: {best_accuracy:.4f})')
        else:
            early_stop_counter += 1
            print(f'  早停计数器: {early_stop_counter}/{patience}')
            
        # 早停检查
        if early_stop_counter >= patience:
            print(f'  入侵者检测准确率在 {patience} 个epoch内未提升，提前停止训练')
            break
    
    print(f"训练完成! 最佳准确率: {best_accuracy:.4f}")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_accuracies, loss_components_history)
    print("训练曲线已保存到 intruder/training_curves.png")
    
    # 绘制入侵者检测准确率变化曲线
    plot_detection_metrics(detection_metrics_history)
    print("入侵者检测准确率变化曲线已保存到 intruder/detection_accuracy.png")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': learnable_detector.state_dict(),
        'best_accuracy': best_accuracy,
    }, 'intruder/final_learnable_intruder_detector.pth')
    print("最终模型已保存到 intruder/final_learnable_intruder_detector.pth")


def main():
    """
    主训练函数
    """
    print("开始训练入侵者检测器...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型路径
    model_path = "identify/best_identify_model.pth"  # 身份识别训练好的模型
    output_path = "intruder/learnable_intruder_detector.pth"



    # 训练入侵者检测器
    train_intruder_detector(model_path, output_path, device)


if __name__ == "__main__":
    main()