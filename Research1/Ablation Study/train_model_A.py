"""
Model A (Baseline) 训练和评估脚本
仅使用源域数据训练,在目标域上测试
"""

import torch
import torch.optim as optim
import torch.nn as nn
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from model_A_baseline import BaselineModel
from Research1.Process.dataloder_ATT import CustomDataset
from Research1.plot_GAN import (
    evaluate_gan_comprehensive,
    compute_time_domain_mse,
    compute_spectral_correlation
)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        pred, _ = model(data)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(pred, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, device):
    """验证模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            pred, _ = model(data)
            
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def evaluate_model(model, target_loader, device):
    """评估模型 - 计算FID, IS, 时域MSE, 频谱CC和跨域准确率"""
    model.eval()
    
    # 1. 计算跨域准确率
    cross_domain_accuracy = validate(model, target_loader, device)
    
    # 2. 收集目标域真实数据和模型特征
    real_samples_list = []
    real_features_list = []
    
    with torch.no_grad():
        for data, _ in target_loader:
            data = data.to(device)
            _, features = model(data)
            real_samples_list.append(data.cpu())
            real_features_list.append(features.cpu())
    
    real_samples = torch.cat(real_samples_list, dim=0)
    real_features = torch.cat(real_features_list, dim=0)
    
    # 3. 基线模型没有生成数据,使用源域数据代替(仅作为对比)
    # 这里的评估指标会很差,因为没有生成器
    # 时域MSE和频谱CC设为最差值
    time_domain_mse = float('inf')  # 无生成数据,设为无穷大
    spectral_correlation = 0.0  # 无生成数据,设为0
    
    # FID和IS也无法计算,因为没有生成样本
    fid = float('inf')
    inception_score = 0.0
    
    metrics = {
        'fid': fid,
        'inception_score': inception_score,
        'time_domain_mse': time_domain_mse,
        'spectral_correlation': spectral_correlation,
        'cross_domain_accuracy': cross_domain_accuracy
    }
    
    return metrics


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    source_data = torch.load('../Data/source_env0_env1_data.pt')
    source_labels = torch.load('../Data/source_env0_env1_labels.pt')
    target_data = torch.load('../Data/target_env2_data.pt')
    target_labels = torch.load('../Data/target_env2_labels.pt')
    
    print(f"源域数据: {source_data.shape}")
    print(f"目标域数据: {target_data.shape}")
    
    # 创建数据加载器
    source_dataset = CustomDataset(source_data, source_labels)
    target_dataset = CustomDataset(target_data, target_labels)
    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = BaselineModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练参数
    num_epochs = 100
    best_accuracy = 0.0
    
    print("\n开始训练 Model A (Baseline)...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        # 仅在源域上训练
        train_loss, train_acc = train_epoch(model, source_loader, criterion, optimizer, device)
        
        # 在目标域上验证
        val_acc = validate(model, target_loader, device)
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
            print(f"  目标域准确率: {val_acc:.2f}% (最佳: {best_accuracy:.2f}%)")
    
    print("\n训练完成!")
    print("=" * 70)
    
    # 评估模型
    print("\n评估 Model A (Baseline)...")
    metrics = evaluate_model(model, target_loader, device)
    
    print("\n" + "="*70)
    print("Model A (Baseline) 评估结果：")
    print("="*70)
    print(f"  ① FID (Fréchet Inception Distance)      : N/A (无生成器)")
    print(f"  ② IS (Inception Score)                  : N/A (无生成器)")
    print(f"  ③ 时域MSE (Time-domain MSE)             : N/A (无生成器)")
    print(f"  ④ 频谱相关性系数 (Spectral Correlation)  : N/A (无生成器)")
    print(f"  ⑤ 跨域准确率 (Cross-domain Accuracy)    : {metrics['cross_domain_accuracy']:.2f}%")
    print("="*70 + "\n")
    
    # 保存结果
    output_dir = 'Ablation Study/model_A'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'model_name': 'Model A (Baseline)',
        'description': 'w/o TFGAN & CAL - 仅使用源域数据训练CNN分类器',
        'metrics': metrics,
        'best_target_accuracy': best_accuracy
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"结果已保存到 {output_dir}/evaluation_results.json")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
