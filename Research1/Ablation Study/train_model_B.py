"""
Model B 训练和评估脚本
w/o Generator, Only CAL - 仅使用交叉注意力,不使用生成器
在少量目标域数据上训练
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from model_B_only_CAL import ModelB_OnlyCAL
from Research1.DataProcess.dataloder_GAN import CustomDataset, select_samples_by_label


def mmd_loss(source_features, target_features):
    """MMD损失"""
    delta = source_features.mean(0) - target_features.mean(0)
    return (delta ** 2).sum()


def train_epoch(model, source_loader, target_loader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    
    max_batches = max(len(source_loader), len(target_loader))
    
    for _ in range(max_batches):
        try:
            src_data, src_labels = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            src_data, src_labels = next(source_iter)
            
        try:
            tgt_data, tgt_labels = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            tgt_data, tgt_labels = next(target_iter)
        
        min_batch = min(src_data.size(0), tgt_data.size(0))
        src_data, src_labels = src_data[:min_batch].to(device), src_labels[:min_batch].to(device)
        tgt_data, tgt_labels = tgt_data[:min_batch].to(device), tgt_labels[:min_batch].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        pred_s, pred_t, F_s, F_t, _ = model(src_data, tgt_data)
        
        # 计算损失
        loss_s = F.cross_entropy(pred_s, src_labels)
        loss_t = F.cross_entropy(pred_t, tgt_labels)
        loss_mmd = mmd_loss(F_s, F_t)
        
        loss = loss_s + 1.5 * loss_t + 0.5 * loss_mmd
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / max_batches


def validate(model, dataloader, device):
    """验证模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            _, pred, _, _, _ = model(torch.zeros_like(data).to(device), data)
            
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    source_data = torch.load('../Data/source_env0_env1_data.pt')
    source_labels = torch.load('../Data/source_env0_env1_labels.pt')
    target_data = torch.load('../Data/target_env2_data.pt')
    target_labels = torch.load('../Data/target_env2_labels.pt')
    
    # 从目标域中选取少量样本 (每类10个)
    selected_target_data, selected_target_labels = select_samples_by_label(
        target_data, target_labels, samples_per_label=10
    )
    
    print(f"源域数据: {source_data.shape}")
    print(f"目标域数据(选取): {selected_target_data.shape}")
    
    # 创建数据加载器
    source_dataset = CustomDataset(source_data, source_labels)
    target_dataset = CustomDataset(selected_target_data, selected_target_labels)
    full_target_dataset = CustomDataset(target_data, target_labels)
    
    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)
    full_target_loader = DataLoader(full_target_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = ModelB_OnlyCAL(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 150
    best_accuracy = 0.0
    
    print("\n开始训练 Model B (w/o Generator, Only CAL)...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, source_loader, target_loader, optimizer, device)
        val_acc = validate(model, full_target_loader, device)
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  目标域准确率: {val_acc:.2f}% (最佳: {best_accuracy:.2f}%)")
    
    print("\n训练完成!")
    print("=" * 70)
    
    print("\n" + "="*70)
    print("Model B (w/o Generator, Only CAL) 评估结果：")
    print("="*70)
    print(f"  ① FID (Fréchet Inception Distance)      : N/A (无生成器)")
    print(f"  ② IS (Inception Score)                  : N/A (无生成器)")
    print(f"  ③ 时域MSE (Time-domain MSE)             : N/A (无生成器)")
    print(f"  ④ 频谱相关性系数 (Spectral Correlation)  : N/A (无生成器)")
    print(f"  ⑤ 跨域准确率 (Cross-domain Accuracy)    : {best_accuracy:.2f}%")
    print("="*70 + "\n")
    
    # 保存结果
    output_dir = 'Ablation Study/model_B'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'model_name': 'Model B (w/o Generator, Only CAL)',
        'description': '仅使用交叉注意力机制,不使用生成器',
        'metrics': {
            'fid': None,
            'inception_score': None,
            'time_domain_mse': None,
            'spectral_correlation': None,
            'cross_domain_accuracy': best_accuracy
        }
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"结果已保存到 {output_dir}/evaluation_results.json")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
