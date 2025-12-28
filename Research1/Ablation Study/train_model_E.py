"""
Model E 训练和评估脚本
TFGAN-CAL (Full) - 完整模型,包含所有组件
使用GAN生成的增强数据 + 交叉注意力机制进行训练
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from Research1.model_GAN import build_model
from Research1.model_ATT import CrossAttentionModel
from Research1.Process.dataloder_GAN import CustomDataset, select_samples_by_label
from Research1.loss_GAN import (
    mmd_loss,
    frequency_consistency_loss,
    wasserstein_generator_loss, 
    wasserstein_discriminator_loss
)
from Research1.plot_GAN import evaluate_gan_comprehensive


def train_gan(E, G, D, D_spec, source_loader, target_loader, target_data_cache, target_labels_cache,
              optimizer_E, optimizer_G, optimizer_D, optimizer_D_spec,
              num_epochs=200, device='cuda'):
    """训练GAN阶段"""
    print("\n[阶段1] 训练GAN生成器...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        E.train()
        G.train()
        D.train()
        D_spec.train()
        
        metrics = {'d_loss': 0.0, 'g_loss': 0.0, 'mmd': 0.0, 'freq': 0.0}
        num_batches = 0
        
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        total_batches = len(source_loader)
        
        for batch_idx in range(total_batches):
            try:
                x_s, _ = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                x_s, _ = next(source_iter)
            
            if x_s.size(0) < 4:
                continue
                
            x_s = x_s.to(device)
            batch_size = x_s.size(0)
            
            try:
                x_t_real, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                x_t_real, _ = next(target_iter)
            
            x_t_real = x_t_real.to(device)
            if x_t_real.size(0) < batch_size:
                idx = torch.randint(0, target_data_cache.size(0), (batch_size,)).tolist()
                x_t_real = target_data_cache[idx]
            
            with torch.no_grad():
                target_features = E(x_t_real)
            
            # 训练判别器
            for _ in range(3):
                optimizer_D.zero_grad()
                optimizer_D_spec.zero_grad()
                
                with torch.no_grad():
                    x_hat_t = G(x_s, target_features)
                
                d_loss1, _, _ = wasserstein_discriminator_loss(D, x_t_real, x_hat_t, lambda_gp=10.0, device=device)
                d_loss2, _, _ = wasserstein_discriminator_loss(D_spec, x_t_real, x_hat_t, lambda_gp=10.0, device=device)
                
                d_loss = d_loss1 + d_loss2
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(D_spec.parameters(), max_norm=1.0)
                optimizer_D.step()
                optimizer_D_spec.step()
            
            metrics['d_loss'] += d_loss.item()
            
            # 训练生成器
            optimizer_G.zero_grad()
            optimizer_E.zero_grad()
            
            target_features = E(x_t_real)
            x_hat_t = G(x_s, target_features)
            
            g_adv = wasserstein_generator_loss(D, x_hat_t) + wasserstein_generator_loss(D_spec, x_hat_t)
            gen_feat = E(x_hat_t)
            mmd = mmd_loss(target_features.detach(), gen_feat)
            freq = frequency_consistency_loss(x_t_real, x_hat_t)
            
            g_loss = g_adv + 20.0 * mmd + 5.0 * freq
            g_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(E.parameters(), max_norm=1.0)
            optimizer_G.step()
            optimizer_E.step()
            
            metrics['g_loss'] += g_loss.item()
            metrics['mmd'] += mmd.item()
            metrics['freq'] += freq.item()
            num_batches += 1
        
        for key in metrics:
            metrics[key] /= max(num_batches, 1)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  D损失: {metrics['d_loss']:.4f} | G损失: {metrics['g_loss']:.4f}")
            print(f"  MMD: {metrics['mmd']:.6f} | 频域: {metrics['freq']:.6f}")
    
    print("\nGAN训练完成!")


def generate_synthetic_data(E, G, source_loader, target_loader, num_samples=900, device='cuda'):
    """生成合成数据"""
    E.eval()
    G.eval()
    
    synthetic_data = []
    synthetic_labels = []
    generated_count = 0
    
    # 提取目标域特征
    all_features = []
    with torch.no_grad():
        for x_t_real, _ in target_loader:
            x_t_real = x_t_real.to(device)
            features = E(x_t_real)
            all_features.append(features)
    target_features = torch.cat(all_features, dim=0).to(device)
    
    with torch.no_grad():
        for x_s, source_labels_batch in source_loader:
            if generated_count >= num_samples:
                break
            
            x_s = x_s.to(device)
            x_hat_t = G(x_s, target_features)
            
            synthetic_data.append(x_hat_t.cpu())
            synthetic_labels.append(source_labels_batch[:x_hat_t.size(0)])
            generated_count += x_hat_t.size(0)
    
    if synthetic_data:
        synthetic_data = torch.cat(synthetic_data, dim=0)[:num_samples]
        synthetic_labels = torch.cat(synthetic_labels, dim=0)[:num_samples]
    
    return synthetic_data, synthetic_labels


def train_attention_model(model, source_loader, augmented_target_loader, optimizer, num_epochs=150, device='cuda'):
    """训练交叉注意力模型"""
    print("\n[阶段2] 训练交叉注意力分类模型...")
    print("=" * 70)
    
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        source_iter = iter(source_loader)
        target_iter = iter(augmented_target_loader)
        max_batches = max(len(source_loader), len(augmented_target_loader))
        
        for _ in range(max_batches):
            try:
                src_data, src_labels = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                src_data, src_labels = next(source_iter)
            
            try:
                tgt_data, tgt_labels = next(target_iter)
            except StopIteration:
                target_iter = iter(augmented_target_loader)
                tgt_data, tgt_labels = next(target_iter)
            
            min_batch = min(src_data.size(0), tgt_data.size(0))
            src_data, src_labels = src_data[:min_batch].to(device), src_labels[:min_batch].to(device)
            tgt_data, tgt_labels = tgt_data[:min_batch].to(device), tgt_labels[:min_batch].to(device)
            
            optimizer.zero_grad()
            pred_s, pred_t, F_s, F_t, _ = model(src_data, tgt_data)
            
            loss_s = F.cross_entropy(pred_s, src_labels)
            loss_t = F.cross_entropy(pred_t, tgt_labels)
            
            # MMD损失
            delta = F_s.mean(0) - F_t.mean(0)
            loss_mmd = (delta ** 2).sum()
            
            loss = loss_s + 1.5 * loss_t + 0.5 * loss_mmd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in augmented_target_loader:
                data, labels = data.to(device), labels.to(device)
                _, pred, _, _, _ = model(torch.zeros_like(data).to(device), data)
                _, predicted = torch.max(pred, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  训练损失: {total_loss/max_batches:.4f}")
            print(f"  目标域准确率: {accuracy:.2f}% (最佳: {best_accuracy:.2f}%)")
    
    print("\n交叉注意力模型训练完成!")
    return best_accuracy


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    source_data = torch.load('../Data/source_env0_env1_data.pt')
    source_labels = torch.load('../Data/source_env0_env1_labels.pt')
    target_data = torch.load('../Data/target_env2_data.pt')
    target_labels = torch.load('../Data/target_env2_labels.pt')
    
    source_dataset = CustomDataset(source_data, source_labels)
    source_loader = DataLoader(source_dataset, batch_size=100, shuffle=True)
    
    selected_target_data, selected_target_labels = select_samples_by_label(
        target_data, target_labels, samples_per_label=10
    )
    target_dataset = CustomDataset(selected_target_data, selected_target_labels)
    target_loader = DataLoader(target_dataset, batch_size=100, shuffle=True)
    
    print(f"源域数据: {source_data.shape}")
    print(f"目标域数据(选取): {selected_target_data.shape}")
    
    # 缓存目标域数据
    all_target_data = []
    all_target_labels = []
    for x_t_real, labels in target_loader:
        all_target_data.append(x_t_real)
        all_target_labels.append(labels)
    target_data_cache = torch.cat(all_target_data, dim=0).to(device)
    target_labels_cache = torch.cat(all_target_labels, dim=0).to(device)
    
    # 阶段1: 训练GAN
    E, G, D, D_spec = build_model()
    E.to(device)
    G.to(device)
    D.to(device)
    D_spec.to(device)
    
    optimizer_E = optim.Adam(E.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_D_spec = optim.Adam(D_spec.parameters(), lr=1e-4, betas=(0.0, 0.9))
    
    train_gan(E, G, D, D_spec, source_loader, target_loader, target_data_cache, target_labels_cache,
              optimizer_E, optimizer_G, optimizer_D, optimizer_D_spec, num_epochs=200, device=device)
    
    # 评估GAN
    print("\n评估GAN质量...")
    gan_metrics = evaluate_gan_comprehensive(E, G, source_loader, target_loader, device=device)
    
    print("\nGAN质量评估结果：")
    print(f"  FID: {gan_metrics.get('fid', -1):.4f}")
    print(f"  IS: {gan_metrics.get('inception_score', -1):.4f}")
    print(f"  时域MSE: {gan_metrics.get('time_domain_mse', -1):.6f}")
    print(f"  频谱CC: {gan_metrics.get('spectral_correlation', -1):.4f}")
    
    # 生成合成数据
    print("\n生成合成数据...")
    synthetic_data, synthetic_labels = generate_synthetic_data(E, G, source_loader, target_loader, num_samples=900, device=device)
    print(f"已生成 {len(synthetic_data)} 个合成样本")
    
    # 合并数据
    augmented_data = torch.cat([selected_target_data, synthetic_data], dim=0)
    augmented_labels = torch.cat([selected_target_labels, synthetic_labels], dim=0)
    print(f"增强后目标域数据: {augmented_data.shape}")
    
    # 阶段2: 训练交叉注意力模型
    augmented_dataset = CustomDataset(augmented_data, augmented_labels)
    augmented_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
    
    attention_model = CrossAttentionModel(num_classes=10).to(device)
    optimizer_att = optim.Adam(attention_model.parameters(), lr=1e-4)
    
    best_accuracy = train_attention_model(attention_model, source_loader, augmented_loader, optimizer_att, num_epochs=150, device=device)
    
    # 保存结果
    print("\n" + "="*70)
    print("Model E (TFGAN-CAL, Full) 评估结果：")
    print("="*70)
    print(f"  ① FID (Fréchet Inception Distance)      : {gan_metrics.get('fid', -1):.4f}")
    print(f"  ② IS (Inception Score)                  : {gan_metrics.get('inception_score', -1):.4f}")
    print(f"  ③ 时域MSE (Time-domain MSE)             : {gan_metrics.get('time_domain_mse', -1):.6f}")
    print(f"  ④ 频谱相关性系数 (Spectral Correlation)  : {gan_metrics.get('spectral_correlation', -1):.4f}")
    print(f"  ⑤ 跨域准确率 (Cross-domain Accuracy)    : {best_accuracy:.2f}%")
    print("="*70 + "\n")
    
    output_dir = 'Ablation Study/model_E'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'model_name': 'Model E (TFGAN-CAL, Full)',
        'description': '完整模型 - GAN数据增强 + 交叉注意力',
        'metrics': {
            'fid': gan_metrics.get('fid', -1),
            'inception_score': gan_metrics.get('inception_score', -1),
            'time_domain_mse': gan_metrics.get('time_domain_mse', -1),
            'spectral_correlation': gan_metrics.get('spectral_correlation', -1),
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
