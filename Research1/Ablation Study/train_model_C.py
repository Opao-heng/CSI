"""
Model C 训练和评估脚本
w/o Freq-D, Time-only TFGAN - 仅使用时域判别器
"""

import torch
import torch.optim as optim
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from model_C_time_only_GAN import build_model
from Research1.DataProcess.dataloder_GAN import CustomDataset, select_samples_by_label
from Research1.loss_GAN import (
    mmd_loss,
    frequency_consistency_loss,
    wasserstein_generator_loss, 
    wasserstein_discriminator_loss
)
from Research1.plot_GAN import evaluate_gan_comprehensive, compute_time_domain_mse, compute_spectral_correlation


def train_epoch(E, G, D, source_loader, target_loader, target_data, target_labels,
                optimizer_E, optimizer_G, optimizer_D,
                lambda_mmd=20.0, lambda_freq=5.0, n_critic=3, device='cuda'):
    """训练一个epoch - Model C仅使用时域判别器"""
    E.train()
    G.train()
    D.train()

    metrics = {'d_loss': 0.0, 'g_loss': 0.0, 'mmd': 0.0, 'freq': 0.0, 'gp': 0.0}
    num_d_batches = 0
    num_g_batches = 0
    
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    total_batches = len(source_loader)
    
    for batch_idx in range(total_batches):
        try:
            x_s, source_labels = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            x_s, source_labels = next(source_iter)
        
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
            idx = torch.randint(0, target_data.size(0), (batch_size,)).tolist()
            x_t_real = target_data[idx].to(device)

        with torch.no_grad():
            target_features = E(x_t_real)
        
        # 训练判别器 (仅时域判别器)
        for _ in range(n_critic):
            optimizer_D.zero_grad()
            
            with torch.no_grad():
                x_hat_t = G(x_s, target_features)
            
            d_loss, w, gp = wasserstein_discriminator_loss(D, x_t_real, x_hat_t, lambda_gp=10.0, device=device)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            optimizer_D.step()
            
            metrics['gp'] += gp
            num_d_batches += 1
        
        metrics['d_loss'] += d_loss.item()

        # 训练生成器
        optimizer_G.zero_grad()
        optimizer_E.zero_grad()
        
        target_features = E(x_t_real)
        x_hat_t = G(x_s, target_features)
        
        # 对抗损失 (仅时域)
        g_adv = wasserstein_generator_loss(D, x_hat_t)
        
        # MMD损失
        gen_feat = E(x_hat_t)
        mmd = mmd_loss(target_features.detach(), gen_feat)
        
        # 频域一致性损失
        freq = frequency_consistency_loss(x_t_real, x_hat_t)
        
        g_loss = g_adv + lambda_mmd * mmd + lambda_freq * freq
        g_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(E.parameters(), max_norm=1.0)
        
        optimizer_G.step()
        optimizer_E.step()
        
        metrics['g_loss'] += g_loss.item()
        metrics['mmd'] += mmd.item()
        metrics['freq'] += freq.item()
        num_g_batches += 1

    # 平均化
    for key in ['d_loss', 'gp']:
        metrics[key] /= max(num_d_batches, 1)
    for key in ['g_loss', 'mmd', 'freq']:
        metrics[key] /= max(num_g_batches, 1)
    
    return metrics


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
    
    # 创建模型 (仅E, G, D)
    E, G, D = build_model()
    E.to(device)
    G.to(device)
    D.to(device)
    
    optimizer_E = optim.Adam(E.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
    
    num_epochs = 200
    
    print("\n开始训练 Model C (w/o Freq-D, Time-only TFGAN)...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        metrics = train_epoch(
            E, G, D, source_loader, target_loader, target_data_cache, target_labels_cache,
            optimizer_E, optimizer_G, optimizer_D,
            lambda_mmd=20.0, lambda_freq=5.0, n_critic=3, device=device
        )
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  D损失: {metrics['d_loss']:.4f} | G损失: {metrics['g_loss']:.4f}")
            print(f"  MMD: {metrics['mmd']:.6f} | 频域: {metrics['freq']:.6f}")
    
    print("\n训练完成!")
    print("=" * 70)
    
    # 评估模型
    print("\n评估 Model C...")
    comprehensive_metrics = evaluate_gan_comprehensive(E, G, source_loader, target_loader, device=device)
    
    print("\n" + "="*70)
    print("Model C (w/o Freq-D, Time-only TFGAN) 评估结果：")
    print("="*70)
    print(f"  ① FID (Fréchet Inception Distance)      : {comprehensive_metrics.get('fid', -1):.4f}")
    print(f"  ② IS (Inception Score)                  : {comprehensive_metrics.get('inception_score', -1):.4f}")
    print(f"  ③ 时域MSE (Time-domain MSE)             : {comprehensive_metrics.get('time_domain_mse', -1):.6f}")
    print(f"  ④ 频谱相关性系数 (Spectral Correlation)  : {comprehensive_metrics.get('spectral_correlation', -1):.4f}")
    print("="*70 + "\n")
    
    # 保存结果
    output_dir = 'Ablation Study/model_C'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'model_name': 'Model C (w/o Freq-D, Time-only TFGAN)',
        'description': '仅使用时域判别器的GAN',
        'metrics': comprehensive_metrics
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"结果已保存到 {output_dir}/evaluation_results.json")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
