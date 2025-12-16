import torch
import os
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Baseline.VAE.model_VAE import build_vae_model
from Baseline.VAE.loss_VAE import combined_vae_loss
from Research1.Process.dataloder_GAN import CustomDataset, select_samples_by_label
from Research1.plot_GAN import (
    plot_training_metrics, 
    save_evaluation_results,
    plot_synthetic_sample_amplitude,
    evaluate_gan_comprehensive
)


"""
执行VAE模型的完整训练流程
"""
def train_and_test(model_path='vae_model.pth', epochs=100, lr=1e-3, num_samples=900, beta=1.0, lambda_freq=0.1):
    # 步骤1: 初始化设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 步骤2: 构建模型
    model = build_vae_model()
    model.to(device)
    
    # 步骤3: 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  VAE模型参数量总量: {total_params:,}")
    
    # 步骤4: 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # 步骤5: 初始化训练记录
    train_loss_history = []
    training_start_time = datetime.now()
    
    # 步骤6: 执行训练循环
    print(f"  正在训练VAE模型...")
    for epoch in range(epochs):
        # 步骤6.1: 执行单个epoch的训练
        loss_dict = train_epoch(
            model, target_loader, optimizer,
            beta=beta, lambda_freq=lambda_freq,
            device=device
        )
        
        # 步骤6.2: 记录训练损失变化
        train_loss_history.append(loss_dict)
        
        # 步骤6.3: 打印训练进度
        elapsed_time = (datetime.now() - training_start_time).total_seconds() / 60
        print(f"  轮数 [{epoch + 1:3d}/{epochs}] | 耗时: {elapsed_time:.1f}分钟")
        print(f"  总损失: {loss_dict['total_loss']:.4f} | "
              f"重构损失: {loss_dict['recon_loss']:.4f} | "
              f"KL损失: {loss_dict['kl_loss']:.4f} | "
              f"频域损失: {loss_dict['freq_loss']:.4f}")
    
    # 步骤7: 保存训练完成后的模型
    print(f"  VAE模型训练完成")
    print("=" * 70 + "\n")
    print(f"  正在保存VAE模型...")
    torch.save(model.state_dict(), model_path)
    
    # 步骤8: 生成合成样本用于数据增强
    print("=" * 70 + "\n")
    print(f"  正在生成合成样本用于数据增强...")
    synthetic_data, synthetic_labels = generate_synthetic_samples(
        model, source_loader, num_samples=num_samples, device=device
    )
    print(f"  已生成 {len(synthetic_data)} 个合成样本")
    
    # 步骤9: 绘制训练指标
    print(f"  正在绘制训练指标...")
    plot_training_metrics(train_loss_history, output_dir='Baseline/VAE')
    
    # 步骤10: 绘制生成样本的幅度图
    print(f"  正在绘制生成样本幅度图...")
    random_idx = np.random.randint(0, len(synthetic_data))
    plot_synthetic_sample_amplitude(synthetic_data, sample_idx=random_idx, output_dir='Baseline/VAE')
    
    # 步骤11: 保存评估结果
    evaluation_results = {
        'model_name': 'VAE',
        'total_params': total_params,
        'final_total_loss': train_loss_history[-1]['total_loss'],
        'final_recon_loss': train_loss_history[-1]['recon_loss'],
        'final_kl_loss': train_loss_history[-1]['kl_loss'],
        'final_freq_loss': train_loss_history[-1]['freq_loss']
    }
    save_evaluation_results(evaluation_results, train_loss_history, 'Baseline/VAE')
    
    return synthetic_data, synthetic_labels


"""
执行单个epoch的VAE训练
"""
def train_epoch(model, data_loader, optimizer, beta=1.0, lambda_freq=0.1, device='cuda'):
    model.train()
    
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    freq_loss_sum = 0.0
    num_batches = 0
    
    for batch_idx, (x, _) in enumerate(data_loader):
        x = x.to(device)
        
        # 前向传播
        recon_x, mu, logvar = model(x)
        
        # 计算损失
        loss, recon_loss, kl_loss, freq_loss = combined_vae_loss(
            recon_x, x, mu, logvar, beta=beta, lambda_freq=lambda_freq
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 累计损失
        total_loss_sum += loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        freq_loss_sum += freq_loss.item()
        num_batches += 1
    
    loss_dict = {
        'total_loss': total_loss_sum / num_batches if num_batches > 0 else 0,
        'recon_loss': recon_loss_sum / num_batches if num_batches > 0 else 0,
        'kl_loss': kl_loss_sum / num_batches if num_batches > 0 else 0,
        'freq_loss': freq_loss_sum / num_batches if num_batches > 0 else 0,
    }
    return loss_dict


"""
使用训练的VAE生成指定数量的合成样本
"""
def generate_synthetic_samples(model, source_loader, num_samples=900, device='cuda'):
    model.eval()
    
    synthetic_data = []
    synthetic_labels = []
    generated_count = 0
    
    with torch.no_grad():
        # 从源域数据的分布中生成样本
        for x_s, source_labels in source_loader:
            if generated_count >= num_samples:
                break
            
            x_s = x_s.to(device)
            
            # 编码到潜在空间
            mu, logvar = model.encoder(x_s)
            
            # 从潜在分布采样
            z = model.reparameterize(mu, logvar)
            
            # 解码生成样本
            generated = model.decoder(z)
            
            # 收集生成的样本和对应的标签
            synthetic_data.append(generated.cpu())
            synthetic_labels.append(source_labels[:generated.size(0)])
            
            generated_count += generated.size(0)
    
    # 合并所有生成样本并截断至指定数量
    if synthetic_data:
        synthetic_data = torch.cat(synthetic_data, dim=0)[:num_samples]
        synthetic_labels = torch.cat(synthetic_labels, dim=0)[:num_samples]
    
    return synthetic_data, synthetic_labels


if __name__ == "__main__":
    # 步骤0: 设置随机种子以确保结果可重现
    print("步骤0: 设置随机种子以确保结果可重现...")
    torch.manual_seed(30)
    np.random.seed(30)
    torch.backends.cudnn.benchmark = True
    print("  随机种子设置成功\n")
    
    # 步骤1: 从磁盘加载源域和目标域数据
    print("步骤1: 正在加载数据文件...")
    source_data = torch.load('Data/source_env0_env1_data.pt')
    source_labels = torch.load('Data/source_env0_env1_labels.pt')
    target_data = torch.load('Data/target_env2_data.pt')
    target_labels = torch.load('Data/target_env2_labels.pt')
    print("  数据文件加载成功\n")
    
    # 步骤2: 创建数据加载器
    print("步骤2: 正在创建数据加载器...")
    source_dataset = CustomDataset(source_data, source_labels)
    source_loader = DataLoader(source_dataset, batch_size=100, shuffle=True)
    
    # 从目标域中均匀采样每个标签的样本
    selected_target_data, selected_target_labels = select_samples_by_label(
        target_data, target_labels, samples_per_label=10
    )
    
    print(f"  源域数据: {source_data.shape}")
    print(f"  源域标签: {source_labels.shape}")
    print(f"  目标域数据(已选): {selected_target_data.shape}")
    print(f"  目标域标签(已选): {selected_target_labels.shape}\n")
    
    target_dataset = CustomDataset(selected_target_data, selected_target_labels)
    target_loader = DataLoader(target_dataset, batch_size=100, shuffle=True)
    
    # 步骤3: 执行主训练流程
    print("步骤3: 开始VAE训练...")
    synthetic_data, synthetic_labels = train_and_test(
        model_path='Baseline/VAE/vae_model.pth',
        epochs=50,
        lr=2e-4,
        num_samples=900,
        beta=0.5,
        lambda_freq=0.1
    )
    
    # 步骤4: 打印最终的数据统计结果
    print("\n" + "="*70)
    print("最终数据统计:")
    print(f"源域数据: {source_data.shape}")
    print(f"源域标签: {source_labels.shape}")
    print(f"生成的合成数据: {synthetic_data.shape}")
    print(f"生成的合成标签: {synthetic_labels.shape}")
    print("="*70)
