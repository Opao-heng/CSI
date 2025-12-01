import torch
import os
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Baseline.DCGAN.model_DCGAN import build_dcgan_model
from Research1.Process.dataloder_GAN import CustomDataset, select_samples_by_label
from Research1.plot_GAN import (
    plot_training_metrics, 
    save_evaluation_results,
    plot_synthetic_sample_amplitude
)


"""
执行DCGAN模型的完整训练流程
"""
def train_and_test(model_path='dcgan_model.pth', epochs=100, lr_g=2e-4, lr_d=2e-4, num_samples=900, latent_dim=100):
    # 步骤1: 初始化设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 步骤2: 构建模型
    G, D = build_dcgan_model(latent_dim=latent_dim)
    G.to(device)
    D.to(device)
    
    # 步骤3: 打印模型参数统计
    total_params_G = sum(p.numel() for p in G.parameters())
    total_params_D = sum(p.numel() for p in D.parameters())
    total_params = total_params_G + total_params_D
    print(f"  DCGAN模型参数量总量: {total_params:,}")
    
    # 步骤4: 定义优化器和损失函数
    optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    # 步骤5: 初始化训练记录
    train_loss_history = []
    training_start_time = datetime.now()
    
    # 步骤6: 执行训练循环
    print(f"  正在训练DCGAN模型...")
    for epoch in range(epochs):
        # 步骤6.1: 执行单个epoch的训练
        loss_dict = train_epoch(
            G, D, target_loader, optimizer_G, optimizer_D, criterion,
            latent_dim=latent_dim, device=device
        )
        
        # 步骤6.2: 记录训练损失变化
        train_loss_history.append(loss_dict)
        
        # 步骤6.3: 打印训练进度
        elapsed_time = (datetime.now() - training_start_time).total_seconds() / 60
        print(f"  轮数 [{epoch + 1:3d}/{epochs}] | 耗时: {elapsed_time:.1f}分钟")
        print(f"  生成器损失: {loss_dict['g_loss']:.4f} | "
              f"判别器损失: {loss_dict['d_loss']:.4f}")
    
    # 步骤7: 保存训练完成后的模型
    print(f"  DCGAN模型训练完成")
    print("=" * 70 + "\n")
    print(f"  正在保存DCGAN模型...")
    torch.save({
        'generator': G.state_dict(),
        'discriminator': D.state_dict()
    }, model_path)
    
    # 步骤8: 生成合成样本用于数据增强
    print("=" * 70 + "\n")
    print(f"  正在生成合成样本用于数据增强...")
    synthetic_data, synthetic_labels = generate_synthetic_samples(
        G, source_loader, num_samples=num_samples, latent_dim=latent_dim, device=device
    )
    print(f"  已生成 {len(synthetic_data)} 个合成样本")
    
    # 步骤9: 绘制训练指标
    print(f"  正在绘制训练指标...")
    plot_training_metrics(train_loss_history, output_dir='Baseline/DCGAN')
    
    # 步骤10: 绘制生成样本的幅度图
    print(f"  正在绘制生成样本幅度图...")
    random_idx = np.random.randint(0, len(synthetic_data))
    plot_synthetic_sample_amplitude(synthetic_data, sample_idx=random_idx, output_dir='Baseline/DCGAN')
    
    # 步骤11: 保存评估结果
    evaluation_results = {
        'model_name': 'DCGAN',
        'total_params': total_params,
        'final_g_loss': train_loss_history[-1]['g_loss'],
        'final_d_loss': train_loss_history[-1]['d_loss']
    }
    save_evaluation_results(evaluation_results, train_loss_history, 'Baseline/DCGAN')
    
    return synthetic_data, synthetic_labels


"""
执行单个epoch的DCGAN训练
"""
def train_epoch(G, D, data_loader, optimizer_G, optimizer_D, criterion, latent_dim=100, device='cuda'):
    G.train()
    D.train()
    
    total_g_loss = 0.0
    total_d_loss = 0.0
    num_batches = 0
    
    for batch_idx, (real_data, _) in enumerate(data_loader):
        batch_size = real_data.size(0)
        real_data = real_data.to(device)
        
        # 真实和虚假标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # ================== 训练判别器 ==================
        optimizer_D.zero_grad()
        
        # 真实样本的判别损失
        real_output = D(real_data)
        d_loss_real = criterion(real_output, real_labels)
        
        # 生成假样本
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_data = G(z)
        
        # 假样本的判别损失
        fake_output = D(fake_data.detach())
        d_loss_fake = criterion(fake_output, fake_labels)
        
        # 总判别器损失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # ================== 训练生成器 ==================
        optimizer_G.zero_grad()
        
        # 生成器希望判别器判定假样本为真
        fake_output = D(fake_data)
        g_loss = criterion(fake_output, real_labels)
        
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        optimizer_G.step()
        
        # 累计损失
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        num_batches += 1
    
    loss_dict = {
        'g_loss': total_g_loss / num_batches if num_batches > 0 else 0,
        'd_loss': total_d_loss / num_batches if num_batches > 0 else 0,
    }
    return loss_dict


"""
使用训练的DCGAN生成指定数量的合成样本
"""
def generate_synthetic_samples(G, source_loader, num_samples=900, latent_dim=100, device='cuda'):
    G.eval()
    
    synthetic_data = []
    synthetic_labels = []
    generated_count = 0
    
    with torch.no_grad():
        for _, source_labels_batch in source_loader:
            if generated_count >= num_samples:
                break
            
            batch_size = min(len(source_labels_batch), num_samples - generated_count)
            
            # 从随机噪声生成样本
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_data = G(z)
            
            # 收集生成的样本和对应的标签
            synthetic_data.append(fake_data.cpu())
            synthetic_labels.append(source_labels_batch[:batch_size])
            
            generated_count += batch_size
    
    # 合并所有生成样本
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
    print("步骤3: 开始DCGAN训练...")
    synthetic_data, synthetic_labels = train_and_test(
        model_path='Baseline/DCGAN/dcgan_model.pth',
        epochs=50,
        lr_g=2e-4,
        lr_d=2e-4,
        num_samples=900,
        latent_dim=100
    )
    
    # 步骤4: 打印最终的数据统计结果
    print("\n" + "="*70)
    print("最终数据统计:")
    print(f"源域数据: {source_data.shape}")
    print(f"源域标签: {source_labels.shape}")
    print(f"生成的合成数据: {synthetic_data.shape}")
    print(f"生成的合成标签: {synthetic_labels.shape}")
    print("="*70)
