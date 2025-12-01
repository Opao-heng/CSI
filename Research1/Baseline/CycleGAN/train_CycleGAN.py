import torch
import os
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Baseline.CycleGAN.model_CycleGAN import build_cyclegan_model
from Baseline.CycleGAN.loss_CycleGAN import adversarial_loss, cycle_consistency_loss, identity_loss
from Research1.Process.dataloder_GAN import CustomDataset, select_samples_by_label
from Research1.plot_GAN import (
    plot_training_metrics, 
    save_evaluation_results,
    plot_synthetic_sample_amplitude
)


"""
执行CycleGAN模型的完整训练流程
"""
def train_and_test(model_path='cyclegan_model.pth', epochs=100, lr=2e-4, num_samples=900, lambda_cycle=10.0, lambda_identity=5.0):
    # 步骤1: 初始化设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 步骤2: 构建模型
    G_S2T, G_T2S, D_S, D_T = build_cyclegan_model()
    G_S2T.to(device)
    G_T2S.to(device)
    D_S.to(device)
    D_T.to(device)
    
    # 步骤3: 打印模型参数统计
    total_params_G_S2T = sum(p.numel() for p in G_S2T.parameters())
    total_params_G_T2S = sum(p.numel() for p in G_T2S.parameters())
    total_params_D_S = sum(p.numel() for p in D_S.parameters())
    total_params_D_T = sum(p.numel() for p in D_T.parameters())
    total_params = total_params_G_S2T + total_params_G_T2S + total_params_D_S + total_params_D_T
    print(f"  CycleGAN模型参数量总量: {total_params:,}")
    
    # 步骤4: 定义优化器
    optimizer_G = optim.Adam(list(G_S2T.parameters()) + list(G_T2S.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D_S = optim.Adam(D_S.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_T = optim.Adam(D_T.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # 步骤5: 初始化训练记录
    train_loss_history = []
    training_start_time = datetime.now()
    
    # 步骤6: 执行训练循环
    print(f"  正在训练CycleGAN模型...")
    for epoch in range(epochs):
        # 步骤6.1: 执行单个epoch的训练
        loss_dict = train_epoch(
            G_S2T, G_T2S, D_S, D_T,
            source_loader, target_loader,
            optimizer_G, optimizer_D_S, optimizer_D_T,
            lambda_cycle=lambda_cycle,
            lambda_identity=lambda_identity,
            device=device
        )
        
        # 步骤6.2: 记录训练损失变化
        train_loss_history.append(loss_dict)
        
        # 步骤6.3: 打印训练进度
        elapsed_time = (datetime.now() - training_start_time).total_seconds() / 60
        print(f"  轮数 [{epoch + 1:3d}/{epochs}] | 耗时: {elapsed_time:.1f}分钟")
        print(f"  生成器损失: {loss_dict['g_loss']:.4f} | "
              f"判别器损失: {loss_dict['d_loss']:.4f} | "
              f"循环损失: {loss_dict['cycle_loss']:.4f}")
    
    # 步骤7: 保存训练完成后的模型
    print(f"  CycleGAN模型训练完成")
    print("=" * 70 + "\n")
    print(f"  正在保存CycleGAN模型...")
    torch.save({
        'G_S2T': G_S2T.state_dict(),
        'G_T2S': G_T2S.state_dict(),
        'D_S': D_S.state_dict(),
        'D_T': D_T.state_dict()
    }, model_path)
    
    # 步骤8: 生成合成样本用于数据增强
    print("=" * 70 + "\n")
    print(f"  正在生成合成样本用于数据增强...")
    synthetic_data, synthetic_labels = generate_synthetic_samples(
        G_S2T, source_loader, num_samples=num_samples, device=device
    )
    print(f"  已生成 {len(synthetic_data)} 个合成样本")
    
    # 步骤9: 绘制训练指标
    print(f"  正在绘制训练指标...")
    plot_training_metrics(train_loss_history, output_dir='Baseline/CycleGAN')
    
    # 步骤10: 绘制生成样本的幅度图
    print(f"  正在绘制生成样本幅度图...")
    random_idx = np.random.randint(0, len(synthetic_data))
    plot_synthetic_sample_amplitude(synthetic_data, sample_idx=random_idx, output_dir='Baseline/CycleGAN')
    
    # 步骤11: 保存评估结果
    evaluation_results = {
        'model_name': 'CycleGAN',
        'total_params': total_params,
        'final_g_loss': train_loss_history[-1]['g_loss'],
        'final_d_loss': train_loss_history[-1]['d_loss'],
        'final_cycle_loss': train_loss_history[-1]['cycle_loss']
    }
    save_evaluation_results(evaluation_results, train_loss_history, 'Baseline/CycleGAN')
    
    return synthetic_data, synthetic_labels


"""
执行单个epoch的CycleGAN训练
"""
def train_epoch(G_S2T, G_T2S, D_S, D_T, source_loader, target_loader,
                optimizer_G, optimizer_D_S, optimizer_D_T,
                lambda_cycle=10.0, lambda_identity=5.0, device='cuda'):
    G_S2T.train()
    G_T2S.train()
    D_S.train()
    D_T.train()
    
    total_g_loss = 0.0
    total_d_loss = 0.0
    total_cycle_loss = 0.0
    num_batches = 0
    
    # 将两个数据加载器配对
    for (x_s, _), (x_t, _) in zip(source_loader, target_loader):
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        
        # ================== 训练生成器 ==================
        optimizer_G.zero_grad()
        
        # 生成假样本
        fake_t = G_S2T(x_s)  # 源域 -> 目标域
        fake_s = G_T2S(x_t)  # 目标域 -> 源域
        
        # 对抗损失
        loss_G_S2T = adversarial_loss(D_T(fake_t), True)
        loss_G_T2S = adversarial_loss(D_S(fake_s), True)
        
        # 循环一致性损失
        reconstructed_s = G_T2S(fake_t)  # 源域 -> 目标域 -> 源域
        reconstructed_t = G_S2T(fake_s)  # 目标域 -> 源域 -> 目标域
        loss_cycle_s = cycle_consistency_loss(x_s, reconstructed_s)
        loss_cycle_t = cycle_consistency_loss(x_t, reconstructed_t)
        
        # 身份保持损失
        same_s = G_T2S(x_s)  # 源域 -> 源域（应该不变）
        same_t = G_S2T(x_t)  # 目标域 -> 目标域（应该不变）
        loss_identity_s = identity_loss(x_s, same_s)
        loss_identity_t = identity_loss(x_t, same_t)
        
        # 总生成器损失
        loss_G = (loss_G_S2T + loss_G_T2S +
                  lambda_cycle * (loss_cycle_s + loss_cycle_t) +
                  lambda_identity * (loss_identity_s + loss_identity_t))
        
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(list(G_S2T.parameters()) + list(G_T2S.parameters()), max_norm=1.0)
        optimizer_G.step()
        
        # ================== 训练判别器 ==================
        # 训练D_T
        optimizer_D_T.zero_grad()
        loss_D_real_t = adversarial_loss(D_T(x_t), True)
        loss_D_fake_t = adversarial_loss(D_T(fake_t.detach()), False)
        loss_D_T = (loss_D_real_t + loss_D_fake_t) * 0.5
        loss_D_T.backward()
        optimizer_D_T.step()
        
        # 训练D_S
        optimizer_D_S.zero_grad()
        loss_D_real_s = adversarial_loss(D_S(x_s), True)
        loss_D_fake_s = adversarial_loss(D_S(fake_s.detach()), False)
        loss_D_S = (loss_D_real_s + loss_D_fake_s) * 0.5
        loss_D_S.backward()
        optimizer_D_S.step()
        
        # 累计损失
        total_g_loss += loss_G.item()
        total_d_loss += (loss_D_T.item() + loss_D_S.item())
        total_cycle_loss += (loss_cycle_s.item() + loss_cycle_t.item())
        num_batches += 1
    
    loss_dict = {
        'g_loss': total_g_loss / num_batches if num_batches > 0 else 0,
        'd_loss': total_d_loss / num_batches if num_batches > 0 else 0,
        'cycle_loss': total_cycle_loss / num_batches if num_batches > 0 else 0,
    }
    return loss_dict


"""
使用训练的CycleGAN生成指定数量的合成样本
"""
def generate_synthetic_samples(G_S2T, source_loader, num_samples=900, device='cuda'):
    G_S2T.eval()
    
    synthetic_data = []
    synthetic_labels = []
    generated_count = 0
    
    with torch.no_grad():
        for x_s, source_labels_batch in source_loader:
            if generated_count >= num_samples:
                break
            
            x_s = x_s.to(device)
            
            # 生成目标域样本
            fake_t = G_S2T(x_s)
            
            # 收集生成的样本和对应的标签
            synthetic_data.append(fake_t.cpu())
            synthetic_labels.append(source_labels_batch[:fake_t.size(0)])
            
            generated_count += fake_t.size(0)
    
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
    print("步骤3: 开始CycleGAN训练...")
    synthetic_data, synthetic_labels = train_and_test(
        model_path='Baseline/CycleGAN/cyclegan_model.pth',
        epochs=50,
        lr=2e-4,
        num_samples=900,
        lambda_cycle=10.0,
        lambda_identity=5.0
    )
    
    # 步骤4: 打印最终的数据统计结果
    print("\n" + "="*70)
    print("最终数据统计:")
    print(f"源域数据: {source_data.shape}")
    print(f"源域标签: {source_labels.shape}")
    print(f"生成的合成数据: {synthetic_data.shape}")
    print(f"生成的合成标签: {synthetic_labels.shape}")
    print("="*70)
