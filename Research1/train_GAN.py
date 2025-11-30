import torch
import os
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 导入必要的模块
from Research1.model_GAN import build_model
from Research1.dataloder_GAN import CustomDataset, select_samples_by_label
from Research1.loss_GAN import (
    mmd_loss,
    frequency_consistency_loss,
    wasserstein_discriminator_loss,
    wasserstein_generator_loss
)
from Research1.plot_GAN import plot_training_metrics
from Research1.evaluator_GAN import evaluate_gan_comprehensive


"""
训练一个epoch，采用WGAN-GP对抗训练机制
参数:
  E - 特征提取器
  G - 生成器
  D - 判别器
  source_loader - 源域数据加载器
  target_loader - 目标域数据加载器
  optimizer_E - 特征提取器优化器
  optimizer_G - 生成器优化器
  optimizer_D - 判别器优化器
  lambda_mmd - MMD损失权重，默认5.0
  lambda_feat - 特征匹配损失权重，默认10.0
  lambda_freq - 频域约束损失权重，默认2.0
  lambda_gp - 梯度惩罚权重，默认10.0
  n_critic - 判别器更新次数/生成器更新次数，默认5
  device - 设备类型，默认cuda
返回: tuple - (train_loss_dict, eval_metrics) 训练损失字典和评估指标字典
"""
def train_epoch(E, G, D, D_spec, D_patch, source_loader, target_features, target_data, optimizer_E, optimizer_G, optimizer_D,
                optimizer_D_spec, optimizer_D_patch, lambda_mmd=5.0, lambda_feat=10.0, lambda_freq=2.0, lambda_gp=10.0, 
                n_critic=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 步骤1: 设置模型为训练模式
    E.train()
    G.train()
    D.train()

    # 损失记录
    total_g_loss = 0.0
    total_d_loss = 0.0
    total_mmd_loss = 0.0
    total_feat_loss = 0.0
    total_freq_loss = 0.0
    total_gp_loss = 0.0
    num_batches = 0
    
    # 使用预计算的目标域特征和数据（不再重复提取）

    # 步骤2: 遍历源域数据批次
    critic_iter = 0
    for batch_idx, (x_s, source_labels) in enumerate(source_loader):
        if x_s.size(0) < 2:  # 至少需要2个样本
            continue

        x_s = x_s.to(device)
        batch_size = x_s.size(0)
        
        # 随机采样目标域真实数据
        rand_idx = torch.randperm(target_data.size(0))[:batch_size]
        x_t_real = target_data[rand_idx].to(device)

        # 步骤3: 更新判别器D（每n_critic次更新一次生成器）
        for _ in range(n_critic):
            optimizer_D.zero_grad()
            optimizer_D_spec.zero_grad()
            optimizer_D_patch.zero_grad()
            
            # 生成虚假样本
            with torch.no_grad():
                x_hat_t = G(x_s, target_features)
            
            # 计算三个判别器的 WGAN-GP 损失
            d_loss_main, w_main, gp_main = wasserstein_discriminator_loss(
                D, x_t_real, x_hat_t, lambda_gp=lambda_gp, device=device
            )
            d_loss_spec, w_spec, gp_spec = wasserstein_discriminator_loss(
                D_spec, x_t_real, x_hat_t, lambda_gp=lambda_gp, device=device
            )
            d_loss_patch, w_patch, gp_patch = wasserstein_discriminator_loss(
                D_patch, x_t_real, x_hat_t, lambda_gp=lambda_gp, device=device
            )
            
            d_loss = (d_loss_main + d_loss_spec + d_loss_patch) / 3.0
            d_loss.backward()
            optimizer_D.step()
            optimizer_D_spec.step()
            optimizer_D_patch.step()
            
            total_d_loss += d_loss.item()
            total_gp_loss += ((gp_main + gp_spec + gp_patch) / 3.0).item()
            critic_iter += 1

        # 步骤4: 更新生成器G和特征提取器E
        optimizer_G.zero_grad()
        optimizer_E.zero_grad()
        
        # 提取源域特征（用于感知损失）
        source_features = E(x_s)
        
        # 生成虚假样本
        x_hat_t = G(x_s, target_features)
        
        # 计算各项损失
        # 1. WGAN生成器损失（对抗损失，与多判别器平均）
        g_adv_main = wasserstein_generator_loss(D, x_hat_t)
        g_adv_spec = wasserstein_generator_loss(D_spec, x_hat_t)
        g_adv_patch = wasserstein_generator_loss(D_patch, x_hat_t)
        g_adv_loss = (g_adv_main + g_adv_spec + g_adv_patch) / 3.0
        
        # 2. MMD损失（分布对齐）
        generated_features = E(x_hat_t)
        mmd_loss_value = mmd_loss(target_features, generated_features)
        
        # 3. 简化的特征匹配损失（仅使用MSE）
        feat_loss = torch.nn.functional.mse_loss(generated_features, source_features, reduction='mean')
        
        # 4. 频域一致性损失（新加模块）
        rand_idx_freq = torch.randperm(target_data.size(0))[:batch_size]
        x_t_real_freq = target_data[rand_idx_freq].to(device)
        freq_loss = frequency_consistency_loss(x_t_real_freq, x_hat_t)
        
        # 总生成器损失（加入频域损失）
        g_total_loss = (g_adv_loss + 
                        lambda_mmd * mmd_loss_value + 
                        lambda_feat * feat_loss +
                        lambda_freq * freq_loss)
        
        g_total_loss.backward()
        optimizer_G.step()
        optimizer_E.step()
        
        # 记录损失
        total_g_loss += g_total_loss.item()
        total_mmd_loss += mmd_loss_value.item()
        total_feat_loss += feat_loss.item()
        total_freq_loss += freq_loss.item()  # 记录频域损失
        num_batches += 1

    # 步骤5: 简化评估（不再每个epoch都进行完整评估）
    eval_metrics = {}
    
    # 返回平均损失
    loss_dict = {
        'g_loss': total_g_loss / num_batches if num_batches > 0 else 0,
        'd_loss': total_d_loss / critic_iter if critic_iter > 0 else 0,
        'mmd_loss': total_mmd_loss / num_batches if num_batches > 0 else 0,
        'feat_loss': total_feat_loss / num_batches if num_batches > 0 else 0,
        'freq_loss': total_freq_loss / num_batches if num_batches > 0 else 0,
        'gp_loss': total_gp_loss / critic_iter if critic_iter > 0 else 0
    }
    
    return loss_dict, eval_metrics


"""  
【已优化】简化的评估函数，仅在需要时调用
参数:
  E - 特征提取器
  G - 生成器
  D - 判别器
  source_loader - 源域数据加载器
  target_data - 目标域真实数据
  target_features - 目标域特征张量
  device - 设备类型，默认cuda
返回: dict - 包含核心评估指标的字典
"""
def evaluate_generated_samples_fast(E, G, D, source_loader, target_data, target_features, device='cuda'):
    E.eval()
    G.eval()
    D.eval()
    
    metrics = {}
    with torch.no_grad():
        # 仅对第一个batch进行快速评估
        x_s, _ = next(iter(source_loader))
        x_s = x_s.to(device)
        
        # 生成样本
        x_hat_t = G(x_s, target_features)
        
        # 判别器评分
        metrics['avg_fake_score'] = D(x_hat_t).mean().item()
        
        # 随机采样真实样本评分
        rand_idx = torch.randperm(target_data.size(0))[:x_s.size(0)]
        x_t_real = target_data[rand_idx].to(device)
        metrics['avg_real_score'] = D(x_t_real).mean().item()
        
        # 特征MSE
        source_feat = E(x_s)
        gen_feat = E(x_hat_t)
        metrics['feature_mse'] = torch.mean((source_feat - gen_feat) ** 2).item()
        # 频域 MSE
        metrics['spectral_mse'] = frequency_consistency_loss(x_t_real, x_hat_t).item()
    
    return metrics


"""
生成虚假目标域样本用于数据扩充
参数:
  E - 特征提取器
  G - 生成器
  source_loader - 源域数据加载器
  target_loader - 目标域数据加载器（用于提取目标域特征）
  num_samples - 需要生成的样本数量，默认900
  device - 设备类型，默认cuda
返回: tuple - (synthetic_data, synthetic_labels) 生成的虚假样本数据和标签
"""
def generate_synthetic_samples(E, G, source_loader, target_loader, num_samples=900, device='cuda'):
    # 步骤1: 设置模型为评估模式
    E.eval()
    G.eval()

    synthetic_data = []
    synthetic_labels = []
    generated_count = 0
    
    # 预先提取目标域特征
    all_features = []
    with torch.no_grad():
        for x_t_real, _ in target_loader:
            x_t_real = x_t_real.to(device)
            features = E(x_t_real)
            all_features.append(features)
    target_features = torch.cat(all_features, dim=0).to(device)

    with torch.no_grad():
        # 步骤2: 使用源域数据和目标域特征生成虚假样本
        for x_s, source_labels in source_loader:
            if generated_count >= num_samples:
                break

            x_s = x_s.to(device)
            batch_size = x_s.size(0)

            # 步骤3: 生成虚假样本
            x_hat_t = G(x_s, target_features)

            # 步骤4: 收集生成的样本和对应的标签
            synthetic_data.append(x_hat_t.cpu())
            synthetic_labels.append(source_labels[:x_hat_t.size(0)])

            generated_count += x_hat_t.size(0)

    # 步骤5: 合并所有生成样本并截断至指定数量
    if synthetic_data:
        synthetic_data = torch.cat(synthetic_data, dim=0)[:num_samples]
        synthetic_labels = torch.cat(synthetic_labels, dim=0)[:num_samples]

    return synthetic_data, synthetic_labels


"""
将生成的虚假样本与原始目标域数据合并，并保存为.pt文件
参数:
  synthetic_data - 生成的虚假样本张量，形状为 (N, C, S, T)
  synthetic_labels - 生成的虚假样本标签，形状为 (N,)
  target_loader - 原始目标域数据加载器
  output_dir - 输出目录路径
返回: tuple - (combined_data, combined_labels) 合并后的数据和标签张量
"""
def save_combined_target_data(synthetic_data, synthetic_labels, target_loader, output_dir):
    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 步骤2: 提取原始目标域数据和标签
    target_data_list = []
    target_label_list = []

    for x_t_real, labels in target_loader:
        target_data_list.append(x_t_real)
        target_label_list.append(labels)

    # 步骤3: 合并所有原始目标域数据
    target_data = torch.cat(target_data_list, dim=0)  # (M, C, S, T)
    target_labels = torch.cat(target_label_list, dim=0)  # (M,)

    # 步骤4: 将生成数据和原始数据拼接
    combined_data = torch.cat([target_data, synthetic_data], dim=0)  # (M+N, C, S, T)
    combined_labels = torch.cat([target_labels, synthetic_labels], dim=0)  # (M+N,)

    # 步骤5: 保存合并后的数据到文件
    target_data_path = os.path.join(output_dir, 'target_env2_gan_data.pt')
    target_labels_path = os.path.join(output_dir, 'target_env2_gan_labels.pt')

    torch.save(combined_data, target_data_path)
    torch.save(combined_labels, target_labels_path)

    return combined_data, combined_labels


"""
主训练函数，完整的GAN模型训练和评估流程（使用WGAN-GP）
参数:
  model_path - 模型保存路径，默认'model.pth'
  epochs - 训练轮数，默认100
  lr_g - 生成器和特征提取器学习率，默认1e-4
  lr_d - 判别器学习率，默认4e-4
  num_samples - 需要生成的合成样本数量，默认900
返回: tuple - (E, G, D, synthetic_data, synthetic_labels) 训练完的模型和生成的合成数据
"""
def train_and_test(model_path='model.pth', epochs=100, lr_g=1e-4, lr_d=4e-4, num_samples=900):
    # 步骤1: 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Optimized GAN with WGAN-GP, MMD Loss, Perceptual Loss, and Frequency Constraint")

    # 步骤2: 构建模型
    E, G, D, D_spec, D_patch = build_model()
    E.to(device)
    G.to(device)
    D.to(device)
    D_spec.to(device)
    D_patch.to(device)
    
    # 打印模型参数量
    total_params_E = sum(p.numel() for p in E.parameters())
    total_params_G = sum(p.numel() for p in G.parameters())
    total_params_D = sum(p.numel() for p in D.parameters())
    print(f"Model Parameters - E: {total_params_E:,}, G: {total_params_G:,}, D: {total_params_D:,}")

    # 步骤3: 定义优化器（判别器学习率略高）
    optimizer_E = optim.Adam(E.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optimizer_D_spec = optim.Adam(D_spec.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optimizer_D_patch = optim.Adam(D_patch.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # 步骤4: 初始化训练记录
    train_loss_history = []
    evaluation_metrics = []

    # 预先提取目标域特征（只提取一次，避免重复计算）
    print("提取目标域特征...")
    E.eval()
    all_target_features = []
    all_target_data = []
    with torch.no_grad():
        for x_t_real, _ in target_loader:
            x_t_real = x_t_real.to(device)
            features = E(x_t_real)
            all_target_features.append(features)
            all_target_data.append(x_t_real)
    target_features_cache = torch.cat(all_target_features, dim=0)
    target_data_cache = torch.cat(all_target_data, dim=0)
    E.train()
    print(f"目标域特征提取完成，shape: {target_features_cache.shape}")
    
    # 步骤5: 训练循环
    for epoch in range(epochs):
        # 训练一个epoch（传入预计算的特征）
        loss_dict, eval_metrics = train_epoch(
            E, G, D, D_spec, D_patch, source_loader, target_features_cache, target_data_cache,
            optimizer_E, optimizer_G, optimizer_D, optimizer_D_spec, optimizer_D_patch,
            lambda_mmd=5.0, lambda_feat=10.0, lambda_freq=1.0, 
            lambda_gp=10.0, n_critic=3, device=device
        )

        # 记录训练结果
        train_loss_history.append(loss_dict)
        evaluation_metrics.append(eval_metrics)

        # 打印进度（移除不必要的评估指标输出）
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  G_Loss: {loss_dict['g_loss']:.4f} | D_Loss: {loss_dict['d_loss']:.4f} | MMD: {loss_dict['mmd_loss']:.4f} | Feat: {loss_dict['feat_loss']:.4f} | GP: {loss_dict['gp_loss']:.4f}")
        
        # 每5个epoch进行一次快速评估
        if (epoch + 1) % 5 == 0:
            eval_metrics = evaluate_generated_samples_fast(
                E, G, D, source_loader, target_data_cache, target_features_cache, device
            )
            print(f"  [Eval] Fake: {eval_metrics.get('avg_fake_score', 0):.4f} | Real: {eval_metrics.get('avg_real_score', 0):.4f} | MSE: {eval_metrics.get('feature_mse', 0):.4f} | Spec: {eval_metrics.get('spectral_mse', 0):.4f}")

        # 步骤6: 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'E_state_dict': E.state_dict(),
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'D_spec_state_dict': D_spec.state_dict(),
                'D_patch_state_dict': D_patch.state_dict(),
                'optimizer_E_state_dict': optimizer_E.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'optimizer_D_spec_state_dict': optimizer_D_spec.state_dict(),
                'optimizer_D_patch_state_dict': optimizer_D_patch.state_dict(),
                'train_loss_history': train_loss_history,
                'evaluation_metrics': evaluation_metrics
            }, model_path)
            print(f"  Model checkpoint saved at {model_path}")

    # 步骤7: 训练完成后生成虚假样本
    print("\nGenerating synthetic samples for data augmentation...")
    synthetic_data, synthetic_labels = generate_synthetic_samples(
        E, G, source_loader, target_loader, num_samples=num_samples, device=device
    )
    print(f"Generated {len(synthetic_data)} synthetic samples")

    # 步骤8: 调用绘图模块绘制训练指标
    # 提取生成器损失作为主要损失曲线
    train_losses_for_plot = [loss['g_loss'] for loss in train_loss_history]
    plot_training_metrics(train_losses_for_plot, evaluation_metrics, output_dir='GAN')

    print("\nPerforming comprehensive GAN evaluation...")
    comprehensive_metrics = evaluate_gan_comprehensive(E, G, source_loader, target_loader, device=device)
    print("\nComprehensive GAN Evaluation Results:")
    print(f"  FID Score: {comprehensive_metrics.get('fid', -1):.4f}")
    print(f"  Spectral Fidelity: {comprehensive_metrics.get('spectral_fidelity', -1):.4f}")

    return synthetic_data, synthetic_labels


"""
主程序入口：GAN模型训练和数据生成
1. 设置随机种子保证可重现性
2. 加载源域和目标域数据
3. 创建数据加载器
4. 执行训练
5. 生成合成数据并保存
"""
if __name__ == "__main__":
    # 步骤1: 设置随机种子以保证结果可重现
    torch.manual_seed(40)
    np.random.seed(40)
    torch.backends.cudnn.benchmark = True

    # 步骤2: 加载数据文件
    source_data = torch.load('Data/source_env0_env1_data.pt')
    source_labels = torch.load('Data/source_env0_env1_labels.pt')
    target_data = torch.load('Data/target_env2_data.pt')
    target_labels = torch.load('Data/target_env2_labels.pt')

    # 步骤3: 创建源域数据集和数据加载器
    source_dataset = CustomDataset(source_data, source_labels)
    source_loader = DataLoader(source_dataset, batch_size=100, shuffle=True)

    # 步骤4: 从目标域数据中选择每个标签10个样本
    selected_target_data, selected_target_labels = select_samples_by_label(
        target_data, target_labels, samples_per_label=10
    )

    # 查看数据形状
    print("---------------------------------------------------------------------")
    print(f"Source data shape: {source_data.shape}")
    print(f"Source labels shape: {source_labels.shape}")
    print(f"Selected target data shape: {selected_target_data.shape}")
    print(f"Selected target labels shape: {selected_target_labels.shape}")

    # 步骤5: 创建目标域数据集和数据加载器
    target_dataset = CustomDataset(selected_target_data, selected_target_labels)
    target_loader = DataLoader(target_dataset, batch_size=100, shuffle=True)

    # 步骤6: 开始训练（使用优化后的参数）
    print("\n" + "="*70)
    print("Starting GAN Training with Optimized Architecture and Loss Functions")
    print("="*70)
    synthetic_data, synthetic_labels = train_and_test(
        model_path='GAN/best_gan_model.pth', 
        epochs=10,  # 可根据需要调整
        lr_g=1e-4,  # 生成器学习率
        lr_d=4e-4,  # 判别器学习率
        num_samples=900
    )

    # 步骤8: 合并并保存生成的数据
    output_dir = 'Data'
    target_gan_data, target_gan_labels = save_combined_target_data(synthetic_data, synthetic_labels, target_loader, output_dir)

    # 查看数据形状
    print(f"Merge Generated target data shape: {target_gan_data.shape}")
    print(f"Merge Generated target labels shape: {target_gan_labels.shape}")
    print("---------------------------------------------------------------------")

    # 用于ATTENTION的数据
    print(f"Source data shape: {source_data.shape}")
    print(f"Source labels shape: {source_labels.shape}")
    print(f"target data shape: {target_gan_data.shape}")
    print(f"target labels shape: {target_gan_labels.shape}")