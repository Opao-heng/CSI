import torch
import os
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from Research1.model_GAN import build_model
from Research1.dataloder_GAN import CustomDataset, select_samples_by_label
from Research1.loss_GAN import (
    mmd_loss,
    frequency_consistency_loss,
    wasserstein_discriminator_loss,
    wasserstein_generator_loss
)
from Research1.plot_GAN import plot_training_metrics, save_evaluation_results
from Research1.evaluator_GAN import evaluate_gan_comprehensive

"""
执行完整的GAN模型训练和评估流程
"""
def train_and_test(model_path='model.pth', epochs=100, lr_g=1e-4, lr_d=4e-4, num_samples=900):
    # 步骤1: 初始化设备和打印配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n" + "=" * 70)
    print("GAN 训练配置")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"训练轮数: {epochs}")
    print(f"生成器学习率: {lr_g}, 判别器学习率: {lr_d}")
    print(f"GAN生成样本数量：" + str(num_samples))
    print("=" * 70 + "\n")

    # 步骤2: 构建模型
    E, G, D, D_spec, D_patch = build_model()
    E.to(device)
    G.to(device)
    D.to(device)
    D_spec.to(device)
    D_patch.to(device)

    # 步骤3: 打印模型参数统计
    total_params_E = sum(p.numel() for p in E.parameters())
    total_params_G = sum(p.numel() for p in G.parameters())
    total_params_D = sum(p.numel() for p in D.parameters())
    total_params_D_spec = sum(p.numel() for p in D_spec.parameters())
    total_params_D_patch = sum(p.numel() for p in D_patch.parameters())
    total_params = total_params_E + total_params_G + total_params_D + total_params_D_spec + total_params_D_patch
    print(f"  模型参数量总量: {total_params:,}\n")

    # 步骤4: 定义优化器
    optimizer_E = optim.Adam(E.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optimizer_D_spec = optim.Adam(D_spec.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optimizer_D_patch = optim.Adam(D_patch.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # 步骤5: 初始化训练记录
    train_loss_history = []
    training_start_time = datetime.now()

    # 步骤6: 预先提取目标域特征（只提取一次，避免重复计算）
    print("正在提取目标域特征...")
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
    print(f"目标域特征提取完成\n")

    # 步骤7: 训练循环
    for epoch in range(epochs):
        # 步骤7: 训练循环
        loss_dict = train_epoch(
            E, G, D, D_spec, D_patch, source_loader, target_features_cache, target_data_cache,
            optimizer_E, optimizer_G, optimizer_D, optimizer_D_spec, optimizer_D_patch,
            lambda_mmd=5.0, lambda_freq=1.0,
            lambda_gp=10.0, n_critic=3, device=device
        )

        # 步骤8: 记录训练损失变化
        train_loss_history.append(loss_dict)

        # 步骤9: 打印训练进度
        elapsed_time = (datetime.now() - training_start_time).total_seconds() / 60
        print(f"轮数 [{epoch + 1:3d}/{epochs}] | 耗时: {elapsed_time:.1f}分钟")
        print(f"  生成器损失: {loss_dict['total_loss']:.4f} | 判别器损失: {loss_dict['d_loss']:.4f} | "
              f"MMD: {loss_dict['mmd_loss']:.4f} | " f"频域约束: {loss_dict['freq_loss']:.4f} ")

    # 保存最优模型
    torch.save(E.state_dict(), model_path)

    # 步骤12: 训练完成后生成虚假样本
    print("=" * 70 + "\n")
    print("GAN模型训练完成")
    print("正在生成合成样本用于数据增强...")
    synthetic_data, synthetic_labels = generate_synthetic_samples(
        E, G, source_loader, target_loader, num_samples=num_samples, device=device
    )
    print(f"  已生成 {len(synthetic_data)} 个合成样本\n")

    # 步骤13: 绘制训练指标
    print("正在绘制训练指标...")
    train_losses_for_plot = [loss['g_loss'] for loss in train_loss_history]
    plot_training_metrics(train_losses_for_plot, output_dir='GAN')

    # 步骤14: 执行全面的GAN质量评估
    print("正在执行GAN质量综合评估...")
    comprehensive_metrics = evaluate_gan_comprehensive(E, G, source_loader, target_loader, device=device)
    print("=" * 70)
    print(f"FID分数 (越低越好): {comprehensive_metrics.get('fid', -1):.4f}")
    print(f"频谱保真度 (越低越好): {comprehensive_metrics.get('spectral_fidelity', -1):.4f}")
    print(f"时间相关性-真实样本: {comprehensive_metrics.get('temporal_correlation_real', -1):.4f}")
    print(f"时间相关性-生成样本: {comprehensive_metrics.get('temporal_correlation_fake', -1):.4f}")
    print(f"信道平滑度-真实样本: {comprehensive_metrics.get('channel_smoothness_real', -1):.4f}")
    print(f"信道平滑度-生成样本: {comprehensive_metrics.get('channel_smoothness_fake', -1):.4f}")
    print(f"精确度 (Precision): {comprehensive_metrics.get('precision', -1):.4f}")
    print(f"召回率 (Recall): {comprehensive_metrics.get('recall', -1):.4f}")
    print("=" * 70 + "\n")

    # 步骤15: 保存全面评估结果
    print("步骤4: 正在保存评估结果...")
    save_evaluation_results(comprehensive_metrics, train_loss_history, 'GAN')

    return synthetic_data, synthetic_labels


"""
执行单个epoch的GAN对抗训练
"""
def train_epoch(E, G, D, D_spec, D_patch, source_loader, target_features, target_data, optimizer_E, optimizer_G, optimizer_D, optimizer_D_spec, optimizer_D_patch, lambda_mmd=5.0, lambda_freq=2.0, lambda_gp=10.0, n_critic=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 步骤1: 设置模型为训练模式
    E.train()
    G.train()
    D.train()
    D_spec.train()
    D_patch.train()

    # 步骤2: 初始化各类损失的累计变量
    total_g_loss = 0.0
    total_d_loss = 0.0
    total_mmd_loss = 0.0
    total_freq_loss = 0.0
    num_batches = 0
    critic_iter = 0
    
    # 步骤3: 遍历源域数据批次
    for batch_idx, (x_s, source_labels) in enumerate(source_loader):
        if x_s.size(0) < 2:
            continue

        x_s = x_s.to(device)
        batch_size = x_s.size(0)
        rand_idx = torch.randperm(target_data.size(0))[:batch_size]
        x_t_real = target_data[rand_idx].to(device)

        # 步骤4: 更新判别器D（每n_critic次更新一次生成器）
        for _ in range(n_critic):
            optimizer_D.zero_grad()
            optimizer_D_spec.zero_grad()
            optimizer_D_patch.zero_grad()
            
            # 4.1 生成虚假样本
            with torch.no_grad():
                x_hat_t = G(x_s, target_features)
            
            # 4.2 计算三个判别器的 WGAN-GP 损失
            d_loss_main, w_main, gp_main = wasserstein_discriminator_loss(
                D, x_t_real, x_hat_t, lambda_gp=lambda_gp, device=device
            )
            d_loss_spec, w_spec, gp_spec = wasserstein_discriminator_loss(
                D_spec, x_t_real, x_hat_t, lambda_gp=lambda_gp, device=device
            )
            d_loss_patch, w_patch, gp_patch = wasserstein_discriminator_loss(
                D_patch, x_t_real, x_hat_t, lambda_gp=lambda_gp, device=device
            )
            
            # 4.3 计算平均判别器损失并反向传播
            d_loss = (d_loss_main + d_loss_spec + d_loss_patch) / 3.0
            d_loss.backward()
            optimizer_D.step()
            optimizer_D_spec.step()
            optimizer_D_patch.step()
            
            # 4.4 累计损失记录
            total_d_loss += d_loss.item()
            critic_iter += 1

        # 步骤5: 更新生成器G和特征提取器E
        optimizer_G.zero_grad()
        optimizer_E.zero_grad()
        
        # 5.1 生成虚假样本
        x_hat_t = G(x_s, target_features)
        
        # 5.2 计算WGAN对抗损失
        g_adv_main = wasserstein_generator_loss(D, x_hat_t)
        g_adv_spec = wasserstein_generator_loss(D_spec, x_hat_t)
        g_adv_patch = wasserstein_generator_loss(D_patch, x_hat_t)
        g_adv_loss = (g_adv_main + g_adv_spec + g_adv_patch) / 3.0
        
        # 5.3 计算MMD损失（分布对齐）
        generated_features = E(x_hat_t)
        mmd_loss_value = mmd_loss(target_features, generated_features)
        
        # 5.4 计算频域一致性损失
        rand_idx_freq = torch.randperm(target_data.size(0))[:batch_size]
        x_t_real_freq = target_data[rand_idx_freq].to(device)
        freq_loss = frequency_consistency_loss(x_t_real_freq, x_hat_t)
        
        # 5.5 计算加权的总生成器损失
        g_total_loss = (g_adv_loss + lambda_mmd * mmd_loss_value + lambda_freq * freq_loss)
        
        # 5.7 反向传播与优化器更新
        g_total_loss.backward()
        optimizer_G.step()
        optimizer_E.step()
        
        # 5.8 累计各项损失
        total_g_loss += g_total_loss.item()
        total_mmd_loss += mmd_loss_value.item()
        total_freq_loss += freq_loss.item()
        num_batches += 1

    # 步骤6: 计算并返回平均损失
    # train_epoch函数共使用了以下5类损失函数：
    # 1. total_loss - 生成器总损失，包含WGAN对抗损失、MMD损失和频域一致性损失的加权和
    # 2. d_loss - Wasserstein判别器损失（WGAN-GP框架），用于三个判别器（D、D_spec、D_patch）的平均值
    # 3. mmd_loss - 最大均值差异损失(来自loss_GAN.py的mmd_loss函数)，衡量生成样本特征与目标域特征分布的差异
    # 4. freq_loss - 频域一致性损失(来自loss_GAN.py的frequency_consistency_loss函数)，在频谱上约束生成样本
    loss_dict = {
        'total_loss': total_g_loss / num_batches if num_batches > 0 else 0,
        'd_loss': total_d_loss / critic_iter if critic_iter > 0 else 0,
        'mmd_loss': total_mmd_loss / num_batches if num_batches > 0 else 0,
        'freq_loss': total_freq_loss / num_batches if num_batches > 0 else 0,
    }
    return loss_dict


"""
使用训练的生成器生成虚假样本进行数据增强
"""
def generate_synthetic_samples(E, G, source_loader, target_loader, num_samples=900, device='cuda'):
    # 步骤1: 设置模型为评估模式
    E.eval()
    G.eval()

    synthetic_data = []
    synthetic_labels = []
    generated_count = 0
    
    # 步骤2: 预先提取目标域所有样本的特征（缓存以提高效率）
    all_features = []
    with torch.no_grad():
        for x_t_real, _ in target_loader:
            x_t_real = x_t_real.to(device)
            features = E(x_t_real)
            all_features.append(features)
    target_features = torch.cat(all_features, dim=0).to(device)

    with torch.no_grad():
        # 步骤3: 使用源域数据和目标域特征生成虚假样本
        for x_s, source_labels in source_loader:
            if generated_count >= num_samples:
                break

            x_s = x_s.to(device)
            batch_size = x_s.size(0)

            # 步骤4: 生成与目标域特征匹配的虚假样本
            x_hat_t = G(x_s, target_features)

            # 步骤5: 收集生成的样本和对应的标签
            synthetic_data.append(x_hat_t.cpu())
            synthetic_labels.append(source_labels[:x_hat_t.size(0)])

            generated_count += x_hat_t.size(0)

    # 步骤6: 合并所有生成样本并截断至指定数量
    if synthetic_data:
        synthetic_data = torch.cat(synthetic_data, dim=0)[:num_samples]
        synthetic_labels = torch.cat(synthetic_labels, dim=0)[:num_samples]

    return synthetic_data, synthetic_labels


"""
合并生成的虚假样本与原始目标域数据，并保存为文件
"""
def save_combined_target_data(synthetic_data, synthetic_labels, target_loader, output_dir):
    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 步骤2: 提取数据加载器中的所有原始目标域数据和标签
    target_data_list = []
    target_label_list = []

    for x_t_real, labels in target_loader:
        target_data_list.append(x_t_real)
        target_label_list.append(labels)

    # 步骤3: 合并所有原始目标域数据
    target_data = torch.cat(target_data_list, dim=0)
    target_labels = torch.cat(target_label_list, dim=0)

    # 步骤4: 将原始数据和生成数据拼接
    combined_data = torch.cat([target_data, synthetic_data], dim=0)
    combined_labels = torch.cat([target_labels, synthetic_labels], dim=0)

    # 步骤5: 保存合并后的数据到文件
    target_data_path = os.path.join(output_dir, 'target_env2_gan_data.pt')
    target_labels_path = os.path.join(output_dir, 'target_env2_gan_labels.pt')

    torch.save(combined_data, target_data_path)
    torch.save(combined_labels, target_labels_path)

    return combined_data, combined_labels


if __name__ == "__main__":
    # 步骤0: 设置随机种子以确保结果可重现
    print("步骤0: 设置随机种子以确保结果可重现...")
    torch.manual_seed(40)
    np.random.seed(40)
    torch.backends.cudnn.benchmark = True
    print("  随机种子设置成功\n")

    # 步骤1: 从磁盘加载源域和目标域数据
    print("步骤1: 正在加载数据文件...")
    try:
        source_data = torch.load('Data/source_env0_env1_data.pt')
        source_labels = torch.load('Data/source_env0_env1_labels.pt')
        target_data = torch.load('Data/target_env2_data.pt')
        target_labels = torch.load('Data/target_env2_labels.pt')
        print("  数据文件加载成功\n")
    except FileNotFoundError as e:
        print(f"  错误: 加载数据文件失败: {e}")
        exit(1)

    # 步骤2: 为源域数据创建DataLoader
    print("步骤2: 正在创建数据加载器...")
    source_dataset = CustomDataset(source_data, source_labels)
    source_loader = DataLoader(source_dataset, batch_size=100, shuffle=True)

    # 步骤3: 从目标域中均匀采样每个标签的样本
    selected_target_data, selected_target_labels = select_samples_by_label(target_data, target_labels, samples_per_label=10)

    # 步骤4: 打印加载后的数据形状统计
    print(f"  源域数据: {source_data.shape}")
    print(f"  源域标签: {source_labels.shape}")
    print(f"  目标域数据(已选): {selected_target_data.shape}")
    print(f"  目标域标签(已选): {selected_target_labels.shape}\n")

    # 步骤5: 为选中的目标域数据创建DataLoader
    target_dataset = CustomDataset(selected_target_data, selected_target_labels)
    target_loader = DataLoader(target_dataset, batch_size=100, shuffle=True)

    # 步骤6: 执行主训练流程
    print("步骤3: 开始GAN训练...")
    synthetic_data, synthetic_labels = train_and_test(model_path='GAN/best_gan_model.pth', epochs=10, lr_g=1e-4, lr_d=4e-4, num_samples=900)

    # 步骤7: 合并生成的样本与原始目标域数据
    print("步骤4: 正在合并并保存合成数据...")
    output_dir = 'Data'
    target_gan_data, target_gan_labels = save_combined_target_data(synthetic_data, synthetic_labels, target_loader, output_dir)
    print(f"  合并后数据形状: {target_gan_data.shape}")
    print(f"  合并后标签形状: {target_gan_labels.shape}")
    
    # 步骤8: 打印最终的数据统计结果
    print("\n" + "="*70)
    print("最终数据统计")
    print("="*70)
    print(f"源域数据: {source_data.shape}")
    print(f"源域标签: {source_labels.shape}")
    print(f"目标域数据(增强): {target_gan_data.shape}")
    print(f"目标域标签(增强): {target_gan_labels.shape}")
    print("="*70)
    print("\n流程完成！所有训练和数据增强步骤已执行完毕。")