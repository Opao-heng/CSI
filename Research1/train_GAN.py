import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from Research1.model_GAN import build_model
from Research1.Process.dataloder_GAN import CustomDataset, select_samples_by_label
from Research1.loss_GAN import (
    mmd_loss,
    frequency_consistency_loss,
    wasserstein_generator_loss, 
    wasserstein_discriminator_loss,
)
from Research1.plot_GAN import (
    plot_training_metrics, 
    save_evaluation_results,
    evaluate_gan_comprehensive
)


def train_and_test(model_path='model.pth', epochs=100, lr_g=1e-4, lr_d=1e-4, num_samples=900):
    """
    执行GAN模型的完整训练流程。
    包括模型初始化、训练、合成样本生成和质量评估。
    """

    # 步骤1: 初始化设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 步骤2: 构建模型
    E, G, D, D_spec = build_model()
    E.to(device)
    G.to(device)
    D.to(device)
    D_spec.to(device)

    # 步骤3: 打印模型参数统计
    total_params_E = sum(p.numel() for p in E.parameters())
    total_params_G = sum(p.numel() for p in G.parameters())
    total_params_D = sum(p.numel() for p in D.parameters())
    total_params_D_spec = sum(p.numel() for p in D_spec.parameters())
    total_params = total_params_E + total_params_G + total_params_D + total_params_D_spec
    print(f"  模型参数量总量: {total_params:,}")

    # 步骤4: 定义优化器 - 使用更低的学习率和更稳定的betas
    optimizer_E = optim.Adam(E.parameters(), lr=lr_g, betas=(0.0, 0.9))
    optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.0, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.0, 0.9))
    optimizer_D_spec = optim.Adam(D_spec.parameters(), lr=lr_d, betas=(0.0, 0.9))

    # 步骤5: 初始化训练记录
    train_loss_history = []
    training_start_time = datetime.now()

    # 步骤6: 缓存目标域数据
    print(f"  正在缓存目标域数据...")
    all_target_data = []
    all_target_labels = []
    for x_t_real, labels in target_loader:
        all_target_data.append(x_t_real)
        all_target_labels.append(labels)
    target_data_cache = torch.cat(all_target_data, dim=0).to(device)
    target_labels_cache = torch.cat(all_target_labels, dim=0).to(device)
    print(f"  目标域数据缓存完成: {target_data_cache.shape}")

    # 步骤7: 执行训练循环
    print(f"  正在训练GAN模型 (WGAN-GP)...")
    for epoch in range(epochs):
        # 步骤7.1: 执行单个epoch的训练
        loss_dict = train_epoch(
            E, G, D, D_spec, source_loader, target_loader, target_data_cache, target_labels_cache,
            optimizer_E, optimizer_G, optimizer_D, optimizer_D_spec,
            lambda_mmd=20.0, lambda_freq=5.0, lambda_content=2.0,
            n_critic=3,  # 判别器训练次数
            device=device
        )

        # 步骤7.2: 记录训练损失变化
        train_loss_history.append(loss_dict)

        # 步骤7.3: 打印训练进度
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            elapsed_time = (datetime.now() - training_start_time).total_seconds() / 60
            print(f"  轮数 [{epoch + 1:3d}/{epochs}] | 耗时: {elapsed_time:.1f}分钟")
            print(f"    判别器损失: {loss_dict['d_loss']:.4f} | 生成器对抗损失: {loss_dict['g_adv_loss']:.4f}")
            print(f"    MMD损失: {loss_dict['mmd_loss']:.6f} | 频域损失: {loss_dict['freq_loss']:.6f}")

    # 步骤8: 保存训练完成后的模型
    print(f"  GAN模型训练完成")
    print("=" * 70 + "\n")
    print(f"  正在保存模型...")
    torch.save({
        'E': E.state_dict(),
        'G': G.state_dict(),
        'D': D.state_dict(),
        'D_spec': D_spec.state_dict()
    }, model_path)
    print(f"  模型已保存到: {model_path}")

    # 步骤9: 生成合成样本用于数据增强
    print("=" * 70 + "\n")
    print(f"  正在生成合成样本用于数据增强...")
    synthetic_data, synthetic_labels = generate_synthetic_samples(
        E, G, source_loader, target_loader, num_samples=num_samples, device=device
    )
    print(f"  已生成 {len(synthetic_data)} 个合成样本")

    # 步骤10: 绘制训练指标
    print(f"  正在绘制训练指标...")
    plot_training_metrics(train_loss_history, output_dir='GAN')

    # 步骤11: 执行全面的GAN质量评估（四项指标）
    print(f"  正在执行GAN质量综合评估（四项指标）...")
    comprehensive_metrics = evaluate_gan_comprehensive(E, G, source_loader, target_loader, device=device)
    print("\n" + "="*70)
    print("GAN生成质量评估结果：")
    print("="*70)
    print(f"  ① FID (Fréchet Inception Distance)      : {comprehensive_metrics.get('fid', -1):.4f} (越小越好)")
    print(f"  ② IS (Inception Score)                  : {comprehensive_metrics.get('inception_score', -1):.4f} (越大越好)")
    print(f"  ③ 时域MSE (Time-domain MSE)             : {comprehensive_metrics.get('time_domain_mse', -1):.6f} (越小越好)")
    print(f"  ④ 频谱相关性系数 (Spectral Correlation)  : {comprehensive_metrics.get('spectral_correlation', -1):.4f} (越接近1越好)")
    print("="*70 + "\n")

    # 步骤12: 保存全面评估结果
    save_evaluation_results(comprehensive_metrics, train_loss_history, 'GAN')
    return synthetic_data, synthetic_labels


def train_epoch(E, G, D, D_spec, source_loader, target_loader, target_data, target_labels,
                optimizer_E, optimizer_G, optimizer_D, optimizer_D_spec,
                lambda_mmd=10.0, lambda_freq=5.0, lambda_content=2.0,
                n_critic=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    GAN训练 - WGAN-GP版本（优化版）
    """
    E.train()
    G.train()
    D.train()
    D_spec.train()

    metrics = {'d_loss': 0.0, 'g_loss': 0.0, 'mmd': 0.0, 'freq': 0.0, 'gp': 0.0}
    num_d_batches = 0
    num_g_batches = 0
    
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    
    total_batches = len(source_loader)
    
    for batch_idx in range(total_batches):
        # 获取源域数据
        try:
            x_s, source_labels = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            x_s, source_labels = next(source_iter)
        
        if x_s.size(0) < 4:
            continue

        x_s = x_s.to(device)
        batch_size = x_s.size(0)
        
        # 获取目标域真实数据
        try:
            x_t_real, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            x_t_real, _ = next(target_iter)
        
        x_t_real = x_t_real.to(device)
        if x_t_real.size(0) < batch_size:
            idx = torch.randint(0, target_data.size(0), (batch_size,)).tolist()
            x_t_real = target_data[idx].to(device)

        # ================== 实时提取目标域特征 ==================
        with torch.no_grad():
            target_features = E(x_t_real)
        
        # ================== 训练判别器 (n_critic次) ==================
        for _ in range(n_critic):
            optimizer_D.zero_grad()
            optimizer_D_spec.zero_grad()
            
            with torch.no_grad():
                x_hat_t = G(x_s, target_features)
            
            # WGAN-GP损失
            d_loss1, w1, gp1 = wasserstein_discriminator_loss(D, x_t_real, x_hat_t, lambda_gp=10.0, device=device)
            d_loss2, w2, gp2 = wasserstein_discriminator_loss(D_spec, x_t_real, x_hat_t, lambda_gp=10.0, device=device)
            
            d_loss = d_loss1 + d_loss2
            d_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(D_spec.parameters(), max_norm=1.0)
            
            optimizer_D.step()
            optimizer_D_spec.step()
            
            metrics['gp'] += (gp1 + gp2) / 2
            num_d_batches += 1
        
        metrics['d_loss'] += d_loss.item()

        # ================== 训练生成器 ==================
        optimizer_G.zero_grad()
        optimizer_E.zero_grad()
        
        # 重新提取目标域特征（E在更新）
        target_features = E(x_t_real)
        
        x_hat_t = G(x_s, target_features)
        
        # 对抗损失
        g_adv = wasserstein_generator_loss(D, x_hat_t) + wasserstein_generator_loss(D_spec, x_hat_t)
        
        # MMD损失 - 对齐生成特征和目标域特征
        gen_feat = E(x_hat_t)
        mmd = mmd_loss(target_features.detach(), gen_feat)
        
        # 频域一致性损失
        freq = frequency_consistency_loss(x_t_real, x_hat_t)
        
        # 总损失
        g_loss = g_adv + lambda_mmd * mmd + lambda_freq * freq
        g_loss.backward()
        
        # 梯度裁剪
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
    
    # 返回整个epoch的各项损失指标
    return {
        'd_loss': metrics['d_loss'],  # 判别器损失
        'g_adv_loss': metrics['g_loss'],  # 生成器对抗损失
        'mmd_loss': metrics['mmd'],  # MMD损失
        'freq_loss': metrics['freq']  # 频域损失
    }


def generate_synthetic_samples(E, G, source_loader, target_loader, num_samples=900, device='cuda'):
    """
    使用训练的生成器生成指定数量的合成样本。
    """

    # 步骤1: 设置模型为评估模式
    E.eval()
    G.eval()

    synthetic_data = []
    synthetic_labels = []
    generated_count = 0
    
    # 步骤2: 预先提取目标域特征（缓存以提高效率）
    all_features = []
    with torch.no_grad():
        for x_t_real, _ in target_loader:
            x_t_real = x_t_real.to(device)
            features = E(x_t_real)
            all_features.append(features)
    target_features = torch.cat(all_features, dim=0).to(device)

    with torch.no_grad():
        # 步骤3: 遍历源域数据生成合成样本
        for x_s, source_labels in source_loader:
            if generated_count >= num_samples:
                break

            x_s = x_s.to(device)
            batch_size = x_s.size(0)

            # 步骤3.1: 生成与目标域特征匹配的合成样本
            x_hat_t = G(x_s, target_features)

            # 步骤3.2: 收集生成的样本和对应的标签
            synthetic_data.append(x_hat_t.cpu())
            synthetic_labels.append(source_labels[:x_hat_t.size(0)])

            generated_count += x_hat_t.size(0)

    # 步骤4: 合并所有生成样本并截断至指定数量
    if synthetic_data:
        synthetic_data = torch.cat(synthetic_data, dim=0)[:num_samples]
        synthetic_labels = torch.cat(synthetic_labels, dim=0)[:num_samples]

    return synthetic_data, synthetic_labels


def save_combined_target_data(synthetic_data, synthetic_labels, target_loader, output_dir):
    """
    合并合成样本与原始目标域数据并保存。
    """

    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 步骤2: 提取目标域的所有原始数据和标签
    target_data_list = []
    target_label_list = []

    for x_t_real, labels in target_loader:
        target_data_list.append(x_t_real)
        target_label_list.append(labels)

    # 步骤3: 合并目标域的所有数据
    target_data = torch.cat(target_data_list, dim=0)
    target_labels = torch.cat(target_label_list, dim=0)

    # 步骤4: 拼接原始数据和合成数据
    combined_data = torch.cat([target_data, synthetic_data], dim=0)
    combined_labels = torch.cat([target_labels, synthetic_labels], dim=0)

    # 步骤5: 保存合并后的数据到文件
    target_data_path = os.path.join(output_dir, 'target_env2_gan_data.pt')
    target_labels_path = os.path.join(output_dir, 'target_env2_gan_labels.pt')

    torch.save(combined_data, target_data_path)
    torch.save(combined_labels, target_labels_path)

    return combined_data, combined_labels


if __name__ == "__main__":
    # 步骤1: 加载源域和目标域数据
    print("步骤1: 正在加载数据文件...")
    source_data = torch.load('Data/source_env0_env1_data.pt')
    source_labels = torch.load('Data/source_env0_env1_labels.pt')
    target_data = torch.load('Data/target_env2_data.pt')
    target_labels = torch.load('Data/target_env2_labels.pt')
    print("  数据文件加载成功\n")

    # 步骤2: 为源域数据创建DataLoader
    print("步骤2: 正在创建数据加载器...")
    source_dataset = CustomDataset(source_data, source_labels)
    source_loader = DataLoader(source_dataset, batch_size=100, shuffle=True)

    # 步骤3: 从目标域中均匀采样每个标签的20个样本
    selected_target_data, selected_target_labels = select_samples_by_label(target_data, target_labels, samples_per_label=20)

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
    synthetic_data, synthetic_labels = train_and_test(model_path='GAN/best_gan_model.pth', epochs=150, lr_g=1e-4, lr_d=1e-4, num_samples=900)

    # 步骤7: 合并生成的样本与原始目标域数据
    print("步骤4: 正在合并并保存合成数据...")
    output_dir = 'Data'
    target_gan_data, target_gan_labels = save_combined_target_data(synthetic_data, synthetic_labels, target_loader, output_dir)

    # 步骤8: 打印最终的数据统计结果
    print("\n" + "="*70)
    print("最终数据统计:")
    print(f"源域数据: {source_data.shape}")
    print(f"源域标签: {source_labels.shape}")
    print(f"合并后目标域数据(增强): {target_gan_data.shape}")
    print(f"合并后目标域标签(增强): {target_gan_labels.shape}")
    print("="*70)

