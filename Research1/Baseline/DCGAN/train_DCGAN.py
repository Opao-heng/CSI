import torch
import os
import sys
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Research1.Baseline.DCGAN.model_DCGAN import build_model
from Research1.Process.dataloder_GAN import CustomDataset, select_samples_by_label
from Research1.Baseline.DCGAN.loss_DCGAN import (
    dcgan_discriminator_loss,
    dcgan_generator_loss,
    frequency_consistency_loss,
    mmd_loss
)
from Research1.plot_GAN import (
    evaluate_gan_comprehensive
)


def train_and_test(model_path='DCGAN/best_dcgan_model.pth', epochs=200, lr_g=2e-4, lr_d=2e-4, num_samples=900):
    """
    执行DCGAN模型的完整训练流程
    """
    # 步骤1: 初始化设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 步骤2: 构建模型
    E, G, D = build_model()
    E.to(device)
    G.to(device)
    D.to(device)

    # 步骤3: 打印模型参数统计
    total_params_E = sum(p.numel() for p in E.parameters())
    total_params_G = sum(p.numel() for p in G.parameters())
    total_params_D = sum(p.numel() for p in D.parameters())
    total_params = total_params_E + total_params_G + total_params_D
    print(f"  模型参数量总量: {total_params:,}")

    # 步骤4: 定义优化器
    optimizer_E = optim.Adam(E.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

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
    print(f"  正在训练DCGAN模型...")
    for epoch in range(epochs):
        # 步骤7.1: 执行单个epoch的训练
        loss_dict = train_epoch(
            E, G, D, source_loader, target_loader, target_data_cache,
            optimizer_E, optimizer_G, optimizer_D,
            lambda_freq=5.0, lambda_mmd=10.0,
            device=device
        )

        # 步骤7.2: 记录训练损失变化
        train_loss_history.append(loss_dict)

        # 步骤7.3: 打印训练进度
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            elapsed_time = (datetime.now() - training_start_time).total_seconds() / 60
            print(f"  轮数 [{epoch + 1:3d}/{epochs}] | 耗时: {elapsed_time:.1f}分钟")
            print(f"    D损失: {loss_dict['d_loss']:.4f} | G损失: {loss_dict['g_total_loss']:.4f}")
            print(f"    频域: {loss_dict['freq_loss']:.6f} | MMD: {loss_dict['mmd_loss']:.6f}")

    # 步骤8: 保存训练完成后的模型
    print(f"  DCGAN模型训练完成")
    print("=" * 70 + "\n")
    print(f"  正在保存模型...")
    torch.save({
        'E': E.state_dict(),
        'G': G.state_dict(),
        'D': D.state_dict()
    }, model_path)
    print(f"  模型已保存到: {model_path}")

    # 步骤9: 执行全面的生成质量评估（四项指标）
    print("=" * 70 + "\n")
    print(f"  正在执行生成质量综合评估（四项指标）...")
    comprehensive_metrics = evaluate_dcgan_comprehensive(E, G, source_loader, target_loader, device=device)
    print("\n" + "="*70)
    print("DCGAN生成质量评估结果：")
    print("="*70)
    print(f"  ① FID (Fréchet Inception Distance)      : {comprehensive_metrics.get('fid', -1):.4f} (越小越好)")
    print(f"  ② IS (Inception Score)                  : {comprehensive_metrics.get('inception_score', -1):.4f} (越大越好)")
    print(f"  ③ 时域MSE (Time-domain MSE)             : {comprehensive_metrics.get('time_domain_mse', -1):.6f} (越小越好)")
    print(f"  ④ 频谱相关性系数 (Spectral Correlation)  : {comprehensive_metrics.get('spectral_correlation', -1):.4f} (越接近1越好)")
    print("="*70 + "\n")

    # 步骤10: 保存全面评估结果（仅保存评估指标）
    os.makedirs('DCGAN', exist_ok=True)
    import json
    with open('DCGAN/dcgan_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(comprehensive_metrics, f, indent=4, ensure_ascii=False)

    return comprehensive_metrics


def train_epoch(E, G, D, source_loader, target_loader, target_data,
                optimizer_E, optimizer_G, optimizer_D,
                lambda_freq=5.0, lambda_mmd=10.0,
                device='cuda'):
    """
    DCGAN训练单个epoch
    """
    E.train()
    G.train()
    D.train()

    metrics = {
        'd_loss': 0.0, 'g_loss': 0.0, 'g_adv_loss': 0.0,
        'freq_loss': 0.0, 'mmd_loss': 0.0,
        'd_real': 0.0, 'd_fake': 0.0
    }
    num_batches = 0

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

        # ================== 提取目标域特征 ==================
        target_features = E(x_t_real)

        # ================== 训练判别器 ==================
        optimizer_D.zero_grad()

        # 生成噪声
        z = torch.randn(batch_size, 100).to(device)

        # 生成假样本
        with torch.no_grad():
            fake_samples = G(z, target_features)

        # 判别器损失
        d_loss, d_real, d_fake = dcgan_discriminator_loss(D, x_t_real, fake_samples)
        d_loss.backward()

        optimizer_D.step()

        metrics['d_loss'] += d_loss.item()
        metrics['d_real'] += d_real
        metrics['d_fake'] += d_fake

        # ================== 训练生成器 ==================
        optimizer_G.zero_grad()
        optimizer_E.zero_grad()

        # 生成新噪声
        z = torch.randn(batch_size, 100).to(device)

        # 生成假样本
        fake_samples = G(z, target_features)

        # 对抗损失
        g_adv_loss = dcgan_generator_loss(D, fake_samples)

        # 频域一致性损失
        freq_loss = frequency_consistency_loss(x_t_real, fake_samples)

        # MMD损失
        fake_features = E(fake_samples)
        mmd = mmd_loss(target_features.detach(), fake_features)

        # 总生成器损失
        g_loss = g_adv_loss + lambda_freq * freq_loss + lambda_mmd * mmd
        g_loss.backward()

        optimizer_G.step()
        optimizer_E.step()

        metrics['g_loss'] += g_loss.item()
        metrics['g_adv_loss'] += g_adv_loss.item()
        metrics['freq_loss'] += freq_loss.item()
        metrics['mmd_loss'] += mmd.item()
        num_batches += 1

    # 平均化
    for key in metrics:
        metrics[key] /= max(num_batches, 1)

    return {
        'd_loss': metrics['d_loss'],
        'g_total_loss': metrics['g_loss'],
        'g_adv_loss': metrics['g_adv_loss'],
        'freq_loss': metrics['freq_loss'],
        'mmd_loss': metrics['mmd_loss'],
        'd_real': metrics['d_real'],
        'd_fake': metrics['d_fake']
    }


def generate_synthetic_samples(E, G, source_loader, target_loader, num_samples=900, device='cuda'):
    """
    使用训练的DCGAN生成指定数量的合成样本
    """
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
        for x_s, source_labels in source_loader:
            if generated_count >= num_samples:
                break

            batch_size = x_s.size(0)

            # 生成噪声
            z = torch.randn(batch_size, 100).to(device)

            # 生成样本
            fake_samples = G(z, target_features[:batch_size])

            synthetic_data.append(fake_samples.cpu())
            synthetic_labels.append(source_labels[:fake_samples.size(0)])

            generated_count += fake_samples.size(0)

    if synthetic_data:
        synthetic_data = torch.cat(synthetic_data, dim=0)[:num_samples]
        synthetic_labels = torch.cat(synthetic_labels, dim=0)[:num_samples]

    return synthetic_data, synthetic_labels


def evaluate_dcgan_comprehensive(E, G, source_loader, target_loader, device='cuda'):
    """
    执行DCGAN的全面评估（四项指标）
    """
    E.eval()
    G.eval()

    with torch.no_grad():
        # 收集目标域真实样本
        real_samples_list = []
        for x_t_real, _ in target_loader:
            x_t_real = x_t_real.to(device)
            real_samples_list.append(x_t_real)
        real_samples = torch.cat(real_samples_list, dim=0)  # 不截断，使用所有可用样本

        # 提取目标域特征
        target_features_list = []
        for x_t_real, _ in target_loader:
            x_t_real = x_t_real.to(device)
            features = E(x_t_real)
            target_features_list.append(features)
        target_features = torch.cat(target_features_list, dim=0).to(device)

        # 生成合成样本（与真实样本数量保持一致）
        num_real_samples = real_samples.size(0)
        fake_samples_list = []
        count = 0
        for x_s, _ in source_loader:
            if count >= num_real_samples:
                break
            batch_size = x_s.size(0)
            z = torch.randn(batch_size, 100).to(device)
            fake_samples = G(z, target_features[:batch_size])
            fake_samples_list.append(fake_samples)
            count += batch_size
        fake_samples = torch.cat(fake_samples_list, dim=0)[:num_real_samples]  # 截断到与真实样本相同数量

    return {
        'fid': calculate_fid(E, real_samples, fake_samples, device),
        'inception_score': calculate_inception_score(E, fake_samples, device),
        'time_domain_mse': calculate_time_domain_mse(real_samples, fake_samples),
        'spectral_correlation': calculate_spectral_correlation(real_samples, fake_samples)
    }


def calculate_fid(E, real_samples, fake_samples, device):
    """计算FID"""
    with torch.no_grad():
        real_features = E(real_samples.to(device)).cpu().numpy()
        fake_features = E(fake_samples.to(device)).cpu().numpy()

    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    covmean = scipy.linalg.sqrtm(sigma_real.dot(sigma_fake))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid)


def calculate_inception_score(E, fake_samples, device, splits=10):
    """计算IS"""
    import scipy.stats
    with torch.no_grad():
        features = E(fake_samples.to(device)).cpu().numpy()

    scores = []
    n = len(features)
    split_size = n // splits

    for i in range(splits):
        part = features[i * split_size:(i + 1) * split_size]
        py = np.mean(part, axis=0)
        pyx = part
        kl_d = pyx * (np.log(pyx + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl_d, axis=1))))

    return float(np.mean(scores))


def calculate_time_domain_mse(real_samples, fake_samples):
    """计算时域MSE"""
    mse = torch.mean((real_samples.cpu() - fake_samples.cpu()) ** 2).item()
    return mse


def calculate_spectral_correlation(real_samples, fake_samples):
    """计算频谱相关性"""
    B, C, S, T = real_samples.shape
    real_flat = real_samples.view(B, C * S, T).cpu()
    fake_flat = fake_samples.view(B, C * S, T).cpu()

    real_fft = torch.fft.rfft(real_flat, dim=-1).abs()
    fake_fft = torch.fft.rfft(fake_flat, dim=-1).abs()

    real_mean = real_fft.mean(dim=0).flatten()
    fake_mean = fake_fft.mean(dim=0).flatten()

    corr = np.corrcoef(real_mean.numpy(), fake_mean.numpy())[0, 1]
    return float(corr)


if __name__ == "__main__":
    import scipy.linalg

    # 步骤0: 设置随机种子以确保结果可重现
    print("步骤0: 设置随机种子以确保结果可重现...")
    torch.manual_seed(40)
    np.random.seed(40)
    torch.backends.cudnn.benchmark = True
    print("  随机种子设置成功\n")

    # 步骤1: 从磁盘加载源域和目标域数据
    print("步骤1: 正在加载数据文件...")
    source_data = torch.load('../../Data/source_env0_env1_data.pt')
    source_labels = torch.load('../../Data/source_env0_env1_labels.pt')
    target_data = torch.load('../../Data/target_env2_data.pt')
    target_labels = torch.load('../../Data/target_env2_labels.pt')
    print("  数据文件加载成功\n")

    # 步骤2: 转换数据维度 (N, 56, 3, 6000) -> (N, 3, 56, 6000)
    print("步骤2: 正在转换数据维度...")
    if source_data.shape[1] == 56 and source_data.shape[2] == 3:
        source_data = source_data.permute(0, 2, 1, 3)  # (N, 56, 3, 6000) -> (N, 3, 56, 6000)
    if target_data.shape[1] == 56 and target_data.shape[2] == 3:
        target_data = target_data.permute(0, 2, 1, 3)  # (N, 56, 3, 6000) -> (N, 3, 56, 6000)
    print(f"  转换后源域数据: {source_data.shape}")
    print(f"  转换后目标域数据: {target_data.shape}")

    # 步骤3: 为源域数据创建DataLoader
    print("步骤3: 正在创建数据加载器...")
    source_dataset = CustomDataset(source_data, source_labels)
    source_loader = DataLoader(source_dataset, batch_size=100, shuffle=True)

    # 步骤3: 从目标域中均匀采样每个标签的样本
    selected_target_data, selected_target_labels = select_samples_by_label(target_data, target_labels,
                                                                              samples_per_label=10)

    # 步骤4: 打印加载后的数据形状统计
    print(f"  源域数据: {source_data.shape}")
    print(f"  源域标签: {source_labels.shape}")
    print(f"  目标域数据(已选): {selected_target_data.shape}")
    print(f"  目标域标签(已选): {selected_target_labels.shape}\n")

    # 步骤5: 为选中的目标域数据创建DataLoader
    target_dataset = CustomDataset(selected_target_data, selected_target_labels)
    target_loader = DataLoader(target_dataset, batch_size=100, shuffle=True)

    # 步骤6: 执行主训练流程
    print("步骤3: 开始DCGAN训练...")
    os.makedirs('DCGAN', exist_ok=True)
    synthetic_data, synthetic_labels = train_and_test(
        model_path='DCGAN/best_dcgan_model.pth',
        epochs=2,
        lr_g=2e-4,
        lr_d=2e-4,
        num_samples=900
    )

    print(f"\n生成的合成样本形状: {synthetic_data.shape}")
    print(f"生成的合成标签形状: {synthetic_labels.shape}")
    print("=" * 70)
