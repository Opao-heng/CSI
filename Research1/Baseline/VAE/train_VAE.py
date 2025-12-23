import torch
import os
import sys
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Research1.Baseline.VAE.model_VAE import build_model
from Research1.Process.dataloder_GAN import CustomDataset, select_samples_by_label
from Research1.Baseline.VAE.loss_VAE import vae_loss, frequency_consistency_loss, mmd_loss
from Research1.plot_GAN import (
    plot_training_metrics,
    save_evaluation_results,
    evaluate_gan_comprehensive
)


def train_and_test(model_path='VAE/best_vae_model.pth', epochs=200, lr=1e-4, num_samples=900):
    """
    执行VAE模型的完整训练流程
    """
    # 步骤1: 初始化设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 步骤2: 构建模型
    E, vae = build_model()
    E.to(device)
    vae.to(device)

    # 步骤3: 打印模型参数统计
    total_params_E = sum(p.numel() for p in E.parameters())
    total_params_vae = sum(p.numel() for p in vae.parameters())
    total_params = total_params_E + total_params_vae
    print(f"  模型参数量总量: {total_params:,}")

    # 步骤4: 定义优化器
    optimizer_E = optim.Adam(E.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_vae = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999))

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
    print(f"  正在训练VAE模型...")
    for epoch in range(epochs):
        # 步骤7.1: 执行单个epoch的训练
        loss_dict = train_epoch(
            E, vae, source_loader, target_loader, target_data_cache,
            optimizer_E, optimizer_vae,
            lambda_freq=5.0, lambda_mmd=10.0, beta=1.0,
            device=device
        )

        # 步骤7.2: 记录训练损失变化
        train_loss_history.append(loss_dict)

        # 步骤7.3: 打印训练进度
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            elapsed_time = (datetime.now() - training_start_time).total_seconds() / 60
            print(f"  轮数 [{epoch + 1:3d}/{epochs}] | 耗时: {elapsed_time:.1f}分钟")
            print(f"    总损失: {loss_dict['total_loss']:.4f} | 重建: {loss_dict['recon_loss']:.4f} | KL: {loss_dict['kl_loss']:.6f}")
            print(f"    频域: {loss_dict['freq_loss']:.6f} | MMD: {loss_dict['mmd_loss']:.6f}")

    # 步骤8: 保存训练完成后的模型
    print(f"  VAE模型训练完成")
    print("=" * 70 + "\n")
    print(f"  正在保存模型...")
    torch.save({
        'E': E.state_dict(),
        'vae': vae.state_dict()
    }, model_path)
    print(f"  模型已保存到: {model_path}")

    # 步骤9: 绘制训练指标
    print("=" * 70 + "\n")
    print(f"  正在绘制训练指标...")
    plot_training_metrics(train_loss_history, output_dir='Baseline/VAE/results')

    # 步骤10: 执行全面的生成质量评估（四项指标）
    print(f"  正在执行生成质量综合评估（四项指标）...")
    comprehensive_metrics = evaluate_vae_comprehensive(E, vae, source_loader, target_loader, device=device)
    print("\n" + "="*70)
    print("VAE生成质量评估结果：")
    print("="*70)
    print(f"  ① FID (Fréchet Inception Distance)      : {comprehensive_metrics.get('fid', -1):.4f} (越小越好)")
    print(f"  ② IS (Inception Score)                  : {comprehensive_metrics.get('inception_score', -1):.4f} (越大越好)")
    print(f"  ③ 时域MSE (Time-domain MSE)             : {comprehensive_metrics.get('time_domain_mse', -1):.6f} (越小越好)")
    print(f"  ④ 频谱相关性系数 (Spectral Correlation)  : {comprehensive_metrics.get('spectral_correlation', -1):.4f} (越接近1越好)")
    print("="*70 + "\n")

    # 步骤11: 保存全面评估结果
    save_evaluation_results(comprehensive_metrics, train_loss_history, 'Baseline/VAE/results')

    return comprehensive_metrics


def train_epoch(E, vae, source_loader, target_loader, target_data,
                optimizer_E, optimizer_vae,
                lambda_freq=5.0, lambda_mmd=10.0, beta=1.0,
                device='cuda'):
    """
    VAE训练单个epoch
    """
    E.train()
    vae.train()

    metrics = {'total_loss': 0.0, 'recon_loss': 0.0, 'kl_loss': 0.0, 'freq_loss': 0.0, 'mmd_loss': 0.0}
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

        # ================== 训练VAE ==================
        optimizer_vae.zero_grad()
        optimizer_E.zero_grad()

        # VAE前向传播 - 使用目标域数据
        recon_x, mu, logvar = vae(x_t_real)

        # VAE损失
        total_vae_loss, recon_loss, kl_loss = vae_loss(recon_x, x_t_real, mu, logvar, beta=beta)

        # 频域一致性损失
        freq_loss = frequency_consistency_loss(x_t_real, recon_x)

        # MMD损失
        target_features = E(x_t_real)
        recon_features = E(recon_x)
        mmd = mmd_loss(target_features.detach(), recon_features)

        # 总损失
        total_loss = total_vae_loss + lambda_freq * freq_loss + lambda_mmd * mmd
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(E.parameters(), max_norm=1.0)

        optimizer_vae.step()
        optimizer_E.step()

        metrics['total_loss'] += total_loss.item()
        metrics['recon_loss'] += recon_loss.item()
        metrics['kl_loss'] += kl_loss.item()
        metrics['freq_loss'] += freq_loss.item()
        metrics['mmd_loss'] += mmd.item()
        num_batches += 1

    # 平均化
    for key in metrics:
        metrics[key] /= max(num_batches, 1)

    return metrics


def generate_synthetic_samples(E, vae, source_loader, target_loader, num_samples=900, device='cuda'):
    """
    使用训练的VAE生成指定数量的合成样本
    """
    E.eval()
    vae.eval()

    synthetic_data = []
    synthetic_labels = []
    generated_count = 0

    with torch.no_grad():
        for x_s, source_labels in source_loader:
            if generated_count >= num_samples:
                break

            batch_size = x_s.size(0)

            # 使用VAE生成样本
            recon_x, _, _ = vae(x_s.to(device))

            synthetic_data.append(recon_x.cpu())
            synthetic_labels.append(source_labels[:recon_x.size(0)])

            generated_count += recon_x.size(0)

    if synthetic_data:
        synthetic_data = torch.cat(synthetic_data, dim=0)[:num_samples]
        synthetic_labels = torch.cat(synthetic_labels, dim=0)[:num_samples]

    return synthetic_data, synthetic_labels


def evaluate_vae_comprehensive(E, vae, source_loader, target_loader, device='cuda'):
    """
    执行VAE的全面评估（四项指标）
    """
    E.eval()
    vae.eval()

    with torch.no_grad():
        # 收集目标域真实样本
        real_samples_list = []
        for x_t_real, _ in target_loader:
            x_t_real = x_t_real.to(device)
            real_samples_list.append(x_t_real)
        real_samples = torch.cat(real_samples_list, dim=0)[:500]

        # 生成合成样本
        fake_samples_list = []
        for x_s, _ in source_loader:
            if len(torch.cat(fake_samples_list + [torch.zeros(0, 3, 56, 6000)], dim=0)) >= 500:
                break
            x_s = x_s.to(device)
            recon_x, _, _ = vae(x_s)
            fake_samples_list.append(recon_x)
        fake_samples = torch.cat(fake_samples_list, dim=0)[:500]

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
    source_data = torch.load('Data/source_env0_env1_data.pt')
    source_labels = torch.load('Data/source_env0_env1_labels.pt')
    target_data = torch.load('Data/target_env2_data.pt')
    target_labels = torch.load('Data/target_env2_labels.pt')
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
    print("步骤3: 开始VAE训练...")
    os.makedirs('Baseline/VAE', exist_ok=True)
    synthetic_data, synthetic_labels = train_and_test(
        model_path='Baseline/VAE/best_vae_model.pth',
        epochs=200,
        lr=1e-4,
        num_samples=900
    )

    print(f"\n生成的合成样本形状: {synthetic_data.shape}")
    print(f"生成的合成标签形状: {synthetic_labels.shape}")
    print("=" * 70)
