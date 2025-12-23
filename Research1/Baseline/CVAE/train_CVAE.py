import torch
import os
import sys
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Research1.Baseline.CVAE.model_CVAE import build_model
from Research1.Process.dataloder_GAN import CustomDataset, select_samples_by_label
from Research1.Baseline.CVAE.loss_CVAE import cvae_loss, frequency_consistency_loss, mmd_loss


def train_and_test(model_path='CVAE/best_cvae_model.pth', epochs=200, lr=1e-4, num_samples=900):
    """
    执行CVAE模型的完整训练流程
    """
    # 步骤1: 初始化设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 步骤2: 构建模型
    E, cvae = build_model()
    E.to(device)
    cvae.to(device)

    # 步骤3: 打印模型参数统计
    total_params_E = sum(p.numel() for p in E.parameters())
    total_params_cvae = sum(p.numel() for p in cvae.parameters())
    total_params = total_params_E + total_params_cvae
    print(f"  模型参数量总量: {total_params:,}")

    # 步骤4: 定义优化器
    optimizer_E = optim.Adam(E.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_cvae = optim.Adam(cvae.parameters(), lr=lr, betas=(0.5, 0.999))

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
    print(f"  正在训练CVAE模型...")
    for epoch in range(epochs):
        # 步骤7.1: 执行单个epoch的训练
        loss_dict = train_epoch(
            E, cvae, source_loader, target_loader, target_data_cache,
            optimizer_E, optimizer_cvae,
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
    print(f"  CVAE模型训练完成")
    print("=" * 70 + "\n")
    print(f"  正在保存模型...")
    torch.save({
        'E': E.state_dict(),
        'cvae': cvae.state_dict()
    }, model_path)
    print(f"  模型已保存到: {model_path}")
    
    # 步骤9: 执行全面的生成质量评估（四项指标）
    print("=" * 70 + "\n")
    print(f"  正在执行生成质量综合评估（四项指标）...")
    comprehensive_metrics = evaluate_cvae_comprehensive(E, cvae, source_loader, target_loader, device=device)
    print("\n" + "="*70)
    print("CVAE生成质量评估结果：")
    print("="*70)
    print(f"  ① FID (Fréchet Inception Distance)      : {comprehensive_metrics.get('fid', -1):.4f} (越小越好)")
    print(f"  ② IS (Inception Score)                  : {comprehensive_metrics.get('inception_score', -1):.4f} (越大越好)")
    print(f"  ③ 时域MSE (Time-domain MSE)             : {comprehensive_metrics.get('time_domain_mse', -1):.6f} (越小越好)")
    print(f"  ④ 频谱相关性系数 (Spectral Correlation)  : {comprehensive_metrics.get('spectral_correlation', -1):.4f} (越接近1越好)")
    print("="*70 + "\n")

    # 步骤10: 保存全面评估结果（仅保存评估指标）
    import json
    with open('CVAE/cvae_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(comprehensive_metrics, f, indent=4, ensure_ascii=False)
        
    return comprehensive_metrics


def train_epoch(E, cvae, source_loader, target_loader, target_data,
                optimizer_E, optimizer_cvae,
                lambda_freq=5.0, lambda_mmd=10.0, beta=1.0,
                device='cuda'):
    """
    CVAE训练单个epoch
    """
    E.train()
    cvae.train()

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

        # ================== 提取目标域特征 ==================
        target_features = E(x_t_real)

        # ================== 训练CVAE ==================
        optimizer_cvae.zero_grad()
        optimizer_E.zero_grad()

        # 确保条件特征与输入批次大小匹配
        if target_features.size(0) != batch_size:
            # 如果目标特征数量不足，从缓存中随机采样
            indices = torch.randint(0, target_data.size(0), (batch_size,), device=device)
            target_features = E(target_data[indices])

        # CVAE前向传播
        recon_x, mu, logvar = cvae(x_s, target_features)

        # CVAE损失
        total_cvae_loss, recon_loss, kl_loss = cvae_loss(recon_x, x_s, mu, logvar, beta=beta)

        # 频域一致性损失
        freq_loss = frequency_consistency_loss(x_t_real, recon_x)

        # MMD损失
        recon_features = E(recon_x)
        mmd = mmd_loss(target_features.detach(), recon_features)

        # 总损失
        total_loss = total_cvae_loss + lambda_freq * freq_loss + lambda_mmd * mmd
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(E.parameters(), max_norm=1.0)

        optimizer_cvae.step()
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


def generate_synthetic_samples(E, cvae, source_loader, target_loader, num_samples=900, device='cuda'):
    """
    使用训练的CVAE生成指定数量的合成样本
    """
    # 步骤1: 设置模型为评估模式
    E.eval()
    cvae.eval()

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

            # 步骤3.1: 从目标域特征中采样匹配批次大小
            if target_features.size(0) < batch_size:
                # 如果目标特征不足，重复使用
                repeat_times = (batch_size + target_features.size(0) - 1) // target_features.size(0)
                cond_features = target_features.repeat(repeat_times, 1)[:batch_size]
            else:
                cond_features = target_features[:batch_size]

            # 步骤3.2: 使用CVAE生成样本
            recon_x, _, _ = cvae(x_s, cond_features)

            # 步骤3.2: 收集生成的样本和对应的标签
            synthetic_data.append(recon_x.cpu())
            synthetic_labels.append(source_labels[:recon_x.size(0)])

            generated_count += recon_x.size(0)

    # 步骤4: 合并所有生成样本并截断至指定数量
    if synthetic_data:
        synthetic_data = torch.cat(synthetic_data, dim=0)[:num_samples]
        synthetic_labels = torch.cat(synthetic_labels, dim=0)[:num_samples]

    return synthetic_data, synthetic_labels


def evaluate_cvae_comprehensive(E, cvae, source_loader, target_loader, device='cuda'):
    """
    执行CVAE的全面评估（四项指标）
    """
    E.eval()
    cvae.eval()

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
        for x_s, _ in source_loader:
            if fake_samples_list and len(torch.cat(fake_samples_list, dim=0)) >= num_real_samples:
                break
            x_s = x_s.to(device)
            batch_size = x_s.size(0)
            
            # 确保条件特征与批次大小匹配
            if target_features.size(0) < batch_size:
                repeat_times = (batch_size + target_features.size(0) - 1) // target_features.size(0)
                cond_features = target_features.repeat(repeat_times, 1)[:batch_size]
            else:
                cond_features = target_features[:batch_size]
            
            recon_x, _, _ = cvae(x_s, cond_features)
            fake_samples_list.append(recon_x)
        fake_samples = torch.cat(fake_samples_list, dim=0)[:num_real_samples]  # 截断到与真实样本相同数量

    # 使用原有的评估函数计算四项指标（复用GAN的评估逻辑）
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
    
    # 归一化特征到[0,1]范围，避免负值和log计算错误
    features = features - features.min()
    features = features / (features.max() + 1e-10)
    features = features + 1e-10  # 避免log(0)
    
    scores = []
    n = len(features)
    split_size = n // splits
    
    for i in range(splits):
        part = features[i * split_size:(i + 1) * split_size]
        py = np.mean(part, axis=0)
        py = py / (py.sum() + 1e-10)  # 归一化为概率分布
        pyx = part / (part.sum(axis=1, keepdims=True) + 1e-10)
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

    # 步骤4: 从目标域中均匀采样每个标签的样本
    selected_target_data, selected_target_labels = select_samples_by_label(target_data, target_labels, samples_per_label=10)

    # 步骤5: 打印加载后的数据形状统计
    print(f"  源域数据: {source_data.shape}")
    print(f"  源域标签: {source_labels.shape}")
    print(f"  目标域数据(已选): {selected_target_data.shape}")
    print(f"  目标域标签(已选): {selected_target_labels.shape}\n")

    # 步骤6: 为选中的目标域数据创建DataLoader
    target_dataset = CustomDataset(selected_target_data, selected_target_labels)
    target_loader = DataLoader(target_dataset, batch_size=100, shuffle=True)

    # 步骤7: 执行主训练流程
    print("步骤7: 开始CVAE训练...")
    os.makedirs('CVAE', exist_ok=True)
    comprehensive_metrics = train_and_test(
        model_path='CVAE/best_cvae_model.pth',
        epochs=100,
        lr=1e-4,
        num_samples=900
    )

    print(f"\n评估指标:")
    print(f"  FID: {comprehensive_metrics.get('fid', -1):.4f}")
    print(f"  IS: {comprehensive_metrics.get('inception_score', -1):.4f}")
    print(f"  时域MSE: {comprehensive_metrics.get('time_domain_mse', -1):.6f}")
    print(f"  频谱相关性: {comprehensive_metrics.get('spectral_correlation', -1):.4f}")
    print("="*70)
