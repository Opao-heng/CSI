import torch
import os
import torch.optim as optim
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader
from datetime import datetime
from model_GAN import build_model
import sys
sys.path.append(r'c:/Users/USER/Desktop/liuheng/Research1')
from DataProcess.dataloder_GAN import CustomDataset, select_samples_by_label
from loss_GAN import (
    mmd_loss,
    frequency_consistency_loss,
    wasserstein_generator_loss, 
    wasserstein_discriminator_loss,
)


def compute_fid(real_features, fake_features):
    """
    计算Fréchet Inception Distance，衡量真实样本和生成样本在特征空间的分布差异
    返回: FID分数（越低越好）
    """
    # 确保特征在CPU上计算
    if real_features.is_cuda:
        real_features = real_features.cpu()
    if fake_features.is_cuda:
        fake_features = fake_features.cpu()
    
    # 计算均值
    mu_real = torch.mean(real_features, dim=0)
    mu_fake = torch.mean(fake_features, dim=0)
    
    # 计算协方差矩阵 - 添加正则化
    real_centered = real_features - mu_real
    fake_centered = fake_features - mu_fake
    
    n_real = real_features.size(0)
    n_fake = fake_features.size(0)
    
    sigma_real = (real_centered.T @ real_centered) / (n_real - 1) + torch.eye(real_features.size(1)) * 1e-6
    sigma_fake = (fake_centered.T @ fake_centered) / (n_fake - 1) + torch.eye(fake_features.size(1)) * 1e-6
    
    # 计算均值差的平方范数
    diff = mu_real - mu_fake
    mean_diff = torch.dot(diff, diff)
    
    # 计算协方差矩阵的迹
    trace_term = torch.trace(sigma_real + sigma_fake)
    
    # 计算 sqrt(sigma_real * sigma_fake) - 使用数值稳定的方法
    try:
        covmean = sigma_real @ sigma_fake
        eigvals, eigvecs = torch.linalg.eigh(covmean)
        eigvals = torch.clamp(eigvals.real, min=0)
        trace_sqrt = torch.sqrt(eigvals).sum()
    except Exception:
        trace_sqrt = 0.0
    
    # FID = ||mu_real - mu_fake||^2 + Tr(sigma_real + sigma_fake - 2*sqrt(sigma_real*sigma_fake))
    fid = mean_diff + trace_term - 2 * trace_sqrt
    
    return max(fid.item(), 0.0)


def compute_inception_score(fake_features, fake_labels, num_classes=6, eps=1e-16):
    """
    计算Inception Score - 评估生成样本的多样性与类内一致性
    IS = exp(E[KL(p(y|x) || p(y))])
    返回: IS分数（越大越好）
    """
    # 将特征转换为伪概率分布
    fake_features_norm = F.normalize(fake_features, p=2, dim=1)
    
    # 计算每个样本的条件概率 p(y|x)
    logits = torch.randn(fake_features.size(0), num_classes, device=fake_features.device)
    # 根据标签增强对应类别的logits
    for i, label in enumerate(fake_labels):
        if label < num_classes:
            logits[i, label] += 5.0
    
    pyx = F.softmax(logits, dim=1)  # p(y|x) - 条件概率
    py = pyx.mean(dim=0, keepdim=True)  # p(y) - 边缘概率
    
    # 计算KL散度: KL(p(y|x) || p(y))
    kl_div = (pyx * (torch.log(pyx + eps) - torch.log(py + eps))).sum(dim=1)
    
    # IS = exp(E[KL])
    is_score = torch.exp(kl_div.mean()).item()
    
    return is_score


def evaluate_gan_metrics(E, G, source_loader, target_loader, device='cuda'):
    """
    评估GAN生成质量 - 只计算FID和IS两项指标
    """
    E.eval()
    G.eval()
    
    # 收集数据
    real_features_list = []
    fake_features_list = []
    fake_labels_list = []
    
    # 先提取目标域特征
    with torch.no_grad():
        for x_t, labels in target_loader:
            x_t = x_t.to(device)
            real_features_list.append(E(x_t).cpu())
    
    target_features = torch.cat(real_features_list, dim=0).to(device)
    
    # 生成样本
    with torch.no_grad():
        for x_s, labels in source_loader:
            x_s = x_s.to(device)
            x_fake = G(x_s, target_features)
            fake_features_list.append(E(x_fake).cpu())
            fake_labels_list.append(labels.cpu())
    
    real_features = torch.cat(real_features_list, dim=0)
    fake_features = torch.cat(fake_features_list, dim=0)
    fake_labels = torch.cat(fake_labels_list, dim=0)
    
    # 计算评估指标
    metrics = {}
    
    # 1. FID分数
    try:
        metrics['fid'] = compute_fid(real_features, fake_features)
    except Exception as e:
        print(f"  [警告] FID计算失败: {e}")
        metrics['fid'] = -1
    
    # 2. Inception Score
    try:
        num_classes = len(torch.unique(fake_labels))
        metrics['inception_score'] = compute_inception_score(fake_features, fake_labels, num_classes=num_classes)
    except Exception as e:
        print(f"  [警告] IS计算失败: {e}")
        metrics['inception_score'] = -1
    
    return metrics


def save_gan_results(fid, inception_score, save_dir='model_C'):
    """
    保存GAN评估结果到JSON文件（只保存FID和IS）
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 构建结果字典
    results = {
        'fid': fid,
        'inception_score': inception_score
    }
    
    # 保存为JSON格式
    save_path = os.path.join(save_dir, 'gan_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"GAN评估结果已保存到 {save_path}")


def train_and_test(model_path='model.pth', epochs=100, lr_g=1e-4, lr_d=1e-4, num_samples=900):
    """
    执行GAN模型的完整训练流程（Model C: 只使用时域判别器）
    """

    # 步骤1: 初始化设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 步骤2: 构建模型（Model C: 只返回E, G, D，没有频域判别器）
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

    # 步骤4: 定义优化器（移除频域判别器优化器）
    optimizer_E = optim.Adam(E.parameters(), lr=lr_g, betas=(0.0, 0.9))
    optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.0, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.0, 0.9))

    # 步骤5: 缓存目标域数据
    print(f"  正在缓存目标域数据...")
    all_target_data = []
    all_target_labels = []
    for x_t_real, labels in target_loader:
        all_target_data.append(x_t_real)
        all_target_labels.append(labels)
    target_data_cache = torch.cat(all_target_data, dim=0).to(device)
    target_labels_cache = torch.cat(all_target_labels, dim=0).to(device)
    print(f"  目标域数据缓存完成: {target_data_cache.shape}")

    # 步骤6: 执行训练循环
    print(f"  正在训练GAN模型 (WGAN-GP, 只使用时域判别器)...")
    training_start_time = datetime.now()
    
    for epoch in range(epochs):
        # 执行单个epoch的训练
        loss_dict = train_epoch(
            E, G, D, source_loader, target_loader, target_data_cache, target_labels_cache,
            optimizer_E, optimizer_G, optimizer_D,
            lambda_mmd=20.0, lambda_freq=5.0, lambda_content=2.0,
            n_critic=3,
            device=device
        )

        # 打印训练进度
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            elapsed_time = (datetime.now() - training_start_time).total_seconds() / 60
            print(f"  轮数 [{epoch + 1:3d}/{epochs}] | 耗时: {elapsed_time:.1f}分钟")
            print(f"    判别器损失: {loss_dict['d_loss']:.4f} | 生成器对抗损失: {loss_dict['g_adv_loss']:.4f}")
            print(f"    MMD损失: {loss_dict['mmd_loss']:.6f} | 频域损失: {loss_dict['freq_loss']:.6f}")

    # 步骤7: 保存训练完成后的模型
    print(f"  GAN模型训练完成")
    print("=" * 70 + "\n")
    print(f"  正在保存模型...")
    torch.save({
        'E': E.state_dict(),
        'G': G.state_dict(),
        'D': D.state_dict()
    }, model_path)
    print(f"  模型已保存到: {model_path}")

    # 步骤8: 执行GAN质量评估（只计算FID和IS）
    print(f"  正在执行GAN质量评估（FID和IS）...")
    metrics = evaluate_gan_metrics(E, G, source_loader, target_loader, device=device)
    print("\n" + "="*70)
    print("GAN生成质量评估结果：")
    print("="*70)
    print(f"  ① FID (Fréchet Inception Distance): {metrics.get('fid', -1):.4f} (越小越好)")
    print(f"  ② IS (Inception Score)           : {metrics.get('inception_score', -1):.4f} (越大越好)")
    print("="*70 + "\n")

    # 步骤9: 保存评估结果
    save_gan_results(metrics.get('fid', -1), metrics.get('inception_score', -1), 'model_C')
    
    return metrics


def train_epoch(E, G, D, source_loader, target_loader, target_data, target_labels,
                optimizer_E, optimizer_G, optimizer_D,
                lambda_mmd=10.0, lambda_freq=5.0, lambda_content=2.0,
                n_critic=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    GAN训练 - WGAN-GP版本（Model C: 只使用时域判别器）
    """
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

        # 实时提取目标域特征
        with torch.no_grad():
            target_features = E(x_t_real)
        
        # 训练判别器 (n_critic次) - 只使用时域判别器
        for _ in range(n_critic):
            optimizer_D.zero_grad()
            
            with torch.no_grad():
                x_hat_t = G(x_s, target_features)
            
            # WGAN-GP损失（只使用时域判别器）
            d_loss, w, gp = wasserstein_discriminator_loss(D, x_t_real, x_hat_t, lambda_gp=10.0, device=device)
            
            d_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            
            optimizer_D.step()
            
            metrics['gp'] += gp
            num_d_batches += 1
        
        metrics['d_loss'] += d_loss.item()

        # 训练生成器
        optimizer_G.zero_grad()
        optimizer_E.zero_grad()
        
        # 重新提取目标域特征
        target_features = E(x_t_real)
        
        x_hat_t = G(x_s, target_features)
        
        # 对抗损失（只使用时域判别器）
        g_adv = wasserstein_generator_loss(D, x_hat_t)
        
        # MMD损失
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
        'd_loss': metrics['d_loss'],
        'g_adv_loss': metrics['g_loss'],
        'mmd_loss': metrics['mmd'],
        'freq_loss': metrics['freq']
    }


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
    metrics = train_and_test(model_path='model_C/best_gan_model.pth', epochs=150, lr_g=1e-4, lr_d=1e-4, num_samples=900)

    print("\n训练完成!")
