import torch
import os
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 导入必要的模块
from Research1.model_GAN import build_model
from Research1.dataloder_GAN import CustomDataset, select_samples_by_label
from Research1.loss_GAN import kl_divergence_loss, feature_matching_loss, discriminator_loss
from Research1.plot_GAN import plot_training_metrics


"""
训练一个epoch，采用正确的GAN对抗训练机制
参数:
  E - 特征提取器
  G - 生成器
  D - 判别器
  source_loader - 源域数据加载器
  target_loader - 目标域数据加载器
  optimizer_E - 特征提取器优化器
  optimizer_G - 生成器优化器
  optimizer_D - 判别器优化器
  lambda_E - KL散度损失权重，默认0.5
  lambda_feat - 特征匹配损失权重，默认0.5
  device - 设备类型，默认cuda
返回: tuple - (train_loss, eval_metrics) 训练损失和评估指标字典
"""
def train_epoch(E, G, D, source_loader, target_loader, optimizer_E, optimizer_G, optimizer_D,
                lambda_E=0.5, lambda_feat=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 步骤1: 设置模型为训练模式
    E.train()
    G.train()
    D.train()

    total_loss = 0.0
    num_batches = 0

    # 步骤2: 遍历源域数据批次
    for x_s, source_labels in source_loader:
        if x_s.size(0) < 100:  # 如果当前批次不足100组，则跳过
            continue

        x_s = x_s.to(device)
        x_t_real = selected_target_data.cuda()

        # 步骤3: 提取目标域所有数据的特征
        E.eval()
        all_features = []
        with torch.no_grad():
            for x_t_real, _ in target_loader:
                x_t_real = x_t_real.to(device)
                features = E(x_t_real)  # 提取特征
                all_features.append(features.cpu())
        target_features = torch.cat(all_features, dim=0).to(device)

        # 步骤4: 更新生成器G和特征提取器E
        optimizer_G.zero_grad()
        optimizer_E.zero_grad()

        # 计算KL散度损失和特征匹配损失
        L_E = kl_divergence_loss(target_features)
        L_feat = feature_matching_loss(G, E, x_s, target_features)

        # 反向传播并更新
        L_total = lambda_E * L_E + lambda_feat * L_feat
        L_total.backward()
        optimizer_G.step()
        optimizer_E.step()

        total_loss += L_total.item()
        num_batches += 1

        # 步骤5: 更新判别器D
        optimizer_D.zero_grad()

        # 生成虚假样本
        x_hat_t = G(x_s, target_features)  # 使用完整的目标域特征

        # 计算判别器损失并反向传播
        L_D = discriminator_loss(D, x_t_real, x_hat_t.detach())
        L_D.backward()
        optimizer_D.step()

    # 步骤6: 评估生成样本质量
    eval_metrics = evaluate_generated_samples(E, G, D, source_loader, target_loader, target_features, device)
    return total_loss / num_batches if num_batches > 0 else 0.0, eval_metrics


"""
评估生成样本的质量，包括判别器评分、特征匹配度和多样性
参数:
  E - 特征提取器
  G - 生成器
  D - 判别器
  source_loader - 源域数据加载器
  target_loader - 目标域数据加载器
  target_features - 目标域特征张量
  device - 设备类型，默认cuda
返回: dict - 包含各种评估指标的字典 (avg_fake_score, avg_real_score, feature_mse, avg_cosine_similarity, feature_diversity)
"""
def evaluate_generated_samples(E, G, D, source_loader, target_loader, target_features, device='cuda'):
    # 步骤1: 设置模型为评估模式
    E.eval()
    G.eval()
    D.eval()

    metrics = {}

    with torch.no_grad():
        # 步骤2: 评估生成样本质量，获取判别器对真实和生成样本的评分
        discriminator_scores = []
        real_scores = []

        for x_s, _ in source_loader:
            x_s = x_s.to(device)
            # 生成虚假样本
            x_hat_t = G(x_s, target_features)
            # 获取判别器对生成样本的评分
            fake_score = D(x_hat_t)
            discriminator_scores.append(fake_score.mean().item())

        # 获取判别器对真实样本的评分
        for x_t, _ in target_loader:
            x_t = x_t.to(device)
            real_score = D(x_t)
            real_scores.append(real_score.mean().item())

        metrics['avg_fake_score'] = np.mean(discriminator_scores) if discriminator_scores else 0
        metrics['avg_real_score'] = np.mean(real_scores) if real_scores else 0

        # 步骤3: 评估特征匹配度，计算源域特征与生成样本特征的距离和相似度
        source_features_list = []
        generated_features_list = []

        for x_s, _ in source_loader:
            x_s = x_s.to(device)
            # 提取源域特征
            source_features = E(x_s)
            source_features_list.append(source_features.cpu())

            # 生成并提取生成样本特征
            x_hat_t = G(x_s, target_features)
            generated_features = E(x_hat_t)
            generated_features_list.append(generated_features.cpu())

        if source_features_list and generated_features_list:
            source_features_all = torch.cat(source_features_list, dim=0)
            generated_features_all = torch.cat(generated_features_list, dim=0)

            # 计算特征距离 (MSE)
            feature_mse = torch.mean((source_features_all - generated_features_all) ** 2).item()
            metrics['feature_mse'] = feature_mse

            # 计算特征相似度 (余弦相似度)
            cos_sim = torch.nn.functional.cosine_similarity(source_features_all, generated_features_all, dim=1)
            metrics['avg_cosine_similarity'] = torch.mean(cos_sim).item()

        # 步骤4: 评估多样性，计算生成样本特征的方差
        if 'generated_features_all' in locals():
            gen_feature_variance = torch.var(generated_features_all, dim=0).mean().item()
            metrics['feature_diversity'] = gen_feature_variance

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
def generate_synthetic_samples(E, G, source_loader, num_samples=900, device='cuda'):
    # 步骤1: 设置模型为评估模式
    E.eval()
    G.eval()

    synthetic_data = []
    synthetic_labels = []
    generated_count = 0

    with torch.no_grad():
        # 步骤2: 使用源域数据和目标域特征生成虚假样本
        for x_s, source_labels in source_loader:
            if generated_count >= num_samples:
                break

            x_s = x_s.to(device)
            batch_size = x_s.size(0)

            # 步骤3: 提取完整的目标域特征
            E.eval()
            all_features = []
            with torch.no_grad():
                for x_t_real, _ in target_loader:
                    x_t_real = x_t_real.to(device)
                    features = E(x_t_real)  # 提取特征
                    all_features.append(features.cpu())
            target_features = torch.cat(all_features, dim=0).to(device)
            f_t = target_features

            # 步骤4: 生成虚假样本
            x_hat_t = G(x_s, f_t)

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
主训练函数，完整的GAN模型训练和评估流程
参数:
  model_path - 模型保存路径，默认'model.pth'
  epochs - 训练轮数，默认100
  lr - 学习率，默认0.001
  num_samples - 需要生成的合成样本数量，默认900
返回: tuple - (E, G, D, synthetic_data, synthetic_labels) 训练完的模型和生成的合成数据
"""
def train_and_test(model_path='model.pth', epochs=100, lr=0.001, num_samples=900):
    # 步骤1: 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 步骤2: 构建模型
    E, G, D = build_model()
    E.to(device)
    G.to(device)
    D.to(device)

    # 步骤3: 定义优化器
    optimizer_E = optim.Adam(E.parameters(), lr=lr)
    optimizer_G = optim.Adam(G.parameters(), lr=lr)
    optimizer_D = optim.Adam(D.parameters(), lr=lr)

    # 步骤4: 初始化训练记录
    train_losses = []
    evaluation_metrics = []

    # 步骤5: 训练循环
    for epoch in range(epochs):
        # 训练一个epoch并评估
        train_loss , eval_metrics = train_epoch(
            E, G, D, source_loader, target_loader,
            optimizer_E, optimizer_G, optimizer_D,
            lambda_E=0.5, lambda_feat=1.0, device=device
        )

        # 记录训练结果
        train_losses.append(train_loss)
        evaluation_metrics.append(eval_metrics)

        # 打印进度和评估指标
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Eval Metrics:")
        print(f"    Avg Fake Score: {eval_metrics.get('avg_fake_score', 0):.4f}")
        print(f"    Avg Real Score: {eval_metrics.get('avg_real_score', 0):.4f}")
        print(f"    Feature MSE: {eval_metrics.get('feature_mse', 0):.4f}")
        print(f"    Cosine Similarity: {eval_metrics.get('avg_cosine_similarity', 0):.4f}")
        print(f"    Feature Diversity: {eval_metrics.get('feature_diversity', 0):.4f}")

        # 步骤6: 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'E_state_dict': E.state_dict(),
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'optimizer_E_state_dict': optimizer_E.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, model_path)
            print(f"Model saved at {model_path}")

    # 步骤7: 训练完成后生成虚假样本
    print("Generating synthetic samples for data augmentation...")
    synthetic_data, synthetic_labels = generate_synthetic_samples(
        E, G, source_loader, num_samples=num_samples, device=device
    )

    # 步骤8: 调用绘图模块绘制训练指标
    plot_training_metrics(train_losses, evaluation_metrics, output_dir='GAN')

    return E, G, D, synthetic_data, synthetic_labels


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

    # 步骤6: 开始训练
    E, G, D, synthetic_data, synthetic_labels = train_and_test(model_path='GAN/best_gan_model.pth', epochs=10, lr=0.001, num_samples=900)

    # 步骤7: 合并并保存生成的数据
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