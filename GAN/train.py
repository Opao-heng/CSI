import torch
import os
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 导入必要的模块
from model import build_model
from pre_process.dataloder_GAN import CustomDataset, select_samples_by_label
from loss import kl_divergence_loss, feature_matching_loss, discriminator_loss


def train_epoch(E, G, D, source_loader, target_loader, optimizer_E, optimizer_G, optimizer_D,
                lambda_E=0.5, lambda_feat=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    训练一个epoch，采用正确的GAN对抗训练机制
    """
    E.train()
    G.train()
    D.train()

    total_loss = 0.0
    num_batches = 0

    # 从源域中选择样本
    for x_s, source_labels in source_loader:
        if x_s.size(0) < 100:  # 如果当前批次不足100组，则跳过
            continue

        x_s = x_s.to(device)
        x_t_real = selected_target_data.cuda()

        # 1. 提取目标域所有数据的特征
        E.eval()
        all_features = []
        with torch.no_grad():
            for x_t_real, _ in target_loader:
                x_t_real = x_t_real.to(device)
                features = E(x_t_real)  # 提取特征
                all_features.append(features.cpu())
        target_features = torch.cat(all_features, dim=0).to(device)

        # 2. 更新生成器 G 和 特征提取器 E
        optimizer_G.zero_grad()
        optimizer_E.zero_grad()

        # 计算总体损失
        L_E = kl_divergence_loss(target_features)
        L_feat = feature_matching_loss(G, E, x_s, target_features)

        # 更新生成器和特征提取器
        L_total = lambda_E * L_E + lambda_feat * L_feat
        L_total.backward()
        optimizer_G.step()
        optimizer_E.step()

        total_loss += L_total.item()
        num_batches += 1

        # 3. 更新判别器 D
        optimizer_D.zero_grad()

        # 生成虚假样本
        x_hat_t = G(x_s, target_features)  # 使用完整的目标域特征

        # 计算判别器损失
        L_D = discriminator_loss(D, x_t_real, x_hat_t.detach())
        L_D.backward()
        optimizer_D.step()

    eval_metrics = evaluate_generated_samples(E, G, D, source_loader, target_loader, target_features, device)
    return total_loss / num_batches if num_batches > 0 else 0.0, eval_metrics


def evaluate_generated_samples(E, G, D, source_loader, target_loader, target_features, device='cuda'):
    """
    评估生成样本的质量

    Args:
        E: 特征提取器
        G: 生成器
        D: 判别器
        source_loader: 源域数据加载器
        target_loader: 目标域数据加载器
        target_features: 目标域特征
        device: 设备类型

    Returns:
        dict: 包含各种评估指标的字典
    """
    E.eval()
    G.eval()
    D.eval()

    metrics = {}

    with torch.no_grad():
        # 1. 生成样本质量评估 (通过判别器得分)
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

        # 2. 特征匹配度评估
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

        # 3. 多样性评估 (生成样本之间的差异性)
        if 'generated_features_all' in locals():
            # 计算生成特征的方差，衡量多样性
            gen_feature_variance = torch.var(generated_features_all, dim=0).mean().item()
            metrics['feature_diversity'] = gen_feature_variance

    return metrics


def generate_synthetic_samples(E, G, source_loader, num_samples=900, device='cuda'):
    """
    生成虚假目标域样本用于数据扩充

    Args:
        E: 特征提取器
        G: 生成器
        source_loader: 源域数据加载器
        target_features: 预先提取的目标域特征 (目标域样本数, 128)
        num_samples: 需要生成的样本数量
        device: 设备类型

    Returns:
        synthetic_data: 生成的虚假样本数据
        synthetic_labels: 生成的虚假样本标签
    """
    E.eval()
    G.eval()

    synthetic_data = []
    synthetic_labels = []
    generated_count = 0

    with torch.no_grad():
        # 使用源域数据和目标域特征生成虚假样本
        for x_s, source_labels in source_loader:
            if generated_count >= num_samples:
                break

            x_s = x_s.to(device)
            batch_size = x_s.size(0)

            # 直接使用完整的目标域特征
            E.eval()
            all_features = []
            with torch.no_grad():
                for x_t_real, _ in target_loader:
                    x_t_real = x_t_real.to(device)
                    features = E(x_t_real)  # 提取特征
                    all_features.append(features.cpu())
            target_features = torch.cat(all_features, dim=0).to(device)
            f_t = target_features

            # 生成虚假样本
            x_hat_t = G(x_s, f_t)

            # 收集生成的样本
            synthetic_data.append(x_hat_t.cpu())
            synthetic_labels.append(source_labels[:x_hat_t.size(0)])

            generated_count += x_hat_t.size(0)

    if synthetic_data:
        synthetic_data = torch.cat(synthetic_data, dim=0)[:num_samples]
        synthetic_labels = torch.cat(synthetic_labels, dim=0)[:num_samples]

    return synthetic_data, synthetic_labels


def save_combined_target_data(synthetic_data, synthetic_labels, target_loader, output_dir):
    """
    将生成的虚假样本与原始目标域数据合并，并保存为 .pt 文件

    Args:
        synthetic_data: 生成的虚假样本张量 (N, C, S, T)
        synthetic_labels: 生成的虚假样本标签 (N,)
        target_loader: 原始目标域数据加载器
        output_dir: 输出目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 提取原始目标域数据和标签
    target_data_list = []
    target_label_list = []

    for x_t_real, labels in target_loader:
        target_data_list.append(x_t_real)
        target_label_list.append(labels)

    # 合并所有原始目标域数据
    target_data = torch.cat(target_data_list, dim=0)  # (M, C, S, T)
    target_labels = torch.cat(target_label_list, dim=0)  # (M,)

    # 合并生成数据和原始数据
    combined_data = torch.cat([target_data, synthetic_data], dim=0)  # (M+N, C, S, T)
    combined_labels = torch.cat([target_labels, synthetic_labels], dim=0)  # (M+N,)

    # 保存到文件
    target_data_path = os.path.join(output_dir, 'target_data.pt')
    target_labels_path = os.path.join(output_dir, 'target_labels.pt')

    torch.save(combined_data, target_data_path)
    torch.save(combined_labels, target_labels_path)

    print(f"Combined data saved to {target_data_path}")
    print(f"Combined labels saved to {target_labels_path}")


def train_and_test(model_path='model.pth', epochs=100, lr=0.001):
    """
    主训练函数，包含评估过程

    Args:
        model_path: 模型保存路径
        epochs: 训练轮数
        lr: 学习率
    """

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 构建模型
    E, G, D = build_model()
    E.to(device)
    G.to(device)
    D.to(device)

    # 定义优化器
    optimizer_E = optim.Adam(E.parameters(), lr=lr)
    optimizer_G = optim.Adam(G.parameters(), lr=lr)
    optimizer_D = optim.Adam(D.parameters(), lr=lr)

    # 训练循环
    train_losses = []
    evaluation_metrics = []

    for epoch in range(epochs):
        # 训练和评估
        train_loss , eval_metrics = train_epoch(
            E, G, D, source_loader, target_loader,
            optimizer_E, optimizer_G, optimizer_D,
            lambda_E=0.5, lambda_feat=1.0, device=device
        )

        # 记录结果
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

        # 保存模型
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

    # 生成虚假样本进行数据扩充
    print("Generating synthetic samples for data augmentation...")
    synthetic_data, synthetic_labels = generate_synthetic_samples(
        E, G, source_loader, num_samples=900, device=device
    )

    # 绘制训练曲线和评估指标
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 绘制评估指标
    epochs_range = range(1, len(evaluation_metrics)+1)

    plt.subplot(2, 3, 2)
    fake_scores = [m.get('avg_fake_score', 0) for m in evaluation_metrics]
    real_scores = [m.get('avg_real_score', 0) for m in evaluation_metrics]
    plt.plot(epochs_range, fake_scores, label='Fake Score')
    plt.plot(epochs_range, real_scores, label='Real Score')
    plt.title('Discriminator Scores')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.subplot(2, 3, 3)
    feature_mse = [m.get('feature_mse', 0) for m in evaluation_metrics]
    plt.plot(epochs_range, feature_mse)
    plt.title('Feature Matching MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')

    plt.subplot(2, 3, 4)
    cosine_sim = [m.get('avg_cosine_similarity', 0) for m in evaluation_metrics]
    plt.plot(epochs_range, cosine_sim)
    plt.title('Average Cosine Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')

    plt.subplot(2, 3, 5)
    diversity = [m.get('feature_diversity', 0) for m in evaluation_metrics]
    plt.plot(epochs_range, diversity)
    plt.title('Feature Diversity')
    plt.xlabel('Epoch')
    plt.ylabel('Variance')

    plt.tight_layout()
    plt.show()

    return E, G, D, synthetic_data, synthetic_labels


# 运行训练
if __name__ == "__main__":
    # 设置随机种子以保证结果可重现
    torch.manual_seed(40)
    np.random.seed(40)
    torch.backends.cudnn.benchmark = True

    # 加载数据文件
    source_data = torch.load('../data/SourceData/source_data.pt')
    source_labels = torch.load('../data/SourceData/source_labels.pt')
    target_data = torch.load('../data/TargetData/target_data.pt')
    target_labels = torch.load('../data/TargetData/target_labels.pt')

    # 创建源域数据集和数据加载器
    source_dataset = CustomDataset(source_data, source_labels)
    source_loader = DataLoader(source_dataset, batch_size=100, shuffle=True)

    # 从目标域数据中选择每个标签10个样本
    selected_target_data, selected_target_labels = select_samples_by_label(
        target_data, target_labels, samples_per_label=10
    )
    # 创建目标域数据集和数据加载器
    target_dataset = CustomDataset(selected_target_data, selected_target_labels)
    target_loader = DataLoader(target_dataset, batch_size=100, shuffle=True)

    # 开始训练
    E, G, D, synthetic_data, synthetic_labels = train_and_test(model_path='best_model.pth', epochs=100, lr=0.001)

    # 合并并保存数据
    output_dir = '../data/TargetData_hat'
    save_combined_target_data(synthetic_data, synthetic_labels, target_loader, output_dir)