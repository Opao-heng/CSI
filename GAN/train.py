import torch
import os
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 导入必要的模块
from model import build_model
from pre_process.dataloder_GAN import source_loader, target_loader
from loss import compute_total_loss, discriminator_loss


def extract_targt_features(E, target_loader, device='cuda'):
    """
    提取目标域所有数据的特征，返回(目标域样本数, 128)的张量

    Args:
        E: 特征提取器
        target_loader: 目标域数据加载器
        device: 设备类型

    Returns:
        target_features: 目标域特征张量 (目标域样本数, 128)
    """
    E.eval()
    all_features = []

    with torch.no_grad():
        for x_t_real, _ in target_loader:
            x_t_real = x_t_real.to(device)
            features = E(x_t_real)  # 提取特征
            all_features.append(features.cpu())

    if all_features:
        target_features = torch.cat(all_features, dim=0)
        return target_features.to(device)

    return None


def train_epoch(E, G, D, source_loader, target_loader, target_features, optimizer_E, optimizer_G, optimizer_D,
                lambda_E=0.5, lambda_feat=1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    训练一个epoch

    Args:
        E: 特征提取器
        G: 生成器
        D: 判别器
        source_loader: 源域数据加载器
        target_loader: 目标域数据加载器
        target_features: 预先提取的目标域特征 (目标域样本数, 128)
        optimizer_E: 特征提取器优化器
        optimizer_G: 生成器优化器
        optimizer_D: 判别器优化器
        lambda_E: 特征提取损失权重
        lambda_feat: 特征匹配损失权重
        device: 设备类型

    Returns:
        epoch_loss: 当前epoch的平均损失
    """
    E.train()
    G.train()
    D.train()

    total_loss = 0.0
    num_batches = 0

    # 直接使用完整的目标域特征
    f_t = target_features

    # 将目标域数据加载器转换为完整的目标域数据
    target_data_list = []
    for x_t_real, _ in target_loader:
        target_data_list.append(x_t_real)
    x_t_real = torch.cat(target_data_list, dim=0).to(device)

    for x_s, source_labels in source_loader:
        x_s = x_s.to(device)
        batch_size = x_s.size(0)

        # 1. 更新判别器 D
        optimizer_D.zero_grad()

        # 生成虚假样本
        x_hat_t = G(x_s, f_t)  # (B, C, S, T)

        # 计算判别器损失
        L_D = discriminator_loss(D, x_t_real, x_hat_t)
        L_D.backward()
        optimizer_D.step()

        # 2. 更新生成器 G 和 特征提取器 E
        optimizer_G.zero_grad()
        optimizer_E.zero_grad()

        # 计算总体损失
        L_total, L_E, L_feat, _ = compute_total_loss(E, G, D, x_s, x_t_real, f_t, lambda_E, lambda_feat)

        # 只更新生成器和特征提取器
        L_total.backward()
        optimizer_G.step()
        optimizer_E.step()

        total_loss += L_total.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def generate_synthetic_samples(E, G, source_loader, target_features, num_samples=900, device='cuda'):
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
    主训练

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

    # 预先提取目标域所有数据的特征
    print("Extracting features from target domain...")
    target_features = extract_targt_features(E, target_loader, device)

    # 训练循环
    train_losses = []
    discrimination_scores = []
    feature_matching_scores = []

    for epoch in range(epochs):
        # 训练
        train_loss = train_epoch(
            E, G, D, source_loader, target_loader, target_features,
            optimizer_E, optimizer_G, optimizer_D,
            lambda_E=0.5, lambda_feat=1.0, device=device
        )

        # 记录结果
        train_losses.append(train_loss)

        # 打印进度
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")

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
        E, G, source_loader, target_features, num_samples=900, device=device
    )

    # 绘制训练曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

    return E, G, D, synthetic_data, synthetic_labels


# 运行训练
if __name__ == "__main__":
    # 设置随机种子以保证结果可重现
    torch.manual_seed(40)
    np.random.seed(40)

    # 开始训练
    E, G, D, synthetic_data, synthetic_labels = train_and_test(model_path='best_model.pth', epochs=20, lr=0.001)
    # 合并并保存数据
    output_dir = '../data/TargetData_hat'
    save_combined_target_data(synthetic_data, synthetic_labels, target_loader, output_dir)