import torch
import torch.nn.functional as F

def kl_divergence_loss(features, target_features):
    """
    KL 散度简化形式用于衡量两个样本特征的分布差异
    L_E = Σ_i Σ_j E(xi^t) log(E(xi^t)/E(xj^t))
    """
    eps = 1e-8
    features = F.softmax(features, dim=-1)  # 归一化
    target_features = F.softmax(target_features, dim=-1)

    # 计算KL散度
    kl_loss = torch.sum(features * torch.log(features / (target_features + eps) + eps), dim=-1)
    return kl_loss.mean()


def feature_matching_loss(G, E, x_s, f_t, x_t_real):
    """
    特征匹配损失 L_feat = ||E(G(x_s, f_t)) - E(x_t_real)||^2
    """
    x_hat_t = G(x_s, f_t)
    f_hat = E(x_hat_t)

    # 从目标域特征中随机采样与源域数据相同数量的样本
    if f_t.size(0) >= x_s.size(0):
        # 随机选择与源域数据相同数量的目标域样本
        indices = torch.randperm(f_t.size(0))[:x_s.size(0)]
        f_real_sampled = f_t[indices]
    else:
        # 如果目标域样本较少，则重复采样
        indices = torch.randint(0, f_t.size(0), (x_s.size(0),))
        f_real_sampled = f_t[indices]

    return F.mse_loss(f_hat, f_real_sampled)


def discriminator_loss(D, x_t_real, x_hat_t):
    """
    判别器损失 L_D = -[log(D(x_t_real)) + log(1 - D(x_hat_t))]
    """
    real_pred = D(x_t_real)
    fake_pred = D(x_hat_t)
    real_loss = -torch.mean(torch.log(real_pred + 1e-8))
    fake_loss = -torch.mean(torch.log(1 - fake_pred + 1e-8))
    return real_loss + fake_loss


def compute_total_loss(E, G, D, x_s, x_t_real, f_t, lambda_E=0.5, lambda_feat=1.0):
    """
    总体损失函数：
    min_{E,G} max_D L_total = λ_E * L_E + λ_feat * L_feat + L_D
    """
    # 计算特征提取损失 L_E
    # 根据论文公式：L_E = (2/n²) × Σ_i Σ_j E(xi^t) × log(E(xi^t)/E(xj^t))
    if f_t.size(0) > 1:
        # 优化实现：使用更高效的计算方式避免O(n²)复杂度
        # 计算每个样本与全局平均特征的KL散度
        mean_features = torch.mean(f_t, dim=0, keepdim=True)
        kl_losses = []

        for i in range(f_t.size(0)):
            kl_loss = kl_divergence_loss(f_t[i:i+1], mean_features)
            kl_losses.append(kl_loss)

        L_E = torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0, device=f_t.device)
    else:
        L_E = torch.tensor(0.0, device=f_t.device)  # 如果只有一个样本，设置为0

    # 计算特征匹配损失 L_feat
    L_feat = feature_matching_loss(G, E, x_s, f_t, x_t_real)

    # 计算判别器损失 L_D
    x_hat_t = G(x_s, f_t)
    L_D = discriminator_loss(D, x_t_real, x_hat_t)

    # 总损失
    L_total = lambda_E * L_E + lambda_feat * L_feat + L_D
    return L_total, L_E, L_feat, L_D
