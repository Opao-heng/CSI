import torch
import torch.nn.functional as F

def kl_divergence_loss(target_features):
    """
    KL 散度用于衡量目标域数据样本特征分布的内部差异
    L_E = \frac{2}{n^2} \sum_{i=1}^{N} \sum_{j=1}^{N} E(x_i^t) \log \left( \frac{E(x_i^t)}{E(x_j^t)} \right)
    """
    eps = 1e-8
    n = target_features.size(0)  # 样本数量

    # 对 target_features 进行归一化
    target_features_normalized = F.softmax(target_features, dim=-1)

    # 计算所有样本对之间的KL散度
    kl_loss = 0.0
    for i in range(n):
        for j in range(n):
            # 计算第i个样本与第j个样本的KL散度
            kl_loss += target_features_normalized[i] * torch.log((target_features_normalized[i] + eps) / (target_features_normalized[j] + eps))

    # 应用公式中的系数
    # kl_loss = (2.0 / (n * n)) * kl_loss

    return kl_loss.sum()


def feature_matching_loss(G, E, x_s, f_t):
    """
    特征匹配损失 L_feat = E||E(G(x_s, f_t)) - E(x_t)||^2
    """
    f_hat = E(G(x_s, f_t))

    # 计算特征匹配损失
    return F.mse_loss(f_hat, f_t, reduction='sum')



def discriminator_loss(D, x_t_real, x_hat_t):
    """
    判别器损失 L_D = -[log(D(x_t_real)) + log(1 - D(x_hat_t))]
    """
    real_pred = D(x_t_real)
    fake_pred = D(x_hat_t)
    real_loss = -torch.mean(torch.log(real_pred + 1e-8))
    fake_loss = -torch.mean(torch.log(1 - fake_pred + 1e-8))
    return real_loss + fake_loss


