import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SelfAttention1d(nn.Module):
    """
    1D 自注意力层（SAGAN 风格）用于长程依赖捕获
    输入形状: (B, C, T)
    """
    def __init__(self, in_channels):
        super(SelfAttention1d, self).__init__()
        self.query = nn.Conv1d(in_channels, max(1, in_channels // 8), kernel_size=1)
        self.key = nn.Conv1d(in_channels, max(1, in_channels // 8), kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, T = x.shape
        q = self.query(x)                # (B, Cq, T)
        k = self.key(x)                  # (B, Cq, T)
        v = self.value(x)                # (B, C, T)
        attn = torch.bmm(q.transpose(1, 2), k)  # (B, T, T)
        attn = F.softmax(attn, dim=-1)
        o = torch.bmm(v, attn)           # (B, C, T)
        return self.gamma * o + x


class SqueezeExcite1d(nn.Module):
    """
    1D Squeeze-and-Excitation 通道注意力
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeExcite1d, self).__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        B, C, T = x.shape
        s = x.mean(dim=-1)           # (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        s = s.unsqueeze(-1)          # (B, C, 1)
        return x * s


class NoiseInjection(nn.Module):
    """
    训练阶段噪声注入，提升多样性与鲁棒性
    """
    def __init__(self):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x)
            return x + self.weight * noise
        return x


class AdaptiveInstanceNorm1d(nn.Module):
    """
    自适应实例归一化(AdaIN)层
    使用目标域特征作为风格条件，调制生成过程
    """
    def __init__(self, num_features, feature_dim):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        # 将特征映射为缩放和偏移参数
        self.fc = nn.Linear(feature_dim, num_features * 2)
        
    def forward(self, x, style):
        """
        参数:
          x - 输入特征图, 形状为 (B, C, T)
          style - 风格特征, 形状为 (B, feature_dim)
        返回: 调制后的特征图
        """
        # 实例归一化
        x = self.norm(x)
        
        # 生成缩放和偏移参数
        style_params = self.fc(style)  # (B, num_features * 2)
        gamma, beta = style_params.chunk(2, dim=1)  # 各为 (B, num_features)
        
        # 应用仿射变换: gamma * x + beta
        gamma = gamma.unsqueeze(-1)  # (B, num_features, 1)
        beta = beta.unsqueeze(-1)    # (B, num_features, 1)
        
        return gamma * x + beta


class FeatureExtractor(nn.Module):
    """
    特征提取器 E: 提取目标域 CSI 的环境特征
    使用 ResNet18 架构进行高层特征提取
    输入: (C, S, T) -> 输出: (d)
    """

    def __init__(self, input_dim=(3, 56, 6000), feature_dim=128):
        super(FeatureExtractor, self).__init__()
        # 假设输入为 (C, S, T)，即 (num_tx_rx, num_subcarriers, num_time)
        # 这里我们将其视为图像输入，使用 ResNet18 结构
        # 简化的特征提取器，减少层数
        self.backbone = nn.Sequential(
            nn.Conv1d(input_dim[0] * input_dim[1], 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool1d(1)  # 使用自适应池化确保输出尺寸为1
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        # x: (B, C, S, T) -> reshape to (B, C*S, T)
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)  # 展平为 (B, C*S, T)

        # 经过骨干网络
        x = self.backbone(x)  # (B, 128, 1)
        x = x.squeeze(-1)  # (B, 128)
        x = self.dropout(x)
        x = self.fc(x)  # (B, d)
        return x


class Generator(nn.Module):
    """
    生成器 G: 融合源域身份特征与目标域环境特征，生成符合目标域分布的虚假样本
    输入: 源域数据 X_s + 目标域特征 F_t
    输出: 虚假目标域数据 X_hat_t
    使用 U-Net 架构 + AdaIN条件调制 + 自注意力/SE/噪声注入 + 跳连
    """

    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000, feature_dim=128):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.time_steps = time_steps

        # 简化的编码器
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels * subcarriers, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool1d(2)
        self.se_e1 = SqueezeExcite1d(64)
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool1d(2)
        self.sa_enc = SelfAttention1d(128)
        self.noise = NoiseInjection()

        # 目标域特征投影
        self.feature_proj = nn.Linear(feature_dim, 128)
        
        # AdaIN层
        self.adain1 = AdaptiveInstanceNorm1d(64, feature_dim)

        # 简化的解码器
        self.dec1 = nn.ConvTranspose1d(128 + 128, 64, kernel_size=2, stride=2)
        self.dec1_norm = nn.BatchNorm1d(64)
        self.dec1_act = nn.ReLU(inplace=True)
        self.sa_dec1 = SelfAttention1d(64)

        # 跳连融合后精炼
        self.refine_conv = nn.Conv1d(64 + 64, 64, kernel_size=1)
        self.refine_norm = nn.BatchNorm1d(64)
        self.refine_act = nn.ReLU(inplace=True)
        
        self.dec2 = nn.ConvTranspose1d(64, in_channels * subcarriers, kernel_size=2, stride=2)
        self.output_act = nn.Tanh()

    def forward(self, x_s, f_t):
        # x_s: (B, C, S, T), f_t: (目标域样本数, d)
        B, C, S, T = x_s.shape

        # 展平为 (B, C*S, T)
        x_s_flat = x_s.view(B, C * S, T)

        # 编码器
        e1 = self.enc1(x_s_flat)      # (B, 64, T)
        e1 = self.se_e1(e1)
        p1 = self.pool1(e1)           # (B, 64, T/2)
        
        e2 = self.enc2(p1)            # (B, 128, T/2)
        e2 = self.sa_enc(e2)
        encoded = self.pool2(e2)      # (B, 128, T/4)
        encoded = self.noise(encoded)

        # 处理目标域特征，使其与编码特征匹配
        if f_t.size(0) >= B:
            selected_f_t = f_t[:B]
        else:
            repeat_times = (B + f_t.size(0) - 1) // f_t.size(0)
            selected_f_t = f_t.repeat(repeat_times, 1)[:B]

        # 投影目标域特征
        projected_f = self.feature_proj(selected_f_t).unsqueeze(-1)  # (B, 128, 1)
        projected_f = projected_f.expand(-1, -1, encoded.size(-1))   # (B, 128, T/4)

        # 融合编码特征和目标域特征
        combined = torch.cat([encoded, projected_f], dim=1)  # (B, 256, T/4)
        
        # 解码器 + AdaIN 调制
        d1 = self.dec1(combined)                  # (B, 64, T/2)
        d1 = self.dec1_norm(d1)
        d1 = self.adain1(d1, selected_f_t)        # AdaIN调制
        d1 = self.dec1_act(d1)
        d1 = self.sa_dec1(d1)
        
        # 跳连（将 e1 下采样到 T/2 并与 d1 融合）
        e1_down = F.avg_pool1d(e1, kernel_size=2, stride=2)  # (B, 64, T/2)
        refined = torch.cat([d1, e1_down], dim=1)            # (B, 128, T/2)
        refined = self.refine_conv(refined)                  # (B, 64, T/2)
        refined = self.refine_norm(refined)
        refined = self.refine_act(refined)
        
        output_flat = self.dec2(refined)                     # (B, C*S, T)
        output_flat = self.output_act(output_flat)

        # 恢复为四维 (B, C, S, T)
        output = output_flat.view(B, C, S, T)
        return output


class Discriminator(nn.Module):
    """
    判别器 D: 区分真实目标域样本 vs 虚假生成样本
    使用谱归一化增强训练稳定性，加入膨胀卷积分支与自注意力/SE
    输入: (B, C, S, T) -> 输出: (B, 1)
    """
    def __init__(self, in_channels=3*56, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        self.use_spectral_norm = use_spectral_norm
        
        # 主干与膨胀分支（多尺度）
        conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        conv1_dilated = nn.Conv1d(in_channels, 64, kernel_size=3, padding=2, dilation=2)
        
        if use_spectral_norm:
            conv1 = spectral_norm(conv1)
            conv1_dilated = spectral_norm(conv1_dilated)
        
        self.conv1 = conv1
        self.conv1_dilated = conv1_dilated
        self.bn1 = nn.BatchNorm1d(64)
        self.bn1d = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.act1d = nn.LeakyReLU(0.2, inplace=True)
        self.pool1 = nn.MaxPool1d(2)
        self.pool1d = nn.MaxPool1d(2)
        self.sa = SelfAttention1d(64)
        
        conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        if use_spectral_norm:
            conv2 = spectral_norm(conv2)
        self.conv2 = conv2
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.se2 = SqueezeExcite1d(128)
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        
        # 最后的全连接层
        fc = nn.Linear(128, 1)
        if use_spectral_norm:
            fc = spectral_norm(fc)
        self.fc = fc

    def forward(self, x):
        # x: (B, C, S, T) -> reshape to (B, C*S, T)
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        
        # 多尺度第一层
        x_base = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x_dilated = self.pool1d(self.act1d(self.bn1d(self.conv1_dilated(x))))
        x = 0.5 * (x_base + x_dilated)
        x = self.sa(x)
        
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.se2(x)
        x = self.pool2(x)  # 自适应池化
        
        x = x.squeeze(-1)  # (B, 128)
        x = self.fc(x)     # (B, 1)
        
        # 注意：WGAN不需要sigmoid，直接输出分数
        return x


# 实例化模型
def build_model():
    E = FeatureExtractor()
    G = Generator()
    D = Discriminator()
    return E, G, D
