import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.backbone = nn.Sequential(
            nn.Conv1d(input_dim[0] * input_dim[1], 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1)  # 使用自适应池化确保输出尺寸为1
        )
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        # x: (B, C, S, T) -> reshape to (B, C*S, T)
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)  # 展平为 (B, C*S, T)

        # 经过骨干网络
        x = self.backbone(x)  # (B, 128, 1)
        x = x.squeeze(-1)  # (B, 128)
        x = self.fc(x)  # (B, d)
        return x


class Generator(nn.Module):
    """
    生成器 G: 融合源域身份特征与目标域环境特征，生成符合目标域分布的虚假样本
    输入: 源域数据 X_s + 目标域特征 F_t
    输出: 虚假目标域数据 X_hat_t
    使用 U-Net 架构（编码器-解码器对称结构）
    """

    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000, feature_dim=128):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.time_steps = time_steps

        # 输入维度为 (B, C, S, T) -> 展平为 (B, C*S, T)
        self.encoder = nn.Sequential(
            # Encoder
            nn.Conv1d(in_channels * subcarriers, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        self.feature_proj = nn.Linear(feature_dim, 256)

        self.decoder = nn.Sequential(
            # Decoder
            nn.ConvTranspose1d(256 + 256, 128, kernel_size=2, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(64, in_channels * subcarriers, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x_s, f_t):
        # x_s: (B, C, S, T), f_t: (目标域样本数, d)
        B, C, S, T = x_s.shape

        # 展平为 (B, C*S, T)
        x_s_flat = x_s.view(B, C * S, T)

        encoded = self.encoder(x_s_flat)  # (B, 256, T//8)

        # 处理目标域特征，使其与编码特征匹配
        if f_t.size(0) >= B:
            # 如果目标域特征数量大于等于批次大小，直接取前B个
            selected_f_t = f_t[:B]
        else:
            # 如果目标域特征数量小于批次大小，重复使用
            repeat_times = (B + f_t.size(0) - 1) // f_t.size(0)
            selected_f_t = f_t.repeat(repeat_times, 1)[:B]

        projected_f = self.feature_proj(selected_f_t).unsqueeze(-1)  # (B, 256, 1)

        # 调整 projected_f 的大小以匹配 encoded 的大小
        if projected_f.size(-1) != encoded.size(-1):
            # 使用 expand 确保尺寸完全匹配
            projected_f = projected_f.expand(-1, -1, encoded.size(-1))

        combined = torch.cat([encoded, projected_f], dim=1)  # (B, 512, T//8)
        output_flat = self.decoder(combined)  # (B, C*S, T)

        # 恢复为四维 (B, C, S, T)
        output = output_flat.view(B, C, S, T)
        return output


class Discriminator(nn.Module):
    """
    判别器 D: 区分真实目标域样本 vs 虚假生成样本
    输入: (B, C, S, T) -> 输出: (B, 1)
    """
    def __init__(self, in_channels=3*56):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # x: (B, C, S, T) -> reshape to (B, C*S, T)
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        x = self.net(x)  # (B, 256, 1)
        x = x.squeeze(-1)  # (B, 256)
        x = self.fc(x)  # (B, 1)
        return torch.sigmoid(x)


# 实例化模型
def build_model():
    E = FeatureExtractor()
    G = Generator()
    D = Discriminator()
    return E, G, D
