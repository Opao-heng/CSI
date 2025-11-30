import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


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
    使用 U-Net 架构 + AdaIN条件调制
    """

    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000, feature_dim=128):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.time_steps = time_steps

        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels * subcarriers, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool1d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool1d(2)

        # 目标域特征投影
        self.feature_proj = nn.Linear(feature_dim, 256)
        
        # AdaIN层，在解码器中使用
        self.adain1 = AdaptiveInstanceNorm1d(128, feature_dim)
        self.adain2 = AdaptiveInstanceNorm1d(64, feature_dim)

        # 解码器
        self.dec1 = nn.ConvTranspose1d(256 + 256, 128, kernel_size=2, stride=2)
        self.dec1_norm = nn.BatchNorm1d(128)
        self.dec1_act = nn.ReLU(inplace=True)
        
        self.dec2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec2_norm = nn.BatchNorm1d(64)
        self.dec2_act = nn.ReLU(inplace=True)
        
        self.dec3 = nn.ConvTranspose1d(64, in_channels * subcarriers, kernel_size=2, stride=2)
        self.output_act = nn.Tanh()

    def forward(self, x_s, f_t):
        # x_s: (B, C, S, T), f_t: (目标域样本数, d)
        B, C, S, T = x_s.shape

        # 展平为 (B, C*S, T)
        x_s_flat = x_s.view(B, C * S, T)

        # 编码器
        e1 = self.enc1(x_s_flat)      # (B, 64, T)
        p1 = self.pool1(e1)            # (B, 64, T/2)
        
        e2 = self.enc2(p1)             # (B, 128, T/2)
        p2 = self.pool2(e2)            # (B, 128, T/4)
        
        e3 = self.enc3(p2)             # (B, 256, T/4)
        encoded = self.pool3(e3)       # (B, 256, T/8)

        # 处理目标域特征，使其与编码特征匹配
        if f_t.size(0) >= B:
            selected_f_t = f_t[:B]
        else:
            repeat_times = (B + f_t.size(0) - 1) // f_t.size(0)
            selected_f_t = f_t.repeat(repeat_times, 1)[:B]

        # 投影目标域特征
        projected_f = self.feature_proj(selected_f_t).unsqueeze(-1)  # (B, 256, 1)
        projected_f = projected_f.expand(-1, -1, encoded.size(-1))   # (B, 256, T/8)

        # 融合编码特征和目标域特征
        combined = torch.cat([encoded, projected_f], dim=1)  # (B, 512, T/8)
        
        # 解码器（使用AdaIN调制）
        d1 = self.dec1(combined)                  # (B, 128, T/4)
        d1 = self.dec1_norm(d1)
        d1 = self.adain1(d1, selected_f_t)        # AdaIN调制
        d1 = self.dec1_act(d1)
        
        d2 = self.dec2(d1)                        # (B, 64, T/2)
        d2 = self.dec2_norm(d2)
        d2 = self.adain2(d2, selected_f_t)        # AdaIN调制
        d2 = self.dec2_act(d2)
        
        output_flat = self.dec3(d2)               # (B, C*S, T)
        output_flat = self.output_act(output_flat)

        # 恢复为四维 (B, C, S, T)
        output = output_flat.view(B, C, S, T)
        return output


class Discriminator(nn.Module):
    """
    判别器 D: 区分真实目标域样本 vs 虚假生成样本
    使用谱归一化增强训练稳定性
    输入: (B, C, S, T) -> 输出: (B, 1)
    """
    def __init__(self, in_channels=3*56, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        self.use_spectral_norm = use_spectral_norm
        
        # 如果启用谱归一化，对所有卷积层应用
        conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        if use_spectral_norm:
            conv1 = spectral_norm(conv1)
            conv2 = spectral_norm(conv2)
            conv3 = spectral_norm(conv3)
        
        self.conv1 = conv1
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = conv2
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = conv3
        self.bn3 = nn.BatchNorm1d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
        # 最后的全连接层，也应用谱归一化
        fc = nn.Linear(256, 1)
        if use_spectral_norm:
            fc = spectral_norm(fc)
        self.fc = fc

    def forward(self, x):
        # x: (B, C, S, T) -> reshape to (B, C*S, T)
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        # 第三层
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)
        
        x = x.squeeze(-1)  # (B, 256)
        x = self.fc(x)     # (B, 1)
        
        # 注意：WGAN不需要sigmoid，直接输出分数
        return x


# 实例化模型
def build_model():
    E = FeatureExtractor()
    G = Generator()
    D = Discriminator()
    return E, G, D
