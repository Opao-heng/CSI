import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    特征提取器 - 与原模型保持一致
    """
    def __init__(self, input_dim=(3, 56, 6000), feature_dim=128):
        super(FeatureExtractor, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(input_dim[0] * input_dim[1], 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        x = self.backbone(x).squeeze(-1)
        return self.fc(x)


class Generator(nn.Module):
    """
    DCGAN生成器 - 深度卷积生成对抗网络
    """
    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000, latent_dim=100, condition_dim=128):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.flat_channels = in_channels * subcarriers
        self.latent_dim = latent_dim
        
        # 条件融合层
        self.condition_fc = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.ReLU(True)
        )
        
        # 初始全连接层: 将噪声和条件映射到特征空间
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 128, 256 * 94),  # 94 = 6000 / (4*4*4)
            nn.BatchNorm1d(256 * 94),
            nn.ReLU(True)
        )
        
        # 转置卷积层（上采样）
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(True)
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv1d(32, self.flat_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, z, condition):
        """
        前向传播
        z: 噪声向量 (B, latent_dim)
        condition: 条件特征 (B, condition_dim)
        """
        B = z.size(0)
        
        # 条件融合
        cond_feat = self.condition_fc(condition)
        h = torch.cat([z, cond_feat], dim=1)
        
        # 全连接层
        h = self.fc(h)
        h = h.view(B, 256, -1)  # (B, 256, 94)
        
        # 转置卷积上采样
        h = self.deconv1(h)  # (B, 128, 376)
        h = self.deconv2(h)  # (B, 64, 1504)
        h = self.deconv3(h)  # (B, 32, 6016)
        
        # 输出
        output = self.output(h)
        
        # 调整到目标时间长度
        if output.size(-1) != 6000:
            output = F.interpolate(output, size=6000, mode='linear', align_corners=False)
        
        return output.view(B, self.in_channels, self.subcarriers, 6000)


class Discriminator(nn.Module):
    """
    DCGAN判别器 - 深度卷积判别网络
    """
    def __init__(self, in_channels=3*56):
        super(Discriminator, self).__init__()
        
        # 卷积层（下采样）
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 全局平均池化
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        
        # 卷积下采样
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # 全局池化
        x = self.pool(x).squeeze(-1)
        
        # 输出
        return self.fc(x)


def build_model():
    """
    构建DCGAN模型
    """
    E = FeatureExtractor()
    G = Generator()
    D = Discriminator()
    return E, G, D
