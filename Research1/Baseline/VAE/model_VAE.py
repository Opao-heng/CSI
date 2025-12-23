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


class Encoder(nn.Module):
    """
    VAE编码器 - 编码CSI数据到潜在空间
    """
    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000, latent_dim=256):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.flat_channels = in_channels * subcarriers
        
        # 编码器网络
        self.enc1 = nn.Sequential(
            nn.Conv1d(self.flat_channels, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 均值和方差层
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
    def forward(self, x):
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2).squeeze(-1)
        
        # 计算均值和对数方差
        mu = self.fc_mu(e3)
        logvar = self.fc_logvar(e3)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    VAE解码器 - 从潜在空间重建CSI数据
    """
    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000, latent_dim=256):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.flat_channels = in_channels * subcarriers
        
        # 初始全连接层
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2)
        )
        
        # 解码器网络
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=6, stride=4, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv1d(64, self.flat_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, z):
        # 全连接层
        h = self.fc(z).unsqueeze(-1)
        
        # 解码
        d1 = self.dec1(h)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        
        # 输出
        output = self.output_conv(d3)
        
        # 调整到目标时间长度
        if output.size(-1) != 6000:
            output = F.interpolate(output, size=6000, mode='linear', align_corners=False)
        
        B = z.size(0)
        return output.view(B, self.in_channels, self.subcarriers, 6000)


class VAE(nn.Module):
    """
    变分自编码器 (Variational Autoencoder)
    """
    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000, latent_dim=256):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, subcarriers, time_steps, latent_dim)
        self.decoder = Decoder(in_channels, subcarriers, time_steps, latent_dim)
        
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # 编码
        mu, logvar = self.encoder(x)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar
    
    def generate(self, num_samples=1, device='cuda'):
        """
        生成新样本
        """
        # 从标准正态分布采样
        z = torch.randn(num_samples, 256).to(device)
        
        # 解码
        generated = self.decoder(z)
        return generated


def build_model():
    """
    构建VAE模型
    """
    E = FeatureExtractor()
    vae = VAE()
    return E, vae
