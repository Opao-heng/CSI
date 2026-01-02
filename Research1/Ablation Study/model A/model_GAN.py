import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class FeatureExtractor(nn.Module):
    """
    特征提取器
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
    生成器 - 移除AdaIN风格注入层（消融实验Model B）
    """
    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000, feature_dim=128):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.flat_channels = in_channels * subcarriers
        
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv1d(self.flat_channels, 128, kernel_size=7, stride=4, padding=3),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=4, padding=2),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 中间处理层
        self.middle = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 特征融合
        self.style_fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512)
        )
        
        # 解码器 - 移除AdaIN，使用普通的InstanceNorm
        self.dec1 = nn.ConvTranspose1d(512 + 512, 256, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.InstanceNorm1d(256)
        
        self.dec2 = nn.ConvTranspose1d(256 + 256, 128, kernel_size=6, stride=4, padding=1)
        self.norm2 = nn.InstanceNorm1d(128)
        
        self.dec3 = nn.ConvTranspose1d(128 + 128, 64, kernel_size=8, stride=4, padding=2)
        self.norm3 = nn.InstanceNorm1d(64)
        
        self.output_conv = nn.Sequential(
            nn.Conv1d(64, self.flat_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x_s, f_t):
        B, C, S, T = x_s.shape
        x = x_s.view(B, C * S, T)
        
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # 中间处理
        h = self.middle(e3)
        
        # 特征融合
        if f_t.size(0) >= B:
            style = f_t[:B]
        else:
            repeat_times = (B + f_t.size(0) - 1) // f_t.size(0)
            style = f_t.repeat(repeat_times, 1)[:B]
        
        style_feat = self.style_fc(style).unsqueeze(-1).expand(-1, -1, h.size(-1))
        h = torch.cat([h, style_feat], dim=1)
        
        # 解码 + 跳连 - 使用普通的InstanceNorm替代AdaIN
        d1 = F.leaky_relu(self.dec1(h), 0.2)
        d1 = self._match_size(d1, e2)
        d1 = self.norm1(d1)  # 替代AdaIN
        d1 = torch.cat([d1, e2], dim=1)
        
        d2 = F.leaky_relu(self.dec2(d1), 0.2)
        d2 = self._match_size(d2, e1)
        d2 = self.norm2(d2)  # 替代AdaIN
        d2 = torch.cat([d2, e1], dim=1)
        
        d3 = F.leaky_relu(self.dec3(d2), 0.2)
        d3 = self._match_size(d3, x)
        d3 = self.norm3(d3)  # 替代AdaIN
        
        output = self.output_conv(d3)
        if output.size(-1) != T:
            output = F.interpolate(output, size=T, mode='linear', align_corners=False)
        
        return output.view(B, C, S, T)
    
    def _match_size(self, x, target):
        if x.size(-1) != target.size(-1):
            x = F.interpolate(x, size=target.size(-1), mode='linear', align_corners=False)
        return x


class Discriminator(nn.Module):
    """
    时域判别器
    """
    def __init__(self, in_channels=3*56, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        
        def make_conv(in_ch, out_ch, kernel_size=4, stride=4, padding=0):
            conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
            return spectral_norm(conv) if use_spectral_norm else conv
        
        self.net = nn.Sequential(
            make_conv(in_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            make_conv(64, 128, kernel_size=5, stride=4, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            make_conv(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            make_conv(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = spectral_norm(nn.Linear(512, 1)) if use_spectral_norm else nn.Linear(512, 1)
        
    def forward(self, x):
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        x = self.net(x).squeeze(-1)
        return self.fc(x)


class SpectralDiscriminator(nn.Module):
    """
    频域判别器
    """
    def __init__(self, in_channels=3*56, use_spectral_norm=True):
        super(SpectralDiscriminator, self).__init__()
        
        def make_conv(in_ch, out_ch, kernel_size=4, stride=4, padding=0):
            conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
            return spectral_norm(conv) if use_spectral_norm else conv
        
        self.net = nn.Sequential(
            make_conv(in_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            make_conv(64, 128, kernel_size=5, stride=4, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            make_conv(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            make_conv(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = spectral_norm(nn.Linear(512, 1)) if use_spectral_norm else nn.Linear(512, 1)
        
    def forward(self, x):
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        
        # 转换到频域
        x_fft = torch.fft.rfft(x, dim=-1)
        x_mag = x_fft.abs()
        
        x = self.net(x_mag).squeeze(-1)
        return self.fc(x)


def build_model():
    E = FeatureExtractor()
    G = Generator()
    D = Discriminator()
    D_spectral = SpectralDiscriminator()
    return E, G, D, D_spectral
