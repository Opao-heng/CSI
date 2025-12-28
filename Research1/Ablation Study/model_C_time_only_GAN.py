"""
Model C: w/o Freq-D, Time-only GAN
保留生成器和交叉注意力,但仅使用时域判别器(移除频域判别器)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class AdaptiveInstanceNorm1d(nn.Module):
    """自适应实例归一化层"""
    def __init__(self, num_features, feature_dim):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(feature_dim, num_features * 2)
        
    def forward(self, x, style):
        x = self.norm(x)
        style_params = self.fc(style)
        gamma, beta = style_params.chunk(2, dim=1)
        return (1 + gamma.unsqueeze(-1)) * x + beta.unsqueeze(-1)


class FeatureExtractor(nn.Module):
    """特征提取器"""
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
    """生成器 - 使用AdaIN"""
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
        
        # 解码器 + AdaIN
        self.dec1 = nn.ConvTranspose1d(512 + 512, 256, kernel_size=4, stride=2, padding=1)
        self.adain1 = AdaptiveInstanceNorm1d(256, feature_dim)
        
        self.dec2 = nn.ConvTranspose1d(256 + 256, 128, kernel_size=6, stride=4, padding=1)
        self.adain2 = AdaptiveInstanceNorm1d(128, feature_dim)
        
        self.dec3 = nn.ConvTranspose1d(128 + 128, 64, kernel_size=8, stride=4, padding=2)
        self.adain3 = AdaptiveInstanceNorm1d(64, feature_dim)
        
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
        
        # 解码 + 跳连 + AdaIN
        d1 = F.leaky_relu(self.dec1(h), 0.2)
        d1 = self._match_size(d1, e2)
        d1 = self.adain1(d1, style)
        d1 = torch.cat([d1, e2], dim=1)
        
        d2 = F.leaky_relu(self.dec2(d1), 0.2)
        d2 = self._match_size(d2, e1)
        d2 = self.adain2(d2, style)
        d2 = torch.cat([d2, e1], dim=1)
        
        d3 = F.leaky_relu(self.dec3(d2), 0.2)
        d3 = self._match_size(d3, x)
        d3 = self.adain3(d3, style)
        
        output = self.output_conv(d3)
        if output.size(-1) != T:
            output = F.interpolate(output, size=T, mode='linear', align_corners=False)
        
        return output.view(B, C, S, T)
    
    def _match_size(self, x, target):
        if x.size(-1) != target.size(-1):
            x = F.interpolate(x, size=target.size(-1), mode='linear', align_corners=False)
        return x


class Discriminator(nn.Module):
    """时域判别器 - Model C仅使用此判别器"""
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


def build_model():
    """构建Model C: 特征提取器 + 生成器 + 仅时域判别器"""
    E = FeatureExtractor()
    G = Generator()
    D = Discriminator()
    return E, G, D
