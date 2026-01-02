import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


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


class ResidualBlock(nn.Module):
    """
    残差块 - 用于CycleGAN生成器
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm1d(channels)
        )
        
    def forward(self, x):
        return x + self.block(x)


class Generator_S2T(nn.Module):
    """
    CycleGAN生成器: 源域 -> 目标域
    """
    def __init__(self, in_channels=3, subcarriers=56, num_residual_blocks=3):  # 从6降低到3
        super(Generator_S2T, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.flat_channels = in_channels * subcarriers
        
        # 初始卷积层
        self.initial = nn.Sequential(
            nn.Conv1d(self.flat_channels, 64, kernel_size=7, padding=3),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # 下采样层
        self.down1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        residual_blocks = []
        for _ in range(num_residual_blocks):
            residual_blocks.append(ResidualBlock(256))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # 上采样层
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv1d(64, self.flat_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x):
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        
        # 编码
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        
        # 残差变换
        x = self.residual_blocks(x)
        
        # 解码
        x = self.up1(x)
        x = self.up2(x)
        output = self.output(x)
        
        # 调整到目标时间长度
        if output.size(-1) != T:
            output = F.interpolate(output, size=T, mode='linear', align_corners=False)
        
        return output.view(B, C, S, T)


class Generator_T2S(nn.Module):
    """
    CycleGAN生成器: 目标域 -> 源域
    """
    def __init__(self, in_channels=3, subcarriers=56, num_residual_blocks=3):  # 从6降低到3
        super(Generator_T2S, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.flat_channels = in_channels * subcarriers
        
        # 初始卷积层
        self.initial = nn.Sequential(
            nn.Conv1d(self.flat_channels, 64, kernel_size=7, padding=3),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # 下采样层
        self.down1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        residual_blocks = []
        for _ in range(num_residual_blocks):
            residual_blocks.append(ResidualBlock(256))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # 上采样层
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv1d(64, self.flat_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x):
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        
        # 编码
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        
        # 残差变换
        x = self.residual_blocks(x)
        
        # 解码
        x = self.up1(x)
        x = self.up2(x)
        output = self.output(x)
        
        # 调整到目标时间长度
        if output.size(-1) != T:
            output = F.interpolate(output, size=T, mode='linear', align_corners=False)
        
        return output.view(B, C, S, T)


class Discriminator(nn.Module):
    """
    PatchGAN判别器
    """
    def __init__(self, in_channels=3*56, use_spectral_norm=False):
        super(Discriminator, self).__init__()
        
        def make_conv(in_ch, out_ch, kernel_size=4, stride=2, padding=1):
            conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
            return spectral_norm(conv) if use_spectral_norm else conv
        
        self.model = nn.Sequential(
            make_conv(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            make_conv(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            make_conv(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            make_conv(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(512, 1, kernel_size=4, stride=1, padding=1)
        )
        
    def forward(self, x):
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        return self.model(x)


def build_model():
    """
    构建CycleGAN模型
    """
    E = FeatureExtractor()
    G_S2T = Generator_S2T()
    G_T2S = Generator_T2S()
    D_S = Discriminator()
    D_T = Discriminator()
    return E, G_S2T, G_T2S, D_S, D_T
