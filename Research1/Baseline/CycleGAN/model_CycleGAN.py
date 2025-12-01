import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


"""
CycleGAN生成器
作用：将源域CSI数据转换为目标域风格
"""
class CycleGAN_Generator(nn.Module):
    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000):
        super(CycleGAN_Generator, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels * subcarriers, 64, kernel_size=7, padding=3),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(64, in_channels * subcarriers, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x):
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        
        # 编码
        x = self.encoder(x)
        
        # 残差块
        x = self.res_blocks(x)
        
        # 解码
        x = self.decoder(x)
        
        # 恢复原始形状
        return x.view(B, C, S, T)


"""
残差块
作用：增强网络表达能力并稳定训练
"""
class ResidualBlock(nn.Module):
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


"""
CycleGAN判别器
作用：判别样本是真实还是生成的
"""
class CycleGAN_Discriminator(nn.Module):
    def __init__(self, in_channels=3*56, use_spectral_norm=True):
        super(CycleGAN_Discriminator, self).__init__()
        
        # PatchGAN判别器
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv1d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv1d(512, 1, kernel_size=4, padding=1)
        )
        
        # 应用谱归一化
        if use_spectral_norm:
            for layer in self.model:
                if isinstance(layer, nn.Conv1d):
                    spectral_norm(layer)
        
    def forward(self, x):
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        return self.model(x)


"""
构建CycleGAN模型
"""
def build_cyclegan_model():
    # 源域到目标域的生成器
    G_S2T = CycleGAN_Generator()
    # 目标域到源域的生成器
    G_T2S = CycleGAN_Generator()
    # 源域判别器
    D_S = CycleGAN_Discriminator()
    # 目标域判别器
    D_T = CycleGAN_Discriminator()
    
    return G_S2T, G_T2S, D_S, D_T
