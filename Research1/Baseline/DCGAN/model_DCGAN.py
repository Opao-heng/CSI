import torch
import torch.nn as nn


"""
DCGAN生成器
作用：从噪声生成CSI数据
"""
class DCGAN_Generator(nn.Module):
    def __init__(self, latent_dim=100, in_channels=3, subcarriers=56, time_steps=6000):
        super(DCGAN_Generator, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.time_steps = time_steps
        
        # 从潜在向量投影到初始维度
        self.fc = nn.Linear(latent_dim, 256 * (time_steps // 16))
        
        # 生成器主干
        self.generator = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(32, in_channels * subcarriers, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        B = z.size(0)
        
        # 投影潜在向量
        x = self.fc(z).view(B, 256, self.time_steps // 16)
        
        # 通过生成器
        x = self.generator(x)
        
        # 调整到目标时间步长
        if x.size(-1) != self.time_steps:
            x = torch.nn.functional.interpolate(x, size=self.time_steps, mode='linear', align_corners=False)
        
        # 恢复原始形状
        return x.view(B, self.in_channels, self.subcarriers, self.time_steps)


"""
DCGAN判别器
作用：判别样本是真实还是生成的
"""
class DCGAN_Discriminator(nn.Module):
    def __init__(self, in_channels=3*56):
        super(DCGAN_Discriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        
        # 通过判别器
        x = self.discriminator(x).squeeze(-1)
        
        # 输出判别分数
        x = self.fc(x)
        return self.sigmoid(x)


"""
构建DCGAN模型
"""
def build_dcgan_model(latent_dim=100):
    generator = DCGAN_Generator(latent_dim=latent_dim)
    discriminator = DCGAN_Discriminator()
    return generator, discriminator
