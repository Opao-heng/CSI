import torch
import torch.nn as nn
import torch.nn.functional as F


"""
变分自编码器(VAE)编码器
作用：将CSI数据编码为潜在空间的均值和方差参数
"""
class VAE_Encoder(nn.Module):
    def __init__(self, in_channels=3, subcarriers=56, latent_dim=128):
        super(VAE_Encoder, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        
        # 编码器骨干网络
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels * subcarriers, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 均值和方差的全连接层
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
    def forward(self, x):
        B, C, S, T = x.shape
        x = x.view(B, C * S, T)
        
        # 通过编码器
        encoded = self.encoder(x).squeeze(-1)
        
        # 生成均值和对数方差
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        return mu, logvar


"""
变分自编码器(VAE)解码器
作用：从潜在空间重构CSI数据
"""
class VAE_Decoder(nn.Module):
    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000, latent_dim=128):
        super(VAE_Decoder, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.time_steps = time_steps
        
        # 从潜在向量投影到解码器初始维度
        self.fc = nn.Linear(latent_dim, 256 * (time_steps // 16))
        
        # 解码器骨干网络
        self.decoder = nn.Sequential(
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
        
        # 通过解码器
        x = self.decoder(x)
        
        # 调整输出维度到目标时间步长
        if x.size(-1) != self.time_steps:
            x = F.interpolate(x, size=self.time_steps, mode='linear', align_corners=False)
        
        # 恢复原始形状
        return x.view(B, self.in_channels, self.subcarriers, self.time_steps)


"""
完整的VAE模型
作用：整合编码器和解码器，实现重参数化技巧
"""
class VAE(nn.Module):
    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(in_channels, subcarriers, latent_dim)
        self.decoder = VAE_Decoder(in_channels, subcarriers, time_steps, latent_dim)
        
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：z = mu + sigma * epsilon
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
        recon = self.decoder(z)
        
        return recon, mu, logvar
    
    def generate(self, num_samples, device='cuda'):
        """
        从先验分布生成新样本
        """
        z = torch.randn(num_samples, self.encoder.fc_mu.out_features).to(device)
        samples = self.decoder(z)
        return samples


"""
构建VAE模型
"""
def build_vae_model():
    model = VAE()
    return model
