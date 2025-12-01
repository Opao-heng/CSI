import torch
import torch.nn as nn
import torch.nn.functional as F


"""
条件变分自编码器(CVAE)编码器
作用：将CSI数据和条件标签编码为潜在空间的均值和方差参数
"""
class CVAE_Encoder(nn.Module):
    def __init__(self, in_channels=3, subcarriers=56, latent_dim=128, num_classes=30):
        super(CVAE_Encoder, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.num_classes = num_classes
        
        # 标签嵌入层
        self.label_embedding = nn.Embedding(num_classes, 32)
        
        # 编码器骨干网络 (输入包含标签嵌入的通道)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels * subcarriers + 32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 添加dropout层防止过拟合
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 添加dropout层防止过拟合
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 添加dropout层防止过拟合
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 均值和方差的全连接层
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
    def forward(self, x, labels):
        B, S, C, T = x.shape  # 输入数据格式为 (N, 56, 3, 6000)
        # 调整输入数据格式以适应模型 (N, 3, 56, 6000)
        x_adjusted = x.permute(0, 2, 1, 3)  # 从 (N, 56, 3, 6000) 转换为 (N, 3, 56, 6000)
        x_adjusted = x_adjusted.contiguous().view(B, C * S, T)
        
        # 标签嵌入并扩展到时间维度
        label_embed = self.label_embedding(labels)  # (B, 32)
        label_embed = label_embed.unsqueeze(-1).expand(-1, -1, T)  # (B, 32, T)
        
        # 拼接数据和标签嵌入
        x_combined = torch.cat([x_adjusted, label_embed], dim=1)  # (B, C*S+32, T)
        
        # 通过编码器
        encoded = self.encoder(x_combined).squeeze(-1)
        
        # 生成均值和对数方差
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        return mu, logvar


"""
条件变分自编码器(CVAE)解码器
作用：从潜在空间和条件标签重构CSI数据
"""
class CVAE_Decoder(nn.Module):
    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000, latent_dim=128, num_classes=30):
        super(CVAE_Decoder, self).__init__()
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.time_steps = time_steps
        self.num_classes = num_classes
        
        # 标签嵌入层
        self.label_embedding = nn.Embedding(num_classes, 32)
        
        # 从潜在向量和标签投影到解码器初始维度
        self.fc = nn.Linear(latent_dim + 32, 256 * (time_steps // 16))
        
        # 解码器骨干网络
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 添加dropout层防止过拟合
            
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 添加dropout层防止过拟合
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 添加dropout层防止过拟合
            
            nn.ConvTranspose1d(32, in_channels * subcarriers, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z, labels):
        B = z.size(0)
        
        # 标签嵌入
        label_embed = self.label_embedding(labels)  # (B, 32)
        
        # 拼接潜在向量和标签嵌入
        z_combined = torch.cat([z, label_embed], dim=1)  # (B, latent_dim+32)
        
        # 投影潜在向量
        x = self.fc(z_combined).view(B, 256, self.time_steps // 16)
        
        # 通过解码器
        x = self.decoder(x)
        
        # 调整输出维度到目标时间步长
        if x.size(-1) != self.time_steps:
            x = F.interpolate(x, size=self.time_steps, mode='linear', align_corners=False)
        
        # 恢复原始形状 (N, 3, 56, 6000)
        x = x.view(B, self.in_channels, self.subcarriers, self.time_steps)
        
        # 调整输出维度以匹配输入数据格式 (N, 56, 3, 6000)
        x = x.permute(0, 2, 1, 3)  # 从 (N, 3, 56, 6000) 转换为 (N, 56, 3, 6000)
        
        return x


"""
完整的CVAE模型
作用：整合编码器和解码器，实现条件生成
"""
class CVAE(nn.Module):
    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000, latent_dim=128, num_classes=30):
        super(CVAE, self).__init__()
        self.encoder = CVAE_Encoder(in_channels, subcarriers, latent_dim, num_classes)
        self.decoder = CVAE_Decoder(in_channels, subcarriers, time_steps, latent_dim, num_classes)
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, labels):
        # 编码
        mu, logvar = self.encoder(x, labels)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码
        recon = self.decoder(z, labels)
        
        return recon, mu, logvar
    
    def generate(self, num_samples, labels, device='cuda'):
        """
        从先验分布生成指定标签的新样本
        参数:
            num_samples: 生成样本数量
            labels: 样本标签 (B,)
            device: 设备
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z, labels)
        return samples


"""
构建CVAE模型
"""
def build_cvae_model(num_classes=30):
    model = CVAE(num_classes=num_classes)
    return model