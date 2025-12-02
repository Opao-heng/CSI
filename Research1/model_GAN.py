import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class AdaptiveInstanceNorm1d(nn.Module):
    """
    自适应实例归一化(AdaIN)层
    作用：使用目标域特征作为风格条件，对输入特征进行自适应的实例归一化和仿射变换
    """
    def __init__(self, num_features, feature_dim):
        super(AdaptiveInstanceNorm1d, self).__init__()
        # 实例归一化层，不使用可学习的仿射参数
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        # 从风格特征生成gamma和beta参数
        self.fc = nn.Linear(feature_dim, num_features * 2)
        
    def forward(self, x, style):
        # 对输入进行实例归一化
        x = self.norm(x)
        # 从风格特征生成仿射参数
        style_params = self.fc(style)
        # 拆分为gamma和beta参数
        gamma, beta = style_params.chunk(2, dim=1)
        # 应用风格化的仿射变换
        return (gamma.unsqueeze(-1) * x + beta.unsqueeze(-1))


class FeatureExtractor(nn.Module):
    """
    特征提取器（增强版）
    作用：提取目标域CSI的环境特征，将高维CSI数据压缩到低维特征空间
    优化：增加网络深度和残差连接，提升特征表达能力
    """
    def __init__(self, input_dim=(3, 56, 6000), feature_dim=128):
        super(FeatureExtractor, self).__init__()
        # 多层卷积骨干网络：逐步提取和压缩特征（增强版）
        self.backbone = nn.Sequential(
            # 第一层卷积：初步特征提取
            nn.Conv1d(input_dim[0] * input_dim[1], 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # 第二层卷积：进一步特征提取和下采样
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 新增第三层：提升特征表达能力
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # 自适应全局平均池化：统一输出大小
            nn.AdaptiveAvgPool1d(1)
        )
        # 防止过拟合的Dropout层
        self.dropout = nn.Dropout(0.3)
        # 投影到目标特征维度（两层全连接）
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        # 解析输入维度
        B, C, S, T = x.shape
        # 展平通道和子载波维度为单一维度
        x = x.view(B, C * S, T)
        # 通过骨干网络提取特征
        x = self.backbone(x).squeeze(-1)
        # 应用Dropout正则化
        x = self.dropout(x)
        # 投影到目标特征维度
        return self.fc(x)


class Generator(nn.Module):
    """
    生成器网络（优化版）
    作用：融合源域身份特征与目标域环境特征，生成符合目标域分布的虚假样本
    优化：增强特征融合能力，添加注意力机制和更深的架构
    """
    def __init__(self, in_channels=3, subcarriers=56, time_steps=6000, feature_dim=128):
        super(Generator, self).__init__()
        # 保存输入参数
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.time_steps = time_steps
        # 编码器第一层：卷积和归一化（优化激活函数）
        self.enc1 = nn.Sequential(nn.Conv1d(in_channels * subcarriers, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(64), nn.LeakyReLU(0.2, inplace=True))
        self.pool1 = nn.MaxPool1d(2)
        
        # 编码器第二层：进一步降采样和特征提取（优化激活函数）
        self.enc2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(128), nn.LeakyReLU(0.2, inplace=True))
        self.pool2 = nn.MaxPool1d(2)
        
        # 编码器第三层：更深的特征提取
        self.enc3 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(256), nn.LeakyReLU(0.2, inplace=True))
        self.pool3 = nn.MaxPool1d(2)
        
        # 目标域特征投影（增强版：多层映射）
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256)
        )
        
        # 多层自适应实例归一化，用于风格融合
        self.adain1 = AdaptiveInstanceNorm1d(128, feature_dim)
        self.adain2 = AdaptiveInstanceNorm1d(64, feature_dim)
        
        # 解码器第一层：转置卷积上采样
        self.dec1 = nn.ConvTranspose1d(256 + 256, 256, kernel_size=2, stride=2)
        self.dec1_norm = nn.BatchNorm1d(256)
        self.dec1_act = nn.LeakyReLU(0.2, inplace=True)
        
        # 解码器第二层
        self.dec2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2_norm = nn.BatchNorm1d(128)
        
        # 细化卷积1：融合跳连特征
        self.refine_conv1 = nn.Conv1d(128 + 128, 128, kernel_size=1)
        self.refine_norm1 = nn.BatchNorm1d(128)
        self.refine_act1 = nn.LeakyReLU(0.2, inplace=True)
        
        # 解码器第三层：恢复原始分辨率
        self.dec3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec3_norm = nn.BatchNorm1d(64)
        
        # 细化卷积2：融合跳连特征
        self.refine_conv2 = nn.Conv1d(64 + 64, 64, kernel_size=1)
        self.refine_norm2 = nn.BatchNorm1d(64)
        self.refine_act2 = nn.LeakyReLU(0.2, inplace=True)
        
        # 最终输出层
        self.output_conv = nn.Conv1d(64, in_channels * subcarriers, kernel_size=3, padding=1)
        self.output_act = nn.Tanh()

    def forward(self, x_s, f_t):
        # 获取输入维度信息
        B, C, S, T = x_s.shape
        # 展平空间维度以适应1D卷积
        x_s_flat = x_s.view(B, C * S, T)
        
        # 编码阶段：逐层下采样提取源域身份特征（三层）
        e1 = self.enc1(x_s_flat)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        encoded = self.pool3(e3)
        
        # 处理目标域特征：对齐批量大小以匹配源域批量
        if f_t.size(0) >= B:
            selected_f_t = f_t[:B]
        else:
            repeat_times = (B + f_t.size(0) - 1) // f_t.size(0)
            selected_f_t = f_t.repeat(repeat_times, 1)[:B]
        
        # 特征投影与融合：将目标域环境特征投影为相同通道数
        projected_f = self.feature_proj(selected_f_t).unsqueeze(-1)
        projected_f = projected_f.expand(-1, -1, encoded.size(-1))
        combined = torch.cat([encoded, projected_f], dim=1)
        
        # 解码阶段：转置卷积上采样并应用自适应风格化
        d1 = self.dec1(combined)
        d1 = self.dec1_norm(d1)
        d1 = self.dec1_act(d1)
        
        # 第二层解码：应用风格化
        d2 = self.dec2(d1)
        d2 = self.dec2_norm(d2)
        d2 = self.adain1(d2, selected_f_t)
        
        # 跳连融合1：将编码器第二层特征与解码特征融合
        # 确保e2_down与d2具有相同的时间维度
        if e2.size(-1) != d2.size(-1):
            e2_down = F.interpolate(e2, size=d2.size(-1), mode='linear', align_corners=False)
        else:
            e2_down = e2
        refined1 = torch.cat([d2, e2_down], dim=1)
        refined1 = self.refine_conv1(refined1)
        refined1 = self.refine_norm1(refined1)
        refined1 = self.refine_act1(refined1)
        
        # 第三层解码：恢复原始分辨率
        d3 = self.dec3(refined1)
        d3 = self.dec3_norm(d3)
        d3 = self.adain2(d3, selected_f_t)
        
        # 跳连融合2：将编码器第一层特征与解码特征融合
        # 确保e1与d3具有相同的时间维度
        if e1.size(-1) != d3.size(-1):
            e1_resized = F.interpolate(e1, size=d3.size(-1), mode='linear', align_corners=False)
        else:
            e1_resized = e1
        refined2 = torch.cat([d3, e1_resized], dim=1)
        refined2 = self.refine_conv2(refined2)
        refined2 = self.refine_norm2(refined2)
        refined2 = self.refine_act2(refined2)
        
        # 最终输出：进行最后的卷积并应用激活函数
        output_flat = self.output_conv(refined2)
        output_flat = self.output_act(output_flat)
        
        # 恢复原始形状
        return output_flat.view(B, C, S, T)


class Discriminator(nn.Module):
    """
    时域判别器
    作用：区分真实目标域样本和虚假生成样本，为生成器提供训练梯度
    """
    def __init__(self, in_channels=3*56, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        # 保存是否使用谱归一化的标志
        self.use_spectral_norm = use_spectral_norm
        # 第一层卷积：标准卷积分支
        conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        # 第一层卷积：膨胀卷积分支，扩大感受野
        conv1_dilated = nn.Conv1d(in_channels, 64, kernel_size=3, padding=2, dilation=2)
        # 应用谱归一化稳定WGAN训练
        if use_spectral_norm:
            conv1 = spectral_norm(conv1)
            conv1_dilated = spectral_norm(conv1_dilated)
        self.conv1 = conv1
        self.conv1_dilated = conv1_dilated
        # 标准分支的批归一化
        self.bn1 = nn.BatchNorm1d(64)
        # 膨胀分支的批归一化
        self.bn1d = nn.BatchNorm1d(64)
        # 标准分支的激活函数
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        # 膨胀分支的激活函数
        self.act1d = nn.LeakyReLU(0.2, inplace=True)
        # 标准分支的池化
        self.pool1 = nn.MaxPool1d(2)
        # 膨胀分支的池化
        self.pool1d = nn.MaxPool1d(2)
        # 第二层卷积：融合多尺度特征
        conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        if use_spectral_norm:
            conv2 = spectral_norm(conv2)
        self.conv2 = conv2
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        # 全局平均池化
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        # 全连接层：输出判别分数
        fc = nn.Linear(128, 1)
        if use_spectral_norm:
            fc = spectral_norm(fc)
        self.fc = fc

    def forward(self, x):
        # 解析输入维度
        B, C, S, T = x.shape
        # 展平通道和子载波维度为单一维度
        x = x.view(B, C * S, T)
        
        # 多尺度特征融合：标准卷积分支获取局部特征
        x_base = self.pool1(self.act1(self.bn1(self.conv1(x))))
        # 多尺度特征融合：膨胀卷积分支扩大感受野
        x_dilated = self.pool1d(self.act1d(self.bn1d(self.conv1_dilated(x))))
        # 融合两个分支的特征，取平均组合不同感受野的信息
        x = 0.5 * (x_base + x_dilated)
        
        # 第二层处理：进一步下采样和特征提取
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # 全局平均池化获取全局判别信息
        x = self.pool2(x)
        # 输出判别分数
        return self.fc(x.squeeze(-1))


class SpectralDiscriminator(nn.Module):
    """
    频域判别器
    作用：在频谱域区分真实和生成的目标域样本，捕捉频域特征差异
    """
    def __init__(self, in_channels=3*56, use_spectral_norm=True):
        super(SpectralDiscriminator, self).__init__()
        # 保存是否使用谱归一化的标志
        self.use_spectral_norm = use_spectral_norm
        # 第一层卷积：处理频谱信息
        conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        # 第二层卷积：进一步特征提取
        conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        # 应用谱归一化稳定训练
        if use_spectral_norm:
            conv1 = spectral_norm(conv1)
            conv2 = spectral_norm(conv2)
        self.conv1 = conv1
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        # 第一层池化
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = conv2
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        # 全局平均池化
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        # 全连接层：输出判别分数
        fc = nn.Linear(128, 1)
        if use_spectral_norm:
            fc = spectral_norm(fc)
        self.fc = fc

    def forward(self, x):
        # 解析输入维度
        B, C, S, T = x.shape
        # 展平通道和子载波维度为单一维度
        x = x.view(B, C * S, T)
        
        # 转换到频域：使用FFT获取频谱表示
        x_fft = torch.fft.rfft(x, dim=-1)
        # 提取幅度谱信息，频域判别器关注频率特征
        x_mag = x_fft.abs()
        
        # 第一层卷积处理频域特征
        x = self.conv1(x_mag)
        x = self.bn1(x)
        x = self.act1(x)
        # 池化下采样
        x = self.pool1(x)
        # 第二层卷积进一步提取频域判别特征
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # 全局平均池化获取全局频域信息
        x = self.pool2(x)
        # 输出判别分数
        return self.fc(x.squeeze(-1))


def build_model():
    """
    构建GAN模型
    作用：初始化并返回GAN的所有组件
    """

    # 初始化特征提取器：从目标域提取环境特征
    E = FeatureExtractor()
    # 初始化生成器：融合源域和目标域特征生成虚假样本
    G = Generator()
    # 初始化时域判别器：在时间维度区分真假样本
    D = Discriminator()
    # 初始化频域判别器：在频谱维度区分真假样本
    D_spectral = SpectralDiscriminator()
    return E, G, D, D_spectral
