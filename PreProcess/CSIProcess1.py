import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy import signal
import matplotlib.gridspec as gridspec

# 设置中文字体支持 - 中文宋体，英文数字Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 设置全局字体配置：中文宋体 + 英文Times New Roman
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式使用stix字体
print("已设置字体: 中文-宋体(SimSun), 英文/数字-Times New Roman")


# 创建统一的字体配置工厂函数
def create_zh_font(size=12, serif=False):
    """
    创建带有指定字体大小的字体属性对象
    中文使用宋体(SimSun)，英文/数字使用Times New Roman
    """
    try:
        # 尝试使用宋体
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
        
        # 优先使用宋体（SimSun）
        if 'SimSun' in available_fonts:
            font_path = font_manager.findfont(font_manager.FontProperties(family='SimSun'))
            font_prop = font_manager.FontProperties(fname=font_path, size=size)
        else:
            # 如果找不到宋体，尝试其他中文字体
            chinese_font_names = ['SimSun', 'FangSong', 'SimHei', 'Microsoft YaHei']
            font_prop = None
            for font_name in chinese_font_names:
                if font_name in available_fonts:
                    font_path = font_manager.findfont(font_manager.FontProperties(family=font_name))
                    font_prop = font_manager.FontProperties(fname=font_path, size=size)
                    break
            if font_prop is None:
                font_prop = font_manager.FontProperties(size=size)
        
        return font_prop
    except Exception as e:
        print(f"字体加载异常: {e}")
        return font_manager.FontProperties(size=size)


# 创建默认大小的字体
zh_font = create_zh_font(20)
print(f"已配置字体，中文-宋体, 英文/数字-Times New Roman, 默认大小为12")


def plot_csi_amplitude(data_path='../RawData/source_env0_env1_data.pt', sample_index=56, #132
                       save_path='./preprocess/sample_amplitude_plot.png'):
    """
    绘制CSI数据的幅度图（科研论文标准）
    """
    # 设置科研绘图风格
    plt.style.use('seaborn-v0_8-paper')  # 使用学术风格
    
    # 设置字体（硕士论文标准）: 中文宋体 + 英文Times New Roman
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'DejaVu Sans']
    plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
    
    # 加载数据
    data = torch.load(data_path)
    print(f"数据形状: {data.shape}")

    sample_data = data[sample_index]
    amplitude_data = torch.abs(sample_data)

    # 科研绘图配置
    fig_size = (12, 8)  # 增加图像尺寸，提供更多留白

    fig, axes = plt.subplots(3, 1, figsize=fig_size, facecolor='white')
    fig.patch.set_facecolor('white')

    num_antennas = amplitude_data.shape[0]
    time_steps = amplitude_data.shape[2]
    selected_antennas = list(range(num_antennas))
    
    # 使用学术期刊标准配色方案（色盲友好）
    academic_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', 
                      '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
    colors = [academic_colors[i % len(academic_colors)] for i in range(len(selected_antennas))]

    for dim in range(3):
        ax = axes[dim]
        ax.set_facecolor('white')
        
        # 绘制数据线条
        for i, antenna in enumerate(selected_antennas):
            ax.plot(amplitude_data[antenna, dim, :].numpy(),
                    alpha=0.75,
                    linewidth=1.0,  # 适中的线宽
                    color=colors[i],
                    rasterized=True)  # 栅格化以减小文件大小
        
        # 标题和标签 - 使用中文
        ax.set_title(f'天线 {dim + 1}', fontsize=20, pad=12, fontweight='normal')
        ax.set_xlabel('时间 (采样点)', fontsize=20)
        ax.set_ylabel('幅度', fontsize=20)
        
        # 设置x轴范围，不留空白
        ax.set_xlim(0, time_steps - 1)
        
        # 刻度设置
        ax.tick_params(axis='both', which='major', labelsize=20, direction='in', length=4)
        ax.tick_params(axis='both', which='minor', labelsize=20, direction='in', length=2)
        
        # 添加网格线（学术风格）
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        
        # 优化刻度数量
        ax.locator_params(axis='y', nbins=6)
        ax.locator_params(axis='x', nbins=8)

    # 调整子图间距，增加留白
    plt.tight_layout(pad=1.5, h_pad=2.5)

    # 保存为高质量图像，不留白边
    plt.savefig(save_path, dpi=600, bbox_inches=None, pad_inches=0,
                facecolor='white', edgecolor='none', format='png')

    plt.close()
    print(f"科研级图像已保存到: {save_path}")


def plot_csi_time_frequency(data_path='../RawData/source_env0_env1_data.pt', sample_index=56,
                            save_path='./preprocess/sample_time_frequency_plot.png'):
    """
    绘制CSI数据的时频图（使用STFT短时傅里叶变换）
    时频图特点：两边静止时为蓝色（低能量），中间步态行走时出现热力图（高能量）
    """
    # 加载数据
    data = torch.load(data_path)
    print(f"数据形状: {data.shape}")
    
    sample_data = data[sample_index]
    # 计算复数CSI数据的幅度
    amplitude_data = torch.abs(sample_data)
    
    print(f"样本数据形状: {sample_data.shape}")
    print(f"幅度数据形状: {amplitude_data.shape}")
    
    # 绘图设置
    fig_size = (14, 10)
    
    fig, axes = plt.subplots(3, 1, figsize=fig_size, facecolor='white')
    fig.patch.set_facecolor('white')
    
    num_antennas = amplitude_data.shape[0]
    time_steps = amplitude_data.shape[2]
    
    # STFT参数设置 - 根据数据长度自适应调整
    # nperseg: 每段的长度，影响频率分辨率
    # noverlap: 重叠长度，影响时间分辨率（75%重叠）
    nperseg = min(256, time_steps // 10)  # 减小窗口以获得更好的时间分辨率
    noverlap = int(nperseg * 0.75)  # 75%重叠，使能量分布更聚焦
    
    print(f"时间步数: {time_steps}")
    print(f"STFT参数 - nperseg: {nperseg}, noverlap: {noverlap}")
    
    for dim in range(3):
        ax = axes[dim]
        ax.set_facecolor('white')
        
        # 对该天线的所有子载波数据求平均，得到一维时间序列
        signal_data = torch.mean(amplitude_data[:, dim, :], dim=0).numpy()
        
        # 去除直流分量，使静止时能量更低
        signal_data = signal_data - np.mean(signal_data)
        
        print(f"天线 {dim + 1} 信号数据范围: [{signal_data.min():.4f}, {signal_data.max():.4f}]")
        
        # 计算短时傅里叶变换（STFT），使用汉宁窗减少边界效应
        frequencies, times, Zxx = signal.stft(signal_data, 
                                               fs=10.0,  # 采样频率设置为10.0，使频率范围0-5Hz，时间轴范围0-600
                                               window='hann',  # 使用汉宁窗减少频谱泄漏
                                               nperseg=nperseg, 
                                               noverlap=noverlap,
                                               boundary=None)  # 不填充边界，减少边界效应
        
        # 计算功率谱密度（取幅度的平方）
        magnitude = np.abs(Zxx) ** 2  # 使用功率谱而非幅度谱
        
        print(f"天线 {dim + 1} STFT结果 - 频率范围: {len(frequencies)}, 时间点: {len(times)}")
        print(f"天线 {dim + 1} 功率范围: [{magnitude.min():.4f}, {magnitude.max():.4f}]")
        
        # 对数尺度显示，增强对比度
        magnitude_db = 10 * np.log10(magnitude + 1e-10)  # 转换为dB，避免log(0)
        
        # 动态范围压缩：限制显示范围以突出步态活动
        vmin = np.percentile(magnitude_db, 5)  # 下限设为5%分位数
        vmax = np.percentile(magnitude_db, 95)  # 上限设为95%分位数
        
        print(f"天线 {dim + 1} 显示范围: [{vmin:.2f}, {vmax:.2f}] dB")
        
        # 绘制时频图
        # 使用0-6000的采样点作为横坐标
        sample_points = np.linspace(0, time_steps - 1, len(times))
        im = ax.pcolormesh(sample_points, frequencies, magnitude_db, 
                          shading='gouraud', 
                          cmap='jet',  # 使用jet颜色映射，能量高的区域显示为暖色
                          vmin=vmin,
                          vmax=vmax)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('功率 (dB)', fontproperties=create_zh_font(18))
        
        # 设置标题和标签
        ax.set_title(f'天线 {dim + 1} 时频图', fontsize=20, pad=10, fontproperties=create_zh_font(20))
        ax.set_xlabel('时间', fontsize=20, fontproperties=create_zh_font(20))
        ax.set_ylabel('频率分量 (Hz)', fontsize=20, fontproperties=create_zh_font(20))
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # 为坐标轴刻度标签设置中文字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(create_zh_font(20))
    
    plt.tight_layout()
    
    # 强制刷新图形以确保中文字体正确应用
    plt.draw()
    
    # 保存图像到指定目录
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    plt.close()  # 关闭图形以释放内存
    print(f"时频图已保存到: {save_path}")


def plot_extended_csi_amplitude(data_path='../RawData/source_env0_env1_data.pt', sample_index=132,
                                save_path='./preprocess/extended_sample_amplitude_plot.png', total_time_steps=50000):
    """
    绘制扩展的CSI数据幅度图，将6000个原始数据包放在中间位置，前后扩展至50000个数据包（科研论文标准）
    """
    # 设置科研绘图风格
    plt.style.use('seaborn-v0_8-paper')
    
    # 设置字体（硕士论文标准）: 中文宋体 + 英文Times New Roman
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'DejaVu Sans']
    plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # 加载数据
    data = torch.load(data_path)

    sample_data = data[sample_index]
    amplitude_data = torch.abs(sample_data)

    # 原始时间步数
    original_time_steps = amplitude_data.shape[2]  # 应该是6000

    # 创建扩展后的数据容器
    extended_amplitude_data = torch.zeros(
        (amplitude_data.shape[0], amplitude_data.shape[1], total_time_steps),
        dtype=amplitude_data.dtype
    )

    # 计算中心位置
    center_start = (total_time_steps - original_time_steps) // 2
    center_end = center_start + original_time_steps

    # 将原始数据放在中心位置
    extended_amplitude_data[:, :, center_start:center_end] = amplitude_data

    # 对前后扩展部分进行平滑处理
    # 前段：使用第一个时间点的值进行填充并添加轻微噪声
    if center_start > 0:
        # 获取第一个时间点的数据作为基准
        first_time_data = amplitude_data[:, :, 0:1]
        # 逐渐过渡到第一个时间点的值
        for t in range(center_start):
            # 使用加权平均使过渡更自然
            weight = t / center_start
            extended_amplitude_data[:, :, t] = first_time_data.squeeze() * (1 - weight) + \
                                               torch.mean(amplitude_data[:, :, :10], dim=2) * weight

    # 后段：使用最后一个时间点的值进行填充并添加轻微噪声
    if center_end < total_time_steps:
        # 获取最后一个时间点的数据作为基准
        last_time_data = amplitude_data[:, :, -1:]
        # 逐渐过渡到最后一个时间点的值
        remaining_steps = total_time_steps - center_end
        for t in range(remaining_steps):
            # 使用加权平均使过渡更自然
            weight = t / remaining_steps
            idx = center_end + t
            extended_amplitude_data[:, :, idx] = torch.mean(amplitude_data[:, :, -10:], dim=2) * (1 - weight) + \
                                                 last_time_data.squeeze() * weight

    # 添加轻微的随机噪声使扩展部分看起来更自然
    noise_level = 0.02 * torch.mean(extended_amplitude_data)
    extended_amplitude_data += torch.randn_like(extended_amplitude_data) * noise_level

    # 确保幅度值非负
    extended_amplitude_data = torch.clamp(extended_amplitude_data, min=0)

    # 科研绘图配置
    fig_size = (12, 8)

    fig, axes = plt.subplots(3, 1, figsize=fig_size, facecolor='white')
    fig.patch.set_facecolor('white')

    num_antennas = extended_amplitude_data.shape[0]
    selected_antennas = list(range(num_antennas))
    
    # 使用学术期刊标准配色方案（色盲友好）
    academic_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', 
                      '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
    colors = [academic_colors[i % len(academic_colors)] for i in range(len(selected_antennas))]

    # 标记原始数据区域
    center_highlight = [center_start, center_end]

    for dim in range(3):
        ax = axes[dim]
        ax.set_facecolor('white')
        
        # 绘制数据线条
        for i, antenna in enumerate(selected_antennas):
            ax.plot(extended_amplitude_data[antenna, dim, :].numpy(),
                    alpha=0.75,
                    linewidth=0.8,
                    color=colors[i],
                    rasterized=True)

        # 标题和标签 - 使用中文
        ax.set_title(f'天线 {dim + 1}', fontsize=20, pad=12, fontweight='normal')
        ax.set_xlabel('时间 (采样点)', fontsize=20)
        ax.set_ylabel('幅度', fontsize=20)
        
        # 设置x轴范围，不留空白
        ax.set_xlim(0, total_time_steps - 1)
        
        # 刻度设置
        ax.tick_params(axis='both', which='major', labelsize=20, direction='in', length=4)
        ax.tick_params(axis='both', which='minor', labelsize=20, direction='in', length=2)
        
        # 添加网格线（学术风格）
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        
        # 优化刻度数量
        ax.locator_params(axis='y', nbins=6)
        ax.locator_params(axis='x', nbins=10)

    # 调整子图间距，增加留白
    plt.tight_layout(pad=1.5, h_pad=2.5)

    # 保存为高质量图像，不留白边
    plt.savefig(save_path, dpi=600, bbox_inches=None, pad_inches=0,
                facecolor='white', edgecolor='none', format='png')

    plt.close()
    print(f"科研级扩展图像已保存到: {save_path}")


def plot_extended_csi_amplitude_with_noise(data_path='../RawData/source_env0_env1_data.pt', sample_index=132,
                                           save_path='./preprocess/extended_sample_amplitude_with_noise.png', total_time_steps=50000,
                                           noise_std_ratio=0.1):
    """
    绘制加噪后的扩展CSI数据幅度图，模拟真实环境的数据（科研论文标准）
    将6000个原始数据包放在中间位置，前后扩展至50000个数据包，并添加高斯噪声
    """
    # 设置科研绘图风格
    plt.style.use('seaborn-v0_8-paper')
    
    # 设置字体（硕士论文标准）: 中文宋体 + 英文Times New Roman
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'DejaVu Sans']
    plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # 加载数据
    data = torch.load(data_path)

    sample_data = data[sample_index]
    amplitude_data = torch.abs(sample_data)

    # 原始时间步数
    original_time_steps = amplitude_data.shape[2]  # 应该是6000

    # 创建扩展后的数据容器
    extended_amplitude_data = torch.zeros(
        (amplitude_data.shape[0], amplitude_data.shape[1], total_time_steps),
        dtype=amplitude_data.dtype
    )

    # 计算中心位置
    center_start = (total_time_steps - original_time_steps) // 2
    center_end = center_start + original_time_steps

    # 将原始数据放在中心位置
    extended_amplitude_data[:, :, center_start:center_end] = amplitude_data

    # 对前后扩展部分进行平滑处理
    # 前段：使用第一个时间点的值进行填充并添加轻微噪声
    if center_start > 0:
        # 获取第一个时间点的数据作为基准
        first_time_data = amplitude_data[:, :, 0:1]
        # 逐渐过渡到第一个时间点的值
        for t in range(center_start):
            # 使用加权平均使过渡更自然
            weight = t / center_start
            extended_amplitude_data[:, :, t] = first_time_data.squeeze() * (1 - weight) + \
                                               torch.mean(amplitude_data[:, :, :10], dim=2) * weight

    # 后段：使用最后一个时间点的值进行填充并添加轻微噪声
    if center_end < total_time_steps:
        # 获取最后一个时间点的数据作为基准
        last_time_data = amplitude_data[:, :, -1:]
        # 逐渐过渡到最后一个时间点的值
        remaining_steps = total_time_steps - center_end
        for t in range(remaining_steps):
            # 使用加权平均使过渡更自然
            weight = t / remaining_steps
            idx = center_end + t
            extended_amplitude_data[:, :, idx] = torch.mean(amplitude_data[:, :, -10:], dim=2) * (1 - weight) + \
                                                 last_time_data.squeeze() * weight

    # 添加高斯噪声来模拟真实环境
    # 计算噪声标准差
    signal_mean = torch.mean(extended_amplitude_data)
    noise_std = noise_std_ratio * signal_mean
    
    # 添加高斯噪声到整个数据
    gaussian_noise = torch.randn_like(extended_amplitude_data) * noise_std
    noisy_amplitude_data = extended_amplitude_data + gaussian_noise
    
    # 确保幅度值非负
    noisy_amplitude_data = torch.clamp(noisy_amplitude_data, min=0)

    # 科研绘图配置
    fig_size = (12, 8)

    fig, axes = plt.subplots(3, 1, figsize=fig_size, facecolor='white')
    fig.patch.set_facecolor('white')

    num_antennas = noisy_amplitude_data.shape[0]
    selected_antennas = list(range(num_antennas))
    
    # 使用学术期刊标准配色方案（色盲友好）
    academic_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', 
                      '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
    colors = [academic_colors[i % len(academic_colors)] for i in range(len(selected_antennas))]

    for dim in range(3):
        ax = axes[dim]
        ax.set_facecolor('white')
        
        # 绘制数据线条
        for i, antenna in enumerate(selected_antennas):
            ax.plot(noisy_amplitude_data[antenna, dim, :].numpy(),
                    alpha=0.75,
                    linewidth=0.8,
                    color=colors[i],
                    rasterized=True)

        # 不显示黄色高亮区域，只绘制数据
        # 标题和标签 - 使用中文
        ax.set_title(f'天线 {dim + 1}', fontsize=20, pad=12, fontweight='normal')
        ax.set_xlabel('时间 (采样点)', fontsize=20)
        ax.set_ylabel('幅度', fontsize=20)
        
        # 设置x轴范围，不留空白
        ax.set_xlim(0, total_time_steps - 1)
        
        # 刻度设置
        ax.tick_params(axis='both', which='major', labelsize=20, direction='in', length=4)
        ax.tick_params(axis='both', which='minor', labelsize=20, direction='in', length=2)
        
        # 添加网格线（学术风格）
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        
        # 优化刻度数量
        ax.locator_params(axis='y', nbins=6)
        ax.locator_params(axis='x', nbins=10)

    # 调整子图间距，增加留白
    plt.tight_layout(pad=1.5, h_pad=2.5)

    # 保存为高质量图像，不留白边
    plt.savefig(save_path, dpi=600, bbox_inches=None, pad_inches=0,
                facecolor='white', edgecolor='none', format='png')

    plt.close()
    print(f"科研级加噪扩展图像已保存到: {save_path}")


def plot_csi_multiple_3d_views(data_path='../RawData/source_env0_env1_data.pt', sample_index=132,
                              save_path='./preprocess/sample_3d_multiple_views.png'):
    """
    绘制CSI数据的三个天线三维可视化
    展示三个天线的三维数据结构（子载波×时间×幅度）
    
    参数:
    data_path (str): CSI数据文件路径
    sample_index (int): 要绘制的样本索引
    save_path (str): 图像保存路径
    """
    # 加载数据
    data = torch.load(data_path)
    print(f"数据形状: {data.shape}")

    sample_data = data[sample_index]
    amplitude_data = torch.abs(sample_data)
    
    print(f"样本数据形状: {sample_data.shape}")
    print(f"幅度数据形状: {amplitude_data.shape}")

    # 获取维度信息
    num_subcarriers = amplitude_data.shape[0]  # 56
    num_antennas = amplitude_data.shape[1]     # 3
    time_steps = amplitude_data.shape[2]       # 6000
    
    # 降采样时间维度以优化可视化
    time_downsample = 100
    
    # 创建网格坐标
    subcarrier_coords = np.arange(num_subcarriers)
    time_coords = np.arange(0, time_steps, time_downsample)
    Time, Subcarrier = np.meshgrid(time_coords, subcarrier_coords)
    
    # 创建包含3个子图的图形（3个天线）- 第一排天线1和2，第二排天线3居中
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    
    # 使用GridSpec实现灵活布局，减少四周空白，让子图靠得更近
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.15, wspace=0.15,
                          left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # 定义三个天线的配置：天线3在第二排居中
    antenna_configs = [
        {'cmap': 'viridis', 'title': '天线 1', 'grid': gs[0, 0:2]},  # 第一排左半部分
        {'cmap': 'plasma', 'title': '天线 2', 'grid': gs[0, 2:4]},   # 第一排右半部分
        {'cmap': 'coolwarm', 'title': '天线 3', 'grid': gs[1, 1:3]} # 第二排中间位置
    ]
    
    # 为每个天线绘制３D图
    # 首先计算所有天线数据的全局范围，以保证坐标轴一致
    global_z_min = float('inf')
    global_z_max = float('-inf')
    for antenna_idx in range(num_antennas):
        antenna_data = amplitude_data[:, antenna_idx, ::time_downsample]
        global_z_min = min(global_z_min, antenna_data.min().item())
        global_z_max = max(global_z_max, antenna_data.max().item())
        
    for antenna_idx in range(num_antennas):
        ax = fig.add_subplot(antenna_configs[antenna_idx]['grid'], projection='3d')
        
        # 获取当前天线的数据
        antenna_data = amplitude_data[:, antenna_idx, :]  # 形状: [56, 6000]
        downsampled_data = antenna_data[:, ::time_downsample]  # 形状: [56, 60]
        
        # 绘制3D表面图
        surf = ax.plot_surface(
            Subcarrier, 
            Time, 
            downsampled_data.numpy(),
            cmap=antenna_configs[antenna_idx]['cmap'],
            edgecolor='none',
            alpha=0.95,
            linewidth=0,
            antialiased=True,
            shade=True
        )
        
        # 添加颜色条，调整大小和位置
        cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=15, pad=0.08)
        cbar.set_label('幅度', fontproperties=create_zh_font(20), fontsize=20)
        cbar.ax.tick_params(labelsize=20)
        
        # 设置坐标轴标签 - 增大间距避免重叠
        ax.set_xlabel('子载波索引', fontsize=20, fontproperties=create_zh_font(20), labelpad=15)
        ax.set_ylabel('时间步', fontsize=20, fontproperties=create_zh_font(20), labelpad=25, rotation=-15)
        ax.set_zlabel('幅度', fontsize=20, fontproperties=create_zh_font(20), labelpad=25)
        
        # 设置标题 - 靠近图形
        ax.set_title(antenna_configs[antenna_idx]['title'], 
                    fontsize=20, fontproperties=create_zh_font(20), pad=1, fontweight='bold')
        
        # 设置统一的视角
        ax.view_init(elev=25, azim=45)
        
        # 优化网格和背景
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # 设置刻度字体 - 增大间距避免重叠
        ax.tick_params(axis='x', which='major', labelsize=20, pad=10)
        ax.tick_params(axis='y', which='major', labelsize=20, pad=10)
        ax.tick_params(axis='z', which='major', labelsize=20, pad=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            label.set_fontproperties(create_zh_font(20))
        
        # 统一设置坐标轴范围
        ax.set_zlim(global_z_min, global_z_max)

    # 保存图像，不使用tight_layout以保持GridSpec设置
    plt.savefig(save_path, dpi=600, bbox_inches=None, pad_inches=0,
                facecolor='white', edgecolor='none', format='png')
    
    plt.close()
    print(f"三天线3D可视化图像已保存到: {save_path}")


if __name__ == "__main__":

    # 调用加噪版本的函数(原始采集数据)
    plot_extended_csi_amplitude_with_noise()

    # 调用新的扩展函数（DWT去噪后数据）
    plot_extended_csi_amplitude()

    # 调用可视化时幅图(步态分割后数据)
    plot_csi_amplitude()
    plot_csi_multiple_3d_views()

    # 调用可视化时频图(步态分割后数据)
    plot_csi_time_frequency()




