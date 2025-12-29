import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import matplotlib as mpl
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# 设置中文字体支持 - 更直接有效的方式
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 直接设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
print("已设置中文字体: SimHei, Microsoft YaHei")


# 创建特定的中文字体属性对象的工厂函数
def create_zh_font(size=12):
    """
    创建带有指定字体大小的中文字体属性对象

    参数:
    size (int): 字体大小

    返回:
    FontProperties: 配置好的字体属性对象
    """
    try:
        # 尝试使用系统中的中文字体
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
        chinese_font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'STHeiTi', 'STSong']

        for font_name in chinese_font_names:
            if font_name in available_fonts:
                font_path = font_manager.findfont(font_manager.FontProperties(family=font_name))
                return font_manager.FontProperties(fname=font_path, size=size)

        # 如果找不到中文字体，使用默认字体
        return font_manager.FontProperties(size=size)
    except Exception as e:
        print(f"字体加载异常: {e}")
        return font_manager.FontProperties(size=size)


# 创建默认大小的中文字体
zh_font = create_zh_font(12)
print(f"已配置中文字体，默认大小为12")


def plot_csi_amplitude(data_path='../RawData/source_env0_env1_data.pt', sample_index=132, #132
                       save_path='./preprocess/sample_amplitude_plot.png'):
    """
    绘制CSI数据的幅度图

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

    # 绘图设置
    fig_size = (10, 6)

    fig, axes = plt.subplots(3, 1, figsize=fig_size, facecolor='white')
    fig.patch.set_facecolor('white')

    num_antennas = amplitude_data.shape[0]
    time_steps = amplitude_data.shape[2]
    selected_antennas = list(range(num_antennas))
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_antennas)))

    for dim in range(3):
        ax = axes[dim]
        ax.set_facecolor('white')
        for i, antenna in enumerate(selected_antennas):
            ax.plot(amplitude_data[antenna, dim, :].numpy(),
                    alpha=0.8,
                    linewidth=1.2,
                    color=colors[i])
        ax.set_title(f'天线 {dim + 1}', fontsize=12, pad=10, fontproperties=create_zh_font(12))
        ax.set_xlabel('时间', fontsize=10, fontproperties=create_zh_font(10))
        ax.set_ylabel('幅度', fontsize=10, fontproperties=create_zh_font(10))
        ax.tick_params(axis='both', which='major', labelsize=8)
        # 为坐标轴刻度标签也设置中文字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(create_zh_font(8))

    plt.tight_layout()

    # 强制刷新图形以确保中文字体正确应用
    plt.draw()

    # 保存图像到指定目录
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    plt.close()  # 关闭图形以释放内存
    print(f"图像已保存到: {save_path}")


def plot_csi_time_frequency(data_path='../RawData/source_env0_env1_data.pt', sample_index=132,
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
                                               fs=10.0,  # 采样频率设置为10Hz，使频率范围为0-5Hz
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
        im = ax.pcolormesh(times, frequencies, magnitude_db, 
                          shading='gouraud', 
                          cmap='jet',  # 使用jet颜色映射，能量高的区域显示为暖色
                          vmin=vmin,
                          vmax=vmax)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('功率 (dB)', fontproperties=create_zh_font(10))
        
        # 设置标题和标签
        ax.set_title(f'天线 {dim + 1} 时频图', fontsize=12, pad=10, fontproperties=create_zh_font(12))
        ax.set_xlabel('时间', fontsize=10, fontproperties=create_zh_font(10))
        ax.set_ylabel('频率分量 (Hz)', fontsize=10, fontproperties=create_zh_font(10))
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # 为坐标轴刻度标签设置中文字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(create_zh_font(8))
    
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
    绘制扩展的CSI数据幅度图，将6000个原始数据包放在中间位置，前后扩展至50000个数据包
    """
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

    # 绘图设置
    fig_size = (24, 16)  # 增大图像尺寸以使整体更协调

    fig, axes = plt.subplots(3, 1, figsize=fig_size, facecolor='white')
    fig.patch.set_facecolor('white')

    num_antennas = extended_amplitude_data.shape[0]
    selected_antennas = list(range(num_antennas))
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_antennas)))

    # 标记原始数据区域
    center_highlight = [center_start, center_end]

    for dim in range(3):
        ax = axes[dim]
        ax.set_facecolor('white')
        for i, antenna in enumerate(selected_antennas):
            ax.plot(extended_amplitude_data[antenna, dim, :].numpy(),
                    alpha=0.8,
                    linewidth=0.8,
                    color=colors[i])

        # 高亮显示原始数据区域
        ax.axvspan(center_highlight[0], center_highlight[1], alpha=0.2, color='yellow',
                   label='原始6000个数据包区域')

        ax.set_title(f'天线 {dim + 1}', fontsize=28, pad=20, fontproperties=create_zh_font(24), fontweight='bold')
        ax.set_xlabel('时间', fontsize=20, fontproperties=create_zh_font(20))
        ax.set_ylabel('幅度', fontsize=20, fontproperties=create_zh_font(20))
        ax.tick_params(axis='both', which='major', labelsize=14)

        # 设置x轴刻度以便更好地显示
        ax.set_xlim(0, total_time_steps)
        ax.set_xticks(np.linspace(0, total_time_steps, 11))  # 设置11个刻度

        # 为坐标轴刻度标签也设置中文字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(create_zh_font(14))

    # 添加图例
    if dim == 0:  # 只在第一个子图添加图例
        legend = ax.legend(loc='upper right')
        # 为图例文本设置中文字体
        if legend:
            for text in legend.get_texts():
                text.set_fontproperties(create_zh_font(16))

    # 调整子图间距以避免重叠
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08, hspace=0.35)

    # 强制刷新图形以确保中文字体正确应用
    plt.draw()

    # 保存图像到指定目录
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    plt.close()  # 关闭图形以释放内存
    print(f"扩展图像已保存到: {save_path}")


def plot_extended_csi_amplitude_with_noise(data_path='../RawData/source_env0_env1_data.pt', sample_index=132,
                                           save_path='./preprocess/extended_sample_amplitude_with_noise.png', total_time_steps=50000,
                                           noise_std_ratio=0.1):
    """
    绘制加噪后的扩展CSI数据幅度图，模拟真实环境的数据
    将6000个原始数据包放在中间位置，前后扩展至50000个数据包，并添加高斯噪声
    
    参数:
    data_path (str): CSI数据文件路径
    sample_index (int): 要绘制的样本索引
    save_path (str): 图像保存路径
    total_time_steps (int): 总时间步数
    noise_std_ratio (float): 噪声标准差与信号平均值的比例
    """
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

    # 绘图设置
    fig_size = (24, 16)  # 增大图像尺寸以使整体更协调

    fig, axes = plt.subplots(3, 1, figsize=fig_size, facecolor='white')
    fig.patch.set_facecolor('white')

    num_antennas = noisy_amplitude_data.shape[0]
    selected_antennas = list(range(num_antennas))
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_antennas)))

    for dim in range(3):
        ax = axes[dim]
        ax.set_facecolor('white')
        for i, antenna in enumerate(selected_antennas):
            ax.plot(noisy_amplitude_data[antenna, dim, :].numpy(),
                    alpha=0.8,
                    linewidth=0.8,
                    color=colors[i])

        # 不显示黄色高亮区域，只绘制数据
        ax.set_title(f'天线 {dim + 1}', fontsize=28, pad=20, fontproperties=create_zh_font(24), fontweight='bold')
        ax.set_xlabel('时间', fontsize=20, fontproperties=create_zh_font(20))
        ax.set_ylabel('幅度', fontsize=20, fontproperties=create_zh_font(20))
        ax.tick_params(axis='both', which='major', labelsize=14)

        # 设置x轴刻度以便更好地显示
        ax.set_xlim(0, total_time_steps)
        ax.set_xticks(np.linspace(0, total_time_steps, 11))  # 设置11个刻度

        # 为坐标轴刻度标签也设置中文字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(create_zh_font(14))

    # 调整子图间距以避免重叠
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08, hspace=0.35)

    # 强制刷新图形以确保中文字体正确应用
    plt.draw()

    # 保存图像到指定目录
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    plt.close()  # 关闭图形以释放内存
    print(f"加噪扩展图像已保存到: {save_path}")


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
    fig = plt.figure(figsize=(24, 16), facecolor='white')
    
    # 使用GridSpec实现灵活布局
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 定义三个天线的配置：天线3在第二排居中
    antenna_configs = [
        {'cmap': 'viridis', 'title': '天线 1', 'grid': gs[0, 0:2]},  # 第一排左半部分
        {'cmap': 'plasma', 'title': '天线 2', 'grid': gs[0, 2:4]},   # 第一排右半部分
        {'cmap': 'coolwarm', 'title': '天线 3', 'grid': gs[1, 1:3]} # 第二排中间位置
    ]
    
    # 为每个天线绘制3D图
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
        
        # 添加颜色条
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.12)
        cbar.set_label('幅度', fontproperties=create_zh_font(12), fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # 设置坐标轴标签
        ax.set_xlabel('子载波索引', fontsize=13, fontproperties=create_zh_font(13), labelpad=10)
        ax.set_ylabel('时间步', fontsize=13, fontproperties=create_zh_font(13), labelpad=10)
        ax.set_zlabel('幅度', fontsize=13, fontproperties=create_zh_font(13), labelpad=10)
        
        # 设置标题
        ax.set_title(antenna_configs[antenna_idx]['title'], 
                    fontsize=16, fontproperties=create_zh_font(16), pad=20, fontweight='bold')
        
        # 设置统一的视角
        ax.view_init(elev=25, azim=45)
        
        # 优化网格和背景
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # 设置刻度字体
        ax.tick_params(axis='both', which='major', labelsize=11)
        for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            label.set_fontproperties(create_zh_font(11))
    
    # 添加总标题
    fig.suptitle(f'CSI数据三天线三维可视化 - 样本{sample_index}', 
                fontsize=20, fontproperties=create_zh_font(20), y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
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




