import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import matplotlib as mpl

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


def plot_csi_amplitude(data_path='../data/source_env0_env1_data.pt', sample_index=132, save_path='./preprocess/sample_amplitude_plot.png'):
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
    plt.style.use('seaborn-v0_8')
    fig_size = (10, 6)
    
    fig, axes = plt.subplots(3, 1, figsize=fig_size)

    num_antennas = amplitude_data.shape[0]
    time_steps = amplitude_data.shape[2]
    selected_antennas = list(range(num_antennas))
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_antennas)))
    
    for dim in range(3):
        ax = axes[dim]
        for i, antenna in enumerate(selected_antennas):
            ax.plot(amplitude_data[antenna, dim, :].numpy(),
                    alpha=0.8,
                    linewidth=1.2,
                    color=colors[i])
        ax.set_title(f'天线 {dim+1}', fontsize=12, pad=10, fontproperties=create_zh_font(12))
        ax.set_xlabel('时间', fontsize=10, fontproperties=create_zh_font(10))
        ax.set_ylabel('幅度', fontsize=10, fontproperties=create_zh_font(10))
        ax.grid(True, alpha=0.3)
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


def plot_extended_csi_amplitude(data_path='../data/source_env0_env1_data.pt', sample_index=132,  save_path='./preprocess/extended_sample_amplitude_plot.png',  total_time_steps=50000):
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
    plt.style.use('seaborn-v0_8')
    fig_size = (24, 16)  # 增大图像尺寸以使整体更协调
    
    fig, axes = plt.subplots(3, 1, figsize=fig_size)
    
    num_antennas = extended_amplitude_data.shape[0]
    selected_antennas = list(range(num_antennas))
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_antennas)))
    
    # 标记原始数据区域
    center_highlight = [center_start, center_end]
    
    for dim in range(3):
        ax = axes[dim]
        for i, antenna in enumerate(selected_antennas):
            ax.plot(extended_amplitude_data[antenna, dim, :].numpy(),
                    alpha=0.8,
                    linewidth=0.8,
                    color=colors[i])
        
        # 高亮显示原始数据区域
        ax.axvspan(center_highlight[0], center_highlight[1], alpha=0.2, color='yellow', 
                   label='原始6000个数据包区域')
        
        ax.set_title(f'天线 {dim+1}', fontsize=28, pad=20, fontproperties=create_zh_font(24), fontweight='bold')
        ax.set_xlabel('时间', fontsize=20, fontproperties=create_zh_font(20))
        ax.set_ylabel('幅度', fontsize=20, fontproperties=create_zh_font(20))
        ax.grid(True, alpha=0.3)
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


# 示例调用
if __name__ == "__main__":
    plot_csi_amplitude()
    # 调用新的扩展函数
    plot_extended_csi_amplitude()