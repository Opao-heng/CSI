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

def add_gaussian_white_noise(data, snr_db=None, snr_range=(20, 40), noise_scale=0.02):
    """
    为CSI数据添加轻微高斯白噪声进行数据增强（温和型）
    
    根据公式: x' = x + η · noise_scale · std(x)
    其中 η ~ N(0, 1) 是标准高斯白噪声，noise_scale 控制噪声强度（默认0.02 = 2%）
    
    参数:
    RawData (torch.Tensor): 输入CSI数据，形状为 (N, 3, T) 或 (3, T)
    snr_db (float): 指定的SNR值（dB），取值范围20-40（高SNR表示低噪声）。若为None，则随机采样
    snr_range (tuple): SNR随机范围，默认(20, 40)表示较低噪声水平
    noise_scale (float): 噪声缩放因子，相对于信号标准差的比例。默认0.02 = 2%
    
    返回:
    torch.Tensor: 添加噪声后的数据
    float: 实际使用的SNR值（dB）
    """
    # 如果输入是2D数据，添加批次维度
    if data.dim() == 2:
        data = data.unsqueeze(0)  # (3, T) -> (1, 3, T)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # 确定SNR值
    if snr_db is None:
        snr_db = np.random.uniform(snr_range[0], snr_range[1])
    else:
        snr_db = float(snr_db)
        if not (snr_range[0] <= snr_db <= snr_range[1]):
            print(f"警告: SNR {snr_db} dB 超出推荐范围 {snr_range}")
    
    # 复制数据以避免原地修改
    data_augmented = data.clone().float()
    
    # 逐样本处理
    for i in range(data_augmented.shape[0]):
        sample = data_augmented[i]  # (3, T)
        
        # 计算信号标准差
        signal_std = torch.std(sample)
        
        if signal_std < 1e-10:  # 避免除零
            continue
        
        # 生成标准高斯白噪声 η ~ N(0, 1)
        noise = torch.randn_like(sample)
        
        # 应用轻微噪声: x' = x + η · noise_scale · std(x)
        data_augmented[i] = sample + noise * noise_scale * signal_std
    
    if squeeze_output:
        data_augmented = data_augmented.squeeze(0)
    
    return data_augmented, snr_db


def add_multipath_fading_augmentation(data, num_paths=None, rayleigh_ratio=0.7, 
                                      path_delay_range=(0, 100), attenuation_range=(0.1, 1.0),
                                      rician_k_db=5.0):
    """
    基于多径衰落模型进行数据增强（Rayleigh + Rician混合）
    
    模型说明：
    - Rayleigh衰落（非视距场景）：多径信号无主要路径，幅度服从Rayleigh分布
      h(τ,t) = Σ α_{i(t)} δ(τ-τ_i)
    - Rician衰落（视距场景）：包含主路径（视距分量）和散射路径
      h(τ,t) = α_0 δ(τ) + Σ α_{i(t)} δ(τ-τ_i)
      Rician因子K = 10*log10(α_0^2 / Σσ_c^2)
    - 通过傅里叶变换将时域冲激响应转换为频域CSI：H(f,t) = Σ α_{i(t)} e^{-j2πfτ_i}
    
    参数:
    RawData (torch.Tensor): 输入CSI数据，形状为 (N, C, T) 或 (C, T)，其中N为天线/样本数，C为子载波/维度数，T为时间步长
    num_paths (int): 多径数量，范围[3, 10]。默认随机采样
    rayleigh_ratio (float): Rayleigh分量占比，范围[0, 1]。1.0表示纯Rayleigh，0.0表示纯Rician
    path_delay_range (tuple): 路径延迟范围（采样点），默认(0, 100)
    attenuation_range (tuple): 衰减因子范围，默认(0.1, 1.0)。实际路径损耗模型
    rician_k_db (float): Rician因子K（dB），范围[3, 15]dB
    
    返回:
    torch.Tensor: 增强后的数据
    dict: 增强参数（num_paths, rayleigh_ratio等）
    """
    # 处理不同维度的输入
    original_shape = data.shape
    squeeze_output = False
    
    if data.dim() == 2:
        # (C, T) -> (1, C, T)
        data = data.unsqueeze(0)
        squeeze_output = True
    elif data.dim() == 3:
        # (N, C, T) 保持原样
        pass
    else:
        raise ValueError(f"期望输入维度为2或3，得到{data.dim()}")
    
    # 随机确定多径数量
    if num_paths is None:
        num_paths = np.random.randint(3, 11)  # [3, 10]
    
    # 确定参数范围
    if rayleigh_ratio is None:
        rayleigh_ratio = np.random.uniform(0.5, 1.0)  # 倾向于Rayleigh
    
    # 复制数据
    data_augmented = data.clone().float()
    time_steps = data_augmented.shape[-1]
    
    # 逐样本处理
    for sample_idx in range(data_augmented.shape[0]):
        sample = data_augmented[sample_idx]  # (C, T)
        num_dims, T = sample.shape[0], sample.shape[1]
        
        # 生成多径冲激响应参数
        path_delays = np.sort(np.random.randint(path_delay_range[0], path_delay_range[1], num_paths))
        path_attenuations = np.random.uniform(attenuation_range[0], attenuation_range[1], num_paths)
        
        # Rician因子转换（dB到线性）
        k_linear = 10 ** (rician_k_db / 10.0)
        
        # 计算视距分量（Rician主路径）的幅度
        los_component = np.sqrt(k_linear / (k_linear + 1))
        # 计算散射分量的幅度
        scattering_component = np.sqrt(1.0 / (k_linear + 1))
        
        # 对每个天线维度应用多径衰落
        for dim in range(num_dims):
            original_signal = sample[dim, :].clone()  # (T,)
            faded_signal = torch.zeros_like(original_signal)
            
            # Rayleigh分量（散射多径）
            for path_idx, (delay, attenuation) in enumerate(zip(path_delays, path_attenuations)):
                delay_idx = int(delay % time_steps)
                
                # 随机相位
                phase = np.random.uniform(0, 2*np.pi)
                complex_gain = attenuation * np.exp(1j * phase)
                
                # 循环延迟添加
                if delay_idx > 0:
                    delayed_signal = torch.roll(original_signal, delay_idx)
                else:
                    delayed_signal = original_signal
                
                # 应用衰减和Rayleigh分量权重
                faded_signal += scattering_component * complex_gain.real * delayed_signal
            
            # Rician分量（视距主路径）：直射分量
            los_phase = np.random.uniform(0, 2*np.pi)
            los_gain = los_component * np.exp(1j * los_phase)
            faded_signal += los_gain.real * original_signal
            
            # 应用衰减幅度调整（保持能量水平）
            signal_std = torch.std(faded_signal)
            if signal_std > 1e-10:
                faded_signal = faded_signal / signal_std * torch.std(original_signal)
            
            # 更新样本数据
            sample[dim, :] = faded_signal
    
    if squeeze_output:
        data_augmented = data_augmented.squeeze(0)
    
    augmentation_params = {
        'num_paths': num_paths,
        'rayleigh_ratio': rayleigh_ratio,
        'path_delays': path_delays.tolist(),
        'rician_k_db': rician_k_db,
        'method': 'multipath_fading'
    }
    
    return data_augmented, augmentation_params


def add_frequency_selective_fading_augmentation(data, a=None, b=None, fading_type='linear_random'):
    """
    基于频率选择性衰落模型进行数据增强（频域衰落）
    
    模型说明：
    - 对原始CSI幅值序列进行快速傅里叶变换（FFT），得到频域响应 H(f,t)
    - 构造频率选择性衰落函数 G(f) = a - b·f + ξ(f)
      其中 a ∈ [0.5, 1.0]（基础衰减系数）
            b ∈ [0.5×10^-5]（线性衰减斜率）
            ξ(f) ~ N(0, 0.01)（均值为0、方差为0.01的高斯随机变量，模拟随机衰减波动）
    - 频域调制：H'(f,t) = H(f,t) × G(f)
    - 通过逆快速傅里叶变换（IFFT）转换回时域，得到增强后的CSI数据
    
    参数:
    RawData (torch.Tensor): 输入CSI数据，形状为 (N, C, T) 或 (C, T)，其中N为天线/样本数，C为子载波/维度数，T为时间步长
    a (float): 基础衰减系数，范围[0.5, 1.0]。默认随机采样
    b (float): 线性衰减斜率，范围[0.5×10^-5]。默认随机采样
    fading_type (str): 衰落类型，'linear_random'为线性随机衰落
    
    返回:
    torch.Tensor: 增强后的数据
    dict: 增强参数（a, b等）
    """
    # 处理不同维度的输入
    original_shape = data.shape
    squeeze_output = False
    
    if data.dim() == 2:
        # (C, T) -> (1, C, T)
        data = data.unsqueeze(0)
        squeeze_output = True
    elif data.dim() == 3:
        # (N, C, T) 保持原样
        pass
    else:
        raise ValueError(f"期望输入维度为2或3，得到{data.dim()}")
    
    # 随机确定衰减系数
    if a is None:
        a = np.random.uniform(0.5, 1.0)  # 基础衰减系数
    
    if b is None:
        b = np.random.uniform(0.1, 0.5) * 1e-5  # 线性衰减斜率 [0.1×10^-5, 0.5×10^-5]
    
    # 复制数据
    data_augmented = data.clone().float()
    
    # 逐样本处理
    for sample_idx in range(data_augmented.shape[0]):
        sample = data_augmented[sample_idx]  # (C, T)
        num_dims, T = sample.shape[0], sample.shape[1]
        
        # 对每个维度应用频率选择性衰落
        for dim in range(num_dims):
            original_signal = sample[dim, :].clone()  # (T,)
            
            # 执行FFT将时域信号转换到频域
            # FFT会返回复数频域表示
            freq_response = torch.fft.fft(original_signal)
            
            # 获取频率轴（归一化到[0, 1)）
            freqs = np.fft.fftfreq(T)
            freqs = np.abs(freqs)  # 取绝对值以处理负频率
            
            # 构造频率选择性衰落函数 G(f) = a - b·f + ξ(f)
            # 其中 ξ(f) ~ N(0, 0.01)
            fading_func = a - b * freqs * T  # 缩放频率到合理范围
            
            # 添加高斯随机波动
            random_fluctuation = np.random.normal(0, 0.01, T)
            fading_func = fading_func + random_fluctuation
            
            # 确保衰落函数为正（避免负数）
            fading_func = np.clip(fading_func, 0.1, 2.0)  # 限制在[0.1, 2.0]范围内
            
            # 将numpy数组转换为torch张量
            fading_func_tensor = torch.from_numpy(fading_func).float()
            
            # 在频域进行调制：H'(f,t) = H(f,t) × G(f)
            modulated_freq = freq_response * fading_func_tensor
            
            # 通过逆FFT转换回时域
            faded_signal = torch.fft.ifft(modulated_freq).real
            
            # 归一化幅度以保持能量水平
            signal_std = torch.std(faded_signal)
            if signal_std > 1e-10:
                faded_signal = faded_signal / signal_std * torch.std(original_signal)
            
            # 更新样本数据
            sample[dim, :] = faded_signal
    
    if squeeze_output:
        data_augmented = data_augmented.squeeze(0)
    
    augmentation_params = {
        'a': float(a),
        'b': float(b),
        'fading_type': fading_type,
        'method': 'frequency_selective_fading'
    }
    
    return data_augmented, augmentation_params


def plot_frequency_selective_fading_comparison(data_path='../RawData/source_env0_env1_data.pt',
                                               sample_index=132, a=None, b=None,
                                               save_path='./preprocess/frequency_selective_fading_comparison.png'):
    """
    绘制频率选择性衰落增强前后的时频图对比
    
    参数:
    data_path (str): CSI数据文件路径
    sample_index (int): 要绘制的样本索引
    a (float): 基础衰减系数
    b (float): 线性衰减斜率
    save_path (str): 图像保存路径
    """
    import os
    from scipy import signal
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 加载数据
    data = torch.load(data_path)
    print(f"数据形状: {data.shape}")
    
    sample_data = data[sample_index]  # (56, 3, 6000)
    amplitude_data = torch.abs(sample_data)
    
    # 添加频率选择性衰落增强
    augmented_data, aug_params = add_frequency_selective_fading_augmentation(
        sample_data.clone(),
        a=a,
        b=b
    )
    augmented_amplitude = torch.abs(augmented_data)
    
    print(f"增强参数: {aug_params}")
    
    # 选择第一个天线的第一个维度进行对比
    original_signal = amplitude_data[0, 0, :].numpy()  # (6000,)
    augmented_signal = augmented_amplitude[0, 0, :].numpy()  # (6000,)
    
    # 使用短时傅里叶变换（STFT）生成时频图
    nperseg = 512  # 增大窗口大小以提高频率分辨率
    noverlap = 480  # 增大重叠比例（75%重叠）以提高时间分辨率
    
    # 计算原始信号的时频谱
    freqs_orig, times_orig, Sxx_orig = signal.spectrogram(
        original_signal, 
        fs=1.0, 
        nperseg=nperseg, 
        noverlap=noverlap,
        scaling='spectrum'
    )
    
    # 计算增强信号的时频谱
    freqs_aug, times_aug, Sxx_aug = signal.spectrogram(
        augmented_signal, 
        fs=1.0, 
        nperseg=nperseg, 
        noverlap=noverlap,
        scaling='spectrum'
    )
    
    # 取对数以便可视化
    Sxx_orig_db = 10 * np.log10(np.abs(Sxx_orig) + 1e-10)
    Sxx_aug_db = 10 * np.log10(np.abs(Sxx_aug) + 1e-10)
    
    # 绘图设置
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # 上图: 增强前的时频图
    ax_before = axes[0]
    ax_before.set_facecolor('white')
    # 将时间索引映射到实际值
    time_indices_orig = times_orig * (len(original_signal) - 1)
    # 使用插值方法轻滑热力图
    im1 = ax_before.pcolormesh(time_indices_orig, freqs_orig[:len(freqs_orig)//2], 
                                Sxx_orig_db[:len(freqs_orig)//2, :],
                                shading='auto', cmap='jet', rasterized=True)
    ax_before.set_ylabel('频率分量', fontsize=11, fontproperties=create_zh_font(11))
    ax_before.set_title('原始CSI时频图', fontsize=14, fontweight='bold', fontproperties=create_zh_font(14))
    ax_before.set_xlabel('时间索引', fontsize=11, fontproperties=create_zh_font(11))
    cbar1 = plt.colorbar(im1, ax=ax_before)
    cbar1.set_label('幅度(dB)', fontsize=10, fontproperties=create_zh_font(10))
    ax_before.tick_params(axis='both', which='major', labelsize=9)
    for label in ax_before.get_xticklabels() + ax_before.get_yticklabels():
        label.set_fontproperties(create_zh_font(9))
    
    # 下图: 增强后的时频图
    ax_after = axes[1]
    ax_after.set_facecolor('white')
    # 将时间索引映射到实际值
    time_indices_aug = times_aug * (len(augmented_signal) - 1)
    im2 = ax_after.pcolormesh(time_indices_aug, freqs_aug[:len(freqs_aug)//2], 
                               Sxx_aug_db[:len(freqs_aug)//2, :],
                               shading='auto', cmap='jet', rasterized=True)
    ax_after.set_ylabel('频率分量', fontsize=11, fontproperties=create_zh_font(11))
    ax_after.set_title(f'频率选择性衰落时频图 (a={aug_params["a"]:.4f}, b={aug_params["b"]:.2e})', 
                      fontsize=14, fontweight='bold', fontproperties=create_zh_font(14))
    ax_after.set_xlabel('时间索引', fontsize=11, fontproperties=create_zh_font(11))
    cbar2 = plt.colorbar(im2, ax=ax_after)
    cbar2.set_label('幅度(dB)', fontsize=10, fontproperties=create_zh_font(10))
    ax_after.tick_params(axis='both', which='major', labelsize=9)
    for label in ax_after.get_xticklabels() + ax_after.get_yticklabels():
        label.set_fontproperties(create_zh_font(9))
    
    plt.tight_layout()
    plt.draw()
    
    # 保存图像 - 提高为DPI以轴读务画丝螪f化效果
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')
    plt.close()
    
    print(f"对比图已保存到: {save_path}")


def plot_multipath_fading_comparison(data_path='../RawData/source_env0_env1_data.pt',
                                     sample_index=132, num_paths=5, rician_k_db=5.0,
                                     save_path='./preprocess/multipath_fading_comparison.png'):
    """
    绘制多径衰落增强前后的对比图
    
    参数:
    data_path (str): CSI数据文件路径
    sample_index (int): 要绘制的样本索引
    num_paths (int): 多径数量
    rician_k_db (float): Rician因子K（dB）
    save_path (str): 图像保存路径
    """
    import os
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 加载数据
    data = torch.load(data_path)
    print(f"数据形状: {data.shape}")
    
    sample_data = data[sample_index]  # 上一个维度：(56, 3, 6000)
    amplitude_data = torch.abs(sample_data)
    
    # 添加多径衰落增强
    augmented_data, aug_params = add_multipath_fading_augmentation(
        sample_data.clone(), 
        num_paths=num_paths,
        rician_k_db=rician_k_db
    )
    augmented_amplitude = torch.abs(augmented_data)
    
    print(f"增强参数: {aug_params}")
    
    # 绘图设置
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    num_antennas = amplitude_data.shape[0]  # 56个天线
    selected_antennas = list(range(num_antennas))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_antennas)))
    
    # 上图: 增强前 - 原始数据
    ax_before = axes[0]
    ax_before.set_facecolor('white')
    for antenna in selected_antennas:
        # amplitude_data: (56, 3, 6000)
        # 绘制每个天线的第一个维度
        ax_before.plot(amplitude_data[antenna, 0, :].numpy(),
                      alpha=0.6, linewidth=0.8, color=colors[antenna % len(colors)])
    ax_before.set_title('原始CSI数据', fontsize=14, fontweight='bold', fontproperties=create_zh_font(14))
    ax_before.set_xlabel('时间索引', fontsize=11, fontproperties=create_zh_font(11))
    ax_before.set_ylabel('幅度', fontsize=11, fontproperties=create_zh_font(11))
    ax_before.tick_params(axis='both', which='major', labelsize=9)
    for label in ax_before.get_xticklabels() + ax_before.get_yticklabels():
        label.set_fontproperties(create_zh_font(9))
    
    # 下图: 增强后 - 多径衰落
    ax_after = axes[1]
    ax_after.set_facecolor('white')
    for antenna in selected_antennas:
        # augmented_amplitude: (56, 3, 6000)
        # 绘制每个天线的第一个维度
        ax_after.plot(augmented_amplitude[antenna, 0, :].numpy(),
                     alpha=0.6, linewidth=0.8, color=colors[antenna % len(colors)])
    ax_after.set_title(f'多径衰落CSI数据 (多径数={num_paths}, Rician_K={rician_k_db}dB)', 
                      fontsize=14, fontweight='bold', fontproperties=create_zh_font(14))
    ax_after.set_xlabel('时间索引', fontsize=11, fontproperties=create_zh_font(11))
    ax_after.set_ylabel('幅度', fontsize=11, fontproperties=create_zh_font(11))
    ax_after.tick_params(axis='both', which='major', labelsize=9)
    for label in ax_after.get_xticklabels() + ax_after.get_yticklabels():
        label.set_fontproperties(create_zh_font(9))
    
    plt.tight_layout()
    plt.draw()
    
    # 保存图像
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"对比图已保存到: {save_path}")


def plot_noise_augmentation_comparison(data_path='../RawData/source_env0_env1_data.pt',
                                       sample_index=132, snr_db=25, noise_scale=0.02,
                                       save_path='./preprocess/noise_augmentation_comparison.png'):
    """
    绘制数据增强前后的对比图（母线类噪声）
    
    参数:
    data_path (str): CSI数据文件路径
    sample_index (int): 要绘制的样本索引
    snr_db (float): SNR值（dB）
    noise_scale (float): 噪声缩放因子（默认0.02 = 2%）
    save_path (str): 图像保存路径
    """
    import os
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 加载数据
    data = torch.load(data_path)
    print(f"数据形状: {data.shape}")
    
    sample_data = data[sample_index]  # (56, 3, 6000)
    amplitude_data = torch.abs(sample_data)
    
    # 添加噪声增强
    augmented_data, actual_snr = add_gaussian_white_noise(sample_data, snr_db=snr_db, noise_scale=noise_scale)
    augmented_amplitude = torch.abs(augmented_data)
    
    print(f"使用SNR: {actual_snr:.2f} dB, 噪声缩放: {noise_scale}")
    
    # 绘图设置
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    num_antennas = amplitude_data.shape[0]  # 56个天线
    selected_antennas = list(range(num_antennas))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_antennas)))
    
    # 上图: 增强前 - 原始数据
    ax_before = axes[0]
    ax_before.set_facecolor('white')
    for antenna in selected_antennas:
        # 绘制第一个维度的幅度
        ax_before.plot(amplitude_data[antenna, 0, :].numpy(),
                      alpha=0.6, linewidth=0.8, color=colors[antenna % len(colors)])
    ax_before.set_title('原始CSI数据', fontsize=14, fontweight='bold', fontproperties=create_zh_font(14))
    ax_before.set_xlabel('时间索引', fontsize=11, fontproperties=create_zh_font(11))
    ax_before.set_ylabel('幅度', fontsize=11, fontproperties=create_zh_font(11))
    ax_before.tick_params(axis='both', which='major', labelsize=9)
    for label in ax_before.get_xticklabels() + ax_before.get_yticklabels():
        label.set_fontproperties(create_zh_font(9))
    
    # 下图: 增强后 - 添加轻微噪声
    ax_after = axes[1]
    ax_after.set_facecolor('white')
    for antenna in selected_antennas:
        # 绘制第一个维度的幅度
        ax_after.plot(augmented_amplitude[antenna, 0, :].numpy(),
                     alpha=0.6, linewidth=0.8, color=colors[antenna % len(colors)])
    ax_after.set_title(f'添加噪声后的CSI数据 (SNR={actual_snr:.1f}dB, 噪声缩放={noise_scale})', 
                      fontsize=14, fontweight='bold', fontproperties=create_zh_font(14))
    ax_after.set_xlabel('时间索引', fontsize=11, fontproperties=create_zh_font(11))
    ax_after.set_ylabel('幅度', fontsize=11, fontproperties=create_zh_font(11))
    ax_after.tick_params(axis='both', which='major', labelsize=9)
    for label in ax_after.get_xticklabels() + ax_after.get_yticklabels():
        label.set_fontproperties(create_zh_font(9))
    
    plt.tight_layout()
    plt.draw()
    
    # 保存图像
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"对比图已保存到: {save_path}")


def plot_csi_amplitude(data_path='../RawData/source_env0_env1_data.pt', sample_index=132,
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


if __name__ == '__main__':
    import os
    
    data_path = '../RawData/source_env0_env1_data.pt'
    output_dir = './preprocess'
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(data_path):
        # 1. 执行高斯白噪声增强
        print("\n" + "="*60)
        print("正在绘制高斯白噪声增强对比图...")
        print("="*60 + "\n")
        
        try:
            plot_noise_augmentation_comparison(
                data_path=data_path,
                sample_index=132,
                snr_db=25,
                noise_scale=0.02,
                save_path=os.path.join(output_dir, 'noise_augmentation_comparison.png')
            )
            print("✓ 高斯噪声对比图生成成功！")
        except Exception as e:
            print(f"✗ 无法生成高斯噪声对比图: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. 执行多径衰落增强
        print("\n" + "="*60)
        print("正在绘制多径衰落增强对比图...")
        print("="*60 + "\n")
        
        try:
            plot_multipath_fading_comparison(
                data_path=data_path,
                sample_index=132,
                num_paths=5,
                rician_k_db=5.0,
                save_path=os.path.join(output_dir, 'multipath_fading_comparison.png')
            )
            print("✓ 多径衰落对比图生成成功！")
        except Exception as e:
            print(f"✗ 无法生成多径衰落对比图: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. 执行频率选择性衰落增强
        print("\n" + "="*60)
        print("正在绘制频率选择性衰落增强对比图...")
        print("="*60 + "\n")
        
        try:
            plot_frequency_selective_fading_comparison(
                data_path=data_path,
                sample_index=132,
                a=None,  # 随机采样
                b=None,  # 随机采样
                save_path=os.path.join(output_dir, 'frequency_selective_fading_comparison.png')
            )
            print("✓ 频率选择性衰落对比图生成成功！")
        except Exception as e:
            print(f"✗ 无法生成频率选择性衰落对比图: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n✗ 数据文件不存在: {data_path}")
        print("提示: 请确保CSI数据文件在 ../RawData/source_env0_env1_data.pt")
