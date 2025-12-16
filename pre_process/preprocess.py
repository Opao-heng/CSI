import torch
import numpy as np
import pywt


def normalize_csi_data(csi_amplitude):
    """
    对CSI幅度数据进行归一化处理
    输入形状: (num_persons, num_samples, num_tx_rx, num_time, num_subcarriers)
    """
    normalized_data = torch.zeros_like(csi_amplitude)

    # 对每个样本的每个子载波进行归一化
    for person_idx in range(csi_amplitude.shape[0]):
        for sample_idx in range(csi_amplitude.shape[1]):
            for tx_rx_idx in range(csi_amplitude.shape[2]):
                for subcarrier_idx in range(csi_amplitude.shape[4]):
                    # 获取当前子载波的数据
                    data = csi_amplitude[person_idx, sample_idx, tx_rx_idx, :, subcarrier_idx]

                    # 计算均值和标准差
                    mean_val = torch.mean(data)
                    std_val = torch.std(data)

                    # 归一化
                    if std_val > 0:
                        normalized_data[person_idx, sample_idx, tx_rx_idx, :, subcarrier_idx] = (data - mean_val) / std_val
                    else:
                        normalized_data[person_idx, sample_idx, tx_rx_idx, :, subcarrier_idx] = data - mean_val

    return normalized_data

def dwt_denoise_csi(csi_amplitude, wavelet='db4', level=4):
    """
    使用离散小波变换(DWT)对CSI数据进行去噪
    输入形状: (num_persons, num_samples, num_tx_rx, num_time, num_subcarriers)
    """
    denoised_data = torch.zeros_like(csi_amplitude)

    # 对每个样本的每个子载波进行DWT去噪
    for person_idx in range(csi_amplitude.shape[0]):
        for sample_idx in range(csi_amplitude.shape[1]):
            for tx_rx_idx in range(csi_amplitude.shape[2]):
                for subcarrier_idx in range(csi_amplitude.shape[4]):
                    # 获取当前子载波的数据
                    data = csi_amplitude[person_idx, sample_idx, tx_rx_idx, :, subcarrier_idx].numpy()

                    # 执行DWT
                    coeffs = pywt.wavedec(data, wavelet, level=level)

                    # 阈值去噪（软阈值）
                    threshold = np.std(coeffs[-level]) * np.sqrt(2 * np.log(len(data)))
                    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

                    # 重构信号
                    denoised_signal = pywt.waverec(coeffs, wavelet)

                    # 确保长度一致
                    if len(denoised_signal) > len(data):
                        denoised_signal = denoised_signal[:len(data)]
                    elif len(denoised_signal) < len(data):
                        padded_signal = np.pad(denoised_signal, (0, len(data) - len(denoised_signal)),
                                             mode='constant', constant_values=0)
                        denoised_signal = padded_signal

                    denoised_data[person_idx, sample_idx, tx_rx_idx, :, subcarrier_idx] = torch.tensor(denoised_signal, dtype=torch.float32)

    return denoised_data

def extract_gait_segment(csi_amplitude, target_length=6000):
    """
    提取步态片段，确保时间维度为target_length
    输入形状: (num_persons, num_samples, num_tx_rx, num_time, num_subcarriers)
    """
    current_time = csi_amplitude.shape[3]

    if current_time == target_length:
        return csi_amplitude
    elif current_time > target_length:
        # 如果当前长度大于目标长度，截取中间部分
        start_idx = (current_time - target_length) // 2
        end_idx = start_idx + target_length
        return csi_amplitude[:, :, :, start_idx:end_idx, :]
    else:
        # 如果当前长度小于目标长度，进行填充
        pad_length = target_length - current_time
        pad_before = pad_length // 2
        pad_after = pad_length - pad_before

        # 使用边缘值进行填充
        padded_data = torch.nn.functional.pad(
            csi_amplitude,
            (0, 0, pad_before, pad_after, 0, 0, 0, 0, 0, 0),
            mode='replicate'
        )
        return padded_data

def add_gaussian_noise(csi_data, noise_level=0.01):
    """
    添加高斯白噪声
    """
    noise = torch.randn_like(csi_data) * noise_level
    return csi_data + noise

def simulate_multipath_fading(csi_data, max_delay=5):
    """
    模拟多径衰落
    """
    faded_data = torch.zeros_like(csi_data)

    for person_idx in range(csi_data.shape[0]):
        for sample_idx in range(csi_data.shape[1]):
            for tx_rx_idx in range(csi_data.shape[2]):
                for subcarrier_idx in range(csi_data.shape[4]):
                    # 获取当前子载波的数据
                    data = csi_data[person_idx, sample_idx, tx_rx_idx, :, subcarrier_idx]

                    # 创建简单的多径信道响应（指数衰减）
                    channel_length = min(max_delay, len(data) // 10)
                    channel = torch.exp(-torch.arange(channel_length, dtype=torch.float32) / (channel_length / 2))
                    channel = channel / torch.sum(channel)  # 归一化

                    # 应用卷积
                    if len(data) >= len(channel):
                        faded_signal = torch.nn.functional.conv1d(
                            data.unsqueeze(0).unsqueeze(0),
                            channel.flip(0).unsqueeze(0).unsqueeze(0),
                            padding=len(channel) - 1
                        ).squeeze()

                        # 调整长度
                        faded_signal = faded_signal[:len(data)]
                        faded_data[person_idx, sample_idx, tx_rx_idx, :, subcarrier_idx] = faded_signal
                    else:
                        faded_data[person_idx, sample_idx, tx_rx_idx, :, subcarrier_idx] = data

    return faded_data

def add_frequency_selective_fading(csi_data, coherence_bandwidth_ratio=0.1):
    """
    添加频率选择性衰落
    """
    faded_data = csi_data.clone()
    num_subcarriers = csi_data.shape[4]

    # 计算相干带宽内的子载波数
    coherence_subcarriers = max(1, int(num_subcarriers * coherence_bandwidth_ratio))

    for person_idx in range(csi_data.shape[0]):
        for sample_idx in range(csi_data.shape[1]):
            for tx_rx_idx in range(csi_data.shape[2]):
                # 为每个相干块生成不同的衰落因子
                for block_start in range(0, num_subcarriers, coherence_subcarriers):
                    block_end = min(block_start + coherence_subcarriers, num_subcarriers)

                    # 生成随机衰落因子
                    fading_factor = 0.5 + 0.5 * torch.rand(1).item()

                    # 应用衰落
                    faded_data[person_idx, sample_idx, tx_rx_idx, :, block_start:block_end] *= fading_factor

    return faded_data

def augment_csi_data(csi_data):
    """
    对CSI数据应用三种数据增强方法
    输入形状: (num_persons, num_samples, num_tx_rx, num_time, num_subcarriers)
    输出形状: (num_persons, num_samples*4, num_tx_rx, num_time, num_subcarriers)
    """
    # 原始数据
    original_data = csi_data

    # 1. 添加高斯白噪声
    gaussian_noise_data = add_gaussian_noise(csi_data, noise_level=0.02)

    # 2. 模拟多径衰落
    multipath_fading_data = simulate_multipath_fading(csi_data, max_delay=3)

    # 3. 添加频率选择性衰落
    freq_selective_fading_data = add_frequency_selective_fading(csi_data, coherence_bandwidth_ratio=0.15)

    # 拼接所有增强后的数据
    augmented_data = torch.cat([
        original_data,
        gaussian_noise_data,
        multipath_fading_data,
        freq_selective_fading_data
    ], dim=1)

    # 对应地扩展标签
    num_persons = csi_data.shape[0]
    num_original_samples = csi_data.shape[1]
    labels = np.arange(0, num_persons).reshape(-1, 1) * np.ones((1, num_original_samples))
    labels = torch.tensor(labels)

    # 扩展标签以匹配增强后的数据
    augmented_labels = torch.cat([labels, labels, labels, labels], dim=1)

    return augmented_data, augmented_labels

def preprocess_csi_data(csi_amplitude):
    """
    完整的CSI数据预处理流程
    1. 归一化
    2. DWT去噪
    3. 步态分割
    4. 数据增强
    """
    print("开始CSI数据预处理...")

    # 1. 归一化
    print("1. 执行归一化...")
    normalized_data = normalize_csi_data(csi_amplitude)

    # 2. DWT去噪
    print("2. 执行DWT去噪...")
    denoised_data = dwt_denoise_csi(normalized_data)

    # 3. 步态分割
    print("3. 执行步态分割...")
    segmented_data = extract_gait_segment(denoised_data, target_length=6000)

    # 4. 数据增强
    print("4. 执行数据增强...")
    augmented_data, augmented_labels = augment_csi_data(segmented_data)

    print("预处理完成!")
    print(f"原始数据形状: {csi_amplitude.shape}")
    print(f"增强后数据形状: {augmented_data.shape}")

    return augmented_data, augmented_labels
