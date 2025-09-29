import torch
import matplotlib.pyplot as plt
import numpy as np

# 设置字体支持（可选：确保负号正常）
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
data = torch.load('../data/SourceData/source_data.pt')
print(f"数据形状: {data.shape}")

sample_index = 132
sample_data = data[sample_index]
amplitude_data = torch.abs(sample_data)

# 绘图设置
plt.style.use('seaborn-v0_8')
fig_size = (10, 6)

fig, axes = plt.subplots(3, 1, figsize=fig_size)
fig.suptitle(f'Sample {sample_index} Amplitude Plot (by Antenna Dimension)', fontsize=14, fontweight='bold')

num_antennas = amplitude_data.shape[0]
time_steps = amplitude_data.shape[2]
selected_antennas = list(range(num_antennas))
colors = plt.cm.Set1(np.linspace(0, 1, len(selected_antennas)))

for dim in range(3):
    ax = axes[dim]
    for i, antenna in enumerate(selected_antennas):
        ax.plot(amplitude_data[antenna, dim, :].numpy(),
                alpha=0.8,
                linewidth=1.2,
                color=colors[i])
    ax.set_title(f'Dimension {dim+1}', fontsize=12, pad=10)
    ax.set_xlabel('Time Series', fontsize=10)
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=8)

plt.tight_layout()
plt.show()
