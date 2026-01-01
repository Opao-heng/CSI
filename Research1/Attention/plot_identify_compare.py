import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 全局中文字体与绘图风格设置
plt.rcParams['axes.unicode_minus'] = False

# 中文采用宋体，英文、数字采用 Times New Roman
try:
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    chinese_fonts = ['SimSun', 'SimHei', 'Microsoft YaHei', 'FangSong', '宋体']
    
    font_set = False
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.serif'] = ['Times New Roman', font_name]
            plt.rcParams['font.sans-serif'] = ['Times New Roman', font_name]
            plt.rcParams['mathtext.fontset'] = 'stix'
            font_set = True
            break
    
    if not font_set:
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
except:
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (12, 7)


def _ensure_dir(save_path: str) -> None:
    """确保保存路径所在目录存在。"""
    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def plot_cross_domain_identity_comparison(
    method_names,
    metric_values_source,
    metric_values_cross,
    metric_name: str = "身份识别准确率(%)",
    title: str = "源域与跨域身份识别性能对比",
    save_path: str = "cross_domain_identity_comparison.png",
) -> None:
    """绘制源域与跨域身份识别性能对比柱状图。

    参数:
        method_names: 算法名称列表，例如
            ["CSIID", "Deep-WiID", "GaitID", "CAUTION", "Gait-Enhance", "GiWiD", "TFGAN-CAL"]
        metric_values_source: 源域识别准确率列表（单位: %），长度需与 method_names 一致
        metric_values_cross: 跨域识别准确率列表（单位: %），长度需与 method_names 一致
        metric_name: y 轴标题，默认 "识别准确率(%)"
        title: 图表标题
        save_path: 图像保存路径
    """
    if len(method_names) != len(metric_values_source) or len(method_names) != len(metric_values_cross):
        raise ValueError("method_names、metric_values_source 与 metric_values_cross 长度不一致")

    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(15, 8))

    # 定义柱子宽度和位置偏移
    x = range(len(method_names))
    bar_width = 0.32
    x_offset = [-bar_width / 2 - 0.01, bar_width / 2 + 0.01]
    
    # 定义顶会论文风格配色方案
    color_source = '#2E86AB'
    color_cross = '#A23B72'
    
    # 绘制两组柱子
    bars_source = ax.bar(
        [xi + x_offset[0] for xi in x],
        metric_values_source,
        width=bar_width,
        label='源域身份识别',
        color=color_source,
        edgecolor='#1a1a1a',
        linewidth=0.8,
        alpha=0.9
    )
    
    bars_cross = ax.bar(
        [xi + x_offset[1] for xi in x],
        metric_values_cross,
        width=bar_width,
        label='跨域身份识别',
        color=color_cross,
        edgecolor='#1a1a1a',
        linewidth=0.8,
        alpha=0.9
    )

    # 设置坐标轴与标题（学术风格）
    ax.set_xticks(x)
    # X轴标签（算法名称）使用Times New Roman
    ax.set_xticklabels(method_names, rotation=0, fontsize=20, ha='center', fontfamily='Times New Roman')
    # Y轴标题（中文）使用SimSun
    ax.set_ylabel(metric_name, fontsize=20, labelpad=10, fontfamily='SimSun')
    # X轴标题（中文）使用SimSun
    # ax.set_xlabel("对比算法", fontsize=20, labelpad=10, fontfamily='SimSun')
    # 图表标题（中文）使用SimSun
    # ax.set_title(title, fontsize=20, pad=25, fontfamily='SimSun')

    # 网格与坐标轴美化（简洁学术风格，只保留左侧和底部边框）
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.6, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['left'].set_color('#1a1a1a')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['bottom'].set_color('#1a1a1a')
    
    # 设置y轴刻度样式（数字默认用Times New Roman）
    ax.tick_params(axis='both', which='major', labelsize=20, length=6, width=1.0)
    
    # 设置y轴范围，留出空间显示数值
    all_values = metric_values_source + metric_values_cross
    if all_values:
        y_max = max(all_values)
        ax.set_ylim(0, y_max * 1.2)

    # 在源域柱子上标注数值
    for bar, value in zip(bars_source, metric_values_source):
        height = bar.get_height()
        y_offset = max(all_values) * 0.025 if all_values else height * 0.025
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            f"{value:.2f}",
            ha='center',
            va='bottom',
            fontsize=20
        )
    
    # 在跨域柱子上标注数值
    for bar, value in zip(bars_cross, metric_values_cross):
        height = bar.get_height()
        y_offset = max(all_values) * 0.025 if all_values else height * 0.025
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            f"{value:.2f}",
            ha='center',
            va='bottom',
            fontsize=20
        )
    
    # 添加图例（中文标签用SimSun）
    legend = ax.legend(fontsize=20, loc='upper right', frameon=True, fancybox=False, edgecolor='#1a1a1a', framealpha=1.0, shadow=False)
    for text in legend.get_texts():
        text.set_fontfamily('SimSun')

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"源域与跨域身份识别性能对比图已保存到: {save_path}")


def plot_cross_domain_identity_comparison_default(
    save_path: str = "cross_domain_identity_comparison.png",
) -> None:
    """使用占位数据绘制源域与跨域身份识别性能对比图。
    """
    # 方法名称（按照指定顺序从左到右排列）
    method_names = [
        "TFGAN-CAL",
        "GiWiD",
        "Deep-WiID",
        "CSIID",
        "CAUTION",
        "GaitID",
        "Gait-Enhance",
    ]

    # 源域识别准确率
    metric_values_source = [
        97.36,
        88.56,
        82.78,
        80.45,
        82.68,
        79.34,
        76.46,
    ]

    # 跨域识别准确率
    metric_values_cross = [
        99.00,
        83.56,
        36.78,
        40.45,
        56.68,
        45.34,
        52.46,
    ]

    plot_cross_domain_identity_comparison(
        method_names=method_names,
        metric_values_source=metric_values_source,
        metric_values_cross=metric_values_cross,
        metric_name="识别准确率(%)",
        title="源域与跨域身份识别性能对比",
        save_path=save_path,
    )


if __name__ == "__main__":
    # 直接使用占位数据绘制
    plot_cross_domain_identity_comparison_default()
