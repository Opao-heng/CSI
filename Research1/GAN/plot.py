import os
import json
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import font_manager

# 全局中文字体与绘图风格设置（参考顶会论文风格）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0


def create_zh_font(size: int = 12) -> font_manager.FontProperties:
    """创建指定大小的中文字体属性对象。"""
    try:
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
        chinese_font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'STHeiTi', 'STSong']

        for font_name in chinese_font_names:
            if font_name in available_fonts:
                font_path = font_manager.findfont(font_manager.FontProperties(family=font_name))
                return font_manager.FontProperties(fname=font_path, size=size)

        return font_manager.FontProperties(size=size)
    except Exception as e:  # noqa: BLE001
        print(f"字体加载异常: {e}")
        return font_manager.FontProperties(size=size)


def _ensure_dir(save_path: str) -> None:
    """确保保存路径所在目录存在。"""
    dir_name = os.path.dirname(save_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def plot_cross_domain_identity_comparison(
    method_names,
    metric_values,
    metric_name: str = "跨域身份识别准确率(%)",
    title: str = "跨域身份识别性能对比",
    save_path: str = "cross_domain_identity_comparison.png",
) -> None:
    """绘制跨域身份识别性能对比柱状图。

    参数:
        method_names: 算法名称列表，例如
            ["CSIID", "Deep-WiID", "GaitID", "CAUTION", "Gait-Enhance", "GiWiD", "TFGAN-CAL"]
        metric_values: 对应的性能指标数值列表（如识别准确率，单位: %），长度需与 method_names 一致
        metric_name: y 轴标题，默认 "跨域身份识别准确率(%)"
        title: 图表标题
        save_path: 图像保存路径
    """
    if len(method_names) != len(metric_values):
        raise ValueError("method_names 与 metric_values 长度不一致")

    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(14, 8))

    # 定义顶会论文风格配色方案
    colors = ['#E74C3C', '#3498DB', '#95A5A6', '#95A5A6', '#95A5A6', '#95A5A6', '#95A5A6']
    
    x = range(len(method_names))
    bars = ax.bar(x, metric_values, color=colors, edgecolor='#2C3E50', linewidth=1.8, alpha=0.9, width=0.65)

    # 设置坐标轴与标题（学术风格）
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15, fontsize=13, ha='right', fontproperties=create_zh_font(13))
    ax.set_ylabel(metric_name, fontsize=16, fontweight='bold', fontproperties=create_zh_font(16), labelpad=10)
    ax.set_xlabel("对比算法", fontsize=16, fontweight='bold', fontproperties=create_zh_font(16), labelpad=10)
    ax.set_title(title, fontsize=20, fontweight='bold', pad=25, fontproperties=create_zh_font(20))

    # 网格与坐标轴美化（顶会论文风格）
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['left'].set_color('#2C3E50')
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('#2C3E50')
    
    # 设置y轴刻度样式
    ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.0, colors='#2C3E50')
    
    # 设置y轴范围，留出空间显示数值
    if metric_values:
        y_max = max(metric_values)
        ax.set_ylim(0, y_max * 1.15)

    # 在柱子上方标注数值（更精致的样式）
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(metric_values) * 0.02 if metric_values else height * 0.02,
            f"{value:.2f}",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            fontproperties=create_zh_font(12),
            color='#2C3E50'
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"跨域身份识别性能对比图已保存到: {save_path}")


def plot_cross_domain_identity_comparison_from_json(
    json_path: str = "cross_domain_identity_results.json",
    metric_key: str = "accuracy",
    metric_name: Optional[str] = None,
    title: str = "跨域身份识别性能对比",
    save_path: Optional[str] = None,
) -> None:
    """从 JSON 文件加载结果并绘制对比图。

    JSON 推荐结构示例:
    {
        "metric_name": "跨域身份识别准确率(%)",
        "metric_key": "accuracy",
        "results": {
            "CSIID": 72.35,
            "Deep-WiID": 78.12,
            "GaitID": 80.45,
            "CAUTION": 82.90,
            "Gait-Enhance": 84.10,
            "GiWiD": 86.30,
            "TFGAN-CAL": 90.50
        }
    }

    - 若不存在 "results" 键，则默认将最外层字典视为 {方法名: 数值}
    - metric_key 仅在嵌套结构时使用
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到结果文件: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 解析 metric_name
    metric_name_final = metric_name or data.get("metric_name", "跨域身份识别准确率(%)")

    # 解析 results
    if "results" in data:
        results = data["results"]
    else:
        # 直接将最外层字典视为 {方法名: 数值} 结构
        results = data

    # 支持两种结构:
    # 1) "TFGAN-CAL": 90.5
    # 2) "TFGAN-CAL": {"accuracy": 90.5, "f1": 88.3, ...}
    method_names = []
    metric_values = []
    for name, value in results.items():
        if isinstance(value, dict):
            if metric_key not in value:
                raise KeyError(f"方法 {name} 的结果中缺少指定 metric_key: {metric_key}")
            metric_values.append(float(value[metric_key]))
        else:
            metric_values.append(float(value))
        method_names.append(name)

    if not save_path:
        save_path = "cross_domain_identity_comparison.png"

    plot_cross_domain_identity_comparison(
        method_names=method_names,
        metric_values=metric_values,
        metric_name=metric_name_final,
        title=title,
        save_path=save_path,
    )


def plot_cross_domain_identity_comparison_default(
    save_path: str = "cross_domain_identity_comparison.png",
) -> None:
    """使用占位数据绘制跨域身份识别性能对比图。
    """
    # 方法名称（TFGAN-CAL放在最左边，GiWiD第二列）
    method_names = [
        "TFGAN-CAL",
        "GiWiD",
        "Gait-Enhance",
        "CAUTION",
        "GaitID",
        "Deep-WiID",
        "CSIID",
    ]

    # TODO: 将下列占位数值替换为真实实验结果 (单位: %)
    metric_values = [
        82.50,  # TFGAN-CAL
        73.56,  # GiWiD
        67.46,  # Gait-Enhance
        56.68,  # CAUTION
        45.34,  # GaitID
        26.78,  # Deep-WiID
        20.45,  # CSIID
    ]

    plot_cross_domain_identity_comparison(
        method_names=method_names,
        metric_values=metric_values,
        metric_name="跨域身份识别准确率(%)",
        title="跨域身份识别性能对比",
        save_path=save_path,
    )


if __name__ == "__main__":
    # 优先尝试从 JSON 文件加载结果；如不存在则使用占位数据
    default_json_path = "cross_domain_identity_results.json"

    if os.path.exists(default_json_path):
        print(f"检测到结果文件: {default_json_path}, 正在从 JSON 绘图...")
        plot_cross_domain_identity_comparison_from_json(default_json_path)
    else:
        print("未检测到 JSON 结果文件，将使用占位数据绘制，请在代码中填写真实指标值后重新运行。")
        plot_cross_domain_identity_comparison_default()
