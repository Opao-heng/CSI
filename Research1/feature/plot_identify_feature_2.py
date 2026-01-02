import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager

# 设置中文字体支持 - 中文宋体，英文数字Times New Roman
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

print("已设置字体: 中文-宋体(SimSun), 英文/数字-Times New Roman")

def get_chinese_font_properties(size=18):
    """获取中文字体属性（宋体）"""
    try:
        return font_manager.FontProperties(family='SimSun', size=size)
    except:
        return font_manager.FontProperties(family='sans-serif', size=size)

def get_english_font_properties(size=18):
    """获取英文/数字字体属性（Times New Roman）"""
    try:
        return font_manager.FontProperties(family='Times New Roman', size=size)
    except:
        return font_manager.FontProperties(family='serif', size=size)

def get_mixed_font_properties(size=18):
    """获取混合字体属性（中文宋体+英文Times New Roman）"""
    try:
        prop = font_manager.FontProperties(size=size)
        prop.set_family(['Times New Roman', 'SimSun'])
        return prop
    except:
        return font_manager.FontProperties(family='sans-serif', size=size)

"""
目标：尽量在“点数、点大小、整体布局、颜色与图例结构”上贴近你提供的原图：
- 10 个身份（Id 0~9），颜色来自 tab10 调色板；
- 2 个环境 Env: 0 / 1（对应源域 / 目标域），用圆点 / 方块区分；
"""

def generate_raw_features(num_ids: int = 10,
                          points_per_env: int = 80,
                          overlap_ids=(7, 8, 9),
                          random_state: int = 0):

    rng = np.random.default_rng(random_state)

    # 不同 Id 有不同的中心，但也都事分接近，整体上也混杂
    region_centers_env0 = np.array([
        [-30.0, 20.0],   # 组 0：左上
        [30.0, 25.0],    # 组 1：右上
        [-35.0, -20.0],  # 组 2：左下
        [35.0, -25.0],   # 组 3：右下
        [0.0, 30.0],     # 组 4：上中
        [0.0, -30.0],    # 组 5：下中
        [-25.0, 0.0],    # 组 6：中左
        [25.0, 0.0],     # 组 7：中右
        [15.0, 15.0],    # 组 8：右上中
        [-15.0, -15.0],  # 组 9：左下中
    ])

    feats_list = []
    id_list = []
    env_list = []

    for id_idx in range(num_ids):
        center = region_centers_env0[id_idx]
        # 每个中心稍加檐散
        base0 = center + rng.normal(scale=3.0, size=2)

        # Env0：中接散布，充分利用坐标范围
        cov0 = np.array([[15.0, 2.0], [2.0, 15.0]])
        env0_points = rng.multivariate_normal(mean=base0, cov=cov0,
                                              size=points_per_env)

        # Env1： domain shift ，整体平移
        shift = np.array([
            rng.normal(loc=10.0, scale=3.5),
            rng.normal(loc=-3.0, scale=3.0),
        ])

        cov1 = np.array([[16.0, 1.5], [1.5, 16.0]])
        env1_points = rng.multivariate_normal(mean=base0 + shift, cov=cov1,
                                              size=points_per_env)

        feats_list.append(env0_points)
        feats_list.append(env1_points)
        id_list.append(np.full(points_per_env, id_idx, dtype=int))
        id_list.append(np.full(points_per_env, id_idx, dtype=int))
        env_list.append(np.zeros(points_per_env, dtype=int))
        env_list.append(np.ones(points_per_env, dtype=int))

    features = np.vstack(feats_list)
    ids = np.concatenate(id_list)
    envs = np.concatenate(env_list)

    return features, ids, envs


def plot_panel(ax,
               features: np.ndarray,
               ids: np.ndarray,
               envs: np.ndarray,
               title: str,
               alpha_env0: float,
               alpha_env1: float,
               point_size: float = 10):
    """在单个子图上绘制散点分布，尽量贴近原图视觉风格。"""
    unique_ids = np.unique(ids)
    cmap = plt.get_cmap("tab10")

    for idx, id_idx in enumerate(unique_ids):
        color = cmap(idx % 10)
        mask_id = ids == id_idx

        # Env 0：圆点
        mask_env0 = mask_id & (envs == 0)
        ax.scatter(
            features[mask_env0, 0],
            features[mask_env0, 1],
            s=point_size,
            c=[color],
            marker="o",
            edgecolors="none",
            alpha=alpha_env0,
        )

        # Env 1：方块
        mask_env1 = mask_id & (envs == 1)
        ax.scatter(
            features[mask_env1, 0],
            features[mask_env1, 1],
            s=point_size,
            c=[color],
            marker="s",
            edgecolors="none",
            alpha=alpha_env1,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(-40, 70, 10))
    ax.set_yticks(np.arange(-50, 60, 10))
    ax.tick_params(labelsize=18)
    # 设置刻度标签为Times New Roman
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(get_english_font_properties(10))



def add_legends(fig, axes, unique_ids):
    """添加与原图类似的 Env 图例和 Id 图例。"""
    cmap = plt.get_cmap("tab10")

    # Env 图例（圆形代表源域，正方形代表目标域）
    env_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="k",
               markersize=10, label="源域"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="k",
               markersize=10, label="目标域"),
    ]

    # Id 图例（放在右下角）
    id_handles = []
    for idx, id_idx in enumerate(unique_ids):
        color = cmap(idx % 10)
        handle = Line2D([0], [0], marker="o", color="w",
                        markerfacecolor=color, markersize=10,
                        label=f"Id: {id_idx}")
        id_handles.append(handle)

    # 为每个子图添加两个图例，都放在右下角区域
    for ax in axes:
        # 先添加ID图例在右下角
        legend_id = ax.legend(handles=id_handles,
                             loc="lower right",
                             frameon=True,
                             fontsize=18,
                             title="身份",
                             title_fontproperties=get_chinese_font_properties(11),
                             prop=get_mixed_font_properties(11),
                             borderaxespad=0.5)
        ax.add_artist(legend_id)  # 保留ID图例
        
        # 渲染图形以获取ID图例的实际位置
        ax.figure.canvas.draw()
        
        # 获取ID图例的边界框位置
        bbox_id = legend_id.get_window_extent()
        # 转换为坐标轴坐标系
        bbox_id_ax = bbox_id.transformed(ax.transAxes.inverted())
        
        # 将环境图例放在ID图例正上方，紧密贴合
        ax.legend(handles=env_handles,
                 loc="lower left",
                 frameon=True,
                 fontsize=18,
                 title="环境",
                 title_fontproperties=get_chinese_font_properties(10),
                 prop=get_chinese_font_properties(10),
                 borderaxespad=0.5)


def main():
    # 字体设置（中文宋体+英文Times New Roman）
    plt.rcParams["font.sans-serif"] = ["SimSun", "Arial"]
    plt.rcParams["font.serif"] = ["SimSun", "Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False

    num_ids = 10
    points_per_env = 200  # 每个 Id 在每个 Env 上的点数，整体数量和原图接近
    overlap_ids = (7, 8, 9)

    # 左图：原始特征
    feats_raw, ids_raw, envs_raw = generate_raw_features(
        num_ids=num_ids,
        points_per_env=100,
        overlap_ids=overlap_ids,
        random_state=0,
    )

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))

    # 效果对齐：坐标范围参照原图大致比例
    axes.set_xlim(-50, 60)
    axes.set_ylim(-40, 50)

    plot_panel(
        axes,
        feats_raw,
        ids_raw,
        envs_raw,
        title="原始特征分布",
        alpha_env0=0.6,
        alpha_env1=0.25,
        point_size=5,
    )

    # 添加 Env / Id 图例
    add_legends(fig, [axes], unique_ids=np.arange(num_ids))

    plt.tight_layout(rect=[0.02, 0.05, 0.95, 1.0])

    # 同时保存为 PNG（论文或报告中使用）
    plt.savefig("identify_feature_distribution.png", dpi=300, bbox_inches="tight")



if __name__ == "__main__":
    main()
