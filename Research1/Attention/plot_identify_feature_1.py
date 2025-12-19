import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

"""
目标：绘制不同颜色点凌乱分布的特征图
- 10 个身份（Id 0~9），使用 tab10 调色板
- 2 个环境 Env: 0 / 1（源域 / 目标域），用圆点 / 方块区分
- 点分布凌乱，不同身份混杂在一起
"""

def generate_messy_features(num_ids: int = 10,
                            points_per_env: int = 100,
                            random_state: int = 0):
    """
    生成凌乱分布的特征点
    - 每个身份分布在不同的区域组合，造成颜色交错凌乱分布
    - 有疏有密，保留空白区域
    - 环境1相对环境0有一定的平移
    """
    rng = np.random.default_rng(random_state)
    
    feats_list = []
    id_list = []
    env_list = []
    
    # 定义多个候选区域中心，不规则分布，避免环形或对称
    all_regions = [
        np.array([-35.0, 20.0]),   # 左上
        np.array([-15.0, 35.0]),   # 偏左顶部
        np.array([30.0, 30.0]),    # 右上角
        np.array([40.0, -10.0]),   # 右侧偏下
        np.array([-40.0, -30.0]),  # 左下角
        np.array([15.0, -25.0]),   # 中下偏右
        np.array([-5.0, 5.0]),     # 中心偏左
        np.array([25.0, 15.0]),    # 右侧偏上中
    ]
    
    # 为每个身份随机分配2-3个区域，让不同身份分布在不同位置
    id_region_map = {}
    # 为每个身份生成固定的奇怪形状参数
    id_shape_params = {}
    
    for id_idx in range(num_ids):
        # 每个身份随机选择2-3个区域
        num_regions = rng.choice([2, 3])
        selected_regions = rng.choice(len(all_regions), size=num_regions, replace=False)
        id_region_map[id_idx] = [all_regions[i] for i in selected_regions]
        
        # 为每个身份生成固定的奇怪形状参数
        shape_type = rng.choice(['ellipse', 'crescent', 'L_shape', 'scattered', 'curved'])
        angle = rng.uniform(0, 2 * np.pi)  # 旋转角度
        stretch_x = rng.uniform(4.0, 12.0)  # X方向拉伸
        stretch_y = rng.uniform(4.0, 12.0)  # Y方向拉伸
        asymmetry = rng.uniform(0.3, 0.9)  # 不对称程度
        
        id_shape_params[id_idx] = {
            'type': shape_type,
            'angle': angle,
            'stretch_x': stretch_x,
            'stretch_y': stretch_y,
            'asymmetry': asymmetry
        }
    
    for id_idx in range(num_ids):
        env0_points = []
        regions = id_region_map[id_idx]
        shape_params = id_shape_params[id_idx]
        
        # 每个身份的点分布在其选定的几个区域中
        for _ in range(points_per_env):
            # 随机选择该身份的一个区域
            region = regions[rng.integers(0, len(regions))]
            
            # 根据该身份的固定形状参数生成奇怪的分布
            shape_type = shape_params['type']
            angle = shape_params['angle']
            stretch_x = shape_params['stretch_x']
            stretch_y = shape_params['stretch_y']
            asymmetry = shape_params['asymmetry']
            
            if shape_type == 'ellipse':
                # 椭圆形：拉长的椭圆
                offset_x = rng.normal(0, stretch_x)
                offset_y = rng.normal(0, stretch_y * asymmetry)
                
            elif shape_type == 'crescent':
                # 新月形：弯曲分布
                r = rng.uniform(0, stretch_x)
                theta = rng.uniform(-np.pi/2, np.pi/2)
                offset_x = r * np.cos(theta) + r * asymmetry
                offset_y = r * np.sin(theta)
                
            elif shape_type == 'L_shape':
                # L形：两个方向
                if rng.random() < 0.5:
                    offset_x = rng.normal(0, stretch_x)
                    offset_y = rng.normal(0, stretch_y * 0.3)
                else:
                    offset_x = rng.normal(0, stretch_x * 0.3)
                    offset_y = rng.normal(0, stretch_y)
                    
            elif shape_type == 'scattered':
                # 分散形：多个小团
                cluster_offset = rng.choice([[-stretch_x*0.5, 0], [stretch_x*0.5, 0], 
                                            [0, -stretch_y*0.5], [0, stretch_y*0.5]])
                offset_x = rng.normal(cluster_offset[0], stretch_x * 0.3)
                offset_y = rng.normal(cluster_offset[1], stretch_y * 0.3)
                
            else:  # 'curved'
                # 弧形：曲线分布
                t = rng.uniform(-1, 1)
                offset_x = t * stretch_x
                offset_y = (t**2) * stretch_y * asymmetry + rng.normal(0, stretch_y * 0.2)
            
            # 旋转变换，制造倾斜的分布
            rotated_x = offset_x * np.cos(angle) - offset_y * np.sin(angle)
            rotated_y = offset_x * np.sin(angle) + offset_y * np.cos(angle)
            
            point = region + np.array([rotated_x, rotated_y])
            env0_points.append(point)
        
        env0_points = np.array(env0_points)
        
        # Env1：基于Env0添加domain shift
        env1_points = env0_points.copy()
        # 每个点添加不同的随机偏移
        shift = rng.normal(loc=[10.0, -5.0], scale=[6.0, 5.0], 
                          size=(points_per_env, 2))
        env1_points += shift
        
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


def plot_messy_distribution(ax,
                            features: np.ndarray,
                            ids: np.ndarray,
                            envs: np.ndarray,
                            title: str,
                            alpha_env0: float = 0.6,
                            alpha_env1: float = 0.25,
                            point_size: float = 5):
    """绘制凌乱分布的散点图"""
    unique_ids = np.unique(ids)
    cmap = plt.get_cmap("tab10")
    
    for idx, id_idx in enumerate(unique_ids):
        color = cmap(idx % 10)
        mask_id = ids == id_idx
        
        # Env 0：圆点（源域）
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
        
        # Env 1：方块（目标域）
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
    
    # 设置坐标轴样式
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    
    # 设置坐标轴刻度
    ax.set_xticks(np.arange(-40, 70, 10))
    ax.set_yticks(np.arange(-50, 60, 10))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=10, pad=10)


def add_dual_legends(fig, axes, unique_ids):
    """添加环境图例和身份图例"""
    cmap = plt.get_cmap("tab10")
    
    # 环境图例（圆形=源域，正方形=目标域）
    env_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="k",
               markersize=6, label="源域"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="k",
               markersize=6, label="目标域"),
    ]
    
    # 身份图例
    id_handles = []
    for idx, id_idx in enumerate(unique_ids):
        color = cmap(idx % 10)
        handle = Line2D([0], [0], marker="o", color="w",
                        markerfacecolor=color, markersize=6,
                        label=f"Id: {id_idx}")
        id_handles.append(handle)
    
    # 为每个子图添加两个图例
    for ax in axes:
        # 先添加身份图例在右下角
        legend_id = ax.legend(handles=id_handles,
                             loc="lower right",
                             frameon=True,
                             fontsize=8,
                             title="身份",
                             borderaxespad=0.5)
        ax.add_artist(legend_id)  # 保留身份图例
        
        # 渲染图形以获取身份图例的实际位置
        ax.figure.canvas.draw()
        
        # 获取身份图例的边界框位置
        bbox_id = legend_id.get_window_extent()
        # 转换为坐标轴坐标系
        bbox_id_ax = bbox_id.transformed(ax.transAxes.inverted())
        
        # 将环境图例放在身份图例正上方
        ax.legend(handles=env_handles,
                 loc="lower right",
                 bbox_to_anchor=(1.0, bbox_id_ax.y1),
                 frameon=True,
                 fontsize=8,
                 title="环境",
                 borderaxespad=0.5)


def main():
    # 字体设置
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    
    num_ids = 10
    points_per_env = 100  # 每个身份在每个环境上的点数
    
    # 生成凌乱分布的特征数据
    features, ids, envs = generate_messy_features(
        num_ids=num_ids,
        points_per_env=points_per_env,
        random_state=0,
    )
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    # 设置坐标范围
    ax.set_xlim(-50, 60)
    ax.set_ylim(-40, 50)
    
    # 绘制凌乱分布图
    plot_messy_distribution(
        ax,
        features,
        ids,
        envs,
        title="",
        alpha_env0=0.6,
        alpha_env1=0.25,
        point_size=5,
    )
    
    # 添加图例
    add_dual_legends(fig, [ax], unique_ids=np.arange(num_ids))
    
    plt.tight_layout(rect=[0.02, 0.05, 0.95, 1.0])
    
    # 保存图片
    plt.savefig("identify_feature_messy_distribution.png", dpi=300, bbox_inches="tight")
    
    plt.show()


if __name__ == "__main__":
    main()
