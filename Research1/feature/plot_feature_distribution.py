import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from matplotlib import font_manager
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

# 设置中文字体支持 - 中文宋体，英文数字Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 设置全局字体配置：使用font fallback机制实现中英文分离
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

print("已设置字体: 中文-宋体(SimSun), 英文/数字-Times New Roman")


def get_chinese_font_properties(size=18):
    """
    获取中文字体属性（宋体）
    """
    try:
        return font_manager.FontProperties(family='SimSun', size=size)
    except:
        return font_manager.FontProperties(family='sans-serif', size=size)

def get_english_font_properties(size=18):
    """
    获取英文/数字字体属性（Times New Roman）
    """
    try:
        return font_manager.FontProperties(family='Times New Roman', size=size)
    except:
        return font_manager.FontProperties(family='serif', size=size)

def get_mixed_font_properties(size=18):
    """
    获取混合字体属性（中文宋体+英文Times New Roman）
    通过设置fallback实现中英文分离
    """
    try:
        prop = font_manager.FontProperties(size=size)
        prop.set_family(['Times New Roman', 'SimSun'])
        return prop
    except:
        return font_manager.FontProperties(family='sans-serif', size=size)

def create_zh_font(size=18):
    """
    创建带有指定字体大小的中文字体属性对象（宋体）
    """
    return get_chinese_font_properties(size)


def generate_user_clusters(num_users=10, samples_per_user=100, feature_dim=128):
    """
    为10个用户生成不同的特征分布群（真实数据）
    每个用户的特征分布是一个高斯分布，中心不同
    
    参数:
    - num_users: 用户数量（默认10个）
    - samples_per_user: 每个用户的样本数
    - feature_dim: 特征维度
    
    返回:
    - features: 形状为 (num_users * samples_per_user, feature_dim) 的特征数组
    - labels: 对应的用户标签
    """
    np.random.seed(42)
    
    all_features = []
    all_labels = []
    
    # 为每个用户生成独立的特征分布群
    # 使用自然分散的分布策略，避免规则圆形排列
    user_centers = []
    
    # 预定义10个用户在高维空间中自然分散的位置
    # 打破椭圆形排列，使用更随机、分散的布局策略
    # 根据语义区域要求：绿色(User 2)置于中心，紫色(User 4)偏右，蓝色(User 0)偏下
    # 大幅增加各圈之间的距离，避免相互靠近或重叠
    # 
    # 坐标说明：(x, y, z) - 高维特征空间的前3个维度坐标
    #   - x: 第1维度坐标，影响t-SNE降维后的水平位置
    #   - y: 第2维度坐标，影响t-SNE降维后的垂直位置  
    #   - z: 第3维度坐标，增加高维空间的区分度
    # 
    # 颜色说明：使用tab20调色板，按User ID顺序分配颜色
    predefined_positions = [
        (5.0, -18.0, 3.0),     # User 0: 蓝色(深蓝) - 下方偏右（按要求偏下方）
        (-18.0, 16.0, -5.0),   # User 1: 橙色 - 左上角远端
        (1.0, 2.0, 1.0),       # User 2: 绿色(深绿) - 中心位置（按要求在中心）
        (20.0, 12.0, 6.0),     # User 3: 红色 - 右上角远端
        (17.0, -6.0, -4.0),    # User 4: 紫色 - 右侧偏下（按要求偏右侧）
        (-22.0, -10.0, 3.5),   # User 5: 棕色 - 左下角远端
        (10.0, 20.0, -6.0),    # User 6: 粉色 - 上方偏右远端
        (-10.0, -20.0, 5.0),   # User 7: 灰色 - 左下方远端
        (-5.0, 10.0, -2.0),    # User 8: 黄绿色 - 左上方
        (18.0, -16.0, 2.0)     # User 9: 青色 - 右下角远端
    ]
    
    for i in range(num_users):
        # 创建多维中心
        center = np.zeros(feature_dim)
        
        # 使用预定义的不规则位置
        if i < len(predefined_positions):
            x, y, z = predefined_positions[i]
            center[0] = x
            center[1] = y
            center[2] = z
        else:
            # 如果用户数超过预定义，使用随机分布
            center[0] = np.random.uniform(-3.5, 3.5)
            center[1] = np.random.uniform(-3.5, 3.5)
            center[2] = np.random.uniform(-2.0, 2.0)
        
        # 在其他维度上也分布一些信息
        center[3:min(6, feature_dim)] = np.random.randn(min(3, feature_dim-3)) * 0.6
        
        user_centers.append(center)
    
    # 为每个用户生成样本
    for user_id in range(num_users):
        center = user_centers[user_id]
        # 以该用户的中心为均值，生成高斯分布的样本
        std = 2.7  # 增大标准差，使每个用户分布圈的范围更大
        user_features = np.random.normal(center, std, (samples_per_user, feature_dim))
        
        all_features.append(user_features)
        all_labels.extend([user_id] * samples_per_user)
    
    features = np.vstack(all_features)
    labels = np.array(all_labels)
    
    return features, labels


def generate_matched_synthetic_data(real_features, real_labels, num_samples=500, num_users=10):
    """
    生成与真实数据贴合的合成数据
    合成数据会分别贴合真实的10个用户分布群
    
    参数:
    - real_features: 真实特征数据
    - real_labels: 真实标签
    - num_samples: 生成的合成样本数量
    - num_users: 用户数量
    
    返回:
    - synthetic_features: 生成的合成特征
    - synthetic_labels: 对应的用户标签（指示生成数据属于哪个用户）
    """
    np.random.seed(43)
    
    synthetic_features = []
    synthetic_labels = []
    
    # 为每个用户生成样本（均匀分配样本数）
    samples_per_user = num_samples // num_users
    
    for user_id in range(num_users):
        # 获取该用户真实样本的统计信息
        user_mask = real_labels == user_id
        user_real_features = real_features[user_mask]
        
        if len(user_real_features) > 0:
            # 计算用户特征的均值和标准差
            user_mean = np.mean(user_real_features, axis=0)
            user_std = np.std(user_real_features, axis=0)
            
            # 基于真实分布生成合成样本，贴合该用户的分布
            # 不扩大方差，直接使用真实分布的标准差
            synthetic_user_features = np.random.normal(
                user_mean, 
                user_std,  # 直接使用真实标准差，使生成数据紧密贴合
                (samples_per_user, len(user_mean))
            )
            
            synthetic_features.append(synthetic_user_features)
            synthetic_labels.extend([user_id] * samples_per_user)
    
    synthetic_features = np.vstack(synthetic_features)
    synthetic_labels = np.array(synthetic_labels)
    
    # 添加剩余样本数（如果总数不能被用户数整除）
    remaining = num_samples - len(synthetic_features)
    if remaining > 0:
        # 从所有样本中随机采样剩余部分
        random_indices = np.random.choice(len(synthetic_features), remaining)
        extra_samples = synthetic_features[random_indices]
        extra_labels = synthetic_labels[random_indices]
        synthetic_features = np.vstack([synthetic_features, extra_samples])
        synthetic_labels = np.concatenate([synthetic_labels, extra_labels])
    
    return synthetic_features[:num_samples], synthetic_labels[:num_samples]


def plot_feature_distribution_tsne_improved(real_features, real_labels, synthetic_features=None, 
                                            synthetic_labels=None, output_path='TFGAN/feature_distribution_tsne_improved.png'):
    """
    改进版的特征分布t-SNE可视化
    清晰显示10个用户的特征分布群和生成数据的贴合情况
    生成数据会用相同的颜色和不同的标记表示，与对应用户的分布贴合
    
    参数:
    - real_features: 真实特征 (N, D)
    - real_labels: 真实标签 (N,)
    - synthetic_features: 合成特征 (M, D)，可选
    - synthetic_labels: 合成标签 (M,)，指示生成数据属于哪个用户，可选
    - output_path: 输出图片路径
    """
    import os
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # 数据预处理
    if isinstance(real_features, torch.Tensor):
        real_features = real_features.cpu().detach().numpy()
    if isinstance(real_labels, torch.Tensor):
        real_labels = real_labels.cpu().detach().numpy()
    if synthetic_features is not None and isinstance(synthetic_features, torch.Tensor):
        synthetic_features = synthetic_features.cpu().detach().numpy()
    if synthetic_labels is not None and isinstance(synthetic_labels, torch.Tensor):
        synthetic_labels = synthetic_labels.cpu().detach().numpy()
    
    # 合并所有特征用于t-SNE降维
    if synthetic_features is not None:
        all_features = np.vstack([real_features, synthetic_features])
        type_labels = np.array([0] * len(real_features) + [1] * len(synthetic_features))
    else:
        all_features = real_features
        type_labels = np.zeros(len(real_features))
    
    # t-SNE降维
    print(f"  执行t-SNE降维 (共 {len(all_features)} 个样本)...")
    n_samples = len(all_features)
    perplexity = min(30, max(5, n_samples // 20))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                n_iter=1500, learning_rate='auto', init='pca', verbose=1)
    features_2d = tsne.fit_transform(all_features)
    
    # 分离真实和合成样本
    real_2d = features_2d[type_labels == 0]
    if synthetic_features is not None:
        synthetic_2d = features_2d[type_labels == 1]
    else:
        synthetic_2d = None
    
    # 创建高质量的图形
    plt.figure(figsize=(16, 12))
    
    # 获取唯一用户并分配颜色
    unique_users = np.unique(real_labels)
    num_users = len(unique_users)
    colors = plt.cm.tab20(np.linspace(0, 1, max(num_users, 10)))
    
    # 绘制真实样本（每个用户一个分布群）
    for idx, user_id in enumerate(sorted(unique_users)):
        mask = real_labels == user_id
        real_user_2d = real_2d[mask]
        
        # 绘制真实样本点（填充圆形）
        plt.scatter(real_user_2d[:, 0], real_user_2d[:, 1],
                   c=[colors[idx]], label=f'User {int(user_id)} (真实)',
                   alpha=0.7, s=80, edgecolors='black', linewidth=0.8, marker='o')
        
        # 计算并绘制该用户的置信椭圆（显示分布的轮廓）
        if len(real_user_2d) > 2:
            mean_x = real_user_2d[:, 0].mean()
            mean_y = real_user_2d[:, 1].mean()
            
            cov = np.cov(real_user_2d.T)
            if cov.ndim == 1:
                cov = np.array([[cov[0], 0], [0, cov[1] if len(cov) > 1 else cov[0]]])
            
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            # 绘制2个标准差的椭圆
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 4 * np.sqrt(np.abs(eigenvalues))
            
            ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle,
                            facecolor='none', edgecolor=colors[idx], linewidth=2,
                            linestyle='--', alpha=0.6)
            plt.gca().add_patch(ellipse)
    
    # 绘制合成样本（对应到各个用户）
    if synthetic_2d is not None and len(synthetic_2d) > 0:
        if synthetic_labels is not None:
            # 为每个用户绘制对应的生成样本
            for idx, user_id in enumerate(sorted(unique_users)):
                mask = synthetic_labels == user_id
                if np.any(mask):
                    synthetic_user_2d = synthetic_2d[mask]
                    plt.scatter(synthetic_user_2d[:, 0], synthetic_user_2d[:, 1],
                               c=[colors[idx]], label=f'User {int(user_id)} (生成)',
                               alpha=0.5, s=60, edgecolors=colors[idx], linewidth=0.5, marker='^')
        else:
            # 如果没有合成标签，用红色表示所有生成样本
            plt.scatter(synthetic_2d[:, 0], synthetic_2d[:, 1],
                       c='red', label='生成样本',
                       alpha=0.4, s=40, edgecolors='darkred', linewidth=0.5, marker='^')
    
    # 设置标题和标签（使用混合字体：中文宋体，数字Times New Roman）
    # plt.xlabel('t-SNE成分1', fontsize=24, fontweight='bold', fontproperties=get_mixed_font_properties(24))
    # plt.ylabel('t-SNE成分2', fontsize=24, fontweight='bold', fontproperties=get_mixed_font_properties(24))
    
    # 设置刻度标签为Times New Roman（数字）
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(get_english_font_properties(24))
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    # 设置图例（使用混合字体）
    plt.legend(fontsize=14, loc='best', frameon=True, fancybox=True, shadow=True, ncol=2, prop=get_mixed_font_properties(14))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  特征分布图已保存到: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("\n=== 生成模拟数据 ===")
    
    # 步骤1: 真实用户特征分布（10个用户）
    print("真实用户特征分布...")
    real_features, real_labels = generate_user_clusters(
        num_users=10, 
        samples_per_user=35,  # 每个用户35个样本，避免过度拥挤
        feature_dim=128
    )
    print(f"  真实特征形状: {real_features.shape}, 标签数: {len(np.unique(real_labels))}")
    
    # 步骤2: 生成与真实数据贴合的合成数据（分别贴合各用户）
    print("\n真实数据贴合的合成数据...")
    synthetic_features, synthetic_labels = generate_matched_synthetic_data(
        real_features,
        real_labels,
        num_samples=250,  # 生成250个合成样本，每个用户约25个
        num_users=10
    )
    print(f"  合成特征形状: {synthetic_features.shape}")
    
    # 步骤3: 绘制改进版本（带置信椭圆，生成数据按用户颜色显示）
    print("\n绘制改进版特征分布图（带置信椭圆）...")
    plot_feature_distribution_tsne_improved(
        real_features,
        real_labels,
        synthetic_features,
        synthetic_labels,
        output_path='feature_distribution_tsne_improved.png'
    )
    
    print("\n=== 所有图表生成完成！ ===")
    print("特征分布图已保存到: feature_distribution_tsne_improved.png")
