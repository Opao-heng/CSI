import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from matplotlib import font_manager
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

# 设置中文字体支持
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']


def create_zh_font(size=12):
    """
    创建带有指定字体大小的中文字体属性对象
    """
    try:
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
        chinese_font_names = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'STHeiTi', 'STSong']
        
        for font_name in chinese_font_names:
            if font_name in available_fonts:
                font_path = font_manager.findfont(font_manager.FontProperties(family=font_name))
                return font_manager.FontProperties(fname=font_path, size=size)
        
        return font_manager.FontProperties(size=size)
    except Exception as e:
        print(f"字体加载异常: {e}")
        return font_manager.FontProperties(size=size)


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
                                            synthetic_labels=None, output_path='GAN/feature_distribution_tsne_improved.png'):
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
                   c=[colors[idx]], label=f'User {int(user_id)} (Real)',
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
                               c=[colors[idx]], label=f'User {int(user_id)} (Generated)',
                               alpha=0.5, s=60, edgecolors=colors[idx], linewidth=0.5, marker='^')
        else:
            # 如果没有合成标签，用红色表示所有生成样本
            plt.scatter(synthetic_2d[:, 0], synthetic_2d[:, 1],
                       c='red', label='Generated Samples',
                       alpha=0.4, s=40, edgecolors='darkred', linewidth=0.5, marker='^')
    
    # 设置标题和标签（使用中文字体）

    plt.xlabel('t-SNE成分1', fontsize=14, fontweight='bold', fontproperties=create_zh_font(14))
    plt.ylabel('t-SNE成分2', fontsize=14, fontweight='bold', fontproperties=create_zh_font(14))    
    
    # 设置图例（使用中文字体）
    plt.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True, ncol=3, prop=create_zh_font(10))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  特征分布图已保存到: {output_path}")
    plt.close()


if __name__ == "__main__":
    """
    演示代码：使用模拟数据生成feature_distribution_tsne图
    """
    print("\n=== 生成模拟数据 ===")
    
    # 步骤1: 生成真实用户特征分布（10个用户）
    print("生成真实用户特征分布...")
    real_features, real_labels = generate_user_clusters(
        num_users=10, 
        samples_per_user=35,  # 每个用户35个样本，避免过度拥挤
        feature_dim=128
    )
    print(f"  真实特征形状: {real_features.shape}, 标签数: {len(np.unique(real_labels))}")
    
    # 步骤2: 生成与真实数据贴合的合成数据（分别贴合各用户）
    print("\n生成与真实数据贴合的合成数据...")
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
        output_path='GAN/feature_distribution_tsne_improved.png'
    )
    
    print("\n=== 所有图表生成完成！ ===")
    print("特征分布图已保存到: GAN/feature_distribution_tsne_improved.png")
