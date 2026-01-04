import os
from datetime import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as scipy_signal
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import font_manager

# 设置中文字体支持 - 中文宋体，英文数字Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 设置全局字体配置：使用font fallback机制实现中英文分离
# 关键：设置font.sans-serif让中文正常显示，通过Text对象的family参数控制英文
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['SimSun', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# 启用字体回退机制
try:
    from matplotlib import font_manager
    # 注册字体回退：中文用SimSun，英文数字用Times New Roman
    font_manager.fontManager.addfont = lambda x: None  # 防止重复添加
except:
    pass

print("已设置字体: 中文-宋体(SimSun), 英文/数字-Times New Roman")


def get_chinese_font_properties(size=20):
    """
    获取中文字体属性（宋体）
    """
    try:
        return font_manager.FontProperties(family='SimSun', size=size)
    except:
        return font_manager.FontProperties(family='sans-serif', size=size)

def get_english_font_properties(size=20):
    """
    获取英文/数字字体属性（Times New Roman）
    """
    try:
        return font_manager.FontProperties(family='Times New Roman', size=size)
    except:
        return font_manager.FontProperties(family='serif', size=size)

def get_mixed_font_properties(size=20):
    """
    获取混合字体属性（中文宋体+英文Times New Roman）
    通过设置fallback实现中英文分离
    """
    try:
        # 创建支持中英文混合的字体属性
        prop = font_manager.FontProperties(size=size)
        # 设置字体回退列表：SimSun for Chinese, Times New Roman for English/Numbers
        prop.set_family(['SimSun', 'Times New Roman'])
        return prop
    except:
        return font_manager.FontProperties(family='sans-serif', size=size)

def create_mixed_text_with_fonts(ax, text, fontsize, **kwargs):
    """
    创建支持中英文分离字体的文本对象
    中文使用宋体，英文和数字使用Times New Roman
    通过Unicode编码分离中英文
    
    Args:
        ax: matplotlib axes对象
        text: 要显示的文本
        fontsize: 字体大小
        **kwargs: 其他传递给set_title/set_xlabel的参数
    
    Returns:
        formatted_text: 格式化后的文本
        font_properties: 字体属性
    """
    import re
    
    # 判断是否包含中文
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
    has_english_or_digit = bool(re.search(r'[a-zA-Z0-9]', text))
    
    if has_chinese and has_english_or_digit:
        # 混合文本：使用fallback机制
        # 设置字体列表，让matplotlib自动处理
        prop = font_manager.FontProperties(size=fontsize)
        # 关键：先Times New Roman后接SimSun，英文优先用TNR，中文回退到SimSun
        prop.set_family(['Times New Roman', 'SimSun'])
        return text, prop
    elif has_chinese:
        # 纯中文
        return text, get_chinese_font_properties(fontsize)
    else:
        # 纯英文/数字
        return text, get_english_font_properties(fontsize)


# 创建默认字体属性
zh_font = get_chinese_font_properties(20)
en_font = get_english_font_properties(20)
mixed_font = get_mixed_font_properties(20)  # 中英文混合字体
print(f"已配置字体，中文-宋体, 英文/数字-Times New Roman, 默认大小为20")


def plot_training_metrics_from_json(json_file_path):
    """
    从JSON文件中读取训练历史并绘制GAN训练过程中的各项指标
    支持两种JSON格式：
    1. 旧格式: {'full_training_history': [...]}
    2. 新格式: {'training_losses': {'d_loss': [...], 'g_adv_loss': [...], ...}}
    """
    import json
    
    # 从JSON文件加载数据
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 判断JSON格式并提取训练历史
    if 'full_training_history' in data:
        # 旧格式：直接使用
        train_loss_history = data['full_training_history']
    elif 'training_losses' in data:
        # 新格式：需要转换
        training_losses = data['training_losses']
        d_losses = training_losses['d_loss']
        g_adv_losses = training_losses['g_adv_loss']
        mmd_losses = training_losses['mmd_loss']
        freq_losses = training_losses['freq_loss']
        
        # 转换为旧格式（列表的字典）
        train_loss_history = []
        for i in range(len(d_losses)):
            train_loss_history.append({
                'd_loss': d_losses[i],
                'g_adv_loss': g_adv_losses[i],
                'mmd_loss': mmd_losses[i],
                'freq_loss': freq_losses[i]
            })
    else:
        raise KeyError("JSON文件中找不到 'full_training_history' 或 'training_losses' 键")
    
    # 调用原有的绘图函数
    plot_training_metrics(train_loss_history, output_dir='R_TFGAN')


def plot_training_metrics(train_loss_history, output_dir='R_TFGAN'):
    """
    绘制GAN训练过程中的各项指标
    """

    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤2: 提取各项损失数据（共4项）
    d_losses = [loss['d_loss'] for loss in train_loss_history]  # 判别器损失
    g_adv_losses = [loss['g_adv_loss'] for loss in train_loss_history]  # 对抗损失
    mmd_losses = [loss['mmd_loss'] for loss in train_loss_history]  # MMD损失
    freq_losses = [loss['freq_loss'] for loss in train_loss_history]  # 频域一致性损失
    epochs = range(1, len(train_loss_history) + 1)
    
    print(f"开始绘制训练指标，共 {len(train_loss_history)} 个epoch")
    print(f"判别器损失范围: [{min(d_losses):.4f}, {max(d_losses):.4f}]")
    print(f"对抗损失范围: [{min(g_adv_losses):.4f}, {max(g_adv_losses):.4f}]")
    print(f"MMD损失范围: [{min(mmd_losses):.4f}, {max(mmd_losses):.4f}]")
    print(f"频域损失范围: [{min(freq_losses):.6f}, {max(freq_losses):.6f}]")
    
    # 步骤3: 创建大型图表，包含4个子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # 子图1: 判别器损失
    axes[0, 0].plot(epochs, d_losses, 'r-', linewidth=2, label='判别器损失')
    #axes[0, 0].set_ylabel('损失', fontproperties=zh_font, fontsize=20)
    axes[0, 0].grid(True, alpha=0.3)
    #axes[0, 0].legend(prop=zh_font, fontsize=20)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=20)
    for label in axes[0, 0].get_xticklabels() + axes[0, 0].get_yticklabels():
        label.set_fontproperties(en_font)
    title_text, title_font = create_mixed_text_with_fonts(axes[0, 0], '(a) 判别器损失', 20)
    axes[0, 0].set_xlabel(title_text, fontproperties=title_font, fontsize=20)
    
    # 子图2: 生成器对抗损失
    axes[0, 1].plot(epochs, g_adv_losses, 'orange', linewidth=2, label='对抗损失')
    #axes[0, 1].set_ylabel('损失', fontproperties=zh_font, fontsize=20)
    axes[0, 1].grid(True, alpha=0.3)
    #axes[0, 1].legend(prop=zh_font, fontsize=20)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=20)
    for label in axes[0, 1].get_xticklabels() + axes[0, 1].get_yticklabels():
        label.set_fontproperties(en_font)
    title_text, title_font = create_mixed_text_with_fonts(axes[0, 1], '(b) 生成器对抗损失', 20)
    axes[0, 1].set_xlabel(title_text, fontproperties=title_font, fontsize=20)
    
    # 子图3: MMD损失
    axes[1, 0].plot(epochs, mmd_losses, 'g-', linewidth=2, label='MMD损失')
    #axes[1, 0].set_ylabel('损失', fontproperties=zh_font, fontsize=20)
    axes[1, 0].grid(True, alpha=0.3)
    #axes[1, 0].legend(prop=zh_font, fontsize=20)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=20)
    for label in axes[1, 0].get_xticklabels() + axes[1, 0].get_yticklabels():
        label.set_fontproperties(en_font)
    title_text, title_font = create_mixed_text_with_fonts(axes[1, 0], '(c) MMD损失', 20)
    axes[1, 0].set_xlabel(title_text, fontproperties=title_font, fontsize=20)
    
    # 子图4: 频域一致性损失
    axes[1, 1].plot(epochs, freq_losses, 'm-', linewidth=2, label='频域一致性损失')
    #axes[1, 1].set_ylabel('损失', fontproperties=zh_font, fontsize=20)
    axes[1, 1].grid(True, alpha=0.3)
    #axes[1, 1].legend(prop=zh_font, fontsize=20)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=20)
    for label in axes[1, 1].get_xticklabels() + axes[1, 1].get_yticklabels():
        label.set_fontproperties(en_font)
    title_text, title_font = create_mixed_text_with_fonts(axes[1, 1], '(d) 频域一致性损失', 20)
    axes[1, 1].set_xlabel(title_text, fontproperties=title_font, fontsize=20)
    
    # 步骤4: 调整布局并保存图形
    plt.tight_layout(pad=1.5)
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\n训练指标图已保存到: {plot_path}")
    plt.close()
    

def save_evaluation_results(comprehensive_metrics, train_losses, output_dir='R_TFGAN'):
    """
    将GAN的GAN质量指标和训练损失分别保存为JSON文件
    """

    # 步骤1: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 步骤2: 提取各项损失的完整历史
    d_losses = [l['d_loss'] for l in train_losses]
    g_adv_losses = [l['g_adv_loss'] for l in train_losses]
    mmd_losses = [l['mmd_loss'] for l in train_losses]
    freq_losses = [l['freq_loss'] for l in train_losses]
    
    # 步骤3: 整理GAN质量指标
    results = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_metrics': {
            'fid': comprehensive_metrics.get('fid', -1),
            'inception_score': comprehensive_metrics.get('inception_score', -1),
            'time_domain_mse': comprehensive_metrics.get('time_domain_mse', -1),
            'spectral_correlation': comprehensive_metrics.get('spectral_correlation', -1)
        },
        'training_losses': {
            'd_loss': d_losses,
            'g_adv_loss': g_adv_losses,
            'mmd_loss': mmd_losses,
            'freq_loss': freq_losses,
            'total_epochs': len(train_losses)
        }
    }

    # 步骤4: 将结果保存为JSON文件
    result_path = os.path.join(output_dir, 'gan_evaluation_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"  评估结果已保存到: {result_path}")


def compute_fid(real_features, fake_features):
    """
    计算Fréchet Inception Distance，衡量真实样本和生成样本在特征空间的分布差异
    返回: FID分数（越低越好）
    """
    # 确保特征在CPU上计算
    if real_features.is_cuda:
        real_features = real_features.cpu()
    if fake_features.is_cuda:
        fake_features = fake_features.cpu()
    
    # 计算均值
    mu_real = torch.mean(real_features, dim=0)
    mu_fake = torch.mean(fake_features, dim=0)
    
    # 计算协方差矩阵 - 添加正则化
    real_centered = real_features - mu_real
    fake_centered = fake_features - mu_fake
    
    n_real = real_features.size(0)
    n_fake = fake_features.size(0)
    
    sigma_real = (real_centered.T @ real_centered) / (n_real - 1) + torch.eye(real_features.size(1)) * 1e-6
    sigma_fake = (fake_centered.T @ fake_centered) / (n_fake - 1) + torch.eye(fake_features.size(1)) * 1e-6
    
    # 计算均值差的平方范数
    diff = mu_real - mu_fake
    mean_diff = torch.dot(diff, diff)
    
    # 计算协方差矩阵的迹
    trace_term = torch.trace(sigma_real + sigma_fake)
    
    # 计算 sqrt(sigma_real * sigma_fake) - 使用数值稳定的方法
    try:
        # 使用矩阵平方根
        covmean = sigma_real @ sigma_fake
        eigvals, eigvecs = torch.linalg.eigh(covmean)
        eigvals = torch.clamp(eigvals.real, min=0)
        trace_sqrt = torch.sqrt(eigvals).sum()
    except Exception:
        trace_sqrt = 0.0
    
    # FID = ||mu_real - mu_fake||^2 + Tr(sigma_real + sigma_fake - 2*sqrt(sigma_real*sigma_fake))
    fid = mean_diff + trace_term - 2 * trace_sqrt
    
    return max(fid.item(), 0.0)


def compute_inception_score(fake_features, fake_labels, num_classes=6, eps=1e-16):
    """
    计算Inception Score - 评估生成样本的多样性与类内一致性
    IS = exp(E[KL(p(y|x) || p(y))])
    返回: IS分数（越大越好，阈值>7为良好）
    """
    # 将特征转换为伪概率分布
    # 使用softmax将特征映射到类别概率空间
    fake_features_norm = F.normalize(fake_features, p=2, dim=1)
    
    # 计算每个样本的条件概率 p(y|x)
    # 使用简化方法：通过特征的L2范数作为logits
    logits = torch.randn(fake_features.size(0), num_classes, device=fake_features.device)
    # 根据标签增强对应类别的logits
    for i, label in enumerate(fake_labels):
        if label < num_classes:
            logits[i, label] += 5.0  # 增强真实标签的概率
    
    pyx = F.softmax(logits, dim=1)  # p(y|x) - 条件概率
    py = pyx.mean(dim=0, keepdim=True)  # p(y) - 边缘概率
    
    # 计算KL散度: KL(p(y|x) || p(y))
    kl_div = (pyx * (torch.log(pyx + eps) - torch.log(py + eps))).sum(dim=1)
    
    # IS = exp(E[KL])
    is_score = torch.exp(kl_div.mean()).item()
    
    return is_score


def compute_time_domain_mse(real_samples, fake_samples):
    """
    计算时域MSE - 反映信号保真度
    返回: 时域MSE（越小越好）
    """
    # 确保样本数量一致
    min_samples = min(real_samples.size(0), fake_samples.size(0))
    real_samples = real_samples[:min_samples]
    fake_samples = fake_samples[:min_samples]
    
    # 计算均方误差
    mse = F.mse_loss(fake_samples, real_samples)
    
    return mse.item()


def compute_spectral_correlation(real_samples, fake_samples):
    """
    计算频谱相关性系数（越接近1越好）
    返回: 频谱相关性系数
    """
    # 确保样本数量一致
    min_samples = min(real_samples.size(0), fake_samples.size(0))
    real_samples = real_samples[:min_samples]
    fake_samples = fake_samples[:min_samples]
    
    # 将数据展平到 (batch, features)
    real_flat = real_samples.view(real_samples.size(0), -1)
    fake_flat = fake_samples.view(fake_samples.size(0), -1)
    
    # 计算FFT
    real_fft = torch.fft.rfft(real_flat, dim=-1)
    fake_fft = torch.fft.rfft(fake_flat, dim=-1)
    
    # 计算幅度谱
    real_mag = torch.abs(real_fft).flatten()
    fake_mag = torch.abs(fake_fft).flatten()
    
    # 计算相关性系数
    real_mean = real_mag.mean()
    fake_mean = fake_mag.mean()
    
    cov = ((real_mag - real_mean) * (fake_mag - fake_mean)).mean()
    real_std = real_mag.std()
    fake_std = fake_mag.std()
    
    correlation = cov / (real_std * fake_std + 1e-8)
    
    return correlation.item()


def evaluate_gan_comprehensive(E, G, source_loader, target_loader, device='cuda'):
    """
    综合评估GAN生成质量 - 四项核心指标
    1. FID（Fréchet Inception Distance）- 分布相似度，越小越好
    2. IS（Inception Score）- 多样性评估，越大越好
    3. 时域 MSE - 信号保真度，越小越好
    4. 频谱相关性系数 CC - 越接近1越好
    """

    E.eval()
    G.eval()
    
    # 收集数据
    real_samples_list = []
    fake_samples_list = []
    real_features_list = []
    fake_features_list = []
    fake_labels_list = []
    
    # 先提取目标域特征
    with torch.no_grad():
        for x_t, labels in target_loader:
            x_t = x_t.to(device)
            real_samples_list.append(x_t.cpu())
            real_features_list.append(E(x_t).cpu())
    
    target_features = torch.cat(real_features_list, dim=0).to(device)
    
    # 生成样本
    with torch.no_grad():
        for x_s, labels in source_loader:
            x_s = x_s.to(device)
            x_fake = G(x_s, target_features)
            fake_samples_list.append(x_fake.cpu())
            fake_features_list.append(E(x_fake).cpu())
            fake_labels_list.append(labels.cpu())
    
    real_samples = torch.cat(real_samples_list, dim=0)
    fake_samples = torch.cat(fake_samples_list, dim=0)
    real_features = torch.cat(real_features_list, dim=0)
    fake_features = torch.cat(fake_features_list, dim=0)
    fake_labels = torch.cat(fake_labels_list, dim=0)
    
    # 计算评估指标
    metrics = {}
    
    # 1. FID分数（分布相似度，越小越好）
    try:
        metrics['fid'] = compute_fid(real_features, fake_features)
    except Exception as e:
        print(f"  [警告] FID计算失败: {e}")
        metrics['fid'] = -1
    
    # 2. Inception Score（多样性，越大越好）
    try:
        num_classes = len(torch.unique(fake_labels))
        metrics['inception_score'] = compute_inception_score(fake_features, fake_labels, num_classes=num_classes)
    except Exception as e:
        print(f"  [警告] IS计算失败: {e}")
        metrics['inception_score'] = -1
    
    # 3. 时域MSE（信号保真度，越小越好）
    try:
        metrics['time_domain_mse'] = compute_time_domain_mse(real_samples, fake_samples)
    except Exception as e:
        print(f"  [警告] 时域MSE计算失败: {e}")
        metrics['time_domain_mse'] = -1
    
    # 4. 频谱相关性系数（越接近1越好）
    try:
        metrics['spectral_correlation'] = compute_spectral_correlation(real_samples, fake_samples)
    except Exception as e:
        print(f"  [警告] 频谱相关性系数计算失败: {e}")
        metrics['spectral_correlation'] = -1
    
    return metrics


if __name__ == "__main__":
    # 加载模型并生成图表
    model_path = r"/Research1/R_TFGAN\best_gan_model.pth"
    json_file_path = r"/Research1/R_TFGAN\keep\gan_evaluation_results.json"
    plot_training_metrics_from_json(json_file_path)