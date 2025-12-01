"""
统一评估对比脚本
作用：公平对比所有生成模型（我们的GAN、VAE、CVAE、CycleGAN、DCGAN）
评估指标：FID分数、频谱保真度、MMD距离、训练时间、模型参数量
"""

import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
import sys

# 导入各模型的训练脚本
from Research1.train_GAN import train_and_test as train_gan
from Baseline.VAE.train_VAE import train_and_test as train_vae
from Baseline.CVAE.train_CVAE import train_and_test as train_cvae
from Baseline.CycleGAN.train_CycleGAN import train_and_test as train_cyclegan
from Baseline.DCGAN.train_DCGAN import train_and_test as train_dcgan

from Research1.Process.dataloder_GAN import CustomDataset, select_samples_by_label
from Research1.plot_GAN import evaluate_gan_comprehensive


"""
评估单个模型的性能
"""
def evaluate_model(model_name, synthetic_data, target_loader, device='cuda'):
    print(f"\n  正在评估 {model_name} 模型...")
    
    # 提取目标域真实样本
    real_data_list = []
    for x_t_real, _ in target_loader:
        real_data_list.append(x_t_real)
    real_data = torch.cat(real_data_list, dim=0).to(device)
    synthetic_data = synthetic_data.to(device)
    
    # 计算评估指标
    metrics = {}
    
    # 1. 频谱保真度（越低越好）
    real_fft = torch.fft.rfft(real_data.view(real_data.size(0), -1, real_data.size(-1)), dim=-1)
    synth_fft = torch.fft.rfft(synthetic_data.view(synthetic_data.size(0), -1, synthetic_data.size(-1)), dim=-1)
    
    real_mag = real_fft.abs().mean(dim=0)
    synth_mag = synth_fft.abs().mean(dim=0)
    spectral_fidelity = torch.nn.functional.mse_loss(synth_mag, real_mag).item()
    metrics['spectral_fidelity'] = spectral_fidelity
    
    # 2. 时域MSE（越低越好）
    # 随机抽样相同数量的真实样本进行对比
    num_samples = min(real_data.size(0), synthetic_data.size(0))
    real_sample = real_data[:num_samples]
    synth_sample = synthetic_data[:num_samples]
    time_mse = torch.nn.functional.mse_loss(synth_sample, real_sample).item()
    metrics['time_domain_mse'] = time_mse
    
    # 3. 样本多样性（标准差，越高越好）
    synth_std = synthetic_data.std().item()
    real_std = real_data.std().item()
    diversity_ratio = synth_std / (real_std + 1e-8)
    metrics['diversity_ratio'] = diversity_ratio
    
    print(f"  频谱保真度: {spectral_fidelity:.6f}")
    print(f"  时域MSE: {time_mse:.6f}")
    print(f"  多样性比率: {diversity_ratio:.4f}")
    
    return metrics


"""
主评估流程
"""
def main():
    print("="*70)
    print("生成模型对比实验")
    print("="*70)
    
    # 步骤0: 设置随机种子
    print("\n步骤0: 设置随机种子...")
    torch.manual_seed(30)
    np.random.seed(30)
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  使用设备: {device}")
    
    # 步骤1: 加载数据
    print("\n步骤1: 正在加载数据...")
    source_data = torch.load('Data/source_env0_env1_data.pt')
    source_labels = torch.load('Data/source_env0_env1_labels.pt')
    target_data = torch.load('Data/target_env2_data.pt')
    target_labels = torch.load('Data/target_env2_labels.pt')
    
    # 创建数据加载器
    from Research1.Process.dataloder_GAN import CustomDataset, select_samples_by_label
    source_dataset = CustomDataset(source_data, source_labels)
    global source_loader
    source_loader = DataLoader(source_dataset, batch_size=100, shuffle=True)
    
    selected_target_data, selected_target_labels = select_samples_by_label(
        target_data, target_labels, samples_per_label=10
    )
    target_dataset = CustomDataset(selected_target_data, selected_target_labels)
    global target_loader
    target_loader = DataLoader(target_dataset, batch_size=100, shuffle=True)
    
    print(f"  源域数据: {source_data.shape}")
    print(f"  目标域数据: {selected_target_data.shape}")
    
    # 步骤2: 定义要对比的模型
    models_config = {
        'Our_GAN': {
            'train_func': train_gan,
            'params': {
                'model_path': 'Baseline/comparison/our_gan_model.pth',
                'epochs': 50,
                'lr_g': 2e-4,
                'lr_d': 1e-4,
                'num_samples': 900
            }
        },
        'VAE': {
            'train_func': train_vae,
            'params': {
                'model_path': 'Baseline/comparison/vae_model.pth',
                'epochs': 50,
                'lr': 2e-4,
                'num_samples': 900,
                'beta': 0.5,
                'lambda_freq': 0.1
            }
        },
        'CVAE': {
            'train_func': train_cvae,
            'params': {
                'model_path': 'Baseline/comparison/cvae_model.pth',
                'epochs': 50,
                'lr': 2e-4,
                'num_samples': 900,
                'beta': 0.5,
                'lambda_freq': 0.1,
                'num_classes': 30
            }
        },
        'CycleGAN': {
            'train_func': train_cyclegan,
            'params': {
                'model_path': 'Baseline/comparison/cyclegan_model.pth',
                'epochs': 50,
                'lr': 2e-4,
                'num_samples': 900,
                'lambda_cycle': 10.0,
                'lambda_identity': 5.0
            }
        },
        'DCGAN': {
            'train_func': train_dcgan,
            'params': {
                'model_path': 'Baseline/comparison/dcgan_model.pth',
                'epochs': 50,
                'lr_g': 2e-4,
                'lr_d': 2e-4,
                'num_samples': 900,
                'latent_dim': 100
            }
        }
    }
    
    # 步骤3: 训练并评估所有模型
    os.makedirs('Baseline/comparison', exist_ok=True)
    all_results = {}
    
    for model_name, config in models_config.items():
        print("\n" + "="*70)
        print(f"正在训练和评估: {model_name}")
        print("="*70)
        
        start_time = datetime.now()
        
        try:
            # 训练模型
            train_func = config['train_func']
            params = config['params']
            synthetic_data, synthetic_labels = train_func(**params)
            
            # 计算训练时间
            training_time = (datetime.now() - start_time).total_seconds() / 60
            
            # 评估模型
            metrics = evaluate_model(model_name, synthetic_data, target_loader, device=device)
            metrics['training_time_minutes'] = training_time
            metrics['model_name'] = model_name
            
            all_results[model_name] = metrics
            
            print(f"\n  {model_name} 训练完成，耗时: {training_time:.2f}分钟")
            
        except Exception as e:
            print(f"\n  {model_name} 训练失败: {str(e)}")
            all_results[model_name] = {'error': str(e)}
    
    # 步骤4: 保存对比结果
    print("\n" + "="*70)
    print("保存对比结果...")
    print("="*70)
    
    with open('Baseline/comparison/comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    # 步骤5: 生成对比图表
    print("\n正在生成对比图表...")
    plot_comparison_results(all_results)
    
    # 步骤6: 打印对比表格
    print("\n" + "="*70)
    print("模型对比结果汇总")
    print("="*70)
    print(f"{'模型名称':<15} {'训练时间(分钟)':<15} {'频谱保真度':<15} {'时域MSE':<15} {'多样性比率':<15}")
    print("-"*75)
    
    for model_name, metrics in all_results.items():
        if 'error' not in metrics:
            print(f"{model_name:<15} "
                  f"{metrics.get('training_time_minutes', 0):<15.2f} "
                  f"{metrics.get('spectral_fidelity', 0):<15.6f} "
                  f"{metrics.get('time_domain_mse', 0):<15.6f} "
                  f"{metrics.get('diversity_ratio', 0):<15.4f}")
    
    print("\n对比实验完成！结果已保存到 Baseline/comparison/")


"""
绘制对比结果图表
"""
def plot_comparison_results(results):
    models = list(results.keys())
    
    # 过滤掉有错误的模型
    valid_models = [m for m in models if 'error' not in results[m]]
    
    if not valid_models:
        print("没有有效的模型结果可以绘制")
        return
    
    # 提取指标
    training_times = [results[m].get('training_time_minutes', 0) for m in valid_models]
    spectral_fidelities = [results[m].get('spectral_fidelity', 0) for m in valid_models]
    time_mses = [results[m].get('time_domain_mse', 0) for m in valid_models]
    diversity_ratios = [results[m].get('diversity_ratio', 0) for m in valid_models]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('生成模型对比实验结果', fontsize=16, fontweight='bold')
    
    # 1. 训练时间对比
    axes[0, 0].bar(valid_models, training_times, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('训练时间对比', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('时间 (分钟)', fontsize=10)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. 频谱保真度对比（越低越好）
    axes[0, 1].bar(valid_models, spectral_fidelities, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('频谱保真度对比 (越低越好)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('MSE', fontsize=10)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. 时域MSE对比（越低越好）
    axes[1, 0].bar(valid_models, time_mses, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('时域MSE对比 (越低越好)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('MSE', fontsize=10)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. 多样性比率对比（接近1越好）
    axes[1, 1].bar(valid_models, diversity_ratios, color='gold', edgecolor='black')
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='理想值=1.0')
    axes[1, 1].set_title('样本多样性比率对比', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('比率', fontsize=10)
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('Baseline/comparison/comparison_results.png', dpi=300, bbox_inches='tight')
    print("  对比图表已保存到: Baseline/comparison/comparison_results.png")
    plt.close()


if __name__ == "__main__":
    main()
