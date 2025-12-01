"""
对比结果可视化脚本
用于从已保存的结果生成更详细的对比图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os


def load_comparison_results(results_path='Baseline/comparison/comparison_results.json'):
    """加载对比实验结果"""
    if not os.path.exists(results_path):
        print(f"错误: 未找到结果文件 {results_path}")
        print("请先运行对比实验: python Baseline/compare_models.py")
        return None
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


def plot_detailed_comparison(results, output_dir='Baseline/comparison'):
    """生成详细的对比图表"""
    
    # 过滤有效结果
    valid_models = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_models:
        print("没有有效的模型结果")
        return
    
    models = list(valid_models.keys())
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建综合对比图
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 训练时间对比
    ax1 = fig.add_subplot(gs[0, 0])
    times = [valid_models[m].get('training_time_minutes', 0) for m in models]
    bars1 = ax1.barh(models, times, color='#3498db', edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('训练时间 (分钟)', fontsize=11, fontweight='bold')
    ax1.set_title('训练时间对比', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    for i, v in enumerate(times):
        ax1.text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=9)
    
    # 2. 频谱保真度对比
    ax2 = fig.add_subplot(gs[0, 1])
    spectral = [valid_models[m].get('spectral_fidelity', 0) for m in models]
    bars2 = ax2.barh(models, spectral, color='#e74c3c', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('频谱保真度 MSE (越低越好)', fontsize=11, fontweight='bold')
    ax2.set_title('频谱保真度对比', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    # 标记最优值
    best_idx = np.argmin(spectral)
    bars2[best_idx].set_color('#2ecc71')
    
    # 3. 时域MSE对比
    ax3 = fig.add_subplot(gs[0, 2])
    time_mse = [valid_models[m].get('time_domain_mse', 0) for m in models]
    bars3 = ax3.barh(models, time_mse, color='#f39c12', edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('时域MSE (越低越好)', fontsize=11, fontweight='bold')
    ax3.set_title('时域MSE对比', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    # 标记最优值
    best_idx = np.argmin(time_mse)
    bars3[best_idx].set_color('#2ecc71')
    
    # 4. 多样性比率对比
    ax4 = fig.add_subplot(gs[1, 0])
    diversity = [valid_models[m].get('diversity_ratio', 0) for m in models]
    bars4 = ax4.barh(models, diversity, color='#9b59b6', edgecolor='black', linewidth=1.5)
    ax4.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='理想值=1.0')
    ax4.set_xlabel('多样性比率', fontsize=11, fontweight='bold')
    ax4.set_title('样本多样性比率对比', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    ax4.legend()
    
    # 5. 雷达图 - 综合性能
    ax5 = fig.add_subplot(gs[1, 1:], projection='polar')
    
    # 归一化指标到0-1范围（越接近1越好）
    metrics_normalized = {}
    for model in models:
        # 训练时间：越少越好，归一化后反转
        time_norm = 1 - (valid_models[model].get('training_time_minutes', 0) / max(times) if max(times) > 0 else 0)
        # 频谱保真度：越少越好，归一化后反转
        spec_norm = 1 - (valid_models[model].get('spectral_fidelity', 0) / max(spectral) if max(spectral) > 0 else 0)
        # 时域MSE：越少越好，归一化后反转
        mse_norm = 1 - (valid_models[model].get('time_domain_mse', 0) / max(time_mse) if max(time_mse) > 0 else 0)
        # 多样性：越接近1越好
        div_val = valid_models[model].get('diversity_ratio', 0)
        div_norm = 1 - abs(div_val - 1.0) if div_val > 0 else 0
        
        metrics_normalized[model] = [time_norm, spec_norm, mse_norm, div_norm]
    
    categories = ['训练效率', '频谱质量', '时域质量', '样本多样性']
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    colors_radar = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for idx, model in enumerate(models):
        values = metrics_normalized[model]
        values += values[:1]
        ax5.plot(angles, values, 'o-', linewidth=2, label=model, color=colors_radar[idx % len(colors_radar)])
        ax5.fill(angles, values, alpha=0.15, color=colors_radar[idx % len(colors_radar)])
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories, fontsize=10)
    ax5.set_ylim(0, 1)
    ax5.set_title('综合性能雷达图\n(越靠外越好)', fontsize=12, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax5.grid(True)
    
    # 6. 排名表格
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # 计算综合得分
    scores = {}
    for model in models:
        score = sum(metrics_normalized[model]) / len(metrics_normalized[model])
        scores[model] = score
    
    # 排序
    ranked_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 创建表格数据
    table_data = [['排名', '模型名称', '综合得分', '训练时间(分)', '频谱保真度', '时域MSE', '多样性比率']]
    
    for rank, (model, score) in enumerate(ranked_models, 1):
        row = [
            f'{rank}',
            model,
            f'{score:.4f}',
            f'{valid_models[model].get("training_time_minutes", 0):.2f}',
            f'{valid_models[model].get("spectral_fidelity", 0):.6f}',
            f'{valid_models[model].get("time_domain_mse", 0):.6f}',
            f'{valid_models[model].get("diversity_ratio", 0):.4f}'
        ]
        table_data.append(row)
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.08, 0.15, 0.12, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置第一名行的颜色
    for i in range(len(table_data[0])):
        table[(1, i)].set_facecolor('#2ecc71')
        table[(1, i)].set_text_props(weight='bold')
    
    ax6.set_title('模型综合排名表', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('生成模型对比实验 - 详细结果', fontsize=16, fontweight='bold', y=0.98)
    
    # 保存图表
    output_path = os.path.join(output_dir, 'detailed_comparison_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"详细对比图表已保存到: {output_path}")
    plt.close()


def print_summary(results):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("对比实验结果摘要")
    print("="*80)
    
    valid_models = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_models:
        print("没有有效的模型结果")
        return
    
    # 找出各项指标的最优模型
    best_time = min(valid_models.items(), key=lambda x: x[1].get('training_time_minutes', float('inf')))
    best_spectral = min(valid_models.items(), key=lambda x: x[1].get('spectral_fidelity', float('inf')))
    best_mse = min(valid_models.items(), key=lambda x: x[1].get('time_domain_mse', float('inf')))
    
    # 多样性最接近1.0的模型
    best_diversity = min(valid_models.items(), 
                         key=lambda x: abs(x[1].get('diversity_ratio', 0) - 1.0))
    
    print(f"\n✅ 训练速度最快: {best_time[0]} ({best_time[1]['training_time_minutes']:.2f}分钟)")
    print(f"✅ 频谱质量最佳: {best_spectral[0]} (MSE={best_spectral[1]['spectral_fidelity']:.6f})")
    print(f"✅ 时域质量最佳: {best_mse[0]} (MSE={best_mse[1]['time_domain_mse']:.6f})")
    print(f"✅ 样本多样性最佳: {best_diversity[0]} (比率={best_diversity[1]['diversity_ratio']:.4f})")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("正在加载对比实验结果...")
    results = load_comparison_results()
    
    if results:
        print("正在生成详细对比图表...")
        plot_detailed_comparison(results)
        print_summary(results)
        print("\n可视化完成！")
    else:
        print("\n请先运行对比实验:")
        print("  python Baseline/compare_models.py")
