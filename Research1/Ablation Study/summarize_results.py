"""
消融实验结果汇总脚本
读取所有模型的评估结果并生成对比表格
"""

import os
import json
import pandas as pd


def load_model_results(model_dir):
    """加载单个模型的评估结果"""
    result_file = os.path.join(model_dir, 'evaluation_results.json')
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def format_metric(value):
    """格式化指标值"""
    if value is None or value == float('inf'):
        return 'N/A'
    elif isinstance(value, (int, float)):
        if abs(value) < 0.01:
            return f'{value:.6f}'
        else:
            return f'{value:.4f}'
    return str(value)


def main():
    base_dir = 'Ablation Study'
    models = {
        'Model A (Baseline1)': os.path.join(base_dir, 'model_A'),
        'Model B (w/o Generator)': os.path.join(base_dir, 'model_B'),
        'Model C (w/o Freq-D)': os.path.join(base_dir, 'model_C'),
        'Model D (w/o AdaIN)': os.path.join(base_dir, 'model_D'),
        'Model E (Full)': os.path.join(base_dir, 'model_E'),
    }
    
    # 收集结果
    results_data = []
    for model_name, model_dir in models.items():
        result = load_model_results(model_dir)
        if result:
            metrics = result.get('metrics', {})
            results_data.append({
                '模型': model_name,
                'FID↓': format_metric(metrics.get('fid')),
                'IS↑': format_metric(metrics.get('inception_score')),
                '时域MSE↓': format_metric(metrics.get('time_domain_mse')),
                '频谱CC↑': format_metric(metrics.get('spectral_correlation')),
                '跨域准确率(%)↑': format_metric(metrics.get('cross_domain_accuracy'))
            })
        else:
            print(f"警告: 未找到 {model_name} 的结果文件")
    
    # 创建DataFrame
    df = pd.DataFrame(results_data)
    
    # 打印表格
    print("\n" + "=" * 100)
    print("消融实验结果对比表")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    print("\n注: ↓表示越小越好, ↑表示越大越好")
    print("\n模型说明:")
    print("  Model A: Baseline1 - 仅使用源域数据训练CNN分类器")
    print("  Model B: w/o Generator - 仅使用交叉注意力,不使用生成器")
    print("  Model C: w/o Freq-D - 仅使用时域判别器,移除频域判别器")
    print("  Model D: w/o AdaIN - 移除AdaIN,仅使用concat融合")
    print("  Model E: Full - 完整模型(TFGAN + CAL)")
    print()
    
    # 保存为CSV
    csv_path = os.path.join(base_dir, 'ablation_results_summary.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存到: {csv_path}")
    
    # 保存为JSON
    json_path = os.path.join(base_dir, 'ablation_results_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
    print(f"结果已保存到: {json_path}")


if __name__ == "__main__":
    main()
