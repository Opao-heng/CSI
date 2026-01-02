"""
批量运行所有消融实验的主脚本
"""

import subprocess
import sys
import os


def run_experiment(model_name, script_name):
    """运行单个实验"""
    print("\n" + "=" * 80)
    print(f"开始运行: {model_name}")
    print("=" * 80)
    
    try:
        # 使用subprocess运行训练脚本
        result = subprocess.run(
            [sys.executable, "-m", f"Research1.Ablation Study.{script_name}"],
            cwd=r"c:\Users\USER\Desktop\liuheng",
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✓ {model_name} 训练完成")
            return True
        else:
            print(f"\n✗ {model_name} 训练失败")
            return False
    except Exception as e:
        print(f"\n✗ {model_name} 运行出错: {e}")
        return False


def main():
    print("=" * 80)
    print("TFGAN-CAL 消融实验 - 批量运行")
    print("=" * 80)
    print("\n本脚本将依次运行以下5个模型的训练和评估:")
    print("  1. Model A (Baseline1)")
    print("  2. Model B (w/o Generator, Only CAL)")
    print("  3. Model C (w/o Freq-D, Time-only TFGAN)")
    print("  4. Model D (w/o AdaIN, Concat-only TFGAN)")
    print("  5. Model E (TFGAN-CAL, Full)")
    print("\n预计总耗时: 6-10小时")
    
    # 确认是否继续
    response = input("\n是否继续? (y/n): ")
    if response.lower() != 'y':
        print("已取消运行")
        return
    
    # 定义实验列表
    experiments = [
        ("Model A (Baseline1)", "train_model_A"),
        ("Model B (w/o Generator, Only CAL)", "train_model_B"),
        ("Model C (w/o Freq-D, Time-only TFGAN)", "train_model_C"),
        ("Model D (w/o AdaIN, Concat-only TFGAN)", "train_model_D"),
        ("Model E (TFGAN-CAL, Full)", "train_model_E"),
    ]
    
    results = {}
    
    # 运行所有实验
    for model_name, script_name in experiments:
        success = run_experiment(model_name, script_name)
        results[model_name] = success
    
    # 生成汇总结果
    print("\n" + "=" * 80)
    print("生成实验结果汇总...")
    print("=" * 80)
    
    try:
        subprocess.run(
            [sys.executable, "-m", "Research1.Ablation Study.summarize_results"],
            cwd=r"c:\Users\USER\Desktop\liuheng"
        )
    except Exception as e:
        print(f"汇总结果生成失败: {e}")
    
    # 打印最终报告
    print("\n" + "=" * 80)
    print("所有实验运行完成!")
    print("=" * 80)
    print("\n实验结果:")
    for model_name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {status} - {model_name}")
    
    print("\n结果文件:")
    print(f"  - CSV: Ablation Study/ablation_results_summary.csv")
    print(f"  - JSON: Ablation Study/ablation_results_summary.json")
    print()


if __name__ == "__main__":
    main()
