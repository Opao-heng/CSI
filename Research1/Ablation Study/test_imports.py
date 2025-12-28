"""
快速测试脚本 - 验证所有模型是否可以正确导入
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def test_imports():
    """测试所有模型的导入"""
    print("=" * 70)
    print("测试消融实验模块导入...")
    print("=" * 70)
    
    tests = [
        ("Model A (Baseline)", "from model_A_baseline import BaselineModel"),
        ("Model B (Only CAL)", "from model_B_only_CAL import ModelB_OnlyCAL"),
        ("Model C (Time-only GAN)", "from model_C_time_only_GAN import build_model"),
        ("Model D (Concat-only)", "from model_D_concat_only import build_model"),
    ]
    
    success_count = 0
    total_count = len(tests)
    
    for model_name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✓ {model_name} - 导入成功")
            success_count += 1
        except Exception as e:
            print(f"✗ {model_name} - 导入失败: {e}")
    
    print("\n" + "=" * 70)
    print(f"测试结果: {success_count}/{total_count} 成功")
    print("=" * 70)
    
    if success_count == total_count:
        print("\n所有模型导入测试通过! ✓")
        return True
    else:
        print("\n部分模型导入失败! ✗")
        return False


if __name__ == "__main__":
    # 切换到正确的工作目录
    os.chdir(os.path.dirname(__file__))
    test_imports()
