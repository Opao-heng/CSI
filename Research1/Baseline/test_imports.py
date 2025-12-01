"""
测试所有模型能否正常导入
用于验证代码的完整性
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    print("="*70)
    print("测试Baseline模型导入")
    print("="*70)
    
    all_passed = True
    
    # 测试VAE
    print("\n[1/4] 测试VAE模型...")
    try:
        from Baseline.VAE.model_VAE import build_vae_model
        from Baseline.VAE.loss_VAE import vae_loss, combined_vae_loss
        model = build_vae_model()
        print("  ✅ VAE模型导入成功")
    except Exception as e:
        print(f"  ❌ VAE模型导入失败: {e}")
        all_passed = False
    
    # 测试CVAE
    print("\n[2/4] 测试CVAE模型...")
    try:
        from Baseline.CVAE.model_CVAE import build_cvae_model
        model = build_cvae_model()
        print("  ✅ CVAE模型导入成功")
    except Exception as e:
        print(f"  ❌ CVAE模型导入失败: {e}")
        all_passed = False
    
    # 测试CycleGAN
    print("\n[3/4] 测试CycleGAN模型...")
    try:
        from Baseline.CycleGAN.model_CycleGAN import build_cyclegan_model
        from Baseline.CycleGAN.loss_CycleGAN import adversarial_loss, cycle_consistency_loss
        G_S2T, G_T2S, D_S, D_T = build_cyclegan_model()
        print("  ✅ CycleGAN模型导入成功")
    except Exception as e:
        print(f"  ❌ CycleGAN模型导入失败: {e}")
        all_passed = False
    
    # 测试DCGAN
    print("\n[4/4] 测试DCGAN模型...")
    try:
        from Baseline.DCGAN.model_DCGAN import build_dcgan_model
        G, D = build_dcgan_model()
        print("  ✅ DCGAN模型导入成功")
    except Exception as e:
        print(f"  ❌ DCGAN模型导入失败: {e}")
        all_passed = False
    
    # 总结
    print("\n" + "="*70)
    if all_passed:
        print("✅ 所有模型导入测试通过！")
        print("可以开始运行对比实验了。")
    else:
        print("❌ 部分模型导入失败，请检查错误信息。")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
