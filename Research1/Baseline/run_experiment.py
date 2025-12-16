"""
快速启动脚本
用于快速运行单个baseline模型或对比实验
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='生成模型对比实验快速启动脚本')
    parser.add_argument(
        '--model',
        type=str,
        choices=['vae', 'cvae', 'cyclegan', 'dcgan', 'our_gan', 'all'],
        default='all',
        help='选择要训练的模型 (默认: all - 运行对比实验)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='训练轮数 (默认: 50)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=900,
        help='生成样本数量 (默认: 900)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("生成模型对比实验 - 快速启动")
    print("="*70)
    print(f"选择模型: {args.model}")
    print(f"训练轮数: {args.epochs}")
    print(f"生成样本数: {args.num_samples}")
    print("="*70 + "\n")
    
    if args.model == 'all':
        print("运行完整对比实验...")
        from Baseline.compare_models import main as compare_main
        compare_main()
        
    elif args.model == 'vae':
        print("训练VAE模型...")
        os.system(f'python Baseline/VAE/train_VAE.py')
        
    elif args.model == 'cvae':
        print("训练CVAE模型...")
        os.system(f'python Baseline/CVAE/train_CVAE.py')
        
    elif args.model == 'cyclegan':
        print("训练CycleGAN模型...")
        os.system(f'python Baseline/CycleGAN/train_CycleGAN.py')
        
    elif args.model == 'dcgan':
        print("训练DCGAN模型...")
        os.system(f'python Baseline/DCGAN/train_DCGAN.py')
        
    elif args.model == 'our_gan':
        print("训练我们的GAN模型...")
        os.system(f'python Research1/train_GAN.py')
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)


if __name__ == "__main__":
    main()
