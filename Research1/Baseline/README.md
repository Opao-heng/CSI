# 生成模型对比实验

本文件夹包含了与我们提出的GAN模型进行对比的先进生成模型实现。

## 目录结构

```
Baseline/
├── VAE/                    # 变分自编码器
│   ├── model_VAE.py       # VAE模型定义
│   ├── loss_VAE.py        # VAE损失函数
│   └── train_VAE.py       # VAE训练脚本
├── CVAE/                   # 条件变分自编码器
│   ├── model_CVAE.py      # CVAE模型定义
│   └── train_CVAE.py      # CVAE训练脚本
├── CycleGAN/               # 循环一致性GAN
│   ├── model_CycleGAN.py  # CycleGAN模型定义
│   ├── loss_CycleGAN.py   # CycleGAN损失函数
│   └── train_CycleGAN.py  # CycleGAN训练脚本
├── DCGAN/                  # 深度卷积GAN
│   ├── model_DCGAN.py     # DCGAN模型定义
│   └── train_DCGAN.py     # DCGAN训练脚本
└── compare_models.py       # 统一对比评估脚本
```

## 对比模型介绍

### 1. VAE (变分自编码器)
- **特点**: 概率生成模型，通过学习潜在空间的分布来生成样本
- **优势**: 训练稳定，生成样本多样性好
- **劣势**: 生成样本可能较模糊

### 2. CVAE (条件变分自编码器)
- **特点**: VAE的条件版本，可根据标签生成特定类别的样本
- **优势**: 可控生成，适合需要标签信息的场景
- **劣势**: 需要标签信息，模型更复杂

### 3. CycleGAN (循环一致性GAN)
- **特点**: 无需配对数据的域迁移模型，通过循环一致性损失保证转换质量
- **优势**: 不需要严格的数据配对，适合跨域迁移
- **劣势**: 训练复杂，需要两个生成器和两个判别器

### 4. DCGAN (深度卷积GAN)
- **特点**: 经典的深度卷积GAN架构
- **优势**: 结构简单，训练相对稳定
- **劣势**: 生成质量可能不如更先进的模型

### 5. Our GAN (我们的模型)
- **特点**: 基于域适应的GAN，包含特征提取器、生成器和双判别器
- **优势**: 
  - 双判别器（时域+频域）增强判别能力
  - MMD损失实现分布对齐
  - 频域一致性损失保证频谱质量
  - 自适应实例归一化融合域特征

## 使用方法

### 方法1：单独训练某个模型

```bash
# 训练VAE
python Baseline/VAE/train_VAE.py

# 训练CVAE
python Baseline/CVAE/train_CVAE.py

# 训练CycleGAN
python Baseline/CycleGAN/train_CycleGAN.py

# 训练DCGAN
python Baseline/DCGAN/train_DCGAN.py

# 训练我们的GAN
python Research1/train_GAN.py
```

### 方法2：使用快速启动脚本（推荐）

```bash
# 运行完整对比实验（所有模型）
python Baseline/run_experiment.py --model all

# 或者单独训练某个模型
python Baseline/run_experiment.py --model vae --epochs 50
python Baseline/run_experiment.py --model cvae --epochs 50
python Baseline/run_experiment.py --model cyclegan --epochs 50
python Baseline/run_experiment.py --model dcgan --epochs 50
python Baseline/run_experiment.py --model our_gan --epochs 50
```

### 方法3：运行统一对比实验

```bash
# 一次性训练和评估所有模型
python Baseline/compare_models.py
```

这将自动：
1. 依次训练所有5个模型
2. 生成900个合成样本
3. 评估每个模型的性能指标
4. 生成对比图表和JSON结果文件

### 方法4：可视化已有结果

如果已经运行过对比实验，可以使用可视化脚本生成更详细的对比图表：

```bash
# 生成详细的对比图表（包括雷达图、排名表等）
python Baseline/visualize_comparison.py
```

## 评估指标

所有模型将在以下指标上进行对比：

1. **训练时间** (分钟)
   - 越少越好

2. **频谱保真度** (MSE)
   - 越低越好
   - 衡量生成样本与真实样本在频域的差异

3. **时域MSE** (MSE)
   - 越低越好
   - 衡量生成样本与真实样本在时域的差异

4. **样本多样性比率**
   - 接近1.0越好
   - 衡量生成样本的多样性是否与真实样本相当

5. **模型参数量**
   - 反映模型复杂度

## 输出结果

运行对比实验后，将在 `Baseline/comparison/` 文件夹生成：

- `comparison_results.json` - 详细的数值结果
- `comparison_results.png` - 可视化对比图表
- 各模型的训练历史和生成样本

## 实验配置

所有模型使用相同的实验配置以确保公平对比：

- **训练轮数**: 50 epochs
- **生成样本数**: 900
- **批量大小**: 100
- **随机种子**: 30 (确保可重现)
- **源域数据**: source_env0_env1_data.pt
- **目标域数据**: target_env2_data.pt (每类10个样本)

## 注意事项

1. 确保已安装所有依赖包：torch, numpy, matplotlib, sklearn
2. 确保数据文件存在于 `Data/` 文件夹
3. 建议使用GPU加速训练
4. 完整的对比实验可能需要数小时，请耐心等待

## 预期结果

根据设计，我们的GAN模型应在以下方面表现优异：

- ✅ 更好的频谱保真度（双判别器架构）
- ✅ 更强的域适应能力（MMD损失）
- ✅ 更高的生成质量（频域一致性损失）
- ✅ 更稳定的训练过程（谱归一化 + 梯度裁剪）

## 引用

如果您使用了这些baseline模型，请引用相应的原始论文：

- VAE: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
- CVAE: Sohn et al., "Learning Structured Output Representation using Deep Conditional Generative Models", NeurIPS 2015
- CycleGAN: Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", ICCV 2017
- DCGAN: Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", ICLR 2016
