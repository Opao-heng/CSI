# TFGAN-CAL 消融实验

本目录包含TFGAN-CAL模型的消融实验代码,用于验证各个组件的必要性。

## 实验设计

根据论文中的消融实验设计,实现了以下5个模型变体:

### Model A (Baseline) - w/o TFGAN & CAL
- **描述**: 基线模型,仅使用源域数据训练CNN分类器
- **组件**: 特征提取器 + 分类器
- **缺失**: 生成器、判别器、交叉注意力机制
- **文件**: `model_A_baseline.py`, `train_model_A.py`

### Model B - w/o Generator, Only CAL
- **描述**: 仅使用交叉注意力机制,不使用生成器进行数据增强
- **组件**: 特征提取器 + 交叉注意力 + 分类器
- **缺失**: GAN生成器和判别器
- **文件**: `model_B_only_CAL.py`, `train_model_B.py`

### Model C - w/o Freq-D, Time-only GAN
- **描述**: 保留生成器,但仅使用时域判别器(移除频域判别器)
- **组件**: 特征提取器 + 生成器(AdaIN) + 时域判别器
- **缺失**: 频域判别器
- **文件**: `model_C_time_only_GAN.py`, `train_model_C.py`

### Model D - w/o AdaIN, Concat-only GAN
- **描述**: 移除AdaIN,仅使用concat方式融合特征
- **组件**: 特征提取器 + 生成器(无AdaIN) + 双判别器
- **缺失**: AdaIN自适应归一化
- **文件**: `model_D_concat_only.py`, `train_model_D.py`

### Model E (TFGAN-CAL, Full)
- **描述**: 完整模型,包含所有组件
- **组件**: 特征提取器 + 生成器(AdaIN) + 时域判别器 + 频域判别器 + 交叉注意力
- **文件**: `train_model_E.py`

## 评估指标

每个模型都会评估以下5个指标:

1. **FID** (Fréchet Inception Distance) - 分布相似度,越小越好
2. **IS** (Inception Score) - 多样性评估,越大越好  
3. **时域MSE** (Time-domain MSE) - 信号保真度,越小越好
4. **频谱CC** (Spectral Correlation) - 频谱相关性,越接近1越好
5. **跨域准确率** (Cross-domain Accuracy) - 目标域分类准确率,越大越好

## 运行实验

### 1. 训练单个模型

```bash
# Model A (Baseline1)
cd "c:\Users\USER\Desktop\liuheng"
python -m "Research1.Ablation Study.train_model_A"

# Model B (w/o Generator)
python -m "Research1.Ablation Study.train_model_B"

# Model C (w/o Freq-D)
python -m "Research1.Ablation Study.train_model_C"

# Model D (w/o AdaIN)
python -m "Research1.Ablation Study.train_model_D"

# Model E (Full)
python -m "Research1.Ablation Study.train_model_E"
```

### 2. 生成结果汇总

所有模型训练完成后,运行汇总脚本:

```bash
python -m "Research1.Ablation Study.summarize_results"
```

这将生成:
- `ablation_results_summary.csv` - CSV格式的对比表格
- `ablation_results_summary.json` - JSON格式的详细结果

## 输出目录结构

```
Ablation Study/
├── model_A/
│   └── evaluation_results.json
├── model_B/
│   └── evaluation_results.json
├── model_C/
│   └── evaluation_results.json
├── model_D/
│   └── evaluation_results.json
├── model_E/
│   └── evaluation_results.json
├── ablation_results_summary.csv
└── ablation_results_summary.json
```

## 预期结果

根据论文中的消融实验结果:

| 模型 | 核心组件 | 跨域准确率 | 生成质量 |
|------|---------|-----------|---------|
| Model A | 无 (Baseline) | 最低 | N/A |
| Model B | w/o Generator | 较低 | N/A |
| Model C | w/o Freq-D | 中等 | 中等 |
| Model D | w/o AdaIN | 中等 | 较低 |
| Model E | Full | 最高 | 最高 |

## 注意事项

1. 所有模型都使用相同的随机种子(42)确保结果可重现
2. Model A和B没有生成器,因此FID、IS、时域MSE、频谱CC指标为N/A
3. 训练时间:
   - Model A/B: 约30-60分钟
   - Model C/D: 约1-2小时
   - Model E: 约2-3小时(包含GAN训练和注意力模型训练)
4. 需要CUDA支持以获得合理的训练速度

## 依赖要求

- PyTorch >= 1.10.0
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- scipy

## 引用

如果使用此消融实验代码,请引用原始论文中的消融实验部分。
