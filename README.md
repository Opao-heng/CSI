```markdown
# CSI 项目文档

## 项目简介
本项目旨在利用 CSI（Channel State Information） 数据进行深度学习建模，主要研究方向包括注意力机制（Attention）、生成对抗网络（GAN）、身份识别（Identify）和入侵检测（Intruder Detection）。项目包含多个研究模块，每个模块都有对应的数据预处理、模型构建、训练和评估代码。

## 目录结构
```
Research1/                # 研究方向一：注意力机制与 GAN
  Attention/              # 注意力机制相关模型训练结果
  GAN/                    # GAN 模型训练结果
  dataloder_ATT.py        # 注意力机制数据加载器
  loss_ATT.py             # 注意力机制损失函数
  model_ATT.py            # 注意力机制模型定义
  train_ATT.py            # 注意力机制训练脚本
  loss_GAN.py             # GAN 损失函数
  model_GAN.py            # GAN 模型定义
  train_GAN.py            # GAN 训练脚本

Research2/                # 研究方向二：身份识别与入侵检测
  identify/               # 身份识别模型训练结果
  intruder/               # 入侵检测模型训练结果
  dataloader_identify.py  # 身份识别数据加载器
  model_identify.py       # 身份识别模型定义
  loss_identify.py        # 身份识别损失函数
  train_identify.py       # 身份识别训练脚本
  plot_identity.py        # 身份识别结果可视化
  dataloader_intruder.py  # 入侵检测数据加载器
  model_intruder.py       # 入侵检测模型定义
  train_intruder.py       # 入侵检测训练脚本
  plot_intruder.py        # 入侵检测结果可视化

pre_process/              # 数据预处理模块
  SourceData.py           # 源域数据加载
  TargetData.py           # 目标域数据加载
  data.py                 # 数据处理与分组
  preprocess.py           # CSI 数据预处理与增强
  CSI_fig.py              # CSI 数据可视化

picture/                  # 项目相关图表
```

## 功能模块说明

### Research1 - 注意力机制与 GAN
- **注意力机制**：实现跨域特征对齐与分类，包含特征提取器、多头注意力模块、交叉注意力模块等组件。
- **GAN**：用于生成目标域样本，包含特征提取器、生成器、判别器等模块。

### Research2 - 身份识别与入侵检测
- **身份识别**：基于 CSI 数据进行用户身份识别，包含特征提取、注意力模块、投影头、分类器等组件。
- **入侵检测**：检测未知用户入侵行为，包含传统 OpenMax 方法、可学习阈值检测器等模块。

### pre_process - 数据预处理
- CSI 数据加载、归一化、去噪、分段、数据增强等操作。
- 支持源域与目标域数据的联合处理。

## 使用说明

### 环境依赖
- Python 3.8+
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib
- TorchVision（如需图像处理）

### 安装依赖
```bash
pip install torch numpy scikit-learn matplotlib torchvision
```

### 训练模型
#### 注意力机制
```bash
cd Research1
python train_ATT.py
```

#### GAN
```bash
cd Research1
python train_GAN.py
```

#### 身份识别
```bash
cd Research2
python train_identify.py
```

#### 入侵检测
```bash
cd Research2
python train_intruder.py
```

### 可视化结果
#### 身份识别结果
```bash
cd Research2
python plot_identity.py
```

#### 入侵检测结果
```bash
cd Research2
python plot_intruder.py
```

## 模型保存与加载
训练过程中模型会自动保存为 `.pth` 文件，路径如下：
- `Research1/Attention/best_attention_model.pth`
- `Research1/GAN/best_gan_model.pth`
- `Research2/identify/best_identify_model.pth`
- `Research2/intruder/best_intruder_detector.pth`

可通过 `torch.load()` 加载模型：
```python
import torch
model = torch.load('Research1/Attention/best_attention_model.pth')
```

## 数据格式说明
CSI 数据格式为 `[batch_size, 56, 3, 6000]`，表示：
- `batch_size`: 批次大小
- `56`: 子载波数量
- `3`: 天线对组合（Tx-Rx）
- `6000`: 时间步长

## 注意事项
- 所有模型训练脚本默认使用 GPU（如可用），可通过修改 `device` 参数切换为 CPU。
- 数据路径需根据实际数据存放位置修改。
- 可视化脚本依赖训练结果文件 `training_history.json`，请确保训练完成后运行。

## 许可证
本项目采用 MIT License，请参阅 [LICENSE](LICENSE) 文件。
```