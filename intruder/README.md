# 综合入侵者检测系统

## 系统概述

本系统实现了一个综合入侵者检测模型，结合了以下三种检测方法：

1. **传统OpenMax检测** - 基于极值理论建模已知用户特征分布
2. **传统能量检测** - 基于模型输出总置信度的能量检测方法
3. **深度学习阈值优化检测** - 利用训练好的已知用户身份特征进行阈值优化

## 模型架构

### ComprehensiveIntruderDetector (综合入侵者检测器)

综合入侵者检测器结合了多种检测方法的优点：

- **TraditionalOpenMax**: 基于极值理论的开放集识别方法
- **TraditionalEnergyDetector**: 基于模型输出能量的检测方法
- **LearnableOpenMax**: 可学习的OpenMax检测器
- **LearnableEnergyDetector**: 可学习的能量检测器
- **LearnableThresholdDetector**: 可学习的多维阈值检测器

所有检测结果通过一个融合网络进行加权融合，得到最终的入侵者检测结果。

## 文件结构

```
intruder/
├── model_intruder.py          # 入侵者检测模型定义
├── train_intruder.py          # 入侵者检测模型训练脚本
├── test_intruder.py           # 入侵者检测模型测试脚本
├── example_usage.py           # 使用示例
├── intruder_data_loader.py    # 数据加载器
├── README.md                  # 说明文档
├── identify/                  # 身份识别模型权重文件
│   └── best_identify_model.pth
└── outputs/                   # 训练输出文件
    ├── comprehensive_intruder_detector.pth
    ├── final_comprehensive_intruder_detector.pth
    ├── training_curves.png
    ├── score_distribution.png
    ├── confusion_matrix_*.png
    └── test_results.json
```

## 训练流程

1. 首先确保身份识别模型已训练完成，权重文件位于 `intruder/identify/best_identify_model.pth`

2. 训练综合入侵者检测模型:
   ```
   python train_intruder.py
   ```

## 测试流程

运行测试脚本评估模型性能:
```
python test_intruder.py
```

## 使用示例

查看使用示例了解如何在代码中使用模型:
```
python example_usage.py
```

## 输出文件

训练和测试过程会生成以下文件:

- `comprehensive_intruder_detector.pth` - 最佳模型权重
- `final_comprehensive_intruder_detector.pth` - 最终模型权重
- `training_curves.png` - 训练曲线图
- `score_distribution.png` - 决策分数分布图
- `test_results.json` - 测试结果
- `confusion_matrix_*.png` - 混淆矩阵图

## 模型特点

1. **多方法融合**: 结合传统方法和深度学习方法的优势
2. **可学习参数**: 阈值和检测参数可通过训练优化
3. **开放集识别**: 能够识别未知的入侵者
4. **可视化支持**: 提供详细的性能评估和可视化图表

## 使用说明

1. 确保数据文件已准备就绪
2. 按顺序运行训练脚本
3. 使用测试脚本评估模型性能
4. 查看生成的图表和结果文件

## 模型输入输出

### 输入
- `features`: 身份识别模型提取的特征向量 (batch_size, feature_dim)
- `logits`: 身份识别模型的输出logits (batch_size, num_classes)

### 输出
- `predictions`: 入侵者检测预测结果 (0=合法用户, 1=入侵者)
- `fusion_weights`: 融合后的检测置信度
- 各子检测器的详细输出结果