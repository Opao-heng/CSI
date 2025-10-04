import torch
import numpy as np
import os
from model import IntruderDetectionSystem, ComprehensiveIntruderDetector
from data_loader import load_and_split_data, create_data_loaders


def extract_features(model, data_loader, device):
    """
    从数据加载器中提取特征和标签
    """
    model.eval()
    all_features = []
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            # 移动到设备
            data = data.to(device)
            labels = labels.numpy()
            
            # 提取特征和预测
            outputs = model(data)
            features = outputs['features'].cpu().numpy()
            logits = outputs['logits'].cpu().numpy()
            
            all_features.append(features)
            all_logits.append(logits)
            all_labels.append(labels)
    
    return np.vstack(all_features), np.vstack(all_logits), np.hstack(all_labels)

def train_intruder_detector(model_path, output_path, device):
    """
    训练入侵者检测器（使用专门的入侵者检测验证集）
    """
    print("开始训练入侵者检测器...")

    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        return

    # 初始化模型
    print("加载身份识别模型...")
    identity_model = IntruderDetectionSystem(num_classes=10, feature_dim=128, projection_dim=32).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    identity_model.load_state_dict(checkpoint['model_state_dict'])
    identity_model.eval()
    
    print(f"身份识别模型加载完成")
    
    # 加载数据
    print("加载数据...")
    datasets = load_and_split_data()
    data_loaders = create_data_loaders(datasets, batch_size=32)
    
    # 使用入侵者检测验证集来训练入侵者检测器
    print("使用入侵者检测验证集训练入侵者检测器...")
    val_features, val_logits, val_labels = extract_features(identity_model, data_loaders['intruder_validation'], device)
    
    # 分离合法用户和入侵者数据
    legal_mask = val_labels >= 0
    intruder_mask = val_labels == -1
    
    legal_features = val_features[legal_mask]
    legal_labels = val_labels[legal_mask]
    intruder_features = val_features[intruder_mask]
    intruder_labels = val_labels[intruder_mask]
    
    print(f"  合法用户样本数: {len(legal_features)}")
    print(f"  入侵者样本数: {len(intruder_features)}")
    
    # 初始化综合入侵者检测器
    print("初始化综合入侵者检测器...")
    intruder_detector = ComprehensiveIntruderDetector(num_classes=10, feature_dim=128)
    
    # 训练/拟合入侵者检测器参数（只使用合法用户数据进行拟合）
    print("训练入侵者检测器...")
    intruder_detector.fit(legal_features, legal_labels)
    
    # 在入侵者检测测试集上评估入侵者检测器性能
    print("提取入侵者检测测试集特征...")
    test_features, test_logits, test_labels = extract_features(identity_model, data_loaders['intruder_test'], device)
    
    print("评估入侵者检测器性能...")
    # 使用不同的阈值进行评估
    thresholds = [(0.7, 1.3), (0.8, 1.5), (0.9, 1.7)]
    
    best_f1 = 0.0
    best_threshold = (0.8, 1.5)
    
    for openmax_threshold, energy_threshold in thresholds:
        # 进行预测
        detection_results = intruder_detector.predict(
            test_features, test_logits, openmax_threshold, energy_threshold)
        predictions = detection_results['predictions']
        
        # 计算评估指标
        # 入侵者检测性能
        binary_labels = np.where(test_labels == -1, 0, 1)  # 入侵者=0, 已知用户=1
        binary_predictions = np.where(predictions == -1, 0, 1)  # 入侵者=0, 已知用户=1
        
        # 计算指标
        tp = np.sum((binary_predictions == 0) & (binary_labels == 0))  # 真正例
        fp = np.sum((binary_predictions == 0) & (binary_labels == 1))  # 假正例
        fn = np.sum((binary_predictions == 1) & (binary_labels == 0))  # 假负例
        
        detection_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        detection_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall) if (detection_precision + detection_recall) > 0 else 0.0
        
        print(f"阈值 (OpenMax: {openmax_threshold}, Energy: {energy_threshold}) - "
              f"精确率: {detection_precision:.4f}, 召回率: {detection_recall:.4f}, F1: {detection_f1:.4f}")
        
        if detection_f1 > best_f1:
            best_f1 = detection_f1
            best_threshold = (openmax_threshold, energy_threshold)
    
    print(f"最佳阈值: OpenMax={best_threshold[0]}, Energy={best_threshold[1]}, F1={best_f1:.4f}")
    
    # 保存训练好的入侵者检测器
    print("保存入侵者检测器...")
    torch.save({
        'intruder_detector': intruder_detector,
        'best_threshold': best_threshold,
        'val_features': val_features,
        'val_labels': val_labels
    }, output_path)
    
    print(f"入侵者检测器已保存到 {output_path}")
    
    # 显示一些预测示例
    print("\n=== 预测示例 ===")
    detection_results = intruder_detector.predict(
        test_features, test_logits, best_threshold[0], best_threshold[1])
    predictions = detection_results['predictions']
    
    # 随机选择10个样本显示
    indices = np.random.choice(len(predictions), size=min(10, len(predictions)), replace=False)
    for i in indices:
        true_label = test_labels[i]
        pred_label = predictions[i]
        
        if true_label == -1:
            true_str = "入侵者"
        else:
            true_str = f"用户{true_label}"
            
        if pred_label == -1:
            pred_str = "入侵者"
        else:
            pred_str = f"用户{pred_label}"
            
        correct = "✓" if true_label == pred_label else "✗"
        print(f"  样本: 真实={true_str}, 预测={pred_str} {correct}")


def main():
    """
    主训练函数
    """
    print("开始训练入侵者检测器...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型路径
    model_path = "intruder/best_intruder_model.pth"  # 身份识别训练好的模型
    output_path = "intruder/trained_intruder_detector.pth"
    
    # 训练入侵者检测器
    train_intruder_detector(model_path, output_path, device)


if __name__ == "__main__":
    main()