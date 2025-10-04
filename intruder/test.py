"""
测试入侵者检测系统
在目标域测试集上评估系统性能
"""

import torch
import numpy as np
import os
from model import IntruderDetectionSystem, ComprehensiveIntruderDetector
from data_loader import load_and_split_data, create_data_loaders

class IntruderIdentificationSystem:
    """
    完整的入侵者识别系统
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.intruder_detector = ComprehensiveIntruderDetector(num_classes=10, feature_dim=128)
        self.is_fitted = False
    
    def fit(self, features, labels):
        """
        拟合检测器参数
        """
        print("拟合入侵者检测器参数...")
        self.intruder_detector.fit(features, labels)
        self.is_fitted = True
        print("参数拟合完成")
    
    def predict(self, data_loader, openmax_threshold=0.8, energy_threshold=1.5):
        """
        预测样本类别
        """
        if not self.is_fitted:
            raise RuntimeError("系统尚未拟合参数!")
            
        self.model.eval()
        all_features = []
        all_logits = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                # 移动到设备
                data = data.to(self.device)
                labels = labels.numpy()
                
                # 提取特征和预测
                outputs = self.model(data)
                features = outputs['features'].cpu().numpy()
                logits = outputs['logits'].cpu().numpy()
                
                # 入侵者检测
                detection_results = self.intruder_detector.predict(
                    features, logits, openmax_threshold, energy_threshold)
                final_predictions = detection_results['predictions']
                
                all_features.append(features)
                all_logits.append(logits)
                all_predictions.extend(final_predictions)
                all_labels.extend(labels)
        
        return np.array(all_predictions), np.array(all_labels), np.vstack(all_features)
    
    def evaluate(self, data_loader, openmax_threshold=0.8, energy_threshold=1.5):
        """
        评估系统性能
        """
        predictions, labels, features = self.predict(data_loader, openmax_threshold, energy_threshold)
        
        # 入侵者检测性能
        binary_labels = np.where(labels == -1, 0, 1)  # 入侵者=0, 已知用户=1
        binary_predictions = np.where(predictions == -1, 0, 1)  # 入侵者=0, 已知用户=1
        
        # 计算指标
        detection_accuracy = np.mean(binary_predictions == binary_labels)
        
        tp = np.sum((binary_predictions == 0) & (binary_labels == 0))  # 真正例
        fp = np.sum((binary_predictions == 0) & (binary_labels == 1))  # 假正例
        fn = np.sum((binary_predictions == 1) & (binary_labels == 0))  # 假负例
        
        detection_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        detection_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall) if (detection_precision + detection_recall) > 0 else 0.0
        
        # 身份识别性能（仅针对已知用户）
        known_user_mask = (labels >= 0)
        if np.sum(known_user_mask) > 0:
            known_user_predictions = predictions[known_user_mask]
            known_user_labels = labels[known_user_mask]
            
            identification_accuracy = np.mean(known_user_predictions == known_user_labels)
            
            # 计算宏平均F1分数
            f1_scores = []
            for i in range(10):  # 10个已知用户
                class_mask = (known_user_labels == i)
                if np.sum(class_mask) > 0:
                    tp = np.sum((known_user_predictions == i) & (known_user_labels == i))
                    fp = np.sum((known_user_predictions == i) & (known_user_labels != i))
                    fn = np.sum((known_user_predictions != i) & (known_user_labels == i))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    
                    if (precision + recall) > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0.0
                        
                    f1_scores.append(f1)
                    
            macro_f1 = np.mean(f1_scores) if len(f1_scores) > 0 else 0.0
        else:
            identification_accuracy = 0.0
            macro_f1 = 0.0
            
        overall_accuracy = np.mean(predictions == labels)
        
        metrics = {
            'detection_accuracy': detection_accuracy,
            'detection_precision': detection_precision,
            'detection_recall': detection_recall,
            'detection_f1': detection_f1,
            'identification_accuracy': identification_accuracy,
            'identification_macro_f1': macro_f1,
            'overall_accuracy': overall_accuracy
        }
        
        return metrics

def main():
    """
    主测试函数
    """
    print("开始测试入侵者识别系统...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查模型文件
    model_path = "intruder/best_intruder_model.pth"
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        return
    
    # 初始化模型
    print("加载模型...")
    model = IntruderDetectionSystem(num_classes=10, feature_dim=128, projection_dim=32).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型加载完成")
    
    # 加载数据
    print("加载测试数据...")
    datasets = load_and_split_data()
    data_loaders = create_data_loaders(datasets, batch_size=8)
    
    # 提取身份识别验证集特征用于拟合检测器参数
    print("提取身份识别验证集特征用于参数拟合...")
    val_features = []
    val_labels = []
    
    model.eval()
    with torch.no_grad():
        for data, labels in data_loaders['identity_validation']:
            data = data.to(device)
            outputs = model(data)
            features = outputs['features'].cpu().numpy()
            val_features.append(features)
            val_labels.append(labels.numpy())
    
    val_features = np.vstack(val_features)
    val_labels = np.hstack(val_labels)
    
    # 初始化入侵者识别系统
    print("初始化入侵者识别系统...")
    intruder_system = IntruderIdentificationSystem(model, device)
    
    # 拟合参数
    intruder_system.fit(val_features, val_labels)
    
    # 在入侵者检测测试集上评估
    print("在入侵者检测测试集上评估系统性能...")
    metrics = intruder_system.evaluate(data_loaders['intruder_test'])
    
    # 打印结果
    print("\n=== 系统性能评估结果 ===")
    print(f"入侵者检测准确率: {metrics['detection_accuracy']:.4f}")
    print(f"入侵者检测精确率: {metrics['detection_precision']:.4f}")
    print(f"入侵者检测召回率: {metrics['detection_recall']:.4f}")
    print(f"入侵者检测F1分数: {metrics['detection_f1']:.4f}")
    print(f"身份识别准确率: {metrics['identification_accuracy']:.4f}")
    print(f"身份识别宏平均F1分数: {metrics['identification_macro_f1']:.4f}")
    print(f"整体准确率: {metrics['overall_accuracy']:.4f}")
    
    # 显示一些预测示例
    print("\n=== 预测示例 ===")
    predictions, labels, _ = intruder_system.predict(data_loaders['intruder_test'])
    
    # 随机选择10个样本显示
    indices = np.random.choice(len(predictions), size=min(10, len(predictions)), replace=False)
    for i in indices:
        true_label = labels[i]
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

if __name__ == "__main__":
    main()