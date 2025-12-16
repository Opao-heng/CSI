import torch
import numpy as np
import os
import sys
import json

# 添加上级目录到sys.path以解决导入问题
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'no_contrastive'))

from model_identify_no_contrastive import IdentifyDetectionSystemNoContrastive
from model_intruder_traditional_only import TraditionalIntruderDetectorOnly
from dataloader_intruder import load_intruder_data, create_intruder_data_loaders
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.optim as optim


def validate_intruder_detector(model, identity_model, data_loader, device):
    """
    在验证集上测试入侵者检测器性能
    """
    model.eval()
    identity_model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data, labels, identity_labels = batch

            # 移动到设备
            data = data.to(device)
            labels = labels.to(device)

            # 使用身份识别模型提取特征
            identity_outputs = identity_model(data)
            features = identity_outputs['features']
            logits = identity_outputs['logits']

            # 检查特征和logits的维度，确保至少是2D
            if features.dim() == 1:
                features = features.unsqueeze(0)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            # 确保batch维度一致
            batch_size = data.size(0)
            if features.size(0) != batch_size:
                features = features[:batch_size] if features.size(0) > batch_size else features
            if logits.size(0) != batch_size:
                logits = logits[:batch_size] if logits.size(0) > batch_size else logits

            # 使用传统入侵者检测器
            detector_outputs = model(features, logits, identity_labels)
            predictions = detector_outputs['predictions']
            probabilities = detector_outputs['probabilities']

            # 确保预测结果和标签维度一致
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            if probabilities.dim() == 0:
                probabilities = probabilities.unsqueeze(0)

            # 收集预测结果和真实标签
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    if len(all_predictions) == 0 or len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 处理空数组情况
    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    accuracy = np.mean(all_predictions == all_labels) if len(all_labels) > 0 else 0.0
    
    # 使用zero_division='warn'避免警告
    f1 = f1_score(all_labels, all_predictions, zero_division='warn')
    precision = precision_score(all_labels, all_predictions, zero_division='warn')
    recall = recall_score(all_labels, all_predictions, zero_division='warn')
    
    return accuracy, f1, precision, recall


def test_intruder_detector(model, identity_model, data_loader, device):
    """
    在测试集上测试入侵者检测器性能
    """
    model.eval()
    identity_model.eval()
    
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # 处理不同格式的batch数据
            if len(batch) == 3:
                data, labels, identity_labels = batch
            else:
                data, labels = batch
                identity_labels = None  # 如果没有identity_labels，则设为None
            
            # 移动到设备
            data = data.to(device)
            labels = labels.to(device)
            
            # 使用身份识别模型提取特征
            identity_outputs = identity_model(data)
            features = identity_outputs['features']
            logits = identity_outputs['logits']
            
            # 检查特征和logits的维度，确保至少是2D
            if features.dim() == 1:
                features = features.unsqueeze(0)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            # 确保batch维度一致
            batch_size = data.size(0)
            if features.size(0) != batch_size:
                features = features[:batch_size] if features.size(0) > batch_size else features
            if logits.size(0) != batch_size:
                logits = logits[:batch_size] if logits.size(0) > batch_size else logits
            
            # 使用传统入侵者检测器
            if identity_labels is not None:
                detector_outputs = model(features, logits, identity_labels)
            else:
                # 如果没有identity_labels，创建一个默认的标签数组
                # 假设所有样本都是合法用户（标签为0）
                default_identity_labels = torch.zeros(batch_size, dtype=torch.long, device=features.device)
                detector_outputs = model(features, logits, default_identity_labels)
            predictions = detector_outputs['predictions']
            scores = detector_outputs['probabilities']  # 修改这里，使用正确的键
            
            # 确保预测结果和分数维度一致
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
            
            # 收集预测结果和真实标签
            # 将预测结果转换为numpy并确保是整数类型
            pred_np = predictions.cpu().numpy()
            if pred_np.dtype != np.int64:
                pred_np = (pred_np > 0.5).astype(np.int64)
            all_predictions.extend(pred_np)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    # 计算评估指标
    if len(all_predictions) == 0 or len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, np.array([])
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # 处理空数组情况
    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, np.array([])
    
    accuracy = np.mean(all_predictions == all_labels) if len(all_labels) > 0 else 0.0
    
    # 使用zero_division='warn'避免警告
    f1 = f1_score(all_labels, all_predictions, zero_division='warn')
    precision = precision_score(all_labels, all_predictions, zero_division='warn')
    recall = recall_score(all_labels, all_predictions, zero_division='warn')
    
    return accuracy, f1, precision, recall, all_scores


def train_and_evaluate_intruder_detection_with_ablation_model():
    """
    使用消融实验的身份识别模型训练和评估入侵者检测性能
    """
    print("开始训练和评估消融实验模型的入侵者检测性能...")
    
    # 消融实验标识：仅使用传统OpenMax版本
    print("=" * 60)
    print("消融实验：入侵者检测模型（仅使用传统OpenMax组件）")
    print("训练配置：使用消融实验身份识别模型提取特征，仅使用传统OpenMax进行入侵者检测")
    print("=" * 60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化消融实验的身份识别模型
    print("加载消融实验的身份识别模型...")
    identity_model = IdentifyDetectionSystemNoContrastive(num_classes=10, feature_dim=128, projection_dim=32).to(device)
    
    # 加载模型权重
    model_path = os.path.join(os.path.dirname(__file__), "..", "no_contrastive", "best_identify_model_no_contrastive.pth")  # 使用绝对路径
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        identity_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"消融实验身份识别模型加载完成")
    else:
        print(f"未找到消融实验身份识别模型: {model_path}")
        # 尝试在当前目录查找
        alt_path = os.path.join(os.path.dirname(__file__), "..", "no_contrastive", "best_identify_model_no_contrastive.pth")
        if os.path.exists(alt_path):
            checkpoint = torch.load(alt_path, map_location=device)
            identity_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"消融实验身份识别模型加载完成")
        else:
            print(f"也未找到备选路径的模型: {alt_path}")
            return
    
    identity_model.eval()
    
    # 加载入侵者检测专用数据
    print("加载入侵者检测数据...")
    try:
        # 临时切换到数据加载器所在目录
        current_dir = os.getcwd()
        data_loader_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        os.chdir(data_loader_dir)
        datasets = load_intruder_data()
        # 恢复原始工作目录
        os.chdir(current_dir)
    except FileNotFoundError as e:
        print(f"入侵者检测数据文件未找到: {e}")
        print("请确保入侵者检测数据文件存在于正确的位置。")
        return
    except Exception as e:
        print(f"加载入侵者检测数据时发生错误: {e}")
        return
        
    data_loaders = create_intruder_data_loaders(datasets, batch_size=32)

    # 初始化仅包含传统OpenMax组件的入侵者检测器
    print("初始化入侵者检测模型（仅使用传统OpenMax组件）...")
    traditional_detector = TraditionalIntruderDetectorOnly(num_known_users=10).to(device)
    
    # 保存训练好的入侵者检测模型
    torch.save({
        'model_state_dict': traditional_detector.state_dict(),
    }, 'best_intruder_detector_traditional_only.pth')
    print("入侵者检测模型已保存到 best_intruder_detector_traditional_only.pth")
    
    # 在测试集上评估入侵者检测性能
    print("在测试集上评估入侵者检测性能...")
    test_accuracy, test_f1, test_precision, test_recall, test_scores = test_intruder_detector(
        traditional_detector, identity_model, data_loaders['intruder_test'], device)

    print(f"消融实验模型入侵者检测结果:")
    print(f"  准确率: {test_accuracy:.4f}")
    print(f"  F1分数: {test_f1:.4f}")
    print(f"  精确率: {test_precision:.4f}")
    print(f"  召回率: {test_recall:.4f}")

    # 保存评估结果
    results = {
        'accuracy': test_accuracy,
        'f1_score': test_f1,
        'precision': test_precision,
        'recall': test_recall
    }
    
    output_dir = "."  # 保存在当前目录（与py文件同级）
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'intruder_detection_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("入侵者检测评估结果已保存到 intruder_detection_results.json")


if __name__ == "__main__":
    train_and_evaluate_intruder_detection_with_ablation_model()