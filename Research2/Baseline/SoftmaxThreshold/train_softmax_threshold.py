import os
import sys
import json

import torch

# 将项目根目录加入 sys.path，便于导入 Research2 模块
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Research2.model_identify import IdentifyDetectionSystem
from Research2.Process.dataloader_intruder import load_intruder_data, create_intruder_data_loaders
from model_softmax_threshold import evaluate_softmax_threshold


def main(threshold: float = 0.9):
    """Softmax Threshold (ST) 基线实验入口。

    - 使用已训练好的身份识别模型 `R_Identify/best_identify_model.pth` 提取特征/Logits
    - 对入侵者测试集直接应用 Softmax 阈值进行开放集检测
    - 仅保存 Accuracy / AUROC / F1 三个指标到当前文件夹的 `intruder_detection_results.json`
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载身份识别模型
    print("加载身份识别模型...")
    identity_model = IdentifyDetectionSystem(num_classes=10, feature_dim=128, projection_dim=32).to(device)

    ckpt_path = os.path.join(PROJECT_ROOT, "Research2", "R_Identify", "best_identify_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到身份识别模型权重文件: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    identity_model.load_state_dict(checkpoint["model_state_dict"])
    identity_model.eval()
    print("身份识别模型加载完成。")

    # 加载入侵者检测数据
    print("加载入侵者检测数据...")
    datasets = load_intruder_data()
    data_loaders = create_intruder_data_loaders(datasets, batch_size=32)

    test_loader = data_loaders["intruder_test"]

    # 评估 ST 基线
    print(f"开始评估 Softmax Threshold 基线 (阈值 = {threshold})...")
    accuracy, auroc, f1 = evaluate_softmax_threshold(identity_model, test_loader, device, threshold)

    print("ST 基线在测试集上的性能：")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  AUROC           : {auroc:.4f}")
    print(f"  F1-Score        : {f1:.4f}")

    # 保存评估结果（仅三项指标）
    results = {
        "method": "Softmax Threshold (ST)",
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "f1_score": float(f1),
    }

    # 创建SoftmaxThreshold子文件夹用于保存结果
    output_dir = os.path.join(THIS_DIR, "SoftmaxThreshold")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "intruder_detection_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"评估结果已保存到: {output_path}")


if __name__ == "__main__":
    # 默认阈值 0.9，可根据需要在命令行或调用时调整
    main(threshold=0.9)
