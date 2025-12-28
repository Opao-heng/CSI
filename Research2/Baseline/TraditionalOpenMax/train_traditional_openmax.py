import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from Research2.train_intruder import extract_features
from Research2.model_intruder import TraditionalOpenMax

# 将项目根目录加入 sys.path，便于导入 Research2 模块
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Research2.model_identify import IdentifyDetectionSystem
from Research2.Process.dataloader_intruder import load_intruder_data, create_intruder_data_loaders
from model_traditional_openmax import train_traditional_openmax, evaluate_traditional_openmax


def train_traditional_openmax(identity_model, train_loader, device, num_known_users: int = 10, alpha: int = 3):
    """在合法用户样本上拟合 Traditional OpenMax 模型。

    - 使用身份识别模型提取特征
    - labels 为二分类标签：0-合法用户，1-入侵者
    - identity_labels 为身份 ID：0~num_known_users-1，对应合法用户；入侵者为 -1
    """
    features, _, labels, identity_labels = extract_features(identity_model, train_loader, device)

    if features.size == 0 or labels.size == 0 or identity_labels.size == 0:
        raise RuntimeError("训练数据为空，无法训练 Traditional OpenMax 模型。")

    tom = TraditionalOpenMax(num_known_users=num_known_users, alpha=alpha)
    tom.fit(features, labels, identity_labels)
    return tom


def evaluate_traditional_openmax(identity_model, tom: TraditionalOpenMax, data_loader, device):
    """在给定数据集上评估 Traditional OpenMax 基线。

    - 使用身份识别模型提取特征
    - 使用 TOM 进行入侵者检测
    - 预测规则：tom.predict 返回 predictions (0 合法, 1 入侵者) 与 tail_prob（越大越“像已知类”）
    - AUROC 分数使用 1 - tail_prob 作为“入侵者概率”
    """
    features, _, labels, _ = extract_features(identity_model, data_loader, device)

    if labels.size == 0:
        return 0.0, 0.0, 0.0

    preds, tail_probs = tom.predict(features)  # preds: 0/1, tail_probs: [0,1]

    y_true = labels.astype(int)
    y_pred = preds.astype(int)
    intruder_scores = 1.0 - np.asarray(tail_probs, dtype=float)

    accuracy = float((y_pred == y_true).mean()) if len(y_true) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division="warn"))

    try:
        auroc = float(roc_auc_score(y_true, intruder_scores))
    except ValueError:
        auroc = 0.0

    return accuracy, auroc, f1


def main(num_known_users: int = 10, alpha: int = 3):
    """Traditional OpenMax (TOM) 基线实验入口。

    - 不再使用可学习的综合检测器，仅使用经典 OpenMax 距离+Weibull 建模进行判决
    - 使用身份识别模型提取特征，TOM 只在特征空间上工作
    - 在入侵者测试集上输出 Accuracy / AUROC / F1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载身份识别模型
    print("加载身份识别模型...")
    identity_model = IdentifyDetectionSystem(num_classes=num_known_users, feature_dim=128, projection_dim=32).to(device)

    ckpt_path = os.path.join(PROJECT_ROOT, "Research2", "identify", "best_identify_model.pth")
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

    train_loader = data_loaders["intruder_train"]
    test_loader = data_loaders["intruder_test"]

    # 训练 Traditional OpenMax
    print(f"开始拟合 Traditional OpenMax (num_known_users={num_known_users}, alpha={alpha})...")
    tom = train_traditional_openmax(identity_model, train_loader, device, num_known_users=num_known_users, alpha=alpha)

    # 测试集评估
    print("在测试集上评估 Traditional OpenMax 基线...")
    accuracy, auroc, f1 = evaluate_traditional_openmax(identity_model, tom, test_loader, device)

    print("Traditional OpenMax 基线在测试集上的性能：")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  AUROC           : {auroc:.4f}")
    print(f"  F1-Score        : {f1:.4f}")

    # 保存评估结果（仅三项指标）
    results = {
        "method": "Traditional OpenMax (TOM)",
        "num_known_users": int(num_known_users),
        "alpha": int(alpha),
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "f1_score": float(f1),
    }

    # 创建TraditionalOpenMax子文件夹用于保存结果
    output_dir = os.path.join(THIS_DIR, "TraditionalOpenMax")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "intruder_detection_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"评估结果已保存到: {output_path}")


if __name__ == "__main__":
    main(num_known_users=10, alpha=3)
