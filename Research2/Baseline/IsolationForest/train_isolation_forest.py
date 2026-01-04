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
from model_isolation_forest import train_isolation_forest, evaluate_isolation_forest


def train_isolation_forest(identity_model, train_loader, device, contamination: float = 0.1):
    """使用合法用户样本特征训练 Isolation Forest。

    仅使用标签为 0 的样本（合法用户）进行无监督训练。
    """
    features, _, labels, _ = extract_features(identity_model, train_loader, device)

    if features.size == 0 or labels.size == 0:
        raise RuntimeError("训练集特征为空，无法训练 Isolation Forest。")

    # 只使用合法用户样本
    legal_mask = labels == 0
    legal_features = features[legal_mask]

    if legal_features.size == 0:
        raise RuntimeError("训练集中未找到合法用户样本，无法训练 Isolation Forest。")

    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(legal_features)
    return clf


def evaluate_isolation_forest(identity_model, iforest, data_loader, device):
    """在给定数据集上评估 Isolation Forest 基线性能。

    - 使用身份识别模型提取特征
    - 使用 Isolation Forest 进行异常检测（入侵者检测）
    - 预测规则：IF.predict == -1 视为入侵者 (label=1)，否则为合法用户 (label=0)
    - AUROC 使用 "异常得分" = -decision_function 作为分数
    """
    features, _, labels, _ = extract_features(identity_model, data_loader, device)

    if labels.size == 0:
        return 0.0, 0.0, 0.0

    # sklearn 的 decision_function 越大越正常，这里取负号作为“入侵者得分”
    raw_scores = iforest.decision_function(features)  # shape [N]
    anomaly_scores = -raw_scores

    pred_sign = iforest.predict(features)  # 1: inlier, -1: outlier
    preds = (pred_sign == -1).astype(int)  # 1: 入侵者, 0: 合法用户

    y_true = labels.astype(int)

    accuracy = float((preds == y_true).mean()) if len(y_true) > 0 else 0.0
    f1 = float(f1_score(y_true, preds, zero_division="warn"))

    try:
        auroc = float(roc_auc_score(y_true, anomaly_scores))
    except ValueError:
        auroc = 0.0

    return accuracy, auroc, f1


def main(contamination: float = 0.1):
    """Isolation Forest (IF) 基线实验入口。

    - 使用已训练好的身份识别模型提取特征
    - 仅用合法用户训练 Isolation Forest，无监督建模正常区域
    - 在入侵者测试集上评估 Accuracy / AUROC / F1
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

    train_loader = data_loaders["intruder_train"]
    test_loader = data_loaders["intruder_test"]

    # 训练 Isolation Forest
    print(f"开始训练 Isolation Forest (contamination = {contamination})...")
    iforest = train_isolation_forest(identity_model, train_loader, device, contamination=contamination)

    # 测试集评估
    print("在测试集上评估 Isolation Forest 基线...")
    accuracy, auroc, f1 = evaluate_isolation_forest(identity_model, iforest, test_loader, device)

    print("Isolation Forest 基线在测试集上的性能：")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  AUROC           : {auroc:.4f}")
    print(f"  F1-Score        : {f1:.4f}")

    # 保存评估结果（仅三项指标）
    results = {
        "method": "Isolation Forest (IF)",
        "contamination": float(contamination),
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "f1_score": float(f1),
    }

    # 创廯IsolationForest子文件夹用于保存结果
    output_dir = os.path.join(THIS_DIR, "IsolationForest")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "intruder_detection_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"评估结果已保存到: {output_path}")


if __name__ == "__main__":
    # contamination 近似设为 0.1，可按需要调整
    main(contamination=0.1)
