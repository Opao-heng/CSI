import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from Research2.train_intruder import extract_features
from Research2.model_intruder import TraditionalOpenMax


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
