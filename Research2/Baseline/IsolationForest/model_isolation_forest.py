from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score

from Research2.train_intruder import extract_features


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
