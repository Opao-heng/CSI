import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def evaluate_softmax_threshold(identity_model, data_loader, device, threshold: float = 0.9):
    """在入侵者测试集上基于 Softmax 最大概率进行阈值检测。

    规则：max_softmax_prob < threshold 判定为入侵者 (label=1)，否则为合法用户 (label=0)。
    AUROC 使用 "入侵者概率" = 1 - max_softmax_prob 作为分数。
    """
    identity_model.eval()

    all_labels = []  # 真实标签：0-合法用户，1-入侵者
    all_preds = []   # 预测标签
    all_scores = []  # 入侵者打分（越大越可能是入侵者）

    with torch.no_grad():
        for batch in data_loader:
            # 兼容是否包含 identity_labels 的两种数据格式
            if len(batch) == 3:
                data, labels, _ = batch
            else:
                data, labels = batch

            data = data.to(device)
            labels = labels.to(device)

            outputs = identity_model(data)
            logits = outputs["logits"]  # [B, num_classes]
            probs = torch.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)  # [B]

            # 入侵者概率 = 1 - 最大已知类概率
            intruder_scores = 1.0 - max_probs

            # 阈值判决：低置信度样本视为入侵者
            preds = (max_probs < threshold).long()  # 1: 入侵者, 0: 合法用户

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(intruder_scores.cpu().numpy())

    if len(all_labels) == 0:
        return 0.0, 0.0, 0.0

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_scores = np.array(all_scores)

    accuracy = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division="warn"))

    try:
        auroc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        # 当测试集中只有单一类别时，AUROC 不可计算
        auroc = 0.0

    return accuracy, auroc, f1
