import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


class CROSRReconstructionNet(nn.Module):
    """用于 CROSR 基线的简单重构网络（自编码器）。

    输入：身份识别模型提取的 128 维特征
    目标：仅在合法用户特征上训练，最小化重构误差；以重构误差作为开放集检测得分
    """

    def __init__(self, feature_dim: int = 128):
        super(CROSRReconstructionNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


def train_crosr_autoencoder(identity_model, train_loader, device, feature_dim: int = 128, num_epochs: int = 50, lr: float = 1e-3):
    """在合法用户训练样本上训练 CROSR 自编码器。"""
    identity_model.eval()
    ae = CROSRReconstructionNet(feature_dim=feature_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)

    for epoch in range(num_epochs):
        ae.train()
        total_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            data, labels, _ = batch
            data = data.to(device)
            labels = labels.to(device)

            # 只用合法用户样本 (label=0) 训练自编码器
            with torch.no_grad():
                outputs = identity_model(data)
                features = outputs["features"]  # [B, feature_dim]

            legal_mask = labels == 0
            if legal_mask.sum() == 0:
                continue

            legal_features = features[legal_mask]

            recon = ae(legal_features)
            loss = criterion(recon, legal_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            avg_loss = total_loss / batch_count
        else:
            avg_loss = 0.0

        print(f"[CROSR AE] Epoch {epoch + 1}/{num_epochs}, 训练重构损失: {avg_loss:.6f}")

    return ae


def compute_reconstruction_errors(identity_model, ae_model, data_loader, device):
    """计算给定数据集上每个样本的重构误差及对应标签。"""
    identity_model.eval()
    ae_model.eval()

    all_errors = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            # 数据格式：data, label, identity_label
            if len(batch) == 3:
                data, labels, _ = batch
            else:
                data, labels = batch

            data = data.to(device)
            labels = labels.to(device)

            outputs = identity_model(data)
            features = outputs["features"]  # [B, feature_dim]

            recon = ae_model(features)
            # 逐样本 MSE 作为重构误差
            errors = torch.mean((recon - features) ** 2, dim=1)  # [B]

            all_errors.extend(errors.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if len(all_errors) == 0:
        return np.array([]), np.array([])

    return np.array(all_errors), np.array(all_labels)


def evaluate_crosr(identity_model, ae_model, train_loader, test_loader, device, threshold_percentile: float = 95.0):
    """基于重构误差的 CROSR 开放集检测评估。"""
    # 1) 使用训练集合法用户的重构误差确定阈值
    train_errors, train_labels = compute_reconstruction_errors(identity_model, ae_model, train_loader, device)
    if train_errors.size == 0:
        return 0.0, 0.0, 0.0

    legal_errors = train_errors[train_labels == 0]
    if legal_errors.size == 0:
        return 0.0, 0.0, 0.0

    threshold = float(np.percentile(legal_errors, threshold_percentile))
    print(f"CROSR 基线使用重构误差阈值: {threshold:.6f} (合法用户 {threshold_percentile} 百分位)")

    # 2) 在测试集上计算重构误差并根据阈值判决
    test_errors, test_labels = compute_reconstruction_errors(identity_model, ae_model, test_loader, device)
    if test_errors.size == 0 or test_labels.size == 0:
        return 0.0, 0.0, 0.0

    # 预测规则：重构误差 > 阈值 判为入侵者
    preds = (test_errors > threshold).astype(int)  # 1: 入侵者, 0: 合法用户
    y_true = test_labels.astype(int)

    accuracy = float((preds == y_true).mean()) if len(y_true) > 0 else 0.0
    f1 = float(f1_score(y_true, preds, zero_division="warn"))

    try:
        auroc = float(roc_auc_score(y_true, test_errors))
    except ValueError:
        auroc = 0.0

    return accuracy, auroc, f1
