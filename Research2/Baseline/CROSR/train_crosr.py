import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from model_crosr import CROSRReconstructionNet
from sklearn.metrics import f1_score, roc_auc_score

# 将项目根目录加入 sys.path，便于导入 Research2 模块
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Research2.model_identify import IdentifyDetectionSystem
from Research2.Process.dataloader_intruder import load_intruder_data, create_intruder_data_loaders
from model_crosr import train_crosr_autoencoder, evaluate_crosr


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


def main(num_epochs: int = 50, threshold_percentile: float = 95.0):
    """CROSR (Classification-Reconstruction based OSR) 基线实验入口。

    - 使用身份识别模型提取 128 维特征
    - 在合法用户样本上训练特征自编码器，利用重构误差检测未知样本
    - 只输出 Accuracy / AUROC / F1 三个指标
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

    # 训练 CROSR 自编码器
    print(f"开始训练 CROSR 重构网络，共 {num_epochs} 个 epoch...")
    ae_model = train_crosr_autoencoder(identity_model, train_loader, device, feature_dim=128, num_epochs=num_epochs, lr=1e-3)

    # 评估 CROSR 基线
    print("在测试集上评估 CROSR 基线...")
    accuracy, auroc, f1 = evaluate_crosr(identity_model, ae_model, train_loader, test_loader, device, threshold_percentile=threshold_percentile)

    print("CROSR 基线在测试集上的性能：")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  AUROC           : {auroc:.4f}")
    print(f"  F1-Score        : {f1:.4f}")

    # 保存评估结果（仅三项指标）
    results = {
        "method": "CROSR (Classification-Reconstruction OSR)",
        "num_epochs": int(num_epochs),
        "threshold_percentile": float(threshold_percentile),
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "f1_score": float(f1),
    }

    # 创建CROSR子文件夹用于保存结果
    output_dir = os.path.join(THIS_DIR, "CROSR")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "intruder_detection_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"评估结果已保存到: {output_path}")


if __name__ == "__main__":
    main(num_epochs=50, threshold_percentile=95.0)
