#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


class CSIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SmallCSICNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Reducimos sobre subportadoras, conservando bastante el eje temporal.
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Dropout(0.20),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(str)
    folds = data["folds"].astype(int)
    groups = data["groups"].astype(str)
    return X, y, folds, groups


def plot_cm(cm, classes, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def train_one_model(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    device,
    epochs,
    batch_size,
    lr,
    patience,
):
    train_ds = CSIDataset(X_train, y_train)
    val_ds = CSIDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = SmallCSICNN(
        in_channels=X_train.shape[1],
        num_classes=num_classes,
    ).to(device)

    # Pesos por clase, por si hay pequeño desbalance.
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = -1.0
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        all_true = []
        all_pred = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()

                all_pred.append(pred)
                all_true.append(yb.numpy())

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)

        val_acc = accuracy_score(y_true, y_pred)
        val_bacc = balanced_accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="macro")

        mean_loss = float(np.mean(train_losses))

        history.append({
            "epoch": epoch,
            "train_loss": mean_loss,
            "val_accuracy": float(val_acc),
            "val_balanced_accuracy": float(val_bacc),
            "val_macro_f1": float(val_f1),
        })

        print(
            f"epoch {epoch:03d} | loss={mean_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_bacc={val_bacc:.4f} | val_f1={val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping en epoch {epoch}. Mejor epoch: {best_epoch}")
            break

    model.load_state_dict(best_state)

    return model, history, best_epoch


def predict_model(model, X, device, batch_size):
    ds = CSIDataset(X, np.zeros(X.shape[0], dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    preds = []

    t0 = time.perf_counter()

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)

    elapsed = time.perf_counter() - t0

    return np.concatenate(preds), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/ssb_csi_images_v3_empty_P1_P4.npz")
    parser.add_argument("--out-dir", default="models/ssb_cnn_v3_empty_P1_P4")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y_str, folds, groups = load_npz(args.dataset)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)
    classes = list(encoder.classes_)

    print("\n=== Dataset ===")
    print(f"X: {X.shape}")
    print("Classes:", classes)
    print("Labels:")
    print(pd.Series(y_str).value_counts())
    print("Folds:")
    print(pd.Series(folds).value_counts().sort_index())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    unique_folds = sorted([f for f in np.unique(folds) if f > 0])

    all_true = []
    all_pred = []
    fold_metrics = []

    for test_fold in unique_folds:
        print(f"\n=== Fold {test_fold}: test fold {test_fold} ===")

        trainval_idx = folds != test_fold
        test_idx = folds == test_fold

        X_trainval = X[trainval_idx]
        y_trainval = y[trainval_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        # Validación interna solo para early stopping.
        idx_all = np.arange(X_trainval.shape[0])
        train_idx, val_idx = train_test_split(
            idx_all,
            test_size=0.15,
            random_state=args.seed,
            stratify=y_trainval,
        )

        X_train = X_trainval[train_idx]
        y_train = y_trainval[train_idx]
        X_val = X_trainval[val_idx]
        y_val = y_trainval[val_idx]

        print(f"train={X_train.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]}")

        model, history, best_epoch = train_one_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_classes=len(classes),
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )

        y_pred, pred_time_s = predict_model(
            model,
            X_test,
            device=device,
            batch_size=args.batch_size,
        )

        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        mf1 = f1_score(y_test, y_pred, average="macro")

        print(
            f"Fold {test_fold} RESULT | "
            f"acc={acc:.4f} | bacc={bacc:.4f} | f1={mf1:.4f} | "
            f"pred={1000.0 * pred_time_s / X_test.shape[0]:.4f} ms/window"
        )

        fold_dir = out_dir / f"fold_{test_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

        fold_metrics.append({
            "fold": int(test_fold),
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "macro_f1": float(mf1),
            "best_epoch": int(best_epoch),
            "predict_ms_per_window": float(1000.0 * pred_time_s / X_test.shape[0]),
        })

        all_true.append(y_test)
        all_pred.append(y_pred)

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)

    acc = accuracy_score(y_true_all, y_pred_all)
    bacc = balanced_accuracy_score(y_true_all, y_pred_all)
    mf1 = f1_score(y_true_all, y_pred_all, average="macro")

    cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(len(classes)))
    report = classification_report(
        y_true_all,
        y_pred_all,
        labels=np.arange(len(classes)),
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    print("\n=== CNN CV SUMMARY ===")
    print(f"accuracy={acc:.4f}")
    print(f"balanced_accuracy={bacc:.4f}")
    print(f"macro_f1={mf1:.4f}")
    print(report)

    (out_dir / "classification_report.txt").write_text(report)
    plot_cm(cm, classes, out_dir / "confusion_matrix.png", "CNN CSI v3 confusion matrix")

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_f1": float(mf1),
        "classes": classes,
        "fold_metrics": fold_metrics,
        "confusion_matrix": cm.tolist(),
        "dataset": args.dataset,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Entrenamiento final usando todo el dataset.
    print("\n=== Entrenando modelo final con todo el dataset ===")

    idx_all = np.arange(X.shape[0])
    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=0.10,
        random_state=args.seed,
        stratify=y,
    )

    final_model, final_history, best_epoch = train_one_model(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_val=X[val_idx],
        y_val=y[val_idx],
        num_classes=len(classes),
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    final_model_cpu = final_model.to("cpu")

    artifact = {
        "model_state_dict": final_model_cpu.state_dict(),
        "classes": classes,
        "image_shape": list(X.shape[1:]),
        "model_type": "SmallCSICNN",
        "best_epoch": int(best_epoch),
    }

    torch.save(artifact, out_dir / "cnn_model.pt")

    joblib.dump(
        {
            "classes": classes,
            "label_encoder": encoder,
            "image_shape": list(X.shape[1:]),
            "model_type": "SmallCSICNN",
        },
        out_dir / "metadata.joblib",
    )

    pd.DataFrame(final_history).to_csv(out_dir / "final_history.csv", index=False)

    print(f"\nModelo final guardado en: {out_dir / 'cnn_model.pt'}")


if __name__ == "__main__":
    main()
