#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, TensorDataset


class RxGridSSBCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(7, 2), padding=(3, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(16, 32, kernel_size=(5, 2), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(32, 64, kernel_size=(3, 2), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),

            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mat_variable(path: Path, name: str):
    try:
        mat = scipy.io.loadmat(path)
        if name in mat:
            return mat[name]
    except NotImplementedError:
        pass
    except Exception:
        pass

    with h5py.File(path, "r") as f:
        if name not in f:
            raise KeyError(f"{name} not found in {path}")
        x = f[name][()]

    if hasattr(x, "dtype") and x.dtype.fields is not None:
        if "real" in x.dtype.fields and "imag" in x.dtype.fields:
            x = x["real"] + 1j * x["imag"]

    return np.asarray(x)


def normalize_datassb_shape(data):
    data = np.asarray(data)

    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D dataSSB, got shape {data.shape}")

    if data.shape[0] == 360 and data.shape[1] == 6:
        return data

    if data.shape[-1] == 360 and data.shape[-2] == 6:
        return np.transpose(data, (2, 1, 0))

    raise RuntimeError(f"Cannot infer dataSSB orientation from shape {data.shape}")


def load_valid_mask(path: Path, n: int):
    try:
        valid = load_mat_variable(path, "validMask").squeeze()
        valid = np.asarray(valid).astype(bool).reshape(-1)
        if len(valid) == n:
            return valid
    except Exception:
        pass

    return np.ones(n, dtype=bool)


def infer_block_key(path: Path, dataset_root: Path, label: str):
    rel = path.relative_to(dataset_root / label)
    parts = rel.parts
    if len(parts) >= 2:
        return f"{label}/{parts[0]}"
    return f"{label}/{path.stem}"


def split_from_block_key(block_key: str):
    # Mantiene la filosofía anterior: split por bloque, no por captura aleatoria.
    if "block_05" in block_key or "block_5" in block_key:
        return "test"
    if "block_04" in block_key or "block_4" in block_key:
        return "val"
    return "train"


def load_dataset(dataset_root: Path, labels):
    X_parts = []
    y_parts = []
    meta_rows = []

    label_to_id = {label: i for i, label in enumerate(labels)}

    for label in labels:
        label_dir = dataset_root / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")

        files = sorted(label_dir.rglob("*.mat"))
        if not files:
            raise FileNotFoundError(f"No .mat files found for label {label} in {label_dir}")

        print(f"\nLoading label {label}: {len(files)} files")

        for file_path in files:
            print(f"  {file_path}")

            data = load_mat_variable(file_path, "dataSSB")
            data = normalize_datassb_shape(data)

            n = data.shape[2]
            valid_mask = load_valid_mask(file_path, n)
            valid_idx = np.where(valid_mask)[0]

            if len(valid_idx) == 0:
                print("    WARNING: no valid samples")
                continue

            # rxGridSSB validado:
            # MATLAB: dataSSB(61:300, 2:5, :)
            # Python 0-based: dataSSB[60:300, 1:5, :]
            rx = data[60:300, 1:5, valid_idx]
            rx = np.transpose(rx, (2, 0, 1))  # [N, 240, 4]

            mag = np.abs(rx).astype(np.float32)
            phase = np.angle(rx).astype(np.float32)

            x = np.stack([mag, phase], axis=1).astype(np.float32)  # [N, 2, 240, 4]
            y = np.full((x.shape[0],), label_to_id[label], dtype=np.int64)

            block_key = infer_block_key(file_path, dataset_root, label)
            split = split_from_block_key(block_key)

            for local_i, capture_idx in enumerate(valid_idx):
                meta_rows.append({
                    "filePath": str(file_path),
                    "label": label,
                    "labelId": label_to_id[label],
                    "blockKey": block_key,
                    "split": split,
                    "captureIndex": int(capture_idx + 1),
                    "localIndex": int(local_i),
                })

            X_parts.append(x)
            y_parts.append(y)

            print(f"    valid samples: {x.shape[0]} | split={split} | blockKey={block_key}")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    meta = pd.DataFrame(meta_rows)

    return X, y, meta, label_to_id


def compute_normalization(X_train):
    mean = X_train.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = X_train.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_pred = []
    all_true = []

    for xb, yb in loader:
        xb = xb.to(device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device, dtype=torch.long, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * xb.size(0)
        pred = torch.argmax(logits, dim=1)

        all_pred.append(pred.detach().cpu().numpy())
        all_true.append(yb.detach().cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    acc = accuracy_score(all_true, all_pred)
    loss_avg = total_loss / len(all_true)

    return loss_avg, acc


def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_pred = []
    all_true = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, dtype=torch.float32, non_blocking=True)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)

            all_logits.append(logits.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(yb.numpy())

    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_logits)


def save_confusion_matrix(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="data/dataset_datassb/datassb_side_v1_6labels")
    parser.add_argument("--labels", nargs="+", default=["empty", "P5", "P3"])
    parser.add_argument("--out-dir", type=str, default="results/multiclass_empty_P5_P3_rx")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    dataset_root = Path(args.dataset)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    model_dir = out_dir / "model_rxGridSSB"
    model_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir:    {out_dir}")
    print(f"Labels:        {args.labels}")
    print(f"Device:        {device}")

    X, y, meta, label_to_id = load_dataset(dataset_root, args.labels)

    meta_path = model_dir / "metadata.csv"
    meta.to_csv(meta_path, index=False)

    print("\nDataset loaded:")
    print("X:", X.shape)
    print("y:", y.shape)
    print(meta.groupby(["label", "split"]).size())

    train_idx = meta.index[meta["split"] == "train"].to_numpy()
    val_idx = meta.index[meta["split"] == "val"].to_numpy()
    test_idx = meta.index[meta["split"] == "test"].to_numpy()

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Train/val/test split is empty. Check block_04/block_05 naming.")

    mean, std = compute_normalization(X[train_idx])

    X = ((X - mean) / (std + 1e-8)).astype(np.float32)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_loader = make_loader(X_train, y_train, args.batch_size, True)
    val_loader = make_loader(X_val, y_val, args.batch_size, False)
    test_loader = make_loader(X_test, y_test, args.batch_size, False)

    model = RxGridSSBCNN(num_classes=len(args.labels)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    best_epoch = -1
    bad_epochs = 0
    history = []

    best_path = model_dir / "model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, train=False)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": args.labels,
                "label_to_id": label_to_id,
                "input_shape": [2, 240, 4],
                "mean": mean,
                "std": std,
                "config": vars(args),
                "architecture": "RxGridSSBCNN_multiclass",
            }
            torch.save(checkpoint, best_path)
            print(f"  saved best model -> {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(model_dir / "train_history.csv", index=False)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred, logits = evaluate(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=args.labels,
        output_dict=True,
        zero_division=0,
    )
    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=args.labels,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(args.labels))))

    print("\nTEST RESULTS")
    print("=" * 80)
    print("accuracy:", acc)
    print(report_txt)
    print(cm)

    (model_dir / "classification_report.txt").write_text(report_txt)
    save_confusion_matrix(cm, args.labels, model_dir / "confusion_matrix.png")

    metrics = {
        "accuracy": acc,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "labels": args.labels,
        "label_to_id": label_to_id,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "input_shape": [2, 240, 4],
        "normalization": {
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
        },
    }

    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    dataset_summary = {
        "shape": list(X.shape),
        "n_samples": int(len(X)),
        "labels": {
            label: int((meta["label"] == label).sum())
            for label in args.labels
        },
        "splits": {
            split: int((meta["split"] == split).sum())
            for split in ["train", "val", "test"]
        },
    }

    (out_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    print("\nSaved:")
    print(model_dir)
    print(model_dir / "model.pt")
    print(model_dir / "metrics.json")
    print(model_dir / "confusion_matrix.png")
    print(model_dir / "metadata.csv")


if __name__ == "__main__":
    main()
