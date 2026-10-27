#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from build_sensing_dataset import force_hssb_to_n_240_4, read_matlab_dataset, assign_fold


EPS = 1e-6


def amp_db(h):
    return (20.0 * np.log10(np.abs(h) + EPS)).astype(np.float32)


def clean_complex(h):
    h = np.asarray(h, dtype=np.complex64)
    real = np.nan_to_num(h.real, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    imag = np.nan_to_num(h.imag, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)


def build_empty_baselines(session_dirs):
    empty_dirs = []
    for sdir in session_dirs:
        meta_path = sdir / "metadata.json"
        mat_path = sdir / "session_data.mat"
        if not meta_path.exists() or not mat_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("label") == "empty":
            empty_dirs.append(sdir)

    empty_dirs = sorted(empty_dirs, key=lambda p: p.name)
    empty_index_by_session = {sdir.name: i for i, sdir in enumerate(empty_dirs)}
    baselines = {}

    for sdir in empty_dirs:
        meta = json.loads((sdir / "metadata.json").read_text())
        fold_id = empty_index_by_session[sdir.name] + 1
        h = force_hssb_to_n_240_4(read_matlab_dataset(sdir / "session_data.mat", "hSSB"))
        A = amp_db(clean_complex(h))
        mu = np.median(A, axis=0).astype(np.float32)
        q25 = np.percentile(A, 25, axis=0)
        q75 = np.percentile(A, 75, axis=0)
        sigma = np.maximum(((q75 - q25) / 1.349).astype(np.float32), 0.5)
        baselines[fold_id] = {
            "mu": mu,
            "sigma": sigma,
            "session_id": meta["session_id"],
        }

    return baselines, empty_index_by_session


def make_power_image(h_win, empty_mu, empty_sigma, representation, drop_symbol0):
    h_win = clean_complex(h_win)
    A = amp_db(h_win)

    if drop_symbol0:
        A = A[:, :, 1:]
        empty_mu = empty_mu[:, 1:]
        empty_sigma = empty_sigma[:, 1:]

    delta = A - empty_mu[None, :, :]
    zdelta = delta / empty_sigma[None, :, :]
    attenuation = np.maximum(-delta, 0.0)

    # Convert window x subcarrier x symbol into image subcarrier x time-symbol.
    def flatten_time(x):
        return np.transpose(x, (1, 0, 2)).reshape(x.shape[1], x.shape[0] * x.shape[2])

    A_shape = A - np.median(A, axis=(1, 2), keepdims=True)
    delta_shape = delta - np.median(delta, axis=(1, 2), keepdims=True)

    if representation == "power":
        channels = [flatten_time(A)]
    elif representation == "power_shape":
        channels = [flatten_time(A_shape)]
    elif representation == "delta":
        channels = [flatten_time(delta)]
    elif representation == "delta_shape":
        channels = [flatten_time(delta_shape)]
    elif representation == "zdelta":
        channels = [flatten_time(zdelta)]
    elif representation == "attenuation":
        channels = [flatten_time(attenuation)]
    elif representation == "paper_power_delta":
        channels = [flatten_time(A), flatten_time(delta)]
    elif representation == "full":
        temporal = np.abs(np.diff(A, axis=0))
        temporal = np.pad(temporal, ((0, 1), (0, 0), (0, 0)), mode="constant")
        channels = [
            flatten_time(A),
            flatten_time(delta),
            flatten_time(zdelta),
            flatten_time(attenuation),
            flatten_time(temporal),
        ]
    else:
        raise ValueError(f"Unknown representation: {representation}")

    img = np.stack(channels, axis=0).astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return img


def build_dataset(raw_dir, labels, window, stride, representation, drop_symbol0):
    raw_dir = Path(raw_dir)
    session_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    baselines, empty_index_by_session = build_empty_baselines(session_dirs)

    X, rows = [], []
    for sdir in tqdm(session_dirs, desc="Building windows"):
        meta_path = sdir / "metadata.json"
        mat_path = sdir / "session_data.mat"
        if not meta_path.exists() or not mat_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        label = meta["label"]
        if label not in labels:
            continue

        orientation = meta.get("orientation", "unknown")
        empty_idx = empty_index_by_session.get(sdir.name)
        fold_id = assign_fold(label, orientation, empty_idx)
        if fold_id not in baselines:
            continue

        h = force_hssb_to_n_240_4(read_matlab_dataset(mat_path, "hSSB"))
        if h.shape[0] < window:
            continue

        for start in range(0, h.shape[0] - window + 1, stride):
            end = start + window
            X.append(
                make_power_image(
                    h[start:end],
                    baselines[fold_id]["mu"],
                    baselines[fold_id]["sigma"],
                    representation,
                    drop_symbol0,
                )
            )
            rows.append(
                {
                    "session_dir": sdir.name,
                    "session_id": meta["session_id"],
                    "label": label,
                    "orientation": orientation,
                    "fold_id": fold_id,
                    "window_start": start,
                    "window_end": end - 1,
                }
            )

    return np.stack(X).astype(np.float32), pd.DataFrame(rows)


class CSIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LeNet5G(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(24, 48, kernel_size=5, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(96 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def standardize_from_train(X_train, X_other):
    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std = X_train.std(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(std, 1e-3)
    return (X_train - mean) / std, (X_other - mean) / std, mean, std


def train_model(X_train, y_train, X_val, y_val, num_classes, device, args):
    train_ds = CSIDataset(X_train, y_train)
    val_ds = CSIDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LeNet5G(X_train.shape[1], num_classes).to(device)
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state, best_score, best_epoch = None, -1.0, 0
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        model.eval()
        pred, true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                pred.append(torch.argmax(logits, dim=1).cpu().numpy())
                true.append(yb.numpy())

        y_true = np.concatenate(true)
        y_pred = np.concatenate(pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_accuracy": float(acc),
                "val_balanced_accuracy": float(bacc),
                "val_macro_f1": float(mf1),
            }
        )
        print(
            f"epoch {epoch:03d} loss={np.mean(losses):.4f} "
            f"val_acc={acc:.4f} val_bacc={bacc:.4f} val_f1={mf1:.4f}",
            flush=True,
        )

        if bacc > best_score:
            best_score = bacc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


def predict(model, X, device, batch_size):
    loader = DataLoader(CSIDataset(X, np.zeros(X.shape[0], dtype=np.int64)), batch_size=batch_size)
    pred = []
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _ in loader:
            pred.append(torch.argmax(model(xb.to(device)), dim=1).cpu().numpy())
    elapsed = time.perf_counter() - t0
    return np.concatenate(pred), elapsed


def plot_cm(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted",
        ylabel="True",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--labels", nargs="+", default=["empty", "P1", "P2", "P3", "P4"])
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument(
        "--representation",
        choices=[
            "power",
            "power_shape",
            "delta",
            "delta_shape",
            "zdelta",
            "attenuation",
            "paper_power_delta",
            "full",
        ],
        default="paper_power_delta",
    )
    parser.add_argument("--keep-symbol0", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, metadata = build_dataset(
        args.raw_dir,
        args.labels,
        args.window,
        args.stride,
        args.representation,
        drop_symbol0=not args.keep_symbol0,
    )
    y_str = metadata["label"].to_numpy()
    folds = metadata["fold_id"].to_numpy(dtype=int)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)
    classes = list(encoder.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Dataset", flush=True)
    print(f"X={X.shape}", flush=True)
    print(pd.Series(y_str).value_counts(), flush=True)
    print(pd.Series(folds).value_counts().sort_index(), flush=True)
    print(f"classes={classes}", flush=True)
    print(f"device={device}", flush=True)

    metadata.to_csv(out_dir / "metadata.csv", index=False)
    np.savez_compressed(out_dir / "dataset_cache.npz", X=X, y=y_str, folds=folds)

    all_true, all_pred, fold_metrics = [], [], []
    for test_fold in sorted(np.unique(folds)):
        print(f"\nFold {test_fold}", flush=True)
        trainval_mask = folds != test_fold
        test_mask = folds == test_fold
        idx = np.arange(trainval_mask.sum())
        train_idx, val_idx = train_test_split(
            idx,
            test_size=0.15,
            random_state=args.seed,
            stratify=y[trainval_mask],
        )

        X_trainval = X[trainval_mask]
        y_trainval = y[trainval_mask]
        X_train = X_trainval[train_idx]
        y_train = y_trainval[train_idx]
        X_val = X_trainval[val_idx]
        y_val = y_trainval[val_idx]
        X_test = X[test_mask]
        y_test = y[test_mask]

        X_train, X_val, mean, std = standardize_from_train(X_train, X_val)
        X_test = (X_test - mean) / std

        model, history, best_epoch = train_model(
            X_train, y_train, X_val, y_val, len(classes), device, args
        )
        pred, pred_time_s = predict(model, X_test.astype(np.float32), device, args.batch_size)

        acc = accuracy_score(y_test, pred)
        bacc = balanced_accuracy_score(y_test, pred)
        mf1 = f1_score(y_test, pred, average="macro")
        print(f"fold={test_fold} acc={acc:.4f} bacc={bacc:.4f} f1={mf1:.4f}", flush=True)

        fold_dir = out_dir / f"fold_{test_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(fold_dir / "history.csv", index=False)
        torch.save(
            {
                "model_state_dict": model.to("cpu").state_dict(),
                "classes": classes,
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
                "args": vars(args),
                "best_epoch": int(best_epoch),
            },
            fold_dir / "model.pt",
        )
        model.to(device)

        fold_metrics.append(
            {
                "fold": int(test_fold),
                "accuracy": float(acc),
                "balanced_accuracy": float(bacc),
                "macro_f1": float(mf1),
                "best_epoch": int(best_epoch),
                "predict_ms_per_window": float(1000.0 * pred_time_s / max(1, X_test.shape[0])),
            }
        )
        all_true.append(y_test)
        all_pred.append(pred)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classes": classes,
        "fold_metrics": fold_metrics,
        "confusion_matrix": cm.tolist(),
        "args": vars(args),
    }

    print("\nSummary", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(report, flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "classification_report.txt").write_text(report)
    plot_cm(cm, classes, out_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
