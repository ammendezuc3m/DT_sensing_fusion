#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


def load_fixedscale_dataset(path: Path):
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(str)
    folds = data["folds"].astype(int)

    if X.ndim != 4:
        raise ValueError(f"Expected X with shape N x C x K x T, got {X.shape}")

    return X, y, folds


def summarize_axis(values, axis):
    return np.concatenate(
        [
            np.mean(values, axis=axis),
            np.std(values, axis=axis),
            np.min(values, axis=axis),
            np.max(values, axis=axis),
            np.percentile(values, 25, axis=axis),
            np.percentile(values, 75, axis=axis),
        ],
        axis=-1,
    )


def extract_physical_features(X):
    """
    Convert fixed-scale SSB images into robust tabular descriptors.

    Input:
      X: N x C x 240 x 20

    The model receives:
      - global channel statistics
      - per-subcarrier profiles
      - per-time-column profiles
      - coarse frequency bands
      - first-order gradients over frequency and time

    This keeps the information visible in the plots while reducing the chance
    of memorizing individual adjacent windows.
    """
    X = np.asarray(X, dtype=np.float32)
    n, channels, subcarriers, time_cols = X.shape

    features = []

    global_stats = summarize_axis(X.reshape(n, channels, -1), axis=2)
    features.append(global_stats.reshape(n, -1))

    per_subcarrier = summarize_axis(X, axis=3)
    features.append(per_subcarrier.reshape(n, -1))

    per_time = summarize_axis(np.transpose(X, (0, 1, 3, 2)), axis=3)
    features.append(per_time.reshape(n, -1))

    band_count = 12
    band_width = subcarriers // band_count
    bands = X[:, :, : band_count * band_width, :].reshape(
        n, channels, band_count, band_width, time_cols
    )
    band_stats = summarize_axis(bands.reshape(n, channels, band_count, -1), axis=3)
    features.append(band_stats.reshape(n, -1))

    grad_freq = np.abs(np.diff(X, axis=2))
    grad_time = np.abs(np.diff(X, axis=3))
    features.append(summarize_axis(grad_freq.reshape(n, channels, -1), axis=2).reshape(n, -1))
    features.append(summarize_axis(grad_time.reshape(n, channels, -1), axis=2).reshape(n, -1))

    out = np.concatenate(features, axis=1).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def make_models(seed: int):
    return {
        "pca_rbf_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=160, random_state=seed)),
                (
                    "clf",
                    SVC(
                        C=8.0,
                        gamma="scale",
                        kernel="rbf",
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "pca_logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=200, random_state=seed)),
                (
                    "clf",
                    LogisticRegression(
                        C=1.5,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=250,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def cap_pca_components(model, n_train, n_features):
    if not isinstance(model, Pipeline) or "pca" not in model.named_steps:
        return model

    model = clone(model)
    pca = model.named_steps["pca"]
    if isinstance(pca.n_components, int):
        pca.n_components = min(pca.n_components, n_train - 1, n_features)
    return model


def plot_confusion_matrix(cm, classes, out_path, title):
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
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def evaluate_model(name, estimator, X, y, folds, classes, out_dir):
    all_true = []
    all_pred = []
    fold_metrics = []

    for test_fold in sorted(np.unique(folds)):
        train_idx = folds != test_fold
        test_idx = folds == test_fold

        model = cap_pca_components(
            clone(estimator),
            n_train=int(train_idx.sum()),
            n_features=X.shape[1],
        )

        t0 = time.perf_counter()
        model.fit(X[train_idx], y[train_idx])
        train_time_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = model.predict(X[test_idx])
        predict_time_s = time.perf_counter() - t0

        truth = y[test_idx]
        acc = accuracy_score(truth, pred)
        bacc = balanced_accuracy_score(truth, pred)
        macro_f1 = f1_score(truth, pred, average="macro")

        print(
            f"{name} | fold={test_fold} | acc={acc:.4f} | "
            f"bacc={bacc:.4f} | f1={macro_f1:.4f}",
            flush=True,
        )

        fold_metrics.append(
            {
                "fold": int(test_fold),
                "accuracy": float(acc),
                "balanced_accuracy": float(bacc),
                "macro_f1": float(macro_f1),
                "train_time_s": float(train_time_s),
                "predict_ms_per_window": float(
                    1000.0 * predict_time_s / max(1, int(test_idx.sum()))
                ),
            }
        )
        all_true.append(truth)
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
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classes": list(classes),
        "fold_metrics": fold_metrics,
        "confusion_matrix": cm.tolist(),
    }

    model_dir = out_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "classification_report.txt").write_text(report)
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(cm, classes, model_dir / "confusion_matrix.png", name)

    return metrics


def train_final_model(name, estimator, X, y, classes, out_dir):
    final_dir = out_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    model = cap_pca_components(clone(estimator), n_train=X.shape[0], n_features=X.shape[1])
    t0 = time.perf_counter()
    model.fit(X, y)
    train_time_s = time.perf_counter() - t0

    artifact = {
        "model_name": name,
        "model": model,
        "classes": list(classes),
        "feature_extractor": "extract_physical_features",
        "input_shape": [4, 240, 20],
        "feature_dim": int(X.shape[1]),
    }
    joblib.dump(artifact, final_dir / "model.joblib")

    info = {
        "selected_model": name,
        "train_time_s": float(train_time_s),
        "classes": list(classes),
        "feature_dim": int(X.shape[1]),
    }
    (final_dir / "final_model_info.json").write_text(json.dumps(info, indent=2))
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/ssb_fixedscale_v4_multiclass_all.npz")
    parser.add_argument("--out-dir", default="models/ssb_fixedscale_multiclass")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-slow", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of models to run. Options: pca_rbf_svm pca_logreg extra_trees random_forest",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "features_cache.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        X = cached["X"].astype(np.float32)
        y_str = cached["y"].astype(str)
        folds = cached["folds"].astype(int)
        X_img_shape = tuple(cached["image_shape"].astype(int)) if "image_shape" in cached else None
    else:
        X_img, y_str, folds = load_fixedscale_dataset(Path(args.dataset))
        X_img_shape = X_img.shape
        X = extract_physical_features(X_img)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)
    classes = list(encoder.classes_)

    print("Dataset", flush=True)
    print(f"Images: {X_img_shape}", flush=True)
    print(f"Features: {X.shape}", flush=True)
    print(pd.Series(y_str).value_counts(), flush=True)
    print(pd.Series(folds).value_counts().sort_index(), flush=True)
    print("Classes:", classes, flush=True)

    if not cache_path.exists():
        np.savez_compressed(
            cache_path,
            X=X,
            y=y_str,
            folds=folds,
            classes=np.array(classes),
            image_shape=np.array(X_img_shape, dtype=np.int32),
        )

    models = make_models(args.seed)
    if args.skip_slow:
        models.pop("pca_rbf_svm", None)
    if args.models is not None:
        unknown = sorted(set(args.models) - set(models))
        if unknown:
            raise ValueError(f"Unknown model names: {unknown}")
        models = {name: models[name] for name in args.models}

    metrics = []
    for name, estimator in models.items():
        print(f"\nEvaluating {name}", flush=True)
        metrics.append(evaluate_model(name, estimator, X, y, folds, classes, out_dir))

    summary = pd.DataFrame(
        [
            {
                "model": item["model"],
                "accuracy": item["accuracy"],
                "balanced_accuracy": item["balanced_accuracy"],
                "macro_f1": item["macro_f1"],
            }
            for item in metrics
        ]
    ).sort_values("balanced_accuracy", ascending=False)

    summary.to_csv(out_dir / "model_comparison.csv", index=False)
    (out_dir / "model_comparison.json").write_text(json.dumps(metrics, indent=2))
    print("\nModel comparison")
    print(summary)

    best_name = str(summary.iloc[0]["model"])
    final_info = train_final_model(best_name, models[best_name], X, y, classes, out_dir)
    joblib.dump(encoder, out_dir / "label_encoder.joblib")

    print("\nFinal model")
    print(json.dumps(final_info, indent=2))


if __name__ == "__main__":
    main()
