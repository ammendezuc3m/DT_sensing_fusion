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
from sklearn.svm import LinearSVC, SVC


def load_dataset(npz_path: Path, metadata_path: Path):
    data = np.load(npz_path, allow_pickle=True)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(str)
    folds = data["folds"].astype(int)

    metadata = pd.read_csv(metadata_path)

    return X, y, folds, metadata


def make_models(random_state: int = 42):
    models = {}

    models["pca_linear_svm"] = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=120, random_state=random_state)),
        ("clf", LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=30000,
            random_state=random_state,
        )),
    ])

    models["pca_logreg"] = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=120, random_state=random_state)),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )),
    ])

    models["pca_rbf_svm"] = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=80, random_state=random_state)),
        ("clf", SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=False,
            random_state=random_state,
        )),
    ])

    return models


def safe_pca_components(model, n_train, n_features):
    if "pca" not in model.named_steps:
        return model

    pca = model.named_steps["pca"]

    if isinstance(pca.n_components, int):
        max_comp = max(1, min(n_train - 1, n_features))
        if pca.n_components > max_comp:
            model = clone(model)
            model.named_steps["pca"].n_components = max_comp

    return model


def plot_confusion_matrix(cm, classes, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def evaluate_model(name, pipeline, X, y_enc, folds, classes, out_dir):
    unique_folds = sorted([f for f in np.unique(folds) if f > 0])

    all_true = []
    all_pred = []
    fold_results = []

    for fold in unique_folds:
        train_idx = folds != fold
        test_idx = folds == fold

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y_enc[train_idx]
        y_test = y_enc[test_idx]

        model = clone(pipeline)
        model = safe_pca_components(
            model,
            n_train=X_train.shape[0],
            n_features=X_train.shape[1],
        )

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time_s = time.perf_counter() - t0

        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        fold_results.append({
            "fold": int(fold),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "macro_f1": float(macro_f1),
            "train_time_s": float(train_time_s),
            "predict_time_s": float(predict_time_s),
            "predict_ms_per_window": float(1000.0 * predict_time_s / max(1, X_test.shape[0])),
        })

        all_true.append(y_test)
        all_pred.append(y_pred)

        print(
            f"{name} | fold={fold} | "
            f"acc={acc:.4f} | bacc={bacc:.4f} | f1={macro_f1:.4f} | "
            f"pred={1000.0 * predict_time_s / max(1, X_test.shape[0]):.4f} ms/window"
        )

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)

    acc = accuracy_score(y_true_all, y_pred_all)
    bacc = balanced_accuracy_score(y_true_all, y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro")

    cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(len(classes)))

    report = classification_report(
        y_true_all,
        y_pred_all,
        labels=np.arange(len(classes)),
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    model_dir = out_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        cm,
        classes,
        title=f"{name} confusion matrix",
        out_path=model_dir / "confusion_matrix.png",
    )

    (model_dir / "classification_report.txt").write_text(report)

    metrics = {
        "model": name,
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_f1": float(macro_f1),
        "fold_results": fold_results,
        "classes": list(classes),
        "confusion_matrix": cm.tolist(),
    }

    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def train_final_model(name, pipeline, X, y_enc, classes, out_dir):
    final_dir = out_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    model = clone(pipeline)
    model = safe_pca_components(
        model,
        n_train=X.shape[0],
        n_features=X.shape[1],
    )

    t0 = time.perf_counter()
    model.fit(X, y_enc)
    train_time_s = time.perf_counter() - t0

    n_test = min(1000, X.shape[0])
    X_latency = X[:n_test]

    t0 = time.perf_counter()
    _ = model.predict(X_latency)
    predict_time_s = time.perf_counter() - t0

    artifact = {
        "model_name": name,
        "pipeline": model,
        "classes": list(classes),
        "window_size": 5,
        "input_feature_dim": int(X.shape[1]),
        "notes": (
            "Modelo para clasificacion 5G SSB sensing. "
            "Entrada esperada: vector de features generado desde ventanas hSSB de 5 capturas."
        ),
    }

    joblib.dump(artifact, final_dir / "model.joblib")

    final_info = {
        "selected_model": name,
        "train_time_s": float(train_time_s),
        "latency_test_windows": int(n_test),
        "predict_time_s": float(predict_time_s),
        "predict_ms_per_window": float(1000.0 * predict_time_s / max(1, n_test)),
        "classes": list(classes),
        "input_feature_dim": int(X.shape[1]),
    }

    (final_dir / "final_model_info.json").write_text(json.dumps(final_info, indent=2))

    print("\n=== Modelo final guardado ===")
    print(f"Modelo: {name}")
    print(f"Ruta: {final_dir / 'model.joblib'}")
    print(f"Latencia estimada: {final_info['predict_ms_per_window']:.4f} ms/window")

    return final_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/ssb_windows_v001.npz")
    parser.add_argument("--metadata", default="data/processed/ssb_windows_metadata_v001.csv")
    parser.add_argument("--out-dir", default="models/ssb_v001")
    parser.add_argument("--skip-rbf", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, folds, metadata = load_dataset(Path(args.dataset), Path(args.metadata))

    print("\n=== Dataset cargado ===")
    print(f"X: {X.shape}")
    print("Labels:")
    print(pd.Series(y).value_counts())
    print("Folds:")
    print(pd.Series(folds).value_counts().sort_index())

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    classes = list(encoder.classes_)

    print("\nClases codificadas:")
    for i, c in enumerate(classes):
        print(f"  {i}: {c}")

    models = make_models()

    if args.skip_rbf:
        models.pop("pca_rbf_svm", None)

    all_metrics = []

    print("\n=== Validacion por folds ===")

    for name, model in models.items():
        print(f"\n--- Evaluando {name} ---")
        metrics = evaluate_model(
            name=name,
            pipeline=model,
            X=X,
            y_enc=y_enc,
            folds=folds,
            classes=classes,
            out_dir=out_dir,
        )
        all_metrics.append(metrics)

    summary = pd.DataFrame([
        {
            "model": m["model"],
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
        }
        for m in all_metrics
    ]).sort_values("balanced_accuracy", ascending=False)

    summary.to_csv(out_dir / "model_comparison.csv", index=False)

    print("\n=== Comparacion de modelos ===")
    print(summary)

    (out_dir / "model_comparison.json").write_text(
        json.dumps(all_metrics, indent=2)
    )

    best_name = summary.iloc[0]["model"]
    best_model = models[best_name]

    train_final_model(
        name=best_name,
        pipeline=best_model,
        X=X,
        y_enc=y_enc,
        classes=classes,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
