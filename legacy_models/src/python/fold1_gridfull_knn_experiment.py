#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from build_sensing_dataset import assign_fold, read_matlab_dataset


EPS = 1e-6
LABEL_ORDER = ["empty", "P1", "P2", "P3", "P4"]


def read_complex_dataset(mat_path: Path, name: str):
    data = read_matlab_dataset(mat_path, name)
    data = np.asarray(data)
    if data.dtype.fields is not None:
        raise ValueError(f"{name} was not converted to complex correctly")
    return np.nan_to_num(data.astype(np.complex64), nan=0.0, posinf=0.0, neginf=0.0)


def force_gridfull_to_n_360_6(grid):
    """
    MATLAB saves gridFull as N x 6 x 360 in these files.
    Return N x 360 x 6 so each capture looks like the MATLAB imagesc view.
    """
    grid = np.asarray(grid)
    if grid.ndim != 3:
        raise ValueError(f"Expected 3-D gridFull, got {grid.shape}")
    if 360 not in grid.shape or 6 not in grid.shape:
        raise ValueError(f"Expected dimensions 360 and 6 in gridFull, got {grid.shape}")
    ax_subcarrier = grid.shape.index(360)
    ax_symbol = grid.shape.index(6)
    ax_time = [i for i in range(3) if i not in (ax_subcarrier, ax_symbol)][0]
    return np.transpose(grid, (ax_time, ax_subcarrier, ax_symbol)).astype(np.complex64)


def discover_fold1_sessions(raw_dir: Path):
    session_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    empty_dirs = []
    for sdir in session_dirs:
        meta_path = sdir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("label") == "empty":
            empty_dirs.append(sdir)
    empty_index_by_session = {sdir.name: i for i, sdir in enumerate(sorted(empty_dirs))}

    fold1 = []
    for sdir in session_dirs:
        meta_path = sdir / "metadata.json"
        mat_path = sdir / "session_data.mat"
        if not meta_path.exists() or not mat_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        label = meta["label"]
        orientation = meta.get("orientation", "unknown")
        fold_id = assign_fold(label, orientation, empty_index_by_session.get(sdir.name))
        if fold_id == 1:
            fold1.append((sdir, meta, fold_id))
    return fold1


def make_windows(grid, window_size=2, stride=2):
    windows = []
    starts = []
    for start in range(0, grid.shape[0] - window_size + 1, stride):
        windows.append(grid[start : start + window_size])
        starts.append(start)
    return np.stack(windows).astype(np.complex64), np.array(starts, dtype=np.int32)


def build_dataset(raw_dir: Path, window_size: int, stride: int):
    rows = []
    X_complex = []

    for sdir, meta, fold_id in discover_fold1_sessions(raw_dir):
        grid = force_gridfull_to_n_360_6(read_complex_dataset(sdir / "session_data.mat", "gridFull"))
        windows, starts = make_windows(grid, window_size=window_size, stride=stride)
        for w, start in zip(windows, starts):
            X_complex.append(w)
            rows.append(
                {
                    "session_dir": sdir.name,
                    "session_id": meta["session_id"],
                    "label": meta["label"],
                    "orientation": meta.get("orientation", "unknown"),
                    "fold_id": fold_id,
                    "window_start": int(start),
                    "window_end": int(start + window_size - 1),
                }
            )

    X_complex = np.stack(X_complex).astype(np.complex64)
    metadata = pd.DataFrame(rows)
    return X_complex, metadata


def centered_unwrapped_phase(X):
    phase = np.unwrap(np.angle(X), axis=2).astype(np.float32)
    phase = phase - np.median(phase, axis=(2, 3), keepdims=True)
    return phase.astype(np.float32)


def differential_phase(X):
    return np.angle(X[:, :, 1:, :] * np.conj(X[:, :, :-1, :])).astype(np.float32)


def extract_features(X_complex, mode):
    amp_db = (20.0 * np.log10(np.abs(X_complex) + EPS)).astype(np.float32)
    amp_shape = amp_db - np.median(amp_db, axis=(2, 3), keepdims=True)
    phase = centered_unwrapped_phase(X_complex)
    phase_diff = differential_phase(X_complex)

    if mode == "amplitude":
        parts = [amp_db]
    elif mode == "amplitude_shape":
        parts = [amp_shape]
    elif mode == "complex_real_imag":
        parts = [X_complex.real.astype(np.float32), X_complex.imag.astype(np.float32)]
    elif mode == "amplitude_phase":
        parts = [amp_db, phase]
    elif mode == "amplitude_shape_phase_diff":
        # Pad phase diff from 359 to 360 subcarriers for consistent flattening.
        phase_diff_pad = np.pad(phase_diff, ((0, 0), (0, 0), (0, 1), (0, 0)), mode="constant")
        parts = [amp_shape, phase_diff_pad]
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    flat_parts = [p.reshape(p.shape[0], -1) for p in parts]
    X = np.concatenate(flat_parts, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def make_splits(metadata, y, random_state):
    random_train, random_test = train_test_split(
        np.arange(len(metadata)),
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    temporal_train = []
    temporal_test = []
    for _, group in metadata.groupby("session_id", sort=False):
        idx = group.index.to_numpy()
        split = int(np.floor(0.70 * len(idx)))
        split = max(1, min(split, len(idx) - 1))
        temporal_train.extend(idx[:split])
        temporal_test.extend(idx[split:])

    return {
        "random_stratified": (np.array(random_train), np.array(random_test)),
        "temporal_per_session": (np.array(temporal_train), np.array(temporal_test)),
    }


def evaluate_knn(X, y, splits, feature_mode, random_state):
    rows = []
    best = None
    for split_name, (train_idx, test_idx) in splits.items():
        n_train = len(train_idx)
        pca_options = [None, 10, 50]
        pca_options = [
            n
            for n in pca_options
            if n is None or n < min(n_train, X.shape[1])
        ]
        for pca_components in pca_options:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            if pca_components is not None:
                pca = PCA(
                    n_components=pca_components,
                    whiten=True,
                    random_state=random_state,
                )
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            for k in [1, 5, 11]:
                for metric in ["euclidean", "cosine"]:
                    clf = KNeighborsClassifier(
                        n_neighbors=k,
                        weights="distance",
                        metric=metric,
                        algorithm="brute",
                    )
                    clf.fit(X_train, y[train_idx])
                    pred = clf.predict(X_test)
                    row = {
                        "feature_mode": feature_mode,
                        "split": split_name,
                        "pca_components": pca_components if pca_components is not None else 0,
                        "k": k,
                        "metric": metric,
                        "accuracy": float(accuracy_score(y[test_idx], pred)),
                        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                        "macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
                    }
                    rows.append(row)
                    if best is None or row["balanced_accuracy"] > best["balanced_accuracy"]:
                        best = {
                            **row,
                            "estimator": clf,
                            "train_idx": train_idx,
                            "test_idx": test_idx,
                            "pred": pred,
                        }
    return rows, best


def plot_confusion(cm, classes, path, title):
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
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_embedding_plots(X, y, metadata, classes, out_dir, random_state):
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=random_state)
    Z_pca = pca.fit_transform(X_scaled)

    tsne_n = min(1500, X_scaled.shape[0])
    rng = np.random.default_rng(random_state)
    tsne_idx = np.sort(rng.choice(X_scaled.shape[0], size=tsne_n, replace=False))
    Z_tsne = TSNE(
        n_components=2,
        perplexity=35,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    ).fit_transform(X_scaled[tsne_idx])

    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    def scatter(Z, labels, title, path, meta=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        for c_idx, c in enumerate(classes):
            mask = labels == c_idx
            ax.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.70, label=c)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    scatter(
        Z_pca,
        y,
        f"PCA 2D, explained variance={pca.explained_variance_ratio_.sum():.3f}",
        emb_dir / "pca_2d_by_label.png",
    )
    scatter(Z_tsne, y[tsne_idx], "t-SNE 2D by label", emb_dir / "tsne_2d_by_label.png")

    amp_db = (20.0 * np.log10(np.abs(metadata.attrs["X_complex"]) + EPS)).astype(np.float32)
    phase = centered_unwrapped_phase(metadata.attrs["X_complex"])
    phase_diff = differential_phase(metadata.attrs["X_complex"])
    simple = pd.DataFrame(
        {
            "label": [classes[i] for i in y],
            "amp_median_db": np.median(amp_db, axis=(1, 2, 3)),
            "amp_iqr_db": np.percentile(amp_db, 75, axis=(1, 2, 3))
            - np.percentile(amp_db, 25, axis=(1, 2, 3)),
            "phase_median_rad": np.median(phase, axis=(1, 2, 3)),
            "phase_iqr_rad": np.percentile(phase, 75, axis=(1, 2, 3))
            - np.percentile(phase, 25, axis=(1, 2, 3)),
            "phase_diff_iqr_rad": np.percentile(phase_diff, 75, axis=(1, 2, 3))
            - np.percentile(phase_diff, 25, axis=(1, 2, 3)),
        }
    )
    simple.to_csv(emb_dir / "amplitude_phase_2d_features.csv", index=False)

    for x_col, y_col, fname in [
        ("amp_median_db", "phase_iqr_rad", "amplitude_vs_phase_iqr.png"),
        ("amp_iqr_db", "phase_diff_iqr_rad", "amplitude_iqr_vs_phase_diff_iqr.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for c in classes:
            g = simple[simple["label"] == c]
            ax.scatter(g[x_col], g[y_col], s=14, alpha=0.70, label=c)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"2D amplitude/phase dispersion: {x_col} vs {y_col}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(emb_dir / fname, dpi=180)
        plt.close(fig)

    return {
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "tsne_n_points": int(tsne_n),
    }


def save_neighbor_analysis(X, y, classes, out_dir):
    neigh_dir = out_dir / "neighbors"
    neigh_dir.mkdir(parents=True, exist_ok=True)
    X_scaled = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=8, metric="cosine", algorithm="brute")
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)

    rows = []
    for i in range(X.shape[0]):
        neighbor_labels = y[indices[i, 1:]]
        same = neighbor_labels == y[i]
        rows.append(
            {
                "label": classes[y[i]],
                "mean_neighbor_distance": float(np.mean(distances[i, 1:])),
                "same_label_neighbor_ratio": float(np.mean(same)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(neigh_dir / "nearest_neighbor_dispersion.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    data = [df[df["label"] == c]["same_label_neighbor_ratio"].to_numpy() for c in classes]
    ax.boxplot(data, tick_labels=classes)
    ax.set_ylabel("Ratio of same-label neighbors among 7 nearest")
    ax.set_title("KNN neighborhood purity by class")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(neigh_dir / "neighbor_purity_by_class.png", dpi=180)
    plt.close(fig)

    centroids = []
    for c_idx in range(len(classes)):
        centroids.append(X_scaled[y == c_idx].mean(axis=0))
    centroids = np.stack(centroids)
    centroid_dist = np.zeros((len(classes), len(classes)), dtype=np.float32)
    for i in range(len(classes)):
        for j in range(len(classes)):
            a = centroids[i]
            b = centroids[j]
            centroid_dist[i, j] = 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(centroid_dist)
    fig.colorbar(im, ax=ax, label="Cosine distance")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_title("Class centroid distance")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{centroid_dist[i, j]:.2f}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(neigh_dir / "class_centroid_cosine_distance.png", dpi=180)
    plt.close(fig)

    return {
        "mean_same_label_neighbor_ratio": float(df["same_label_neighbor_ratio"].mean()),
        "mean_neighbor_distance": float(df["mean_neighbor_distance"].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="reports/fold1_gridfull_knn_w2")
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_complex, metadata = build_dataset(Path(args.raw_dir), args.window, args.stride)
    metadata.to_csv(out_dir / "fold1_gridfull_w2_metadata.csv", index=False)
    np.savez_compressed(out_dir / "fold1_gridfull_w2_dataset.npz", X=X_complex, y=metadata["label"].to_numpy())

    encoder = LabelEncoder()
    y = encoder.fit_transform(metadata["label"].to_numpy())
    classes = list(encoder.classes_)
    splits = make_splits(metadata, y, args.seed)

    all_rows = []
    best_overall = None
    best_feature_X = None
    for feature_mode in [
        "amplitude",
        "amplitude_shape",
        "complex_real_imag",
        "amplitude_phase",
        "amplitude_shape_phase_diff",
    ]:
        X = extract_features(X_complex, feature_mode)
        rows, best = evaluate_knn(X, y, splits, feature_mode, args.seed)
        all_rows.extend(rows)
        if best_overall is None or best["balanced_accuracy"] > best_overall["balanced_accuracy"]:
            best_overall = best
            best_feature_X = X

    results = pd.DataFrame(all_rows).sort_values("balanced_accuracy", ascending=False)
    results.to_csv(out_dir / "knn_results.csv", index=False)

    train_idx = best_overall["train_idx"]
    test_idx = best_overall["test_idx"]
    pred = best_overall["pred"]
    y_test = y[test_idx]
    cm = confusion_matrix(y_test, pred, labels=np.arange(len(classes)))
    report = classification_report(
        y_test,
        pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    (out_dir / "best_classification_report.txt").write_text(report)
    plot_confusion(
        cm,
        classes,
        out_dir / "best_confusion_matrix.png",
        f"Best KNN: {best_overall['feature_mode']} | {best_overall['split']}",
    )

    metadata.attrs["X_complex"] = X_complex
    emb_info = save_embedding_plots(best_feature_X, y, metadata, classes, out_dir, args.seed)
    neigh_info = save_neighbor_analysis(best_feature_X, y, classes, out_dir)

    try:
        sil = float(silhouette_score(StandardScaler().fit_transform(best_feature_X), y, metric="cosine"))
    except Exception:
        sil = None

    summary = {
        "dataset": {
            "source": "gridFull",
            "fold": 1,
            "input_shape_complex": list(X_complex.shape),
            "window": args.window,
            "stride": args.stride,
            "classes": classes,
            "label_counts": metadata["label"].value_counts().to_dict(),
        },
        "best_knn": {
            k: v
            for k, v in best_overall.items()
            if k not in {"estimator", "train_idx", "test_idx", "pred"}
        },
        "silhouette_cosine": sil,
        "embedding": emb_info,
        "neighbors": neigh_info,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    readme = f"""# Fold 1 gridFull KNN, 2-sample CSI windows

This experiment uses only fold 1 sessions and the raw MATLAB `gridFull` variable.

Input per sample:

- `gridFull`: `360 subcarriers x 6 OFDM symbols`
- window size: `{args.window}` consecutive CSI captures
- stored shape: `{list(X_complex.shape)}` = windows x 2 x 360 x 6 complex

Best KNN:

- feature mode: `{best_overall['feature_mode']}`
- split: `{best_overall['split']}`
- k: `{best_overall['k']}`
- metric: `{best_overall['metric']}`
- PCA components: `{best_overall['pca_components']}` (`0` means no PCA)
- accuracy: `{best_overall['accuracy']:.4f}`
- balanced accuracy: `{best_overall['balanced_accuracy']:.4f}`
- macro F1: `{best_overall['macro_f1']:.4f}`

Useful files:

- `knn_results.csv`: all KNN attempts.
- `best_confusion_matrix.png`: best confusion matrix.
- `best_classification_report.txt`: sklearn report for the best run.
- `embeddings/pca_2d_by_label.png`: PCA 2D points.
- `embeddings/tsne_2d_by_label.png`: t-SNE 2D points.
- `embeddings/amplitude_vs_phase_iqr.png`: direct amplitude/phase 2D scatter.
- `neighbors/neighbor_purity_by_class.png`: local KNN neighborhood purity.
- `neighbors/class_centroid_cosine_distance.png`: class dispersion/separation.
"""
    (out_dir / "README.md").write_text(readme)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
