#!/usr/bin/env python3

"""
Created by Alberto Martinez Mendez
Universidad Carlos III de Madrid (UC3M)

This code is licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0).

You are free to share and adapt this work for any purpose, even
commercially, as long as you give appropriate credit to the author.

License: https://creativecommons.org/licenses/by/4.0/

Generic 5G SSB sensing analysis and classification pipeline.
Expected directory structure:

data_root/
    label_1/
        session_xxx/
            session_data.h5

    label_2/
        session_xxx/
            session_data.h5

    ...

Any number of labels >= 2 is supported.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from sklearn.preprocessing import StandardScaler

from py3gpp.nrSSS import nrSSS
from py3gpp.nrPBCHDMRS import nrPBCHDMRS
from py3gpp.nrPBCHDMRSIndices import nrPBCHDMRSIndices
from py3gpp.nrChannelEstimate import nrChannelEstimate
from ssb_python.ssb_fisher_torch_inference import (
    export_sklearn_pipeline_as_torch,
)


# ============================================================
# CONSTANTS
# ============================================================

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20

DEFAULT_TOP_K_RE = 64
DEFAULT_RANDOM_TRACES = 10
DEFAULT_SEED = 42

N_PROBE = 41


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generic SSB sensing analysis and "
            "logistic-regression classification pipeline."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Directory containing one subdirectory per label."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/results_ssb_sensing"),
        help="Output directory.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K_RE,
        help="Number of Fisher-selected resource elements.",
    )

    parser.add_argument(
        "--random-traces",
        type=int,
        default=DEFAULT_RANDOM_TRACES,
        help=(
            "Number of random H-magnitude snapshots "
            "to plot per label."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed.",
    )

    return parser.parse_args()


# ============================================================
# GENERIC UTILITIES
# ============================================================

def db20(x):
    """Convert complex magnitude to dB."""
    return 20.0 * np.log10(
        np.maximum(
            np.abs(x),
            1e-12,
        )
    )


def safe_mean(x, axis=None):
    return np.nanmean(
        np.asarray(x),
        axis=axis,
    )


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_dataset(data_root: Path):
    """
    Discover labels and session_data.h5 files.

    Immediate subdirectories under data_root are interpreted
    as class labels.
    """

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root does not exist: {data_root}"
        )

    label_dirs = sorted(
        [
            p
            for p in data_root.iterdir()
            if p.is_dir()
        ],
        key=lambda p: p.name,
    )

    dataset = {}

    for label_dir in label_dirs:
        h5_files = sorted(
            label_dir.rglob("session_data.h5")
        )

        if not h5_files:
            continue

        dataset[label_dir.name] = h5_files

    if len(dataset) < 2:
        raise RuntimeError(
            "At least two labels containing "
            "session_data.h5 are required."
        )

    return dataset


# ============================================================
# H5 LOADING
# ============================================================

def load_rxgrid(path: Path):
    """Load rxGridSSB and H5 attributes."""

    with h5py.File(path, "r") as f:
        if "rxGridSSB" not in f:
            raise KeyError(
                f"rxGridSSB not found in {path}"
            )

        rxgrid = f["rxGridSSB"][()]

        attrs = {
            key: f.attrs[key]
            for key in f.attrs.keys()
        }

    if rxgrid.ndim != 3:
        raise ValueError(
            f"Unexpected rxGridSSB shape in {path}: "
            f"{rxgrid.shape}"
        )

    return rxgrid, attrs


# ============================================================
# PCI DETECTION
# ============================================================

def detect_nid1(rxgrid, nid2):
    """
    Detect NID1 using SSS correlation on several snapshots.
    """

    n_snapshots = rxgrid.shape[2]

    probe_indices = np.unique(
        np.linspace(
            0,
            n_snapshots - 1,
            min(N_PROBE, n_snapshots),
        ).round().astype(int)
    )

    sss_references = np.stack(
        [
            nrSSS(
                3 * nid1 + nid2
            )
            for nid1 in range(336)
        ]
    )

    votes = []

    for snapshot_idx in probe_indices:

        received_sss = rxgrid[
            56:183,
            2,
            snapshot_idx,
        ]

        norm = np.linalg.norm(
            received_sss
        )

        if norm > 1e-12:
            received_sss = (
                received_sss
                / norm
            )

        scores = np.abs(
            sss_references.conj()
            @ received_sss
        ) ** 2

        votes.append(
            int(
                np.argmax(scores)
            )
        )

    votes = np.asarray(
        votes
    )

    values, counts = np.unique(
        votes,
        return_counts=True,
    )

    best_nid1 = int(
        values[
            np.argmax(counts)
        ]
    )

    agreement = float(
        np.mean(
            votes == best_nid1
        )
    )

    return best_nid1, agreement


# ============================================================
# i_bar_SSB DETECTION
# ============================================================

def normalized_correlation(a, b):
    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator < 1e-12:
        return 0.0

    return float(
        np.abs(
            np.vdot(
                b,
                a,
            )
        )
        / denominator
    )


def detect_ibar(rxgrid, pci):
    """
    Detect i_bar_SSB using normalized correlation
    against all eight PBCH-DMRS candidates.
    """

    k_idx, s_idx = nrPBCHDMRSIndices(
        pci,
        style="python",
    )

    k_idx = np.asarray(
        k_idx
    )

    s_idx = np.asarray(
        s_idx
    )

    references = {
        ibar: nrPBCHDMRS(
            pci,
            ibar,
        )
        for ibar in range(8)
    }

    n_snapshots = rxgrid.shape[2]

    probe_indices = np.unique(
        np.linspace(
            0,
            n_snapshots - 1,
            min(N_PROBE, n_snapshots),
        ).round().astype(int)
    )

    votes = []
    all_scores = []

    for snapshot_idx in probe_indices:

        received_dmrs = rxgrid[
            k_idx,
            s_idx,
            snapshot_idx,
        ]

        scores = np.asarray(
            [
                normalized_correlation(
                    received_dmrs,
                    references[ibar],
                )
                for ibar in range(8)
            ]
        )

        all_scores.append(
            scores
        )

        votes.append(
            int(
                np.argmax(scores)
            )
        )

    votes = np.asarray(
        votes
    )

    all_scores = np.asarray(
        all_scores
    )

    values, counts = np.unique(
        votes,
        return_counts=True,
    )

    best_ibar = int(
        values[
            np.argmax(counts)
        ]
    )

    agreement = float(
        np.mean(
            votes == best_ibar
        )
    )

    mean_scores = np.mean(
        all_scores,
        axis=0,
    )

    return (
        best_ibar,
        agreement,
        mean_scores,
    )


# ============================================================
# CHANNEL ESTIMATION
# ============================================================

def build_reference_grid(
    pci,
    ibar,
):
    """
    Build a 240 x 4 reference SSB grid using
    PBCH-DMRS and SSS.
    """

    reference_grid = np.zeros(
        (240, 4),
        dtype=np.complex128,
    )

    k_idx, s_idx = nrPBCHDMRSIndices(
        pci,
        style="python",
    )

    reference_grid[
        np.asarray(k_idx),
        np.asarray(s_idx),
    ] = nrPBCHDMRS(
        pci,
        ibar,
    )

    reference_grid[
        56:183,
        2,
    ] = nrSSS(
        pci
    )

    return reference_grid


def estimate_channel(
    rxgrid,
    pci,
    ibar,
):
    """
    Estimate H for all snapshots.
    """

    reference_grid = (
        build_reference_grid(
            pci,
            ibar,
        )
    )

    n_snapshots = rxgrid.shape[2]

    channel = np.empty(
        (
            240,
            4,
            n_snapshots,
        ),
        dtype=np.complex64,
    )

    for snapshot_idx in range(
        n_snapshots
    ):

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            h, _ = nrChannelEstimate(
                rxGrid=rxgrid[
                    :,
                    :,
                    snapshot_idx,
                ],
                refGrid=reference_grid,
            )

        channel[
            :,
            :,
            snapshot_idx,
        ] = h

        if (
            (snapshot_idx + 1)
            % 100
            == 0
            or snapshot_idx
            == n_snapshots - 1
        ):
            print(
                "      "
                f"H {snapshot_idx + 1}/"
                f"{n_snapshots}"
            )

    return channel


# ============================================================
# FISHER SCORE
# ============================================================

def fisher_score_equal_class(
    class_arrays,
):
    """
    Multi-class Fisher score.

    Each array must have shape:
        [N_samples, N_features]
    """

    class_means = np.stack(
        [
            np.mean(
                x,
                axis=0,
            )
            for x in class_arrays
        ]
    )

    global_mean = np.mean(
        class_means,
        axis=0,
    )

    between_class = np.sum(
        (
            class_means
            - global_mean
        ) ** 2,
        axis=0,
    )

    within_class = np.sum(
        [
            np.var(
                x,
                axis=0,
                ddof=1,
            )
            for x in class_arrays
        ],
        axis=0,
    )

    return (
        between_class
        / np.maximum(
            within_class,
            1e-12,
        )
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def concatenate_label_sessions(
    session_results
):
    """
    Concatenate sessions in deterministic session-path order.
    """

    channels = [
        item["H"]
        for item in session_results
    ]

    return np.concatenate(
        channels,
        axis=2,
    )


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def chronological_split(
    n_samples,
):
    """
    60 / 20 / 20 chronological split.
    """

    n_train = int(
        TRAIN_FRACTION
        * n_samples
    )

    n_validation = int(
        VALIDATION_FRACTION
        * n_samples
    )

    train_indices = np.arange(
        0,
        n_train,
    )

    validation_indices = np.arange(
        n_train,
        n_train
        + n_validation,
    )

    test_indices = np.arange(
        n_train
        + n_validation,
        n_samples,
    )

    return {
        "train": train_indices,
        "validation":
            validation_indices,
        "test": test_indices,
    }


# ============================================================
# SPECTRAL FEATURES
# ============================================================

def compute_spectral_features(
    normalized_magnitude,
):
    """
    Compute the same 9 spectral features used previously.

    Input:
        [N, 240, 4]

    Features:
        low mean
        low std

        mid mean
        mid std

        high mean
        high std

        low - mid
        high - mid
        high - low
    """

    bands = {
        "low": (
            0,
            57,
        ),
        "mid": (
            57,
            168,
        ),
        "high": (
            168,
            240,
        ),
    }

    means = {}
    stds = {}

    for band_name, (
        start,
        stop,
    ) in bands.items():

        band = (
            normalized_magnitude[
                :,
                start:stop,
                :,
            ]
        )

        means[
            band_name
        ] = np.mean(
            band,
            axis=(1, 2),
        )

        stds[
            band_name
        ] = np.std(
            band,
            axis=(1, 2),
        )

    return np.stack(
        [
            means["low"],
            stds["low"],

            means["mid"],
            stds["mid"],

            means["high"],
            stds["high"],

            means["low"]
            - means["mid"],

            means["high"]
            - means["mid"],

            means["high"]
            - means["low"],
        ],
        axis=1,
    )


# ============================================================
# PLOTS
# ============================================================

def plot_random_snapshots(
    magnitude_db,
    labels,
    output_path,
    traces_per_label,
    seed,
):
    """
    Plot random channel-magnitude snapshots.

    Magnitude is averaged across the four SSB OFDM symbols.
    """

    rng = np.random.default_rng(
        seed
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for label in labels:

        x = magnitude_db[
            label
        ]

        n_samples = x.shape[0]

        n_to_plot = min(
            traces_per_label,
            n_samples,
        )

        selected = rng.choice(
            n_samples,
            size=n_to_plot,
            replace=False,
        )

        for trace_idx, sample_idx in enumerate(
            selected
        ):
            profile = np.mean(
                x[sample_idx],
                axis=1,
            )

            trace_label = (
                label
                if trace_idx == 0
                else None
            )

            ax.plot(
                np.arange(240),
                profile,
                alpha=0.55,
                linewidth=1.0,
                label=trace_label,
            )

    ax.set_xlabel(
        "Subcarrier index"
    )

    ax.set_ylabel(
        "|H| [dB]"
    )

    ax.set_title(
        "Random H-magnitude snapshots"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
    )

    plt.close(
        fig
    )


def plot_mean_profiles(
    magnitude_db,
    labels,
    output_path,
    normalized=False,
):
    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for label in labels:

        mean_profile = np.mean(
            magnitude_db[label],
            axis=(0, 2),
        )

        ax.plot(
            np.arange(240),
            mean_profile,
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel(
        "Subcarrier index"
    )

    if normalized:
        ax.set_ylabel(
            "Mean normalized |H| [dB]"
        )

        ax.set_title(
            "Shape-normalized H magnitude profile"
        )
    else:
        ax.set_ylabel(
            "Mean |H| [dB]"
        )

        ax.set_title(
            "Estimated H magnitude profile"
        )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
    )

    plt.close(
        fig
    )


def plot_global_channel_level(
    magnitude_db,
    labels,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for label in labels:

        global_level = np.mean(
            magnitude_db[label],
            axis=(1, 2),
        )

        ax.plot(
            np.arange(
                len(global_level)
            ),
            global_level,
            linewidth=1.0,
            alpha=0.75,
            label=label,
        )

    ax.set_xlabel(
        "Snapshot index"
    )

    ax.set_ylabel(
        "Global |H| [dB]"
    )

    ax.set_title(
        "Temporal evolution of global channel level"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
    )

    plt.close(
        fig
    )


def plot_fisher_heatmap(
    fisher,
    output_path,
):
    fisher_grid = fisher.reshape(
        240,
        4,
    )

    fig, ax = plt.subplots(
        figsize=(7, 10)
    )

    image = ax.imshow(
        fisher_grid,
        aspect="auto",
        origin="lower",
    )

    ax.set_xlabel(
        "OFDM symbol"
    )

    ax.set_ylabel(
        "Subcarrier"
    )

    ax.set_title(
        "Training-only Fisher score per resource element"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Fisher score",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
    )

    plt.close(
        fig
    )


def plot_pca(
    normalized_magnitude,
    labels,
    output_path,
):
    """
    Exploratory PCA only.

    PCA is intentionally fitted on the full dataset for visualization,
    not for classifier training or evaluation.
    """

    x_parts = []
    y_parts = []

    for label in labels:

        x = normalized_magnitude[
            label
        ].reshape(
            len(
                normalized_magnitude[
                    label
                ]
            ),
            -1,
        )

        x_parts.append(
            x
        )

        y_parts.extend(
            [label]
            * len(x)
        )

    x_all = np.vstack(
        x_parts
    )

    y_all = np.asarray(
        y_parts
    )

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(
        x_all
    )

    pca = PCA(
        n_components=2
    )

    x_pca = pca.fit_transform(
        x_scaled
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    for label in labels:

        mask = (
            y_all
            == label
        )

        ax.scatter(
            x_pca[
                mask,
                0,
            ],
            x_pca[
                mask,
                1,
            ],
            s=12,
            alpha=0.55,
            label=label,
        )

    ax.set_xlabel(
        "PC1 "
        f"({pca.explained_variance_ratio_[0] * 100:.1f}%)"
    )

    ax.set_ylabel(
        "PC2 "
        f"({pca.explained_variance_ratio_[1] * 100:.1f}%)"
    )

    ax.set_title(
        "Exploratory PCA of normalized H magnitude"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
    )

    plt.close(
        fig
    )

    return (
        pca.explained_variance_ratio_
    )


def plot_confusion_matrix(
    matrix,
    labels,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(
            max(
                6,
                len(labels) * 1.1,
            ),
            max(
                5,
                len(labels) * 1.0,
            ),
        )
    )

    image = ax.imshow(
        matrix,
    )

    ax.set_xticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )

    ax.set_title(
        "Confusion matrix - Logistic Regression"
    )

    threshold = (
        matrix.max()
        / 2.0
        if matrix.size
        else 0
    )

    for i in range(
        matrix.shape[0]
    ):
        for j in range(
            matrix.shape[1]
        ):

            value = int(
                matrix[
                    i,
                    j,
                ]
            )

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
    )

    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.random.seed(
        args.seed
    )

    # --------------------------------------------------------
    # Discover dataset
    # --------------------------------------------------------

    discovered = discover_dataset(
        args.data_root
    )

    labels = list(
        discovered.keys()
    )

    print()
    print("=" * 80)
    print("DATASET DISCOVERY")
    print("=" * 80)

    print(
        f"Labels detected: {labels}"
    )

    for label in labels:

        print()
        print(
            f"{label}:"
        )

        for session_path in discovered[
            label
        ]:
            print(
                f"  {session_path}"
            )

    # --------------------------------------------------------
    # Estimate H session by session
    # --------------------------------------------------------

    processed_sessions = {
        label: []
        for label in labels
    }

    detection_summary = {}

    for label in labels:

        print()
        print("=" * 80)
        print(
            f"PROCESSING LABEL: {label}"
        )
        print("=" * 80)

        detection_summary[
            label
        ] = []

        for session_path in discovered[
            label
        ]:

            print()
            print(
                f"Session: {session_path}"
            )

            rxgrid, attrs = (
                load_rxgrid(
                    session_path
                )
            )

            nid2 = int(
                attrs.get(
                    "force_nid2",
                    2,
                )
            )

            nid1, nid1_agreement = (
                detect_nid1(
                    rxgrid,
                    nid2,
                )
            )

            pci = (
                3 * nid1
                + nid2
            )

            (
                ibar,
                ibar_agreement,
                ibar_scores,
            ) = detect_ibar(
                rxgrid,
                pci,
            )

            print(
                f"  PCI={pci}"
            )

            print(
                "  NID1 agreement="
                f"{nid1_agreement:.3f}"
            )

            print(
                f"  i_bar_SSB={ibar}"
            )

            print(
                "  i_bar agreement="
                f"{ibar_agreement:.3f}"
            )

            print(
                "  i_bar mean scores:"
            )

            for idx, score in enumerate(
                ibar_scores
            ):
                print(
                    f"    {idx}: "
                    f"{score:.4f}"
                )

            print(
                "  Estimating H..."
            )

            channel = estimate_channel(
                rxgrid,
                pci,
                ibar,
            )

            processed_sessions[
                label
            ].append(
                {
                    "path":
                        str(
                            session_path
                        ),
                    "H":
                        channel,
                    "pci":
                        pci,
                    "nid1":
                        nid1,
                    "nid2":
                        nid2,
                    "ibar":
                        ibar,
                }
            )

            detection_summary[
                label
            ].append(
                {
                    "session":
                        str(
                            session_path
                        ),
                    "snapshots":
                        int(
                            rxgrid.shape[2]
                        ),
                    "pci":
                        int(pci),
                    "nid1":
                        int(nid1),
                    "nid2":
                        int(nid2),
                    "nid1_agreement":
                        float(
                            nid1_agreement
                        ),
                    "ibar":
                        int(ibar),
                    "ibar_agreement":
                        float(
                            ibar_agreement
                        ),
                    "ibar_scores": {
                        str(i):
                            float(v)
                        for i, v
                        in enumerate(
                            ibar_scores
                        )
                    },
                }
            )

    # --------------------------------------------------------
    # Check PCI / i_bar consistency
    # --------------------------------------------------------

    all_pci = set()
    all_ibar = set()

    for label in labels:

        for session in processed_sessions[
            label
        ]:
            all_pci.add(
                session["pci"]
            )

            all_ibar.add(
                session["ibar"]
            )

    if len(all_pci) > 1:
        warnings.warn(
            "Different PCI values were detected "
            "across sessions."
        )

    if len(all_ibar) > 1:
        warnings.warn(
            "Different i_bar_SSB values were detected "
            "across sessions. Interpret cross-class "
            "comparisons carefully."
        )

    # --------------------------------------------------------
    # Concatenate sessions for each label
    # --------------------------------------------------------

    channel_by_label = {}

    for label in labels:

        channel_by_label[
            label
        ] = (
            concatenate_label_sessions(
                processed_sessions[
                    label
                ]
            )
        )

        print(
            f"{label}: "
            f"{channel_by_label[label].shape[2]} "
            "total snapshots"
        )

    # --------------------------------------------------------
    # Magnitude representations
    # --------------------------------------------------------

    magnitude_db = {}
    normalized_magnitude = {}

    for label in labels:

        channel = (
            channel_by_label[
                label
            ]
        )

        magnitude = np.transpose(
            db20(
                channel
            ),
            (2, 0, 1),
        )

        snapshot_mean = np.mean(
            magnitude,
            axis=(1, 2),
            keepdims=True,
        )

        magnitude_db[
            label
        ] = magnitude

        normalized_magnitude[
            label
        ] = (
            magnitude
            - snapshot_mean
        )

    # --------------------------------------------------------
    # DATA ANALYSIS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SIGNAL ANALYSIS")
    print("=" * 80)

    signal_stats = {}

    for label in labels:

        global_level = np.mean(
            magnitude_db[
                label
            ],
            axis=(1, 2),
        )

        signal_stats[
            label
        ] = {
            "n_samples":
                int(
                    len(global_level)
                ),
            "global_H_mean_db":
                float(
                    np.mean(
                        global_level
                    )
                ),
            "global_H_std_db":
                float(
                    np.std(
                        global_level
                    )
                ),
            "global_H_min_db":
                float(
                    np.min(
                        global_level
                    )
                ),
            "global_H_max_db":
                float(
                    np.max(
                        global_level
                    )
                ),
        }

        print()
        print(
            f"{label}:"
        )

        print(
            "  Global H mean = "
            f"{signal_stats[label]['global_H_mean_db']:.4f} dB"
        )

        print(
            "  Global H std  = "
            f"{signal_stats[label]['global_H_std_db']:.4f} dB"
        )

    # --------------------------------------------------------
    # Random snapshot visualization
    # --------------------------------------------------------

    plot_random_snapshots(
        magnitude_db=
            magnitude_db,
        labels=
            labels,
        output_path=
            args.output
            / "01_random_H_magnitude_snapshots.pdf",
        traces_per_label=
            args.random_traces,
        seed=
            args.seed,
    )

    # --------------------------------------------------------
    # Mean profiles
    # --------------------------------------------------------

    plot_mean_profiles(
        magnitude_db=
            magnitude_db,
        labels=
            labels,
        output_path=
            args.output
            / "02_mean_H_magnitude_profile.pdf",
        normalized=False,
    )

    plot_mean_profiles(
        magnitude_db=
            normalized_magnitude,
        labels=
            labels,
        output_path=
            args.output
            / "03_normalized_H_profile.pdf",
        normalized=True,
    )

    # --------------------------------------------------------
    # Global temporal evolution
    # --------------------------------------------------------

    plot_global_channel_level(
        magnitude_db=
            magnitude_db,
        labels=
            labels,
        output_path=
            args.output
            / "04_global_H_level_temporal.pdf",
    )

    # --------------------------------------------------------
    # Chronological splits
    # --------------------------------------------------------

    splits = {}

    for label in labels:

        n_samples = len(
            magnitude_db[
                label
            ]
        )

        splits[
            label
        ] = chronological_split(
            n_samples
        )

        print()
        print(
            f"{label}: "
            f"train="
            f"{len(splits[label]['train'])}, "
            f"validation="
            f"{len(splits[label]['validation'])}, "
            f"test="
            f"{len(splits[label]['test'])}"
        )

    # --------------------------------------------------------
    # Fisher score: TRAIN ONLY
    # --------------------------------------------------------

    training_flat = {}

    for label in labels:

        idx = splits[
            label
        ]["train"]

        training_flat[
            label
        ] = (
            normalized_magnitude[
                label
            ][idx]
            .reshape(
                len(idx),
                -1,
            )
        )

    fisher = (
        fisher_score_equal_class(
            [
                training_flat[
                    label
                ]
                for label
                in labels
            ]
        )
    )

    order = np.argsort(
        fisher
    )[::-1]

    top_k = min(
        args.top_k,
        len(order),
    )

    selected_flat = order[
        :top_k
    ]

    selected_resource_elements = []

    for flat_index in selected_flat:

        subcarrier = int(
            flat_index
            // 4
        )

        symbol = int(
            flat_index
            % 4
        )

        selected_resource_elements.append(
            (
                subcarrier,
                symbol,
            )
        )

    print()
    print("=" * 80)
    print("TOP FISHER RESOURCE ELEMENTS")
    print("=" * 80)

    for rank, (
        subcarrier,
        symbol,
    ) in enumerate(
        selected_resource_elements[
            :20
        ],
        1,
    ):
        print(
            f"{rank:2d}. "
            f"k={subcarrier:3d}, "
            f"s={symbol}, "
            f"Fisher="
            f"{fisher[4 * subcarrier + symbol]:.5f}"
        )

    plot_fisher_heatmap(
        fisher=
            fisher,
        output_path=
            args.output
            / "05_fisher_RE_heatmap.pdf",
    )

    # --------------------------------------------------------
    # PCA exploratory visualization
    # --------------------------------------------------------

    explained_variance = plot_pca(
        normalized_magnitude=
            normalized_magnitude,
        labels=
            labels,
        output_path=
            args.output
            / "06_PCA_normalized_H_magnitude.pdf",
    )

    print()
    print(
        "Exploratory PCA:"
    )

    print(
        f"  PC1 = "
        f"{explained_variance[0] * 100:.3f}%"
    )

    print(
        f"  PC2 = "
        f"{explained_variance[1] * 100:.3f}%"
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    def build_features(
        label,
        indices,
    ):

        x = (
            normalized_magnitude[
                label
            ][indices]
        )

        fisher_features = np.stack(
            [
                x[
                    :,
                    subcarrier,
                    symbol,
                ]
                for (
                    subcarrier,
                    symbol
                )
                in selected_resource_elements
            ],
            axis=1,
        )

        spectral_features = (
            compute_spectral_features(
                x
            )
        )

        return np.hstack(
            [
                fisher_features,
                spectral_features,
            ]
        )

    # --------------------------------------------------------
    # Build train / validation / test matrices
    # --------------------------------------------------------

    label_to_index = {
        label: idx
        for idx, label
        in enumerate(labels)
    }

    prepared = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        x_parts = []
        y_parts = []

        for label in labels:

            indices = splits[
                label
            ][split_name]

            x_label = build_features(
                label,
                indices,
            )

            y_label = np.full(
                len(x_label),
                label_to_index[
                    label
                ],
                dtype=int,
            )

            x_parts.append(
                x_label
            )

            y_parts.append(
                y_label
            )

        prepared[
            split_name
        ] = (
            np.vstack(
                x_parts
            ),
            np.concatenate(
                y_parts
            ),
        )

    # --------------------------------------------------------
    # Logistic regression
    # --------------------------------------------------------

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    x_train, y_train = prepared[
        "train"
    ]

    x_validation, y_validation = prepared[
        "validation"
    ]

    model.fit(
        x_train,
        y_train,
    )

    validation_prediction = model.predict(
        x_validation
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_prediction,
    )

    validation_balanced_accuracy = (
        balanced_accuracy_score(
            y_validation,
            validation_prediction,
        )
    )

    validation_macro_f1 = f1_score(
        y_validation,
        validation_prediction,
        average="macro",
    )

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    print(
        "Accuracy          : "
        f"{validation_accuracy:.4f}"
    )

    print(
        "Balanced accuracy : "
        f"{validation_balanced_accuracy:.4f}"
    )

    print(
        "Macro-F1          : "
        f"{validation_macro_f1:.4f}"
    )

    # --------------------------------------------------------
    # Retrain using train + validation
    # --------------------------------------------------------

    x_train_validation = np.vstack(
        [
            x_train,
            x_validation,
        ]
    )

    y_train_validation = np.concatenate(
        [
            y_train,
            y_validation,
        ]
    )

    model.fit(
        x_train_validation,
        y_train_validation,
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    x_test, y_test = prepared[
        "test"
    ]

    test_prediction = model.predict(
        x_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_prediction,
    )

    test_balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            test_prediction,
        )
    )

    test_macro_f1 = f1_score(
        y_test,
        test_prediction,
        average="macro",
    )

    class_indices = np.arange(
        len(labels)
    )

    confusion = confusion_matrix(
        y_test,
        test_prediction,
        labels=class_indices,
    )

    report = classification_report(
        y_test,
        test_prediction,
        labels=class_indices,
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    print()
    print("=" * 80)
    print("FINAL TEST")
    print("=" * 80)

    print(
        "Accuracy          : "
        f"{test_accuracy:.4f}"
    )

    print(
        "Balanced accuracy : "
        f"{test_balanced_accuracy:.4f}"
    )

    print(
        "Macro-F1          : "
        f"{test_macro_f1:.4f}"
    )

    print()
    print(
        report
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion
    )

    # --------------------------------------------------------
    # Confusion matrix figure
    # --------------------------------------------------------

    plot_confusion_matrix(
        matrix=
            confusion,
        labels=
            labels,
        output_path=
            args.output
            / "07_confusion_matrix.pdf",
    )

    # --------------------------------------------------------
    # Save Fisher REs
    # --------------------------------------------------------

    fisher_rows = []

    for rank, (
        subcarrier,
        symbol,
    ) in enumerate(
        selected_resource_elements,
        1,
    ):

        fisher_rows.append(
            {
                "rank":
                    rank,
                "subcarrier":
                    subcarrier,
                "symbol":
                    symbol,
                "fisher_score":
                    float(
                        fisher[
                            4 * subcarrier
                            + symbol
                        ]
                    ),
            }
        )

    pd.DataFrame(
        fisher_rows
    ).to_csv(
        args.output
        / "selected_resource_elements.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    with (
        args.output
        / "classification_report.txt"
    ).open(
        "w"
    ) as file:

        file.write(
            report
        )

    # --------------------------------------------------------
    # Save confusion matrix CSV
    # --------------------------------------------------------

    pd.DataFrame(
        confusion,
        index=labels,
        columns=labels,
    ).to_csv(
        args.output
        / "confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # Export deployment checkpoint
    # --------------------------------------------------------
    # The online pipeline uses the dominant radio configuration observed in
    # the labelled sessions. It can be overridden at runtime when required.
    radio_pairs = [
        (int(session["pci"]), int(session["ibar"]))
        for label in labels
        for session in processed_sessions[label]
    ]
    (deployment_pci, deployment_ibar), _ = Counter(radio_pairs).most_common(1)[0]
    deployment_checkpoint = args.output / "model_fisher_logreg.pt"
    deployment_joblib = args.output / "model_fisher_logreg.joblib"
    deployment_metadata = {
        "classes": labels,
        "selected_resource_elements": selected_resource_elements,
        "radio_configuration": {
            "pci": deployment_pci,
            "ibar": deployment_ibar,
        },
        "empty_labels": [label for label in labels if label.casefold() == "empty"],
    }
    joblib.dump(
        {
            "pipeline": model,
            **deployment_metadata,
        },
        deployment_joblib,
    )
    export_sklearn_pipeline_as_torch(
        model=model,
        output_path=deployment_checkpoint,
        classes=labels,
        selected_resource_elements=selected_resource_elements,
        radio_configuration=deployment_metadata["radio_configuration"],
        empty_labels=deployment_metadata["empty_labels"],
    )

    # --------------------------------------------------------
    # Save detection summary
    # --------------------------------------------------------

    with (
        args.output
        / "detection_summary.json"
    ).open(
        "w"
    ) as file:

        json.dump(
            detection_summary,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Save complete metrics
    # --------------------------------------------------------

    metrics = {
        "labels":
            labels,

        "number_of_labels":
            len(labels),

        "signal_statistics":
            signal_stats,

        "split": {
            "train_fraction":
                TRAIN_FRACTION,
            "validation_fraction":
                VALIDATION_FRACTION,
            "test_fraction":
                (
                    1.0
                    - TRAIN_FRACTION
                    - VALIDATION_FRACTION
                ),
            "strategy":
                "chronological per label",
        },

        "features": {
            "representation":
                "snapshot-normalized H magnitude in dB",
            "fisher_selected_REs":
                top_k,
            "spectral_features":
                9,
            "total_features":
                top_k + 9,
            "fisher_selection":
                "training data only",
        },

        "model": {
            "type":
                "LogisticRegression",
            "scaling":
                "StandardScaler",
            "class_weight":
                "balanced",
            "max_iter":
                5000,
            "deployment_checkpoint":
                str(deployment_checkpoint),
            "radio_configuration": {
                "pci": deployment_pci,
                "ibar": deployment_ibar,
            },
        },

        "exploratory_pca": {
            "PC1":
                float(
                    explained_variance[0]
                ),
            "PC2":
                float(
                    explained_variance[1]
                ),
        },

        "validation": {
            "accuracy":
                float(
                    validation_accuracy
                ),
            "balanced_accuracy":
                float(
                    validation_balanced_accuracy
                ),
            "macro_f1":
                float(
                    validation_macro_f1
                ),
        },

        "test": {
            "accuracy":
                float(
                    test_accuracy
                ),
            "balanced_accuracy":
                float(
                    test_balanced_accuracy
                ),
            "macro_f1":
                float(
                    test_macro_f1
                ),
            "confusion_matrix":
                confusion.tolist(),
        },
    }

    with (
        args.output
        / "metrics.json"
    ).open(
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Results saved to: "
        f"{args.output}"
    )

    print()
    print(
        "Main files:"
    )

    main_files = [
        "01_random_H_magnitude_snapshots.pdf",
        "02_mean_H_magnitude_profile.pdf",
        "03_normalized_H_profile.pdf",
        "04_global_H_level_temporal.pdf",
        "05_fisher_RE_heatmap.pdf",
        "06_PCA_normalized_H_magnitude.pdf",
        "07_confusion_matrix.pdf",
        "selected_resource_elements.csv",
        "classification_report.txt",
        "confusion_matrix.csv",
        "detection_summary.json",
        "metrics.json",
        "model_fisher_logreg.pt",
        "model_fisher_logreg.joblib",
    ]

    for filename in main_files:
        print(
            f"  {filename}"
        )


if __name__ == "__main__":
    main()
