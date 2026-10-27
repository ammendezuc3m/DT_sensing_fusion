#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_sensing_dataset import force_hssb_to_n_240_4, read_matlab_dataset, assign_fold


EPS = 1e-6
LABEL_ORDER = ["empty", "P1", "P2", "P3", "P4"]


def amp_db(h):
    return (20.0 * np.log10(np.abs(h) + EPS)).astype(np.float32)


def clean_complex(h):
    h = np.asarray(h, dtype=np.complex64)
    real = np.nan_to_num(h.real, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    imag = np.nan_to_num(h.imag, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)


def robust_limits(values, low=2, high=98, symmetric=False):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    if symmetric:
        m = float(np.percentile(np.abs(values), high))
        m = max(m, 1e-3)
        return -m, m
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def load_sessions(raw_dir):
    raw_dir = Path(raw_dir)
    session_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    empty_dirs = []
    rows = []

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

    for sdir in session_dirs:
        meta_path = sdir / "metadata.json"
        mat_path = sdir / "session_data.mat"
        if not meta_path.exists() or not mat_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        label = meta["label"]
        orientation = meta.get("orientation", "unknown")
        empty_idx = empty_index_by_session.get(sdir.name)
        fold_id = assign_fold(label, orientation, empty_idx)

        h = clean_complex(force_hssb_to_n_240_4(read_matlab_dataset(mat_path, "hSSB")))
        A = amp_db(h)

        phase = np.unwrap(np.angle(h), axis=1).astype(np.float32)
        phase = phase - np.median(phase, axis=1, keepdims=True)
        phase_diff = np.angle(h[:, 1:, :] * np.conj(h[:, :-1, :])).astype(np.float32)

        amp_q25, amp_q50, amp_q75 = np.percentile(A, [25, 50, 75], axis=(0, 2))
        phase_q25, phase_q50, phase_q75 = np.percentile(phase, [25, 50, 75], axis=(0, 2))
        phase_diff_q25, phase_diff_q50, phase_diff_q75 = np.percentile(
            phase_diff,
            [25, 50, 75],
            axis=(0, 2),
        )

        rows.append(
            {
                "session_dir": sdir.name,
                "session_id": meta["session_id"],
                "label": label,
                "orientation": orientation,
                "fold_id": int(fold_id),
                "n_captures": int(A.shape[0]),
                "accepted_rate_hz": meta.get("accepted_rate_hz"),
                "A": A,
                "median_power": np.median(A, axis=0).astype(np.float32),
                "mean_power": np.mean(A, axis=0).astype(np.float32),
                "std_power": np.std(A, axis=0).astype(np.float32),
                "amp_q25": amp_q25.astype(np.float32),
                "amp_q50": amp_q50.astype(np.float32),
                "amp_q75": amp_q75.astype(np.float32),
                "phase_q25": phase_q25.astype(np.float32),
                "phase_q50": phase_q50.astype(np.float32),
                "phase_q75": phase_q75.astype(np.float32),
                "phase_diff_q25": phase_diff_q25.astype(np.float32),
                "phase_diff_q50": phase_diff_q50.astype(np.float32),
                "phase_diff_q75": phase_diff_q75.astype(np.float32),
            }
        )

    return rows


def build_fold_empty_baselines(sessions):
    baselines = {}
    for item in sessions:
        if item["label"] == "empty":
            baselines[item["fold_id"]] = {
                "session_id": item["session_id"],
                "median_power": item["median_power"],
                "std_power": np.maximum(item["std_power"], 0.5),
            }
    return baselines


def add_derived_maps(sessions, baselines):
    for item in sessions:
        baseline = baselines[item["fold_id"]]
        delta = item["median_power"] - baseline["median_power"]
        zdelta = delta / baseline["std_power"]
        attenuation = np.maximum(-delta, 0.0)

        item["delta"] = delta.astype(np.float32)
        item["zdelta"] = zdelta.astype(np.float32)
        item["attenuation"] = attenuation.astype(np.float32)
        item["baseline_empty_session_id"] = baseline["session_id"]


def draw_matrix(ax, matrix, title, cmap, vmin, vmax, ylabel=True):
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[0.5, matrix.shape[1] + 0.5, 0.5, matrix.shape[0] + 0.5],
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("SSB OFDM symbol")
    if ylabel:
        ax.set_ylabel("SSB subcarrier")
    ax.set_xticks(np.arange(1, matrix.shape[1] + 1))
    return im


def save_matrix(path, matrix, title, cmap, vmin, vmax, cbar_label):
    fig, ax = plt.subplots(figsize=(5.5, 7.0))
    im = draw_matrix(ax, matrix, title, cmap, vmin, vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_sample_frames(path, item, power_limits, n_frames=6):
    A = item["A"]
    indices = np.linspace(0, A.shape[0] - 1, num=min(n_frames, A.shape[0]), dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    last_im = None
    for ax, idx in zip(axes, indices):
        last_im = draw_matrix(
            ax,
            A[idx],
            f"capture {idx}",
            cmap="turbo",
            vmin=power_limits[0],
            vmax=power_limits[1],
            ylabel=ax in axes[::3],
        )
    for ax in axes[len(indices):]:
        ax.axis("off")
    fig.suptitle(
        f"{item['label']} fold {item['fold_id']} | {item['orientation']} | {item['session_id']}",
        y=0.995,
        fontsize=11,
    )
    cbar = fig.colorbar(last_im, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Power |H| [dB]")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_session_plots(sessions, out_dir, limits):
    session_dir = out_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)

    for item in sessions:
        prefix = f"fold{item['fold_id']}_{item['label']}_{item['session_id']}"
        title_base = f"{item['label']} fold {item['fold_id']} | {item['orientation']}"

        save_matrix(
            session_dir / f"{prefix}_median_power_db.png",
            item["median_power"],
            f"{title_base}\nmedian SSB channel power",
            "turbo",
            *limits["power"],
            cbar_label="Power |H| [dB]",
        )
        save_matrix(
            session_dir / f"{prefix}_delta_vs_empty_db.png",
            item["delta"],
            f"{title_base}\ndelta vs fold empty",
            "coolwarm",
            *limits["delta"],
            cbar_label="Delta power [dB]",
        )
        save_matrix(
            session_dir / f"{prefix}_attenuation_vs_empty_db.png",
            item["attenuation"],
            f"{title_base}\nyellow = more attenuated / blocked",
            "viridis",
            *limits["attenuation"],
            cbar_label="Attenuation vs empty [dB]",
        )
        save_matrix(
            session_dir / f"{prefix}_temporal_std_db.png",
            item["std_power"],
            f"{title_base}\ntemporal std over captures",
            "magma",
            *limits["std"],
            cbar_label="Std [dB]",
        )
        save_sample_frames(
            session_dir / f"{prefix}_sample_power_frames.png",
            item,
            limits["power"],
        )


def save_label_fold_montages(sessions, out_dir, limits):
    label_dir = out_dir / "label_fold_comparisons"
    label_dir.mkdir(parents=True, exist_ok=True)

    by_label = {label: [] for label in LABEL_ORDER}
    for item in sessions:
        by_label.setdefault(item["label"], []).append(item)

    for label, items in by_label.items():
        if not items:
            continue
        items = sorted(items, key=lambda x: (x["fold_id"], x["session_id"]))

        for map_name, cmap, cbar_label, title_suffix in [
            ("median_power", "turbo", "Power |H| [dB]", "median power"),
            ("delta", "coolwarm", "Delta power [dB]", "delta vs empty"),
            ("attenuation", "viridis", "Attenuation [dB]", "yellow = more blocked"),
            ("std_power", "magma", "Std [dB]", "temporal variability"),
        ]:
            n = len(items)
            fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 6.6), sharey=True)
            if n == 1:
                axes = [axes]
            last_im = None
            for ax, item in zip(axes, items):
                last_im = draw_matrix(
                    ax,
                    item[map_name],
                    f"fold {item['fold_id']}\n{item['orientation']}",
                    cmap=cmap,
                    vmin=limits[map_name if map_name in limits else "power"][0],
                    vmax=limits[map_name if map_name in limits else "power"][1],
                    ylabel=ax is axes[0],
                )
            fig.suptitle(f"{label}: {title_suffix}", y=0.995, fontsize=12)
            cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02)
            cbar.set_label(cbar_label)
            fig.savefig(label_dir / f"{label}_{map_name}_by_fold.png", dpi=180, bbox_inches="tight")
            plt.close(fig)


def save_global_attenuation_grid(sessions, out_dir, limits):
    labels = [label for label in LABEL_ORDER if any(s["label"] == label for s in sessions)]
    folds = sorted({s["fold_id"] for s in sessions})
    fig, axes = plt.subplots(
        len(labels),
        len(folds),
        figsize=(4.0 * len(folds), 4.5 * len(labels)),
        sharex=True,
        sharey=True,
    )
    if len(labels) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(folds) == 1:
        axes = np.expand_dims(axes, axis=1)

    last_im = None
    for i, label in enumerate(labels):
        for j, fold in enumerate(folds):
            ax = axes[i, j]
            matches = [s for s in sessions if s["label"] == label and s["fold_id"] == fold]
            if not matches:
                ax.axis("off")
                continue
            item = sorted(matches, key=lambda x: x["session_id"])[0]
            last_im = draw_matrix(
                ax,
                item["attenuation"],
                f"{label} | fold {fold}\n{item['orientation']}",
                cmap="viridis",
                vmin=limits["attenuation"][0],
                vmax=limits["attenuation"][1],
                ylabel=j == 0,
            )
    fig.suptitle("Attenuation vs fold empty: yellow = more blocked", y=0.995, fontsize=13)
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.015)
    cbar.set_label("Attenuation vs empty [dB]")
    fig.savefig(out_dir / "all_labels_attenuation_by_fold.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_subcarrier_profiles(sessions, out_dir):
    profile_dir = out_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(LABEL_ORDER), 1, figsize=(11, 2.3 * len(LABEL_ORDER)), sharex=True)
    for ax, label in zip(axes, LABEL_ORDER):
        items = sorted([s for s in sessions if s["label"] == label], key=lambda x: x["fold_id"])
        for item in items:
            profile = item["attenuation"].mean(axis=1)
            ax.plot(
                np.arange(1, profile.size + 1),
                profile,
                linewidth=1.5,
                label=f"fold {item['fold_id']} {item['orientation']}",
            )
        ax.set_title(f"{label}: mean attenuation per subcarrier")
        ax.set_ylabel("dB")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("SSB subcarrier")
    fig.tight_layout()
    fig.savefig(profile_dir / "attenuation_profiles_by_label.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for label in LABEL_ORDER:
        items = [s for s in sessions if s["label"] == label]
        if not items:
            continue
        profile = np.mean([s["attenuation"].mean(axis=1) for s in items], axis=0)
        ax.plot(np.arange(1, profile.size + 1), profile, linewidth=2, label=label)
    ax.set_title("Average attenuation profile by label")
    ax.set_xlabel("SSB subcarrier")
    ax.set_ylabel("Attenuation vs empty [dB]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(profile_dir / "average_attenuation_profile_by_label.png", dpi=180)
    plt.close(fig)


def plot_quartile_band(ax, x, q25, q50, q75, label, color=None):
    line = ax.plot(x, q50, linewidth=1.8, label=label, color=color)[0]
    band_color = line.get_color()
    ax.fill_between(x, q25, q75, color=band_color, alpha=0.18, linewidth=0)


def save_quartile_profiles(sessions, out_dir):
    quartile_dir = out_dir / "quartile_profiles"
    quartile_dir.mkdir(parents=True, exist_ok=True)

    # Per-label figures comparing folds.
    for label in LABEL_ORDER:
        items = sorted([s for s in sessions if s["label"] == label], key=lambda x: x["fold_id"])
        if not items:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

        for item in items:
            fold_label = f"fold {item['fold_id']} {item['orientation']}"

            x_amp = np.arange(1, item["amp_q50"].size + 1)
            plot_quartile_band(
                axes[0],
                x_amp,
                item["amp_q25"],
                item["amp_q50"],
                item["amp_q75"],
                fold_label,
            )

            x_phase = np.arange(1, item["phase_q50"].size + 1)
            plot_quartile_band(
                axes[1],
                x_phase,
                item["phase_q25"],
                item["phase_q50"],
                item["phase_q75"],
                fold_label,
            )

            x_phase_diff = np.arange(1, item["phase_diff_q50"].size + 1)
            plot_quartile_band(
                axes[2],
                x_phase_diff,
                item["phase_diff_q25"],
                item["phase_diff_q50"],
                item["phase_diff_q75"],
                fold_label,
            )

        axes[0].set_title(f"{label}: amplitude quartiles by subcarrier")
        axes[0].set_ylabel("|H| [dB]")
        axes[1].set_title(f"{label}: centered unwrapped phase quartiles by subcarrier")
        axes[1].set_ylabel("Phase [rad]")
        axes[2].set_title(f"{label}: adjacent-subcarrier differential phase quartiles")
        axes[2].set_ylabel("Diff phase [rad]")
        axes[2].set_xlabel("SSB subcarrier")

        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.savefig(quartile_dir / f"{label}_amplitude_phase_quartiles_by_fold.png", dpi=180)
        plt.close(fig)

    # Global mean-of-quartiles by label. This is not a statistical quartile over
    # all raw samples; it is a compact view of the typical per-session quartile curves.
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    for label in LABEL_ORDER:
        items = [s for s in sessions if s["label"] == label]
        if not items:
            continue

        amp_q25 = np.mean([s["amp_q25"] for s in items], axis=0)
        amp_q50 = np.mean([s["amp_q50"] for s in items], axis=0)
        amp_q75 = np.mean([s["amp_q75"] for s in items], axis=0)
        phase_q25 = np.mean([s["phase_q25"] for s in items], axis=0)
        phase_q50 = np.mean([s["phase_q50"] for s in items], axis=0)
        phase_q75 = np.mean([s["phase_q75"] for s in items], axis=0)
        phase_diff_q25 = np.mean([s["phase_diff_q25"] for s in items], axis=0)
        phase_diff_q50 = np.mean([s["phase_diff_q50"] for s in items], axis=0)
        phase_diff_q75 = np.mean([s["phase_diff_q75"] for s in items], axis=0)

        plot_quartile_band(axes[0], np.arange(1, amp_q50.size + 1), amp_q25, amp_q50, amp_q75, label)
        plot_quartile_band(
            axes[1],
            np.arange(1, phase_q50.size + 1),
            phase_q25,
            phase_q50,
            phase_q75,
            label,
        )
        plot_quartile_band(
            axes[2],
            np.arange(1, phase_diff_q50.size + 1),
            phase_diff_q25,
            phase_diff_q50,
            phase_diff_q75,
            label,
        )

    axes[0].set_title("Average per-session amplitude quartiles by label")
    axes[0].set_ylabel("|H| [dB]")
    axes[1].set_title("Average per-session centered unwrapped phase quartiles by label")
    axes[1].set_ylabel("Phase [rad]")
    axes[2].set_title("Average per-session adjacent-subcarrier differential phase quartiles by label")
    axes[2].set_ylabel("Diff phase [rad]")
    axes[2].set_xlabel("SSB subcarrier")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(quartile_dir / "all_labels_amplitude_phase_quartiles.png", dpi=180)
    plt.close(fig)

    rows = []
    for item in sessions:
        rows.append(
            {
                "session_id": item["session_id"],
                "label": item["label"],
                "orientation": item["orientation"],
                "fold_id": item["fold_id"],
                "amp_q25_global_db": float(np.median(item["amp_q25"])),
                "amp_q50_global_db": float(np.median(item["amp_q50"])),
                "amp_q75_global_db": float(np.median(item["amp_q75"])),
                "phase_q25_global_rad": float(np.median(item["phase_q25"])),
                "phase_q50_global_rad": float(np.median(item["phase_q50"])),
                "phase_q75_global_rad": float(np.median(item["phase_q75"])),
                "phase_diff_q25_global_rad": float(np.median(item["phase_diff_q25"])),
                "phase_diff_q50_global_rad": float(np.median(item["phase_diff_q50"])),
                "phase_diff_q75_global_rad": float(np.median(item["phase_diff_q75"])),
            }
        )
    pd.DataFrame(rows).sort_values(["fold_id", "label", "session_id"]).to_csv(
        quartile_dir / "quartile_profile_summary.csv",
        index=False,
    )


def save_quality_summary(sessions, out_dir):
    rows = []
    for item in sessions:
        rows.append(
            {
                "session_id": item["session_id"],
                "session_dir": item["session_dir"],
                "label": item["label"],
                "orientation": item["orientation"],
                "fold_id": item["fold_id"],
                "n_captures": item["n_captures"],
                "accepted_rate_hz": item["accepted_rate_hz"],
                "median_power_db": float(np.median(item["median_power"])),
                "mean_attenuation_db": float(np.mean(item["attenuation"])),
                "max_attenuation_db": float(np.max(item["attenuation"])),
                "mean_temporal_std_db": float(np.mean(item["std_power"])),
                "baseline_empty_session_id": item["baseline_empty_session_id"],
            }
        )
    df = pd.DataFrame(rows).sort_values(["fold_id", "label", "session_id"])
    df.to_csv(out_dir / "session_visual_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df.pivot_table(
        index="label",
        columns="fold_id",
        values="mean_attenuation_db",
        aggfunc="mean",
    ).reindex(LABEL_ORDER)
    im = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"fold {c}" for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")
    ax.set_title("Mean attenuation by label and fold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean attenuation [dB]")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_attenuation_by_label_fold.png", dpi=180)
    plt.close(fig)


def compute_limits(sessions):
    power = np.concatenate([s["median_power"].reshape(-1) for s in sessions])
    delta = np.concatenate([s["delta"].reshape(-1) for s in sessions])
    attenuation = np.concatenate([s["attenuation"].reshape(-1) for s in sessions])
    std = np.concatenate([s["std_power"].reshape(-1) for s in sessions])
    return {
        "power": robust_limits(power, 2, 98),
        "median_power": robust_limits(power, 2, 98),
        "delta": robust_limits(delta, 2, 98, symmetric=True),
        "attenuation": (0.0, max(1.0, float(np.percentile(attenuation, 98)))),
        "std": (0.0, max(1.0, float(np.percentile(std, 98)))),
        "std_power": (0.0, max(1.0, float(np.percentile(std, 98)))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="reports/ssb_data_exploration")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions(args.raw_dir)
    baselines = build_fold_empty_baselines(sessions)
    add_derived_maps(sessions, baselines)
    limits = compute_limits(sessions)

    save_quality_summary(sessions, out_dir)
    save_session_plots(sessions, out_dir, limits)
    save_label_fold_montages(sessions, out_dir, limits)
    save_global_attenuation_grid(sessions, out_dir, limits)
    save_subcarrier_profiles(sessions, out_dir)
    save_quartile_profiles(sessions, out_dir)

    manifest = {
        "raw_dir": args.raw_dir,
        "out_dir": str(out_dir),
        "n_sessions": len(sessions),
        "limits": {k: [float(v[0]), float(v[1])] for k, v in limits.items()},
        "notes": [
            "The saved hSSB data has shape N x 240 x 4, so these figures show the estimated SSB channel, not the original 360 x 6 rxGrid used in MATLAB live plotting.",
            "For attenuation maps, yellow means more attenuated relative to the empty baseline from the same fold.",
            "Delta maps use red/blue: positive means stronger than empty, negative means weaker than empty.",
            "Quartile phase profiles use unwrapped phase centered per capture/symbol to reduce common phase-offset drift.",
            "Differential phase profiles use angle(H[k+1] * conj(H[k])) and can be more stable than absolute phase.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Saved exploration figures to {out_dir}")
    print(f"Sessions: {len(sessions)}")
    print(f"Summary CSV: {out_dir / 'session_visual_summary.csv'}")


if __name__ == "__main__":
    main()
