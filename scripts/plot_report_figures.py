#!/usr/bin/env python3
"""Generate comparison plots for MORL vs Baseline W&B runs for the interim report."""

import os
from pathlib import Path

# Load WANDB_API_KEY from .env if not already set
if not os.environ.get("WANDB_API_KEY"):
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("WANDB_API_KEY="):
                os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip()
                break

import wandb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# --- Config ---
ENTITY = "abhinayjain-uoa"
PROJECT = "SQL-R1-MORL"
MORL_RUN_ID = "9jzpovht"
BASELINE_RUN_ID = "avl8oqv4"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

PLOT_STYLE = {
    "figure.figsize": (8, 4.5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
}


def fetch_history(api, run_id):
    """Fetch full history for a run."""
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    history = run.history(samples=10000, pandas=True)
    return run, history


def smooth(series, window=5):
    """Simple moving average smoothing."""
    return series.rolling(window=window, min_periods=1).mean()


def plot_metric(ax, morl_hist, base_hist, metric, ylabel, title, use_smooth=True):
    """Plot a single metric for both runs on shared axes."""
    for hist, label, color in [
        (morl_hist, "MORL (PC-Grad)", "#2196F3"),
        (base_hist, "Baseline", "#FF9800"),
    ]:
        if metric not in hist.columns:
            continue
        series = hist[metric].dropna()
        if len(series) == 0:
            continue
        steps = hist.loc[series.index, "_step"] if "_step" in hist.columns else series.index
        ax.plot(steps, series, alpha=0.25, color=color, linewidth=0.8)
        if use_smooth and len(series) > 3:
            ax.plot(steps, smooth(series), color=color, label=label)
        else:
            ax.plot(steps, series, color=color, label=label)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def main():
    plt.rcParams.update(PLOT_STYLE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    print("Fetching MORL run...")
    morl_run, morl_hist = fetch_history(api, MORL_RUN_ID)
    print(f"  {morl_run.name}: {len(morl_hist)} steps")

    print("Fetching Baseline run...")
    base_run, base_hist = fetch_history(api, BASELINE_RUN_ID)
    print(f"  {base_run.name}: {len(base_hist)} steps")

    # --- Fig 1: Match Ratio ---
    fig, ax = plt.subplots()
    plot_metric(ax, morl_hist, base_hist, "reward/match_ratio",
                "Match Ratio", "Fig 1: Match Ratio (Execution Accuracy)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_match_ratio.png", dpi=150)
    plt.close(fig)
    print("Saved fig_match_ratio.png")

    # --- Fig 2: Response Length ---
    fig, ax = plt.subplots()
    plot_metric(ax, morl_hist, base_hist, "response_length/mean",
                "Mean Response Length (tokens)", "Fig 2: Mean Response Length")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_response_length.png", dpi=150)
    plt.close(fig)
    print("Saved fig_response_length.png")

    # --- Fig 3: KL Divergence ---
    fig, ax = plt.subplots()
    plot_metric(ax, morl_hist, base_hist, "actor/kl_loss",
                "KL Loss", "Fig 3: KL Divergence")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_kl_divergence.png", dpi=150)
    plt.close(fig)
    print("Saved fig_kl_divergence.png")

    # --- Fig 4: Entropy ---
    fig, ax = plt.subplots()
    plot_metric(ax, morl_hist, base_hist, "actor/entropy_loss",
                "Entropy Loss", "Fig 4: Entropy")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_entropy.png", dpi=150)
    plt.close(fig)
    print("Saved fig_entropy.png")

    # --- Fig 5: Gradient Norm ---
    fig, ax = plt.subplots()
    plot_metric(ax, morl_hist, base_hist, "actor/grad_norm",
                "Gradient Norm", "Fig 5: Gradient Norm")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_grad_norm.png", dpi=150)
    plt.close(fig)
    print("Saved fig_grad_norm.png")

    # --- Fig 6: Dual-Objective Rewards ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    plot_metric(ax1, morl_hist, base_hist, "reward/accuracy_mean",
                "Mean Accuracy Reward", "Fig 6a: Accuracy Reward")
    plot_metric(ax2, morl_hist, base_hist, "reward/efficiency_mean",
                "Mean Efficiency Reward", "Fig 6b: Efficiency Reward")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_dual_rewards.png", dpi=150)
    plt.close(fig)
    print("Saved fig_dual_rewards.png")

    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
