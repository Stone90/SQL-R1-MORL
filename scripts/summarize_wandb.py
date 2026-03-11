#!/usr/bin/env python3
"""Fetch W&B run data and generate markdown summary reports with health checks."""

import argparse
import json
import os
import sys
from datetime import datetime
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
import pandas as pd
import numpy as np


# ── Health check thresholds ──
ENTROPY_COLLAPSE_THRESHOLD = 0.05   # entropy below this = collapsed
ENTROPY_WARNING_THRESHOLD = 0.08    # entropy below this = warning
KL_DANGER_THRESHOLD = 10.0          # KL growth factor above this = danger
KL_WARNING_THRESHOLD = 5.0          # KL growth factor above this = warning
MATCH_RATIO_DECLINE_THRESHOLD = -0.3  # match ratio drop > this = danger
RESPONSE_LENGTH_GROWTH_THRESHOLD = 0.5  # >50% growth = verbosity hacking


def fetch_run(api, entity, project, run_id):
    """Fetch a W&B run and its full history."""
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history(samples=10000, pandas=True)
    return run, history


def safe_get(history, col, default=None):
    """Get a column from history, returning default if missing."""
    if col in history.columns:
        return history[col]
    return default


def phase_mean(series, start, end):
    """Mean of a series between step indices."""
    if series is None:
        return None
    s = series.iloc[start:end]
    s = s.dropna()
    return s.mean() if len(s) > 0 else None


def rolling_mean(series, window=10):
    """Compute rolling mean of a series."""
    if series is None:
        return None
    return series.dropna().rolling(window=window, min_periods=1).mean()


def trend_direction(series, window=10):
    """Determine trend direction: 'rising', 'falling', or 'stable'."""
    if series is None or len(series.dropna()) < window * 2:
        return "insufficient data"
    vals = series.dropna()
    first_w = vals.iloc[:window].mean()
    last_w = vals.iloc[-window:].mean()
    if first_w == 0:
        return "stable" if last_w == 0 else "rising"
    pct_change = (last_w - first_w) / abs(first_w)
    if pct_change > 0.1:
        return "rising"
    elif pct_change < -0.1:
        return "falling"
    return "stable"


def fmt(val, decimals=4):
    """Format a number, handling None."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if abs(val) < 0.001 and val != 0:
            return f"{val:.6f}"
        return f"{val:.{decimals}f}"
    return str(val)


def fmt_pct(val, decimals=1):
    """Format as percentage."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}%"


def health_check(run, history):
    """Run health checks on a training run. Returns list of (severity, message) tuples."""
    issues = []  # list of (severity, message) where severity is 'CRITICAL', 'WARNING', 'INFO'
    n_steps = len(history)

    if n_steps < 5:
        issues.append(("INFO", f"Only {n_steps} steps logged — too early for health assessment"))
        return issues

    # 1. Entropy collapse check
    entropy = safe_get(history, "actor/entropy_loss")
    if entropy is not None:
        ent_vals = entropy.dropna()
        if len(ent_vals) >= 10:
            first5 = ent_vals.iloc[:5].mean()
            last5 = ent_vals.iloc[-5:].mean()
            if first5 > 0:
                collapse_pct = (first5 - last5) / first5
                if last5 < ENTROPY_COLLAPSE_THRESHOLD:
                    issues.append(("CRITICAL", f"Entropy collapsed to {last5:.4f} (down {collapse_pct*100:.0f}% from {first5:.4f}). Policy is near-deterministic — no meaningful exploration possible."))
                elif last5 < ENTROPY_WARNING_THRESHOLD:
                    issues.append(("WARNING", f"Entropy dropping to {last5:.4f} (down {collapse_pct*100:.0f}% from {first5:.4f}). At risk of collapse."))

    # 2. KL divergence growth
    kl_loss = safe_get(history, "actor/kl_loss")
    if kl_loss is not None:
        kl_vals = kl_loss.dropna()
        if len(kl_vals) >= 20:
            first10 = kl_vals.iloc[:10].mean()
            last10 = kl_vals.iloc[-10:].mean()
            if first10 > 0:
                kl_growth = last10 / first10
                if kl_growth > KL_DANGER_THRESHOLD:
                    issues.append(("CRITICAL", f"KL divergence grew {kl_growth:.1f}x ({first10:.6f} → {last10:.6f}). Policy has drifted far from reference. Consider increasing kl_loss_coef."))
                elif kl_growth > KL_WARNING_THRESHOLD:
                    issues.append(("WARNING", f"KL divergence grew {kl_growth:.1f}x. Monitor closely."))

    # 3. Match ratio decline
    match_ratio = safe_get(history, "reward/match_ratio")
    if match_ratio is not None:
        mr_vals = match_ratio.dropna()
        if len(mr_vals) >= 10:
            peak = mr_vals.max()
            last5 = mr_vals.iloc[-5:].mean()
            if peak > 0:
                decline = (last5 - peak) / peak
                if decline < MATCH_RATIO_DECLINE_THRESHOLD:
                    issues.append(("CRITICAL", f"Match ratio dropped from peak {peak:.3f} to {last5:.3f} ({decline*100:.0f}%). Model is forgetting."))
                # Check trend
                if len(mr_vals) >= 20:
                    recent_trend = trend_direction(match_ratio, window=10)
                    if recent_trend == "falling":
                        issues.append(("WARNING", f"Match ratio trend is falling (recent 10-step avg declining)."))

    # 4. Response length growth (verbosity hacking)
    resp_mean = safe_get(history, "response_length/mean")
    if resp_mean is not None:
        rm_vals = resp_mean.dropna()
        if len(rm_vals) >= 20:
            first10 = rm_vals.iloc[:10].mean()
            last10 = rm_vals.iloc[-10:].mean()
            if first10 > 0:
                growth = (last10 - first10) / first10
                if growth > RESPONSE_LENGTH_GROWTH_THRESHOLD:
                    issues.append(("WARNING", f"Response length grew {growth*100:.0f}% ({first10:.0f} → {last10:.0f} tokens). Possible verbosity/reward hacking."))

    # 5. Checkpoint/validation config check
    config = run.config

    def cfg(key, default=None):
        parts = key.split(".")
        obj = config
        for p in parts:
            if isinstance(obj, dict) and p in obj:
                obj = obj[p]
            else:
                return default
        return obj

    total_steps = cfg("trainer.total_training_steps")
    save_freq = cfg("trainer.save_freq")
    test_freq = cfg("trainer.test_freq")

    if total_steps and save_freq and save_freq > 0:
        if save_freq > total_steps:
            issues.append(("CRITICAL", f"save_freq={save_freq} > total_steps={total_steps}. NO checkpoints will be saved!"))
        elif save_freq > total_steps // 2:
            issues.append(("WARNING", f"save_freq={save_freq} is very infrequent for {total_steps} total steps."))

    if total_steps and test_freq and test_freq > 0:
        if test_freq > total_steps:
            issues.append(("CRITICAL", f"test_freq={test_freq} > total_steps={total_steps}. NO validation will run during training!"))

    # 6. Gradient norm spikes
    grad_norm = safe_get(history, "actor/grad_norm")
    if grad_norm is not None:
        gn_vals = grad_norm.dropna()
        if len(gn_vals) >= 10:
            mean_gn = gn_vals.mean()
            max_gn = gn_vals.max()
            if max_gn > mean_gn * 10 and mean_gn > 0:
                issues.append(("WARNING", f"Gradient norm spike detected: max={max_gn:.2f} vs mean={mean_gn:.2f} ({max_gn/mean_gn:.0f}x)."))

    if not issues:
        issues.append(("INFO", "All health checks passed. Training appears healthy."))

    return issues


def format_health_check(issues):
    """Format health check results as markdown."""
    lines = []
    lines.append("## Health Check")
    lines.append("")

    severity_icons = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🟢"}
    has_critical = any(s == "CRITICAL" for s, _ in issues)
    has_warning = any(s == "WARNING" for s, _ in issues)

    if has_critical:
        lines.append("**RECOMMENDATION: STOP RUN** — Critical issues detected.")
    elif has_warning:
        lines.append("**RECOMMENDATION: MONITOR CLOSELY** — Warnings detected.")
    else:
        lines.append("**STATUS: HEALTHY**")
    lines.append("")

    for severity, msg in issues:
        icon = severity_icons.get(severity, "⚪")
        lines.append(f"- {icon} **{severity}**: {msg}")
    lines.append("")

    return "\n".join(lines)


def generate_report(run, history, entity, project):
    """Generate a markdown report for a single run."""
    config = run.config
    summary = run.summary
    n_steps = len(history)

    # Determine if PC-Grad is enabled
    pc_grad = False
    for key_path in [
        "actor_rollout_ref.actor.enable_pc_grad",
        "enable_pc_grad",
    ]:
        parts = key_path.split(".")
        obj = config
        for p in parts:
            if isinstance(obj, dict) and p in obj:
                obj = obj[p]
            else:
                obj = None
                break
        if obj is not None:
            pc_grad = bool(obj)
            break

    # Extract config values with nested key support
    def cfg(key, default="N/A"):
        parts = key.split(".")
        obj = config
        for p in parts:
            if isinstance(obj, dict) and p in obj:
                obj = obj[p]
            else:
                return default
        return obj

    # Run metadata
    run_name = run.name or run.id
    status = run.state
    created = run.created_at
    runtime_s = run.summary.get("_runtime", None)
    runtime_h = f"{runtime_s / 3600:.2f} hours" if runtime_s else "N/A"

    # Key metric series
    reward_mean = safe_get(history, "reward/mean")
    match_ratio = safe_get(history, "reward/match_ratio")
    acc_mean = safe_get(history, "reward/accuracy_mean")
    eff_mean = safe_get(history, "reward/efficiency_mean")
    pg_loss = safe_get(history, "actor/pg_loss")
    clipfrac = safe_get(history, "actor/pg_clipfrac")
    entropy = safe_get(history, "actor/entropy_loss")
    grad_norm = safe_get(history, "actor/grad_norm")
    kl_loss = safe_get(history, "actor/kl_loss")
    kl_coef = safe_get(history, "actor/kl_coef")
    ppo_kl = safe_get(history, "actor/ppo_kl")
    resp_mean = safe_get(history, "response_length/mean")
    resp_max = safe_get(history, "response_length/max")
    resp_min = safe_get(history, "response_length/min")
    clip_ratio = safe_get(history, "response_length/clip_ratio")
    step_time = safe_get(history, "timing_s/step")
    gen_time = safe_get(history, "timing_s/gen")

    # PC-Grad metrics
    fast_path = safe_get(history, "actor/pc_grad_fast_path")
    conflict_rate = safe_get(history, "actor/pc_grad_conflict_rate")
    pg_loss_acc = safe_get(history, "actor/pg_loss_acc")
    pg_loss_eff = safe_get(history, "actor/pg_loss_eff")
    entropy_acc = safe_get(history, "actor/entropy_loss_acc")
    entropy_eff = safe_get(history, "actor/entropy_loss_eff")

    # Timing components
    ref_time = safe_get(history, "timing_s/ref")
    adv_time = safe_get(history, "timing_s/adv")
    actor_time = safe_get(history, "timing_s/update_actor")

    # Validation scores - scan all columns
    val_cols = [c for c in history.columns if c.startswith("val/")]

    # Phase boundaries
    early_end = min(50, n_steps)
    mid_end = min(150, n_steps)

    # Compute learning signal ratio
    if reward_mean is not None:
        nonzero_steps = (reward_mean.abs() > 1e-9).sum()
        learning_signal_ratio = f"{nonzero_steps}/{n_steps} ({100*nonzero_steps/n_steps:.1f}%)"
    else:
        nonzero_steps = "N/A"
        learning_signal_ratio = "N/A"

    # Final values
    def last_valid(series):
        if series is None:
            return None
        s = series.dropna()
        return s.iloc[-1] if len(s) > 0 else None

    # Build report
    lines = []

    # Title
    algo = "PC-Grad Dual-Objective GRPO" if pc_grad else "Single-Objective GRPO (No PC-Grad)"
    lines.append(f"# {'MORL' if pc_grad else 'Baseline'} Run Analysis: {algo}")
    lines.append("")
    lines.append(f"**W&B Run**: `{entity}/{project}/{run.id}`")
    lines.append(f"**Run Name**: `{run_name}`")
    lines.append(f"**Date**: {created[:10] if created else 'N/A'}")
    lines.append(f"**Status**: {status.capitalize()}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Health Check (new section, always first)
    issues = health_check(run, history)
    lines.append(format_health_check(issues))
    lines.append("---")
    lines.append("")

    # 1. Run Configuration
    lines.append("## 1. Run Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")

    model = cfg("actor_rollout_ref.model.path",
                cfg("model.path",
                    cfg("model_path", "N/A")))
    # Show just the model name if it's a path
    if isinstance(model, str) and "/" in model:
        model_display = model.split("/")[-1]
    else:
        model_display = model

    lines.append(f"| Model | {model_display} |")
    lines.append(f"| Algorithm | GRPO ({'dual-objective' if pc_grad else 'single-objective'}) |")
    lines.append(f"| PC-Grad (MORL) | **{'Enabled' if pc_grad else 'Disabled'}** |")
    lines.append(f"| Learning rate | {cfg('actor_rollout_ref.actor.optim.lr', cfg('actor.optim.lr', 'N/A'))} |")
    lines.append(f"| Train batch size | {cfg('data.train_batch_size', cfg('train_batch_size', 'N/A'))} |")
    lines.append(f"| Rollout n | {cfg('actor_rollout_ref.rollout.n', cfg('rollout.n', 'N/A'))} |")
    lines.append(f"| Temperature | {cfg('actor_rollout_ref.rollout.temperature', cfg('rollout.temperature', 'N/A'))} |")

    kl_type = cfg("algorithm.kl_ctrl.type", cfg("kl_ctrl.type", "N/A"))
    kl_coef_val = cfg("algorithm.kl_ctrl.kl_coef", cfg("kl_ctrl.kl_coef", "N/A"))
    lines.append(f"| KL loss type | {kl_type} (coef={kl_coef_val}) |")

    entropy_coeff_val = cfg("actor_rollout_ref.actor.entropy_coeff", cfg("actor.entropy_coeff", "N/A"))
    lines.append(f"| Entropy coeff | {entropy_coeff_val} |")

    lines.append(f"| FSDP size | {cfg('actor_rollout_ref.actor.fsdp_config.fsdp_size', cfg('fsdp_size', 'N/A'))} |")
    lines.append(f"| Optimizer offload | {cfg('actor_rollout_ref.actor.optim.optimizer_offload', cfg('optimizer_offload', 'N/A'))} |")
    lines.append(f"| GPU memory utilization (vLLM) | {cfg('actor_rollout_ref.rollout.gpu_memory_utilization', cfg('rollout.gpu_memory_utilization', 'N/A'))} |")
    lines.append(f"| Total training steps | {n_steps} |")
    lines.append(f"| Save freq | {cfg('trainer.save_freq', 'N/A')} |")
    lines.append(f"| Test freq | {cfg('trainer.test_freq', 'N/A')} |")
    lines.append(f"| Total runtime | {runtime_h} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. Key Results Summary
    lines.append("## 2. Key Results Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    final_match = last_valid(match_ratio)
    lines.append(f"| Final match ratio | **{fmt(final_match, 3)}** |")

    # Validation scores summary
    if val_cols:
        val_summary_parts = []
        for c in sorted(val_cols):
            s = safe_get(history, c)
            if s is not None:
                vals = s.dropna()
                if len(vals) > 0:
                    val_summary_parts.append(f"{c}: {fmt(vals.iloc[-1], 3)}")
        if val_summary_parts:
            lines.append(f"| Validation scores (latest) | {', '.join(val_summary_parts)} |")

    if reward_mean is not None:
        lines.append(f"| Mean reward (overall) | {fmt(reward_mean.mean(), 4)} |")
    if acc_mean is not None:
        lines.append(f"| Mean accuracy reward | {fmt(acc_mean.mean(), 4)} |")
    if eff_mean is not None:
        lines.append(f"| Mean efficiency reward | {fmt(eff_mean.mean(), 4)} |")

    lines.append(f"| Steps with learning signal | {learning_signal_ratio} |")

    if pc_grad and fast_path is not None:
        fp_vals = fast_path.dropna()
        if len(fp_vals) > 0:
            fp_pct = fp_vals.mean() * 100
            lines.append(f"| PC-Grad fast path usage | {fmt_pct(fp_pct)} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 3. Reward Analysis
    lines.append("## 3. Reward Analysis")
    lines.append("")

    if reward_mean is not None:
        lines.append("### 3.1 Reward Trend")
        lines.append("")
        lines.append("| Phase | Mean Reward | Trend |")
        lines.append("|-------|------------|-------|")
        lines.append(f"| Early (0-{early_end}) | {fmt(phase_mean(reward_mean, 0, early_end))} | — |")
        mid_val = phase_mean(reward_mean, early_end, mid_end)
        early_val = phase_mean(reward_mean, 0, early_end)
        mid_trend = ""
        if mid_val is not None and early_val is not None and early_val != 0:
            pct = (mid_val - early_val) / abs(early_val) * 100
            mid_trend = f"{'↑' if pct > 0 else '↓'} {abs(pct):.0f}%"
        lines.append(f"| Mid ({early_end}-{mid_end}) | {fmt(mid_val)} | {mid_trend} |")
        late_val = phase_mean(reward_mean, mid_end, n_steps)
        late_trend = ""
        if late_val is not None and mid_val is not None and mid_val != 0:
            pct = (late_val - mid_val) / abs(mid_val) * 100
            late_trend = f"{'↑' if pct > 0 else '↓'} {abs(pct):.0f}%"
        lines.append(f"| Late ({mid_end}+) | {fmt(late_val)} | {late_trend} |")
        lines.append("")

        # Reward distribution as histogram bins
        lines.append("### 3.2 Reward Distribution")
        lines.append("")
        reward_vals = reward_mean.dropna()
        if len(reward_vals) > 0:
            bins = np.linspace(reward_vals.min(), reward_vals.max(), min(11, len(reward_vals.unique()) + 1))
            if len(bins) >= 2:
                counts, edges = np.histogram(reward_vals, bins=bins)
                lines.append("| Bin range | Count | Percentage |")
                lines.append("|-----------|-------|-----------|")
                for i in range(len(counts)):
                    if counts[i] > 0:
                        lines.append(f"| [{edges[i]:.3f}, {edges[i+1]:.3f}) | {counts[i]} | {fmt_pct(100*counts[i]/len(reward_vals))} |")
            else:
                lines.append(f"All rewards = {fmt(reward_vals.iloc[0], 4)}")
        lines.append("")

    if match_ratio is not None:
        lines.append("### 3.3 Match Ratio")
        lines.append("")
        mr_vals = match_ratio.dropna()
        if mr_vals.max() > 0:
            lines.append(f"- Peak match ratio: **{fmt(mr_vals.max(), 4)}** (step {mr_vals.idxmax()})")
            lines.append(f"- Final match ratio: {fmt(mr_vals.iloc[-1], 4)}")
            lines.append(f"- Mean match ratio: {fmt(mr_vals.mean(), 4)}")
            lines.append(f"- Trend: **{trend_direction(match_ratio)}**")
        else:
            lines.append("Match ratio was **0.000** for all steps. The model never produced a correct match.")
        lines.append("")

    if acc_mean is not None:
        lines.append("### 3.4 Accuracy Reward")
        lines.append("")
        lines.append("| Phase | Mean |")
        lines.append("|-------|------|")
        lines.append(f"| Early (0-{early_end}) | {fmt(phase_mean(acc_mean, 0, early_end))} |")
        lines.append(f"| Mid ({early_end}-{mid_end}) | {fmt(phase_mean(acc_mean, early_end, mid_end))} |")
        lines.append(f"| Late ({mid_end}+) | {fmt(phase_mean(acc_mean, mid_end, n_steps))} |")
        lines.append("")

    if eff_mean is not None:
        lines.append("### 3.5 Efficiency Reward")
        lines.append("")
        eff_vals = eff_mean.dropna()
        if eff_vals.max() > 0:
            lines.append("| Phase | Mean |")
            lines.append("|-------|------|")
            lines.append(f"| Early (0-{early_end}) | {fmt(phase_mean(eff_mean, 0, early_end))} |")
            lines.append(f"| Mid ({early_end}-{mid_end}) | {fmt(phase_mean(eff_mean, early_end, mid_end))} |")
            lines.append(f"| Late ({mid_end}+) | {fmt(phase_mean(eff_mean, mid_end, n_steps))} |")
        else:
            lines.append("`reward/efficiency_mean = 0.000` for all steps.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # 4. Actor Metrics
    lines.append("## 4. Actor Metrics")
    lines.append("")

    # PG Loss
    if pg_loss is not None:
        lines.append("### 4.1 Policy Gradient Loss")
        lines.append("")
        lines.append("| Phase | Mean | Range |")
        lines.append("|-------|------|-------|")
        for label, s, e in [("Early", 0, early_end), ("Mid", early_end, mid_end), ("Late", mid_end, n_steps)]:
            sl = pg_loss.iloc[s:e].dropna()
            if len(sl) > 0:
                lines.append(f"| {label} ({s}-{e}) | {fmt(sl.mean())} | [{fmt(sl.min())}, {fmt(sl.max())}] |")
        lines.append("")

    # KL Divergence
    if kl_loss is not None:
        lines.append("### 4.2 KL Divergence")
        lines.append("")
        kl_vals = kl_loss.dropna()
        if len(kl_vals) >= 20:
            first10 = kl_vals.iloc[:10].mean()
            last10 = kl_vals.iloc[-10:].mean()
            growth = last10 / first10 if first10 > 0 else float("inf")
            lines.append("| Phase | Mean KL Loss |")
            lines.append("|-------|-------------|")
            lines.append(f"| First 10 steps | {fmt(first10, 6)} |")
            lines.append(f"| Last 10 steps | {fmt(last10, 6)} |")
            lines.append(f"| Growth factor | {fmt(growth, 1)}x |")
            lines.append(f"| Trend | **{trend_direction(kl_loss)}** |")
        else:
            lines.append(f"Mean KL loss: {fmt(kl_vals.mean(), 6)}")
        lines.append("")

    # Entropy
    if entropy is not None:
        lines.append("### 4.3 Entropy")
        lines.append("")
        ent_vals = entropy.dropna()
        if len(ent_vals) >= 20:
            first10 = ent_vals.iloc[:10].mean()
            last10 = ent_vals.iloc[-10:].mean()
            change_pct = 100 * (last10 - first10) / abs(first10) if first10 != 0 else 0
            lines.append(f"- First 10 steps: {fmt(first10, 4)}")
            lines.append(f"- Last 10 steps: {fmt(last10, 4)}")
            lines.append(f"- Change: {'+' if change_pct > 0 else ''}{fmt(change_pct, 1)}%")
            lines.append(f"- Trend: **{trend_direction(entropy)}**")
        else:
            lines.append(f"Mean entropy: {fmt(ent_vals.mean(), 4)}")
        lines.append("")

    # Gradient Norms
    if grad_norm is not None:
        lines.append("### 4.4 Gradient Norms")
        lines.append("")
        gn_vals = grad_norm.dropna()
        if len(gn_vals) >= 20:
            lines.append("| Phase | Mean Grad Norm |")
            lines.append("|-------|---------------|")
            lines.append(f"| First 10 steps | {fmt(gn_vals.iloc[:10].mean(), 4)} |")
            lines.append(f"| Last 10 steps | {fmt(gn_vals.iloc[-10:].mean(), 4)} |")
        lines.append("")

    # Clip Fraction
    if clipfrac is not None:
        cf_vals = clipfrac.dropna()
        lines.append("### 4.5 Clip Fraction")
        lines.append("")
        lines.append(f"Mean: {fmt(cf_vals.mean(), 6)}, Max: {fmt(cf_vals.max(), 6)}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # 5. PC-Grad Analysis (if enabled)
    if pc_grad:
        lines.append("## 5. PC-Grad Analysis")
        lines.append("")

        if fast_path is not None:
            fp_vals = fast_path.dropna()
            fp_count = int(fp_vals.sum())
            fp_total = len(fp_vals)
            fp_pct = 100 * fp_count / fp_total if fp_total > 0 else 0
            lines.append(f"### 5.1 Fast Path: {fmt_pct(fp_pct)} of steps")
            lines.append("")
            lines.append(f"Fast path activated on **{fp_count}/{fp_total}** steps.")
            if fp_pct == 100:
                lines.append("The efficiency objective was completely inert - PC-Grad dual-pass never executed.")
            lines.append("")

        if conflict_rate is not None:
            cr_vals = conflict_rate.dropna()
            if len(cr_vals) > 0 and cr_vals.max() > 0:
                lines.append("### 5.2 Gradient Conflict Rate")
                lines.append("")
                lines.append(f"- Mean conflict rate: {fmt_pct(cr_vals.mean() * 100)}")
                lines.append(f"- Max conflict rate: {fmt_pct(cr_vals.max() * 100)}")
                lines.append(f"- Trend: **{trend_direction(conflict_rate)}**")
                lines.append("")

        if pg_loss_acc is not None or pg_loss_eff is not None:
            lines.append("### 5.3 Per-Objective Losses")
            lines.append("")
            lines.append("| Phase | Accuracy PG Loss | Efficiency PG Loss |")
            lines.append("|-------|-----------------|-------------------|")
            for label, s, e in [("Early", 0, early_end), ("Mid", early_end, mid_end), ("Late", mid_end, n_steps)]:
                acc_v = fmt(phase_mean(pg_loss_acc, s, e)) if pg_loss_acc is not None else "N/A"
                eff_v = fmt(phase_mean(pg_loss_eff, s, e)) if pg_loss_eff is not None else "N/A"
                lines.append(f"| {label} ({s}-{e}) | {acc_v} | {eff_v} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Timing
    section_num = 6 if pc_grad else 5
    lines.append(f"## {section_num}. Timing Breakdown")
    lines.append("")

    if step_time is not None:
        lines.append("| Component | Mean time (s) | % of step |")
        lines.append("|-----------|--------------|-----------|")

        step_mean = step_time.dropna().mean()

        def timing_row(name, series):
            if series is not None:
                s = series.dropna()
                if len(s) > 0:
                    m = s.mean()
                    pct = 100 * m / step_mean if step_mean > 0 else 0
                    lines.append(f"| {name} | {fmt(m, 1)} | {fmt_pct(pct)} |")

        timing_row("Generation (vLLM)", gen_time)
        timing_row("Actor update", actor_time)
        timing_row("Reference forward", ref_time)
        timing_row("Advantage computation", adv_time)
        lines.append(f"| **Total step** | **{fmt(step_mean, 1)}** | **100%** |")

        if runtime_s:
            lines.append(f"| **Total wall time** | **{runtime_h}** | |")
        lines.append("")
    else:
        lines.append("No timing data available.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Response Length
    section_num += 1
    lines.append(f"## {section_num}. Response Length")
    lines.append("")

    if resp_mean is not None:
        rm_vals = resp_mean.dropna()
        if len(rm_vals) >= 20:
            lines.append("| Phase | Mean Length |")
            lines.append("|-------|------------|")
            lines.append(f"| First 10 steps | {fmt(rm_vals.iloc[:10].mean(), 1)} |")
            lines.append(f"| Last 10 steps | {fmt(rm_vals.iloc[-10:].mean(), 1)} |")

            first = rm_vals.iloc[:10].mean()
            last = rm_vals.iloc[-10:].mean()
            change = last - first
            change_pct = 100 * change / first if first > 0 else 0
            lines.append(f"| Change | {'+' if change > 0 else ''}{fmt(change, 1)} ({'+' if change_pct > 0 else ''}{fmt(change_pct, 1)}%) |")
            lines.append(f"| Trend | **{trend_direction(resp_mean)}** |")
        else:
            lines.append(f"Mean response length: {fmt(rm_vals.mean(), 1)}")
        lines.append("")

        if resp_max is not None:
            lines.append(f"Max response length: {fmt(resp_max.dropna().max(), 0)}")
        if resp_min is not None:
            lines.append(f"Min response length: {fmt(resp_min.dropna().min(), 0)}")
        if clip_ratio is not None:
            lines.append(f"Prompt clip ratio: {fmt(clip_ratio.dropna().mean(), 4)}")
        lines.append("")
    else:
        lines.append("No response length data available.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Validation
    section_num += 1
    if val_cols:
        lines.append(f"## {section_num}. Validation Scores")
        lines.append("")

        for col in sorted(val_cols):
            s = safe_get(history, col)
            if s is not None:
                vals = s.dropna()
                if len(vals) > 0:
                    lines.append(f"**{col}**:")
                    # Find which steps had validation scores
                    for idx in vals.index:
                        step_num = history["_step"].iloc[idx] if "_step" in history.columns else idx
                        lines.append(f"- Step {step_num}: {fmt(vals.loc[idx], 4)}")
                    lines.append("")

        lines.append("---")
        lines.append("")

    # Footer
    lines.append(f"*Report generated from W&B API data on {datetime.now().strftime('%Y-%m-%d')}.*")
    lines.append("")

    return "\n".join(lines), run_name


def generate_comparison(runs_data):
    """Generate a comparison table for multiple runs."""
    lines = []
    lines.append("## Cross-Run Comparison")
    lines.append("")

    # Header
    header = "| Metric |"
    sep = "|--------|"
    for _, run, _ in runs_data:
        header += f" {run.name or run.id} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    # Metrics to compare
    comparison_metrics = [
        ("PC-Grad", lambda r, h: "Enabled" if _is_pc_grad(r) else "Disabled"),
        ("Total steps", lambda r, h: str(len(h))),
        ("Final match ratio", lambda r, h: fmt(
            h["reward/match_ratio"].dropna().iloc[-1], 4) if "reward/match_ratio" in h.columns and len(h["reward/match_ratio"].dropna()) > 0 else "N/A"),
        ("Mean reward", lambda r, h: fmt(
            h["reward/mean"].dropna().mean(), 4) if "reward/mean" in h.columns else "N/A"),
        ("Mean accuracy reward", lambda r, h: fmt(
            h["reward/accuracy_mean"].dropna().mean(), 4) if "reward/accuracy_mean" in h.columns else "N/A"),
        ("Mean efficiency reward", lambda r, h: fmt(
            h["reward/efficiency_mean"].dropna().mean(), 4) if "reward/efficiency_mean" in h.columns else "N/A"),
        ("Learning signal %", lambda r, h: fmt_pct(
            100 * (h["reward/mean"].dropna().abs() > 1e-9).sum() / len(h)) if "reward/mean" in h.columns else "N/A"),
        ("Final entropy", lambda r, h: fmt(
            h["actor/entropy_loss"].dropna().iloc[-10:].mean(), 4) if "actor/entropy_loss" in h.columns and len(h["actor/entropy_loss"].dropna()) >= 10 else "N/A"),
        ("Final KL loss", lambda r, h: fmt(
            h["actor/kl_loss"].dropna().iloc[-10:].mean(), 6) if "actor/kl_loss" in h.columns and len(h["actor/kl_loss"].dropna()) >= 10 else "N/A"),
        ("Final grad norm", lambda r, h: fmt(
            h["actor/grad_norm"].dropna().iloc[-10:].mean(), 4) if "actor/grad_norm" in h.columns and len(h["actor/grad_norm"].dropna()) >= 10 else "N/A"),
        ("Resp length trend", lambda r, h: trend_direction(
            h["response_length/mean"]) if "response_length/mean" in h.columns else "N/A"),
        ("Mean step time (s)", lambda r, h: fmt(
            h["timing_s/step"].dropna().mean(), 1) if "timing_s/step" in h.columns else "N/A"),
        ("Total runtime", lambda r, h: f"{r.summary.get('_runtime', 0)/3600:.2f}h" if r.summary.get("_runtime") else "N/A"),
    ]

    for name, fn in comparison_metrics:
        row = f"| {name} |"
        for _, run, history in runs_data:
            try:
                val = fn(run, history)
            except Exception:
                val = "N/A"
            row += f" {val} |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def _is_pc_grad(run):
    config = run.config
    for key_path in ["actor_rollout_ref.actor.enable_pc_grad", "enable_pc_grad"]:
        parts = key_path.split(".")
        obj = config
        for p in parts:
            if isinstance(obj, dict) and p in obj:
                obj = obj[p]
            else:
                obj = None
                break
        if obj is not None:
            return bool(obj)
    return False


def export_metrics_json(run, history, output_path):
    """Export key metrics as JSON for downstream use."""
    n_steps = len(history)
    data = {
        "run_id": run.id,
        "run_name": run.name,
        "status": run.state,
        "n_steps": n_steps,
        "pc_grad": _is_pc_grad(run),
        "metrics": {},
    }

    def series_summary(series):
        if series is None:
            return None
        vals = series.dropna()
        if len(vals) == 0:
            return None
        return {
            "first": float(vals.iloc[0]),
            "last": float(vals.iloc[-1]),
            "mean": float(vals.mean()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "first10_mean": float(vals.iloc[:10].mean()) if len(vals) >= 10 else None,
            "last10_mean": float(vals.iloc[-10:].mean()) if len(vals) >= 10 else None,
        }

    metric_keys = [
        "reward/mean", "reward/match_ratio", "reward/accuracy_mean",
        "reward/efficiency_mean", "actor/entropy_loss", "actor/kl_loss",
        "actor/grad_norm", "actor/pg_loss", "response_length/mean",
        "actor/pc_grad_conflict_rate", "actor/pc_grad_fast_path",
    ]

    for key in metric_keys:
        s = safe_get(history, key)
        summary = series_summary(s)
        if summary is not None:
            data["metrics"][key] = summary

    # Health check
    issues = health_check(run, history)
    data["health"] = [{"severity": s, "message": m} for s, m in issues]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def list_running_runs(api, entity, project):
    """List all currently running runs in a project."""
    runs = api.runs(
        f"{entity}/{project}",
        filters={"state": "running"},
    )
    results = []
    for run in runs:
        results.append({
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "config_summary": {
                "pc_grad": _is_pc_grad(run),
                "lr": _nested_get(run.config, "actor_rollout_ref.actor.optim.lr"),
                "kl_coef": _nested_get(run.config, "algorithm.kl_ctrl.kl_coef"),
            },
        })
    return results


def _nested_get(d, key, default=None):
    """Get a nested key from a dict using dot notation."""
    parts = key.split(".")
    obj = d
    for p in parts:
        if isinstance(obj, dict) and p in obj:
            obj = obj[p]
        else:
            return default
    return obj


def main():
    parser = argparse.ArgumentParser(description="Generate W&B run summary reports")
    parser.add_argument("run_ids", nargs="*", help="W&B run IDs to summarize")
    parser.add_argument("--project", default="SQL-R1-MORL", help="W&B project name")
    parser.add_argument("--entity", default="abhinayjain-uoa", help="W&B entity")
    parser.add_argument("--output-dir", default=".", help="Output directory for reports")
    parser.add_argument("--list-running", action="store_true", help="List all currently running runs")
    parser.add_argument("--json", action="store_true", help="Export key metrics as JSON alongside markdown")
    parser.add_argument("--health-only", action="store_true", help="Only print health check results (no full report)")
    args = parser.parse_args()

    api = wandb.Api()

    # Handle --list-running
    if args.list_running:
        print(f"Fetching running runs from {args.entity}/{args.project}...")
        runs = list_running_runs(api, args.entity, args.project)
        if not runs:
            print("No running runs found.")
        else:
            print(f"\nFound {len(runs)} running run(s):\n")
            for r in runs:
                pc = "MORL (PC-Grad)" if r["config_summary"]["pc_grad"] else "Baseline"
                lr = r["config_summary"].get("lr", "N/A")
                kl = r["config_summary"].get("kl_coef", "N/A")
                print(f"  {r['id']}  {r['name']:<40}  {pc:<20}  lr={lr}  kl={kl}")
            print(f"\nTo analyze: python {sys.argv[0]} {' '.join(r['id'] for r in runs)}")
        if not args.run_ids:
            return

    if not args.run_ids:
        parser.error("run_ids are required (unless using --list-running)")

    runs_data = []

    for run_id in args.run_ids:
        print(f"Fetching run {run_id}...")
        try:
            run, history = fetch_run(api, args.entity, args.project, run_id)
        except Exception as e:
            print(f"Error fetching run {run_id}: {e}", file=sys.stderr)
            continue

        print(f"  Run: {run.name}, Steps: {len(history)}, Status: {run.state}")

        # Health-only mode
        if args.health_only:
            issues = health_check(run, history)
            print(f"\n  Health Check for {run.name} ({run.id}):")
            severity_icons = {"CRITICAL": "X", "WARNING": "!", "INFO": "i"}
            for severity, msg in issues:
                icon = severity_icons.get(severity, " ")
                print(f"    [{icon}] {severity}: {msg}")
            print()
            continue

        report, run_name = generate_report(run, history, args.entity, args.project)

        # Sanitize run name for filename
        safe_name = run_name.replace("/", "-").replace(" ", "-").replace(":", "-")
        runs_data.append((safe_name, run, history))

        # Write individual report
        out_path = f"{args.output_dir}/report-{safe_name}.md"
        with open(out_path, "w") as f:
            f.write(report)
        print(f"  Wrote {out_path}")

        # JSON export
        if args.json:
            json_path = f"{args.output_dir}/metrics-{safe_name}.json"
            export_metrics_json(run, history, json_path)
            print(f"  Wrote {json_path}")

    if args.health_only:
        return

    # If multiple runs, append comparison to each report
    if len(runs_data) > 1:
        comparison = generate_comparison(runs_data)
        for safe_name, run, history in runs_data:
            out_path = f"{args.output_dir}/report-{safe_name}.md"
            with open(out_path, "r") as f:
                content = f.read()
            # Insert comparison before the footer
            footer_marker = "\n*Report generated"
            if footer_marker in content:
                parts = content.rsplit(footer_marker, 1)
                content = parts[0] + "\n" + comparison + "\n---\n" + footer_marker + parts[1]
            else:
                content += "\n" + comparison
            with open(out_path, "w") as f:
                f.write(content)
        print(f"\nComparison table appended to all {len(runs_data)} reports.")

    print("Done.")


if __name__ == "__main__":
    main()
