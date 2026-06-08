"""
Figures for the dairy lactation milk-yield project.

Three plots:
    1. model_comparison.png  - RMSE per model, interpretable vs heavy colour-coded
    2. predicted_vs_actual.png - best interpretable vs best heavy model scatter
    3. feature_correlation.png - correlation of each predictor with milk yield

When the data is the placeholder, a clear watermark is drawn on every figure so
results can never be mistaken for real ones.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import INTERPRETABLE

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Restrained, readable palette.
C_INTERP = "#2a7f62"   # green  - interpretable models
C_HEAVY = "#b5651d"    # ochre  - heavy models
C_ACCENT = "#1f3a5f"   # deep blue


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
        }
    )


def _watermark(fig, is_placeholder: bool) -> None:
    if is_placeholder:
        fig.text(
            0.5,
            0.5,
            "PLACEHOLDER DATA",
            fontsize=42,
            color="grey",
            alpha=0.16,
            ha="center",
            va="center",
            rotation=30,
            fontweight="bold",
            zorder=0,
        )


def plot_model_comparison(results: pd.DataFrame, is_placeholder: bool) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    names = results.index.tolist()
    colors = [C_INTERP if n in INTERPRETABLE else C_HEAVY for n in names]
    ax.barh(names, results["RMSE"], color=colors, edgecolor="black", linewidth=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Cross-validated RMSE (kg of milk)  -  lower is better")
    ax.set_title("Milk-yield prediction error by model")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_INTERP),
        plt.Rectangle((0, 0), 1, 1, color=C_HEAVY),
    ]
    ax.legend(handles, ["Interpretable", "Heavy (tree ensembles)"], loc="lower right")
    for i, v in enumerate(results["RMSE"]):
        ax.text(v, i, f" {v:,.0f}", va="center", fontsize=9)

    _watermark(fig, is_placeholder)
    fig.tight_layout()
    out = FIG_DIR / "model_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_predicted_vs_actual(
    y_true: pd.Series,
    oof_preds: dict[str, np.ndarray],
    results: pd.DataFrame,
    is_placeholder: bool,
) -> Path:
    _style()

    interp_ranked = [n for n in results.index if n in INTERPRETABLE]
    heavy_ranked = [n for n in results.index if n not in INTERPRETABLE]
    best_interp = interp_ranked[0]
    best_heavy = heavy_ranked[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    y = y_true.to_numpy()
    lo, hi = y.min(), y.max()

    for ax, name, color in (
        (axes[0], best_interp, C_INTERP),
        (axes[1], best_heavy, C_HEAVY),
    ):
        p = oof_preds[name]
        ax.scatter(y, p, s=22, alpha=0.6, color=color, edgecolor="white", linewidth=0.4)
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, alpha=0.7)
        r2 = results.loc[name, "R2"]
        rmse = results.loc[name, "RMSE"]
        ax.set_title(f"{name}\nR2 = {r2:.2f}   RMSE = {rmse:,.0f} kg")
        ax.set_xlabel("Actual milk yield (kg)")
    axes[0].set_ylabel("Predicted milk yield (kg)")

    fig.suptitle(
        "Best interpretable vs best heavy model (out-of-fold predictions)",
        fontweight="bold",
    )
    _watermark(fig, is_placeholder)
    fig.tight_layout()
    out = FIG_DIR / "predicted_vs_actual.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_feature_correlation(
    df: pd.DataFrame, features: list[str], target: str, is_placeholder: bool
) -> Path:
    _style()
    corr = df[features + [target]].corr()[target].drop(target).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [C_ACCENT if v >= 0 else "#a23b3b" for v in corr]
    ax.barh(corr.index, corr.values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Pearson correlation with {target}")
    ax.set_title("How each predictor relates to milk yield")
    for i, v in enumerate(corr.values):
        ax.text(v, i, f" {v:+.2f}", va="center", fontsize=9)

    _watermark(fig, is_placeholder)
    fig.tight_layout()
    out = FIG_DIR / "feature_correlation.png"
    fig.savefig(out)
    plt.close(fig)
    return out
