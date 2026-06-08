"""
Run the full dairy lactation milk-yield analysis.

Pipeline:
    1. Load data (real CSV if present, else faithful placeholder)
    2. Nested cross-validation for five models
    3. Save the results table and three figures

Usage:
    python -m src.run_analysis
    # or:  python src/run_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow running both as a module and as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import FEATURES, TARGET, get_xy, load_data
from src.evaluate import evaluate_all
from src.models import build_models, param_grids
from src.plots import (
    plot_feature_correlation,
    plot_model_comparison,
    plot_predicted_vs_actual,
)

METRICS_DIR = Path("results/metrics")


def main() -> None:
    df, is_placeholder = load_data()
    tag = "PLACEHOLDER (synthetic)" if is_placeholder else "REAL"

    print("=" * 64)
    print(f"Dairy lactation milk-yield analysis  |  data: {tag}")
    print(f"rows: {len(df)}   features: {FEATURES}   target: {TARGET}")
    if is_placeholder:
        print("NOTE: results below use synthetic placeholder data.")
        print("      Download the real CSV (see src/data.py) for real numbers.")
    print("=" * 64)

    X, y = get_xy(df)
    models = build_models()
    grids = param_grids()

    print("\nRunning nested cross-validation (this takes a moment)...\n")
    results, oof_preds = evaluate_all(models, grids, X, y)

    print(results.to_string(float_format=lambda v: f"{v:,.3f}"))

    # save metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / "model_results.csv"
    results.to_csv(csv_path)
    print(f"\nsaved metrics -> {csv_path}")

    # figures
    f1 = plot_model_comparison(results, is_placeholder)
    f2 = plot_predicted_vs_actual(y, oof_preds, results, is_placeholder)
    f3 = plot_feature_correlation(df, FEATURES, TARGET, is_placeholder)
    print(f"saved figure  -> {f1}")
    print(f"saved figure  -> {f2}")
    print(f"saved figure  -> {f3}")

    # plain-language takeaway
    best = results.index[0]
    from src.models import INTERPRETABLE

    best_interp = next(n for n in results.index if n in INTERPRETABLE)
    print("\n" + "-" * 64)
    print(f"Best overall model : {best}  (RMSE {results.loc[best, 'RMSE']:,.0f} kg)")
    print(
        f"Best interpretable : {best_interp}  "
        f"(RMSE {results.loc[best_interp, 'RMSE']:,.0f} kg)"
    )
    gap = results.loc[best_interp, "RMSE"] - results.loc[best, "RMSE"]
    print(
        f"Gap to best interpretable: {gap:,.0f} kg "
        f"({100 * gap / results.loc[best, 'RMSE']:.1f}% of best RMSE)"
    )
    print("-" * 64)


if __name__ == "__main__":
    main()
