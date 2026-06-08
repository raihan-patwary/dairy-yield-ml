"""
Data loading for the dairy lactation milk-yield project.

The real dataset is the openly licensed (CC BY 4.0) Holstein-Friesian record set:

    Sarah, P., Tasrifin, D. S., Indrijani, H., & Ruswandi, D. (2024).
    "Dataset for performance of superior dairy cattle sires."
    Mendeley Data, v2. https://doi.org/10.17632/2sm93h8t7y.2

How to use the real data
------------------------
1. Download the file from the DOI above (CC BY 4.0, free).
2. Place the CSV at:  data/raw/dairy_sires.csv
3. Re-run the pipeline. Everything downstream regenerates automatically.

If the real CSV is not present, this module builds a clearly labelled
PLACEHOLDER dataset whose per-lactation means and standard deviations match the
values published in the dataset description. The placeholder lets the repository
run end-to-end and produce figures, but it is NOT the real data and is marked as
such in every output. Swap in the real CSV to get real results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/dairy_sires.csv")

# Per-lactation summary statistics taken verbatim from the dataset description
# (mean +/- SD). Used only to build the faithful placeholder.
LACTATION_STATS = {
    # parity: (milk_yield_kg, lactation_length_d, peak_day, peak_milk_kg, dry_period_d)
    1: dict(yield_=(8029.28, 1112), length=(321.26, 38.48), peak_day=(85.36, 29.25),
            peak_milk=(32.55, 4.16), dry=(51.37, 9.33)),
    2: dict(yield_=(7761.66, 1145), length=(323.66, 43.06), peak_day=(58.43, 21.11),
            peak_milk=(40.79, 5.30), dry=(65.10, 22.69)),
    3: dict(yield_=(7788.92, 1148), length=(326.64, 46.74), peak_day=(61.88, 22.72),
            peak_milk=(43.62, 5.11), dry=(65.00, 20.49)),
    4: dict(yield_=(7484.18, 1133), length=(323.04, 42.23), peak_day=(66.39, 24.26),
            peak_milk=(43.82, 5.68), dry=(65.78, 21.60)),
}

# Numeric predictors and the regression target.
FEATURES = ["parity", "lactation_length_d", "peak_day", "peak_milk_kg", "dry_period_d"]
TARGET = "milk_yield_kg"


def _make_placeholder(n_per_parity: int = 60, seed: int = 42) -> pd.DataFrame:
    """Build a placeholder that reproduces the published per-lactation statistics.

    Peak milk and lactation length drive milk yield through a realistic,
    mostly-linear relationship (a cow that peaks higher and milks longer
    produces more over the lactation), plus mild noise. This makes the
    modelling task genuinely learnable - the point of the demonstration - while
    each variable's marginal mean and SD still match the published figures.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for parity, s in LACTATION_STATS.items():
        ym, ysd = s["yield_"]
        lm, lsd = s["length"]
        pdm, pdsd = s["peak_day"]
        pmm, pmsd = s["peak_milk"]
        drm, drsd = s["dry"]

        # generate predictors at their published marginal distributions
        peak_milk = rng.normal(pmm, pmsd, n_per_parity)
        length = rng.normal(lm, lsd, n_per_parity)
        peak_day = rng.normal(pdm, pdsd, n_per_parity)
        dry = rng.normal(drm, drsd, n_per_parity)

        # build yield from a sensible mechanistic-ish combination, then
        # standardise it back to the published mean and SD for this parity.
        # higher peak and longer lactation -> more total milk;
        # later peak day -> slightly less; longer dry period -> slightly less.
        raw = (
            1.0 * (peak_milk - pmm) / pmsd
            + 0.7 * (length - lm) / lsd
            - 0.25 * (peak_day - pdm) / pdsd
            - 0.20 * (dry - drm) / drsd
        )
        # add modest unexplained variation (keeps R2 realistic, not perfect)
        raw = raw + rng.normal(0, 0.55, n_per_parity)
        # standardise raw -> target mean/SD
        raw = (raw - raw.mean()) / raw.std()
        milk = ym + ysd * raw

        for i in range(n_per_parity):
            rows.append(
                dict(
                    cow_id=f"FH-{parity}-{i:03d}",
                    parity=parity,
                    lactation_length_d=round(float(length[i]), 1),
                    peak_day=round(float(peak_day[i]), 1),
                    peak_milk_kg=round(float(peak_milk[i]), 2),
                    dry_period_d=round(float(dry[i]), 1),
                    milk_yield_kg=round(float(milk[i]), 1),
                )
            )

    df = pd.DataFrame(rows)
    # keep biological values in sane ranges
    df["lactation_length_d"] = df["lactation_length_d"].clip(lower=200)
    df["peak_day"] = df["peak_day"].clip(lower=10)
    df["peak_milk_kg"] = df["peak_milk_kg"].clip(lower=10)
    df["dry_period_d"] = df["dry_period_d"].clip(lower=20)
    df["milk_yield_kg"] = df["milk_yield_kg"].clip(lower=3000)
    df.attrs["is_placeholder"] = True
    return df


def _clean_real(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are not genuine cow records.

    The source spreadsheet leaves summary-statistic rows ("Standard of
    Deviation", "Maximum", "Average", etc.) inside the data range. A real cow id
    is numeric, so we keep only numeric ids and drop biologically impossible
    lactations (no real 305-day-style lactation is below ~2000 kg or shorter
    than 150 days). This is documented so the cleaning is transparent.
    """
    before = len(df)
    if "cow_id" in df.columns:
        ids = df["cow_id"].astype(str).str.strip()
        is_numeric_id = ids.str.replace(".", "", regex=False).str.isdigit()
        df = df[is_numeric_id]
    if TARGET in df.columns:
        df = df[df[TARGET].astype(float) >= 2000]
    if "lactation_length_d" in df.columns:
        df = df[df["lactation_length_d"].astype(float) >= 150]
    removed = before - len(df)
    if removed:
        print(f"[clean] removed {removed} non-cow / impossible rows "
              f"({before} -> {len(df)})")
    return df.reset_index(drop=True)


def load_data(raw_path: Path = RAW_PATH) -> tuple[pd.DataFrame, bool]:
    """Return (dataframe, is_placeholder).

    Loads and cleans the real CSV if present at ``raw_path``; otherwise returns
    the faithful placeholder. The boolean lets callers label every output
    honestly.
    """
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        df = _clean_real(df)
        df.attrs["is_placeholder"] = False
        return df, False

    df = _make_placeholder()
    return df, True


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into the feature matrix X and target vector y."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Dataframe is missing required columns: {missing}. "
            f"Expected features {FEATURES} and target '{TARGET}'."
        )
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


if __name__ == "__main__":
    data, placeholder = load_data()
    tag = "PLACEHOLDER" if placeholder else "REAL"
    print(f"[{tag}] loaded {len(data)} rows, {data.shape[1]} columns")
    print(data.head())
