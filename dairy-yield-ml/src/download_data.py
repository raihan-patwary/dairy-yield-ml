"""
Helper for obtaining the real dataset.

The dataset is openly licensed (CC BY 4.0) and hosted on Mendeley Data:

    Sarah, P., Tasrifin, D. S., Indrijani, H., & Ruswandi, D. (2024).
    "Dataset for performance of superior dairy cattle sires."
    Mendeley Data, v2. https://doi.org/10.17632/2sm93h8t7y.2

Mendeley serves the file through a browser download (behind JavaScript), so it
cannot be reliably fetched from a script. Follow these steps once:

    1. Open https://doi.org/10.17632/2sm93h8t7y.2 in a browser.
    2. Click "Download All" (or the individual data file).
    3. Save / rename the tabular file to:   data/raw/dairy_sires.csv

Expected columns (rename to match if the source uses different headers):

    cow_id, parity, lactation_length_d, peak_day, peak_milk_kg,
    dry_period_d, milk_yield_kg

Once the file is in place, run:

    python -m src.run_analysis

and every result and figure regenerates from the real data automatically.
Until then, the pipeline runs on a clearly-labelled synthetic placeholder.
"""

from pathlib import Path

DOI_URL = "https://doi.org/10.17632/2sm93h8t7y.2"
TARGET_PATH = Path("data/raw/dairy_sires.csv")


def main() -> None:
    if TARGET_PATH.exists():
        print(f"Real dataset already present at {TARGET_PATH}. Nothing to do.")
        return
    print(__doc__)
    print(f"\nReminder: place the CSV at {TARGET_PATH.resolve()}")


if __name__ == "__main__":
    main()
