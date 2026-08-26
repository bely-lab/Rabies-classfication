from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR / "presentation_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD CGPP FILES
# ============================================================

datasets = {}

for file in sorted(DATA_DIR.glob("CGPP_*.xlsx")):

    # Ignore temporary Excel files
    if file.name.startswith("~$"):
        continue

    try:
        df = pd.read_excel(file, engine="openpyxl")

        # Remove CGPP_ and extension
        name = file.stem.replace("CGPP_", "")

        # Remove things like (1), (2), (3)
        import re
        name = re.sub(r"\(\d+\)$", "", name).strip()

        datasets[name] = df

        print(
            f"{name:25s} | "
            f"{len(df):4d} rows | "
            f"{len(df.columns):3d} columns"
        )

    except Exception as e:
        print(f"ERROR reading {file.name}: {e}")


# ============================================================
# 2. DATASET SUMMARY
# ============================================================

summary_rows = []

for name, df in datasets.items():

    parts = name.split("_")

    disease = parts[0] if len(parts) >= 1 else "Unknown"
    population = parts[1] if len(parts) >= 2 else "Unknown"

    summary_rows.append({
        "Dataset": name,
        "Disease": disease,
        "Population": population,
        "Rows": len(df),
        "Columns": len(df.columns)
    })


summary = pd.DataFrame(summary_rows)


print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

if not summary.empty:
    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "dataset_summary.csv",
        index=False
    )


# ============================================================
# 3. RABIES / ANTHRAX / BRUCELLOSIS SUMMARY
# ============================================================

target_diseases = ["Rabies", "Anthrax", "Brucellosis"]

target_summary = summary[
    summary["Disease"].isin(target_diseases)
].copy()

print("\n" + "=" * 80)
print("RABIES / ANTHRAX / BRUCELLOSIS")
print("=" * 80)

print(
    target_summary[
        ["Disease", "Population", "Rows", "Columns"]
    ].to_string(index=False)
)


# ============================================================
# 4. PLOT RECORD COUNTS
# ============================================================

if not target_summary.empty:

    pivot = target_summary.pivot_table(
        index="Disease",
        columns="Population",
        values="Rows",
        aggfunc="sum",
        fill_value=0
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5)
    )

    ax.set_title(
        "CGPP Surveillance Records by Disease and Population"
    )
    ax.set_xlabel("Disease")
    ax.set_ylabel("Number of records")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "records_by_disease_population.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 5. COMMON COLUMNS
# ============================================================

column_sets = {
    name: set(df.columns)
    for name, df in datasets.items()
}


# All six main datasets
main_names = [
    "Rabies_Human",
    "Rabies_Animal",
    "Anthrax_Human",
    "Anthrax_Animal",
    "Brucellosis_Human",
    "Brucellosis_Animal"
]

existing_main = [
    name for name in main_names
    if name in column_sets
]

if existing_main:

    common_columns = set.intersection(
        *[column_sets[name] for name in existing_main]
    )

    print("\n" + "=" * 80)
    print("COMMON COLUMNS ACROSS MAIN DISEASE DATASETS")
    print("=" * 80)

    print(f"Datasets included: {len(existing_main)}")
    print(f"Common columns: {len(common_columns)}")

    for col in sorted(common_columns):
        print(col)

    pd.DataFrame({
        "Common column": sorted(common_columns)
    }).to_csv(
        OUTPUT_DIR / "common_columns_main_datasets.csv",
        index=False
    )


# ============================================================
# 6. IMPORTANT VARIABLES
# ============================================================

keywords = [
    "age",
    "sex",
    "vaccination",
    "immunization",
    "species",
    "owner",
    "region",
    "zone",
    "woreda",
    "kebele",
    "village",
    "gps",
    "illness",
    "disease",
    "outcome",
    "notification",
    "identification"
]

relevant_columns = set()

for df in datasets.values():

    for col in df.columns:

        col_lower = col.lower()

        if any(keyword in col_lower for keyword in keywords):
            relevant_columns.add(col)


print("\n" + "=" * 80)
print("POTENTIALLY RELEVANT VARIABLES")
print("=" * 80)

for col in sorted(relevant_columns):
    print(col)

pd.DataFrame({
    "Potentially relevant variable": sorted(relevant_columns)
}).to_csv(
    OUTPUT_DIR / "potentially_relevant_variables.csv",
    index=False
)


# ============================================================
# 7. IMAGE / MEDIA REFERENCES
# ============================================================

print("\n" + "=" * 80)
print("IMAGE / MEDIA REFERENCES")
print("=" * 80)

image_results = []

for name, df in datasets.items():

    image_columns = [
        col for col in df.columns
        if (
            "image" in col.lower()
            or "media" in col.lower()
        )
    ]

    for col in image_columns:

        values = df[col].fillna("").astype(str).str.strip()

        url_count = values.str.startswith(
            ("http://", "https://")
        ).sum()

        nonempty_count = (
            values != ""
        ).sum()

        image_results.append({
            "Dataset": name,
            "Column": col,
            "Rows": len(df),
            "URL references": int(url_count),
            "Non-empty values": int(nonempty_count)
        })


image_summary = pd.DataFrame(image_results)

if not image_summary.empty:

    print(
        image_summary.to_string(index=False)
    )

    image_summary.to_csv(
        OUTPUT_DIR / "image_media_summary.csv",
        index=False
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

for file in sorted(OUTPUT_DIR.iterdir()):
    print(file.name)

print("\nDone.")