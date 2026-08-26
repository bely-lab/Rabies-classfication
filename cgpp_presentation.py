from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR / "presentation_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 1. LOAD ALL CGPP FILES
# ---------------------------------------------------------

files = sorted(DATA_DIR.glob("CGPP_*.xlsx"))

datasets = {}

for file in files:

    # Ignore temporary Excel files
    if file.name.startswith("~$"):
        continue

    try:
        df = pd.read_excel(file, engine="openpyxl")

        name = file.stem.replace("CGPP_", "")

        datasets[name] = df

        print(f"{name:30s} {len(df):4d} rows | {len(df.columns):3d} columns")

    except Exception as e:
        print(f"ERROR: {file.name}")
        print(e)


# ---------------------------------------------------------
# 2. DATASET SUMMARY
# ---------------------------------------------------------

summary_rows = []

for name, df in datasets.items():

    parts = name.split("_")

    disease = parts[0]
    population = parts[1] if len(parts) > 1 else "Unknown"

    summary_rows.append({
        "Disease": disease,
        "Population": population,
        "Rows": len(df),
        "Columns": len(df.columns)
    })

summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    ["Disease", "Population"]
)

print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
print(summary.to_string(index=False))

summary.to_csv(
    OUTPUT_DIR / "dataset_summary.csv",
    index=False
)


# ---------------------------------------------------------
# 3. DISEASE × POPULATION PLOT
# ---------------------------------------------------------

pivot = summary.pivot_table(
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

ax.set_title("CGPP Surveillance Records by Disease and Population")
ax.set_xlabel("Disease")
ax.set_ylabel("Number of records")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "records_by_disease_population.png",
    dpi=300
)
plt.close()


# ---------------------------------------------------------
# 4. COMMON COLUMNS
# ---------------------------------------------------------

all_column_sets = {
    name: set(df.columns)
    for name, df in datasets.items()
}

common_columns = set.intersection(
    *all_column_sets.values()
)

print("\n" + "=" * 80)
print("COMMON COLUMNS ACROSS ALL LOADED DATASETS")
print("=" * 80)

for col in sorted(common_columns):
    print(col)

pd.DataFrame({
    "Common columns": sorted(common_columns)
}).to_csv(
    OUTPUT_DIR / "common_columns.csv",
    index=False
)


# ---------------------------------------------------------
# 5. RABIES / ANTHRAX / BRUCELLOSIS COMMON COLUMNS
# ---------------------------------------------------------

target_names = [
    "Rabies_Human",
    "Rabies_Animal",
    "Anthrax_Human",
    "Anthrax_Animal",
    "Brucellosis_Human",
    "Brucellosis_Animal"
]

target_sets = [
    all_column_sets[name]
    for name in target_names
    if name in all_column_sets
]

if target_sets:

    target_common = set.intersection(*target_sets)

    print("\n" + "=" * 80)
    print("COMMON COLUMNS ACROSS RABIES / ANTHRAX / BRUCELLOSIS")
    print("=" * 80)

    for col in sorted(target_common):
        print(col)

    pd.DataFrame({
        "Common columns": sorted(target_common)
    }).to_csv(
        OUTPUT_DIR / "rabies_anthrax_brucellosis_common_columns.csv",
        index=False
    )


# ---------------------------------------------------------
# 6. IMAGE / MEDIA REFERENCE INSPECTION
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("IMAGE / MEDIA REFERENCES")
print("=" * 80)

image_results = []

for name, df in datasets.items():

    image_columns = [
        col for col in df.columns
        if "image" in col.lower()
        or "media" in col.lower()
    ]

    for col in image_columns:

        series = df[col].astype(str).str.strip()

        # Count actual URL-looking values
        url_mask = series.str.startswith(
            ("http://", "https://")
        )

        image_results.append({
            "Dataset": name,
            "Column": col,
            "Rows": len(df),
            "URL references": int(url_mask.sum()),
            "Non-empty values": int(
                series.replace(
                    {"": np.nan, "nan": np.nan, "None": np.nan}
                ).notna().sum()
            )
        })

image_summary = pd.DataFrame(image_results)

print(image_summary.to_string(index=False))

image_summary.to_csv(
    OUTPUT_DIR / "image_media_summary.csv",
    index=False
)


# ---------------------------------------------------------
# 7. KEY VARIABLE INSPECTION
# ---------------------------------------------------------

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
    "case",
    "notification",
    "identification"
]

print("\n" + "=" * 80)
print("POTENTIALLY RELEVANT VARIABLES")
print("=" * 80)

relevant_columns = set()

for df in datasets.values():

    for col in df.columns:

        col_lower = col.lower()

        if any(keyword in col_lower for keyword in keywords):
            relevant_columns.add(col)

for col in sorted(relevant_columns):
    print(col)

pd.DataFrame({
    "Potentially relevant variable": sorted(relevant_columns)
}).to_csv(
    OUTPUT_DIR / "potentially_relevant_variables.csv",
    index=False
)


print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

for file in sorted(OUTPUT_DIR.iterdir()):
    print(file.name)

print("\nDone.")