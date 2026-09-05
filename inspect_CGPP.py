from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# LOAD HARMONIZED DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
file = BASE_DIR / "output" / "CGPP_Animal_Harmonized.csv"

df = pd.read_csv(file)

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2}. {col}")


# ============================================================
# 1. MISSING VALUES + UNIQUE VALUES
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUES AND UNIQUE VALUES")
print("=" * 80)

quality = pd.DataFrame({
    "Variable": df.columns,
    "Missing": df.isna().sum().values,
    "Unique": df.nunique(dropna=True).values,
    "Data_Type": df.dtypes.astype(str).values
})

print(quality.to_string(index=False))


# ============================================================
# 2. DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("DUPLICATES")
print("=" * 80)

duplicate_count = df.duplicated().sum()

print("Exact duplicate rows:", duplicate_count)

if duplicate_count > 0:
    print("\nDuplicated rows:")
    print(
        df[df.duplicated(keep=False)]
        .to_string(index=False)
    )


# ============================================================
# 3. CATEGORICAL DISTRIBUTIONS
# ============================================================

categorical = [
    "Disease",
    "Region",
    "Animal_Type",
    "Animal_Ownership",
    "Immunization_Status",
    "Identification_Method",
    "Community_HDA_Identified"
]

for col in categorical:

    if col not in df.columns:
        continue

    print("\n" + "=" * 80)
    print(f"CATEGORICAL VARIABLE: {col}")
    print("=" * 80)

    counts = df[col].value_counts(dropna=False)

    table = pd.DataFrame({
        "Count": counts,
        "Percent": (counts / len(df) * 100).round(2)
    })

    print(table.to_string())


# ============================================================
# 4. CATEGORICAL DISTRIBUTION BY DISEASE
# ============================================================

for col in categorical:

    if col == "Disease" or col not in df.columns:
        continue

    print("\n" + "=" * 80)
    print(f"{col} BY DISEASE")
    print("=" * 80)

    table = pd.crosstab(
        df[col],
        df["Disease"],
        margins=True
    )

    print(table.to_string())


# ============================================================
# 5. NUMERICAL VARIABLES
# ============================================================

numeric = [
    "Latitude",
    "Longitude",
    "Altitude"
]

print("\n" + "=" * 80)
print("NUMERICAL VARIABLE SUMMARY")
print("=" * 80)

for col in numeric:

    if col not in df.columns:
        continue

    print(f"\n--- {col} ---")

    print(
        df[col].describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99
            ]
        ).to_string()
    )


# ============================================================
# 6. NUMERICAL VARIABLES BY DISEASE
# ============================================================

print("\n" + "=" * 80)
print("NUMERICAL VARIABLES BY DISEASE")
print("=" * 80)

for col in numeric:

    if col not in df.columns:
        continue

    print(f"\n--- {col} ---")

    print(
        df.groupby("Disease")[col]
        .agg(
            ["count", "mean", "std", "min", "median", "max"]
        )
        .round(3)
        .to_string()
    )


# ============================================================
# 7. CHECK ALL DATE / REPORTING VARIABLES
# ============================================================

print("\n" + "=" * 80)
print("DATE / REPORTING VARIABLES")
print("=" * 80)

date_columns = [
    col for col in df.columns
    if "date" in col.lower()
    or "report" in col.lower()
    or "time" in col.lower()
]

print("Detected columns:")

if len(date_columns) == 0:
    print(" None")

else:
    for col in date_columns:
        print(" -", col)

for col in date_columns:

    print(f"\n--- {col} ---")

    print("Non-missing:", df[col].notna().sum())
    print("Missing:", df[col].isna().sum())
    print("Unique:", df[col].nunique(dropna=True))

    print("Example values:")

    print(
        df[col]
        .dropna()
        .astype(str)
        .head(15)
        .to_list()
    )


# ============================================================
# 8. CHECK POTENTIAL REPORTING DELAY
# ============================================================

reported_col = "date_reported_by_CV_or_HDA"
report_col = "date_of_the_report"

if reported_col in df.columns and report_col in df.columns:

    print("\n" + "=" * 80)
    print("REPORTING DELAY CHECK")
    print("=" * 80)

    d1 = pd.to_datetime(
        df[reported_col],
        errors="coerce"
    )

    d2 = pd.to_datetime(
        df[report_col],
        errors="coerce"
    )

    valid = d1.notna() & d2.notna()

    print("Records with both dates:", valid.sum())
    print("Records without both dates:", (~valid).sum())

    if valid.sum() > 0:

        delay = (d2 - d1).dt.total_seconds() / 86400

        print("\nDelay summary (days):")

        print(
            delay[valid]
            .describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99
                ]
            )
            .to_string()
        )

        print(
            "\nNegative delays:",
            (delay[valid] < 0).sum()
        )

        print(
            "Zero delays:",
            (delay[valid] == 0).sum()
        )

        print(
            "Positive delays:",
            (delay[valid] > 0).sum()
        )


# ============================================================
# 9. RARE CATEGORIES
# ============================================================

print("\n" + "=" * 80)
print("RARE CATEGORIES (< 5 RECORDS)")
print("=" * 80)

for col in categorical:

    if col not in df.columns:
        continue

    counts = df[col].value_counts()

    rare = counts[counts < 5]

    if len(rare) > 0:

        print(f"\n{col}:")
        print(rare.to_string())


# ============================================================
# 10. FINAL QUICK QUALITY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("QUALITY SUMMARY")
print("=" * 80)

print("Total records:", len(df))
print("Total variables:", len(df.columns))

print(
    "Exact duplicate rows:",
    df.duplicated().sum()
)

print(
    "Variables with missing values:",
    (df.isna().sum() > 0).sum()
)

print(
    "Variables with >10% missing:",
    (df.isna().mean() > 0.10).sum()
)

print(
    "Variables with only one unique value:",
    (df.nunique(dropna=True) <= 1).sum()
)