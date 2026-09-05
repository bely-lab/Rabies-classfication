from pathlib import Path
import pandas as pd

# ============================================================
# 1. LOAD EXISTING COMBINED DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

input_file = BASE_DIR / "output" / "CGPP_Animal_Clean_Combined.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("ORIGINAL DATA")
print("=" * 70)
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 2. HARMONIZE ANIMAL TYPE
# ============================================================

df["Animal_Type"] = df["Animal_Type"].replace({
    "Dog (if it is rabies)": "Dog",
    "Cat  (if it is rabies)": "Cat",
    "Donky/Horse/Mule": "Donkey/Horse/Mule",
    "Other specify": "Other"
})


# ============================================================
# 3. HARMONIZE IMMUNIZATION
# ============================================================

# The original combined dataset contains:
# "Vaccinated against the suspected disease"
# and "Not vaccinated".
#
# Because Disease identifies the source disease, we can recover
# the disease-specific vaccination category.

df["Immunization_Status"] = df.apply(
    lambda row:
        f"{row['Disease']} vaccinated"
        if row["Immunization_Status"] ==
           "Vaccinated against the suspected disease"
        else "Not vaccinated",
    axis=1
)


# ============================================================
# 4. REMOVE NOTIFICATION METHOD
# ============================================================

# Notification method describes how the case was communicated
# (e.g. telephone/verbal/written), rather than a characteristic
# of the disease or animal.

if "Notification_Method" in df.columns:
    df = df.drop(columns=["Notification_Method"])


# ============================================================
# 5. CHECK DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)
print(f"Exact duplicate rows: {duplicate_count}")


# ============================================================
# 6. MISSING-VALUE AUDIT
# ============================================================

missing = pd.DataFrame({
    "Missing": df.isna().sum(),
    "Missing_%": (df.isna().mean() * 100).round(2),
    "Unique": df.nunique(dropna=True)
})

missing = missing.sort_values(
    by="Missing_%",
    ascending=False
)

print("\n" + "=" * 70)
print("MISSING VALUES + UNIQUE VALUES")
print("=" * 70)
print(missing.to_string())


# ============================================================
# 7. CATEGORICAL VALUE INSPECTION
# ============================================================

categorical_vars = [
    "Disease",
    "Region",
    "Animal_Type",
    "Animal_Ownership",
    "Immunization_Status",
    "Identification_Method",
    "Community_HDA_Identified"
]

for col in categorical_vars:

    if col not in df.columns:
        continue

    print("\n" + "=" * 70)
    print(f"{col}")
    print("=" * 70)

    counts = df[col].value_counts(dropna=False)

    result = pd.DataFrame({
        "Count": counts,
        "Percent": (counts / len(df) * 100).round(2)
    })

    print(result.to_string())


# ============================================================
# 8. NUMERICAL VARIABLE INSPECTION
# ============================================================

numeric_vars = [
    "Latitude",
    "Longitude",
    "Altitude"
]

print("\n" + "=" * 70)
print("NUMERICAL VARIABLES")
print("=" * 70)

for col in numeric_vars:

    if col not in df.columns:
        continue

    print(f"\n--- {col} ---")

    print(
        df[col].describe(
            percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        ).to_string()
    )


# ============================================================
# 9. CHECK POSSIBLE REPORTING-DELAY VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("DATE VARIABLES IN CURRENT DATASET")
print("=" * 70)

date_candidates = [
    col for col in df.columns
    if "date" in col.lower()
    or "report" in col.lower()
]

print(date_candidates)

for col in date_candidates:
    print(f"\n--- {col} ---")
    print(f"Non-missing: {df[col].notna().sum()}")
    print(f"Missing: {df[col].isna().sum()}")
    print("Examples:")
    print(df[col].dropna().astype(str).head(10).to_list())


# ============================================================
# 10. SAVE HARMONIZED VERSION
# ============================================================

output_file = BASE_DIR / "output" / "CGPP_Animal_Harmonized.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("HARMONIZED DATASET")
print("=" * 70)
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {output_file}")