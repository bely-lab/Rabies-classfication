from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
input_file = BASE_DIR / "output" / "CGPP_Animal_Harmonized.csv"

df = pd.read_csv(input_file)

print("=" * 80)
print("INITIAL DATA")
print("=" * 80)
print("Shape:", df.shape)


# ============================================================
# 1. REMOVE VARIABLES NOT USED FOR MODELING
# ============================================================

remove_columns = [
    # Reporting / surveillance process
    "Notification_Method",
    "Identification_Method",
    "Community_HDA_Identified",

    # Target leakage:
    # vaccination categories directly encode the suspected disease
    "Immunization_Status"
]

df = df.drop(
    columns=[col for col in remove_columns if col in df.columns],
    errors="ignore"
)

print("\nRemoved variables:")
for col in remove_columns:
    if col not in df.columns:
        print(f" - {col} (not present)")
    else:
        print(f" - {col}")


# ============================================================
# 2. REMOVE EXACT DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print("\n" + "=" * 80)
print("DUPLICATE REMOVAL")
print("=" * 80)

print("Exact duplicates found:", duplicate_count)

df = df.drop_duplicates().reset_index(drop=True)

print("Rows after duplicate removal:", len(df))


# ============================================================
# 3. HANDLE MISSING GEOGRAPHIC VALUES
# ============================================================

numeric_columns = [
    "Latitude",
    "Longitude",
    "Altitude"
]

print("\n" + "=" * 80)
print("MISSING NUMERICAL VALUES")
print("=" * 80)

for col in numeric_columns:

    if col not in df.columns:
        continue

    missing_before = df[col].isna().sum()

    if missing_before > 0:

        median_value = df[col].median()

        df[col] = df[col].fillna(median_value)

        print(
            f"{col}: {missing_before} missing "
            f"-> imputed with median {median_value:.3f}"
        )

    else:
        print(f"{col}: no missing values")


# ============================================================
# 4. HANDLE RARE ANIMAL CATEGORIES
# ============================================================

print("\n" + "=" * 80)
print("RARE ANIMAL CATEGORIES")
print("=" * 80)

if "Animal_Type" in df.columns:

    print("Before:")
    print(df["Animal_Type"].value_counts().to_string())

    # Combine categories with fewer than 5 records
    counts = df["Animal_Type"].value_counts()

    rare_categories = counts[counts < 5].index

    df["Animal_Type"] = df["Animal_Type"].replace(
        rare_categories,
        "Other"
    )

    print("\nRare categories combined:")
    print(list(rare_categories))

    print("\nAfter:")
    print(df["Animal_Type"].value_counts().to_string())


# ============================================================
# 5. FINAL COLUMN ORDER
# ============================================================

final_columns = [
    "Disease",
    "Region",
    "Latitude",
    "Longitude",
    "Altitude",
    "Animal_Type",
    "Animal_Ownership",
    "Animal_Image"
]

# Keep only columns that actually exist
final_columns = [
    col for col in final_columns
    if col in df.columns
]

df = df[final_columns]


# ============================================================
# 6. FINAL DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL CLEAN DATA")
print("=" * 80)

print("Final shape:", df.shape)

print("\nFinal columns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2}. {col}")

print("\nMissing values:")
print(df.isna().sum().to_string())

print("\nExact duplicates:")
print(df.duplicated().sum())

print("\nDisease distribution:")
print(df["Disease"].value_counts().to_string())

print("\nAnimal type distribution:")
print(df["Animal_Type"].value_counts().to_string())


# ============================================================
# 7. SAVE FINAL DATA
# ============================================================

output_csv = BASE_DIR / "output" / "CGPP_Animal_Final_Clean.csv"
output_xlsx = BASE_DIR / "output" / "CGPP_Animal_Final_Clean.xlsx"

df.to_csv(output_csv, index=False)
df.to_excel(output_xlsx, index=False)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print("CSV :", output_csv)
print("Excel:", output_xlsx)