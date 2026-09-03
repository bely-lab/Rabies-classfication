from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

DATA_PATH = Path(
    "/Users/belayneshmossiekndie/Desktop/Haqila/analysis/output/CGPP_Animal_Clean_Combined.xlsx"
)

df = pd.read_excel(DATA_PATH)

print("=" * 70)
print("CGPP CLEAN ANIMAL DATASET")
print("=" * 70)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# --------------------------------------------------
# 2. Disease distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("DISEASE DISTRIBUTION")
print("=" * 70)

print(df["Disease"].value_counts())


# --------------------------------------------------
# 3. Categorical variables
# --------------------------------------------------

categorical_vars = [
    "Region",
    "Animal_Type",
    "Animal_Ownership",
    "Immunization_Status",
    "Identification_Method",
    "Notification_Method",
    "Community_HDA_Identified",
]

for col in categorical_vars:

    print("\n" + "=" * 70)
    print(col)
    print("=" * 70)

    table = pd.crosstab(
        df[col],
        df["Disease"],
        margins=True
    )

    print(table)


# --------------------------------------------------
# 4. Numerical variables
# --------------------------------------------------

numerical_vars = [
    "Latitude",
    "Longitude",
    "Altitude",
]

print("\n" + "=" * 70)
print("NUMERICAL VARIABLES BY DISEASE")
print("=" * 70)

for col in numerical_vars:

    print(f"\n--- {col} ---")

    summary = df.groupby("Disease")[col].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )

    print(summary)


# --------------------------------------------------
# 5. Missing values
# --------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = pd.DataFrame({
    "Missing": df.isna().sum(),
    "Missing_%": (df.isna().mean() * 100).round(1)
})

print(missing)


# --------------------------------------------------
# 6. Image availability
# --------------------------------------------------

print("\n" + "=" * 70)
print("IMAGE AVAILABILITY")
print("=" * 70)

df["Image_Available"] = df["Animal_Image"].notna()

print(
    pd.crosstab(
        df["Disease"],
        df["Image_Available"],
        margins=True
    )
)


# --------------------------------------------------
# 7. Save EDA tables
# --------------------------------------------------

OUTPUT_DIR = DATA_PATH.parent

with pd.ExcelWriter(
    OUTPUT_DIR / "CGPP_Clean_EDA.xlsx",
    engine="openpyxl"
) as writer:

    df["Disease"].value_counts().rename(
        "Count"
    ).to_excel(writer, sheet_name="Disease_Count")

    for col in categorical_vars:

        table = pd.crosstab(
            df[col],
            df["Disease"]
        )

        table.to_excel(
            writer,
            sheet_name=col[:31]
        )

    for col in numerical_vars:

        summary = df.groupby("Disease")[col].agg(
            ["count", "mean", "std", "min", "median", "max"]
        )

        summary.to_excel(
            writer,
            sheet_name=f"{col}_summary"
        )

print("\nEDA results saved to:")
print(OUTPUT_DIR / "CGPP_Clean_EDA.xlsx")