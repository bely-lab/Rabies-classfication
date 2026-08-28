from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

files = {
    "Rabies": DATA_DIR / "CGPP_Rabies_Animal.xlsx",
    "Anthrax": DATA_DIR / "CGPP_Anthrax_Animal.xlsx",
    "Brucellosis": DATA_DIR / "CGPP_Brucellosis_Animal.xlsx",
}

datasets = {}

for name, path in files.items():
    df = pd.read_excel(path)
    datasets[name] = df

    print("=" * 80)
    print(name)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

# 1. Columns common to all three datasets
# ---------------------------------------------------------

common_columns = set.intersection(
    *(set(df.columns) for df in datasets.values())
)

print("\n" + "=" * 80)
print(f"COMMON COLUMNS: {len(common_columns)}")

for column in sorted(common_columns):
    print(column)


# ---------------------------------------------------------
# 2. Columns unique to each dataset
# ---------------------------------------------------------

for name, df in datasets.items():

    other_columns = set.union(
        *(set(other_df.columns)
          for other_name, other_df in datasets.items()
          if other_name != name)
    )

    unique_columns = set(df.columns) - other_columns

    print("\n" + "=" * 80)
    print(f"UNIQUE TO {name.upper()}: {len(unique_columns)}")

    for column in sorted(unique_columns):
        print(column)


# ---------------------------------------------------------
# 3. Look specifically for image/media columns
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("IMAGE / MEDIA COLUMNS")

for name, df in datasets.items():

    image_columns = [
        column
        for column in df.columns
        if any(
            word in column.lower()
            for word in [
                "image",
                "photo",
                "picture",
                "media",
                "attachment"
            ]
        )
    ]

    print(f"\n{name}:")
    for column in image_columns:
        non_missing = df[column].notna().sum()
        non_empty = (
            df[column]
            .astype(str)
            .str.strip()
            .replace("nan", "")
            .ne("")
            .sum()
        )

        print(
            f"  {column}"
            f" | non-missing: {non_missing}/{len(df)}"
            f" | non-empty: {non_empty}/{len(df)}"
        )