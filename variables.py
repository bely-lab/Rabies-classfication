import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

DATA_DIR = Path("/Users/belayneshmossiekndie/Desktop/Haqila/analysis/data")

files = {
    "Rabies": DATA_DIR / "CGPP_Rabies_Animal.xlsx",
    "Anthrax": DATA_DIR / "CGPP_Anthrax_Animal.xlsx",
    "Brucellosis": DATA_DIR / "CGPP_Brucellosis_Animal.xlsx",
}

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

datasets = {}

for disease, path in files.items():
    df = pd.read_excel(path)
    datasets[disease] = df
    print(f"{disease}: {df.shape[0]} rows × {df.shape[1]} columns")

# Check that all have the same columns
columns = list(datasets["Rabies"].columns)

print("\nSame columns across all three datasets:")
print(
    list(datasets["Rabies"].columns)
    == list(datasets["Anthrax"].columns)
    == list(datasets["Brucellosis"].columns)
)

# --------------------------------------------------
# 3. Print ALL 127 columns
# --------------------------------------------------

for i, col in enumerate(columns, start=1):

    print("\n" + "=" * 100)
    print(f"{i}. {col}")
    print("=" * 100)

    for disease, df in datasets.items():

        series = df[col]

        # Treat empty strings as missing
        non_missing = series.notna() & (series.astype(str).str.strip() != "")
        n_present = non_missing.sum()
        completeness = n_present / len(df) * 100

        unique_values = series[non_missing].astype(str).unique()

        print(
            f"\n{disease}: "
            f"{n_present}/{len(df)} populated "
            f"({completeness:.1f}%), "
            f"{len(unique_values)} unique"
        )

        # Show up to 8 actual values
        examples = list(unique_values[:8])

        if examples:
            print("  Examples:", examples)
        else:
            print("  Examples: [EMPTY]")

input("\nPress Enter to finish...")