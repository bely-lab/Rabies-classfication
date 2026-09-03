import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

DATA_DIR = Path("/Users/belayneshmossiekndie/Desktop/Haqila/analysis/data")

OUTPUT_FILE = Path(
    "/Users/belayneshmossiekndie/Desktop/Haqila/analysis/all_columns_output.txt"
)

files = {
    "Rabies": DATA_DIR / "CGPP_Rabies_Animal.xlsx",
    "Anthrax": DATA_DIR / "CGPP_Anthrax_Animal.xlsx",
    "Brucellosis": DATA_DIR / "CGPP_Brucellosis_Animal.xlsx",
}

# Store everything here
output = []

def write(text=""):
    print(text)
    output.append(str(text))

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

datasets = {}

for disease, path in files.items():
    df = pd.read_excel(path)
    datasets[disease] = df
    write(f"{disease}: {df.shape[0]} rows × {df.shape[1]} columns")

columns = list(datasets["Rabies"].columns)

write("\nSame columns across all three datasets:")
write(
    list(datasets["Rabies"].columns)
    == list(datasets["Anthrax"].columns)
    == list(datasets["Brucellosis"].columns)
)

# --------------------------------------------------
# 3. Print ALL columns
# --------------------------------------------------

for i, col in enumerate(columns, start=1):

    write("\n" + "=" * 100)
    write(f"{i}. {col}")
    write("=" * 100)

    for disease, df in datasets.items():

        series = df[col]

        non_missing = series.notna() & (
            series.astype(str).str.strip() != ""
        )

        n_present = non_missing.sum()
        completeness = n_present / len(df) * 100

        unique_values = series[non_missing].astype(str).unique()

        write(
            f"\n{disease}: "
            f"{n_present}/{len(df)} populated "
            f"({completeness:.1f}%), "
            f"{len(unique_values)} unique"
        )

        examples = list(unique_values[:8])

        if examples:
            write("  Examples: " + str(examples))
        else:
            write("  Examples: [EMPTY]")

# --------------------------------------------------
# 4. Save COMPLETE output
# --------------------------------------------------

OUTPUT_FILE.write_text("\n".join(output), encoding="utf-8")

print("\n" + "=" * 100)
print(f"Complete output saved to:")
print(OUTPUT_FILE)
print("=" * 100)