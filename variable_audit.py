from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

files = {
    "Rabies": DATA_DIR / "CGPP_Rabies_Animal.xlsx",
    "Anthrax": DATA_DIR / "CGPP_Anthrax_Animal.xlsx",
    "Brucellosis": DATA_DIR / "CGPP_Brucellosis_Animal.xlsx",
}

datasets = {}

for disease, path in files.items():
    df = pd.read_excel(path, engine="openpyxl")
    datasets[disease] = df


# ---------------------------------------------------------
# Create a compact audit of every common variable
# 

columns = datasets["Rabies"].columns

rows = []

for col in columns:

    row = {
        "Variable": col
    }

    for disease, df in datasets.items():

        s = df[col]

        row[f"{disease}_nonmissing"] = s.notna().sum()
        row[f"{disease}_unique"] = s.dropna().nunique()

    rows.append(row)

audit = pd.DataFrame(rows)


# ---------------------------------------------------------
# Print variables that contain information in all datasets
# ---------------------------------------------------------

print("\n" + "=" * 100)
print("COMMON VARIABLE AUDIT")
print("=" * 100)

for _, row in audit.iterrows():

    print("\n" + row["Variable"])

    print(
        f"  Rabies:      "
        f"{row['Rabies_nonmissing']}/{len(datasets['Rabies'])} non-missing, "
        f"{row['Rabies_unique']} unique"
    )

    print(
        f"  Anthrax:     "
        f"{row['Anthrax_nonmissing']}/{len(datasets['Anthrax'])} non-missing, "
        f"{row['Anthrax_unique']} unique"
    )

    print(
        f"  Brucellosis: "
        f"{row['Brucellosis_nonmissing']}/{len(datasets['Brucellosis'])} non-missing, "
        f"{row['Brucellosis_unique']} unique"
    )


# ---------------------------------------------------------
# Save for later inspection
# ---------------------------------------------------------

output = DATA_DIR / "cgpp_variable_audit.csv"
audit.to_csv(output, index=False)

print("\nSaved:")
print(output)