from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

files = {
    "Rabies": DATA_DIR / "CGPP_Rabies_Animal.xlsx",
    "Anthrax": DATA_DIR / "CGPP_Anthrax_Animal.xlsx",
    "Brucellosis": DATA_DIR / "CGPP_Brucellosis_Animal.xlsx",
}

records = []

for disease, path in files.items():

    if not path.exists():
        print(f"WARNING: File not found: {path}")
        continue

    try:
        df = pd.read_excel(path, engine="openpyxl")

        records.append({
            "Disease": disease,
            "Rows": len(df),
            "Columns": len(df.columns),
        })

    except Exception as e:
        print(f"ERROR reading {path.name}: {e}")

summary = pd.DataFrame(
    records,
    columns=["Disease", "Rows", "Columns"]
)

print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print(summary.to_string(index=False))