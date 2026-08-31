from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent

# Find all Excel files anywhere inside the project directory

excel_files = list(PROJECT_DIR.rglob("*.xlsx"))

print("\nFound Excel files:")
for f in excel_files:
    print(" -", f)
# Select the three animal datasets

wanted = {
    "CGPP_Rabies_Animal.xlsx": "Rabies",
    "CGPP_Anthrax_Animal.xlsx": "Anthrax",
    "CGPP_Brucellosis_Animal.xlsx": "Brucellosis",
}

records = []

for filename, disease in wanted.items():

    matches = [f for f in excel_files if f.name == filename]

    if not matches:
        print(f"\nWARNING: Could not find {filename}")
        continue

    path = matches[0]

    try:
        df = pd.read_excel(path, engine="openpyxl")

        # Disease is assigned from the source filename.
        # It is NOT a column in the original dataset.
        
        records.append({
            "Disease": disease,
            "Rows": len(df),
            "Variables": len(df.columns),
        })

        print(f"\n{disease}")
        print("File:", path)
        print("Rows:", len(df))
        print("Variables:", len(df.columns))

    except Exception as e:
        print(f"ERROR reading {path}: {e}")

summary = pd.DataFrame(records)

print("\n" + "=" * 80)
print("CGPP DATASET SUMMARY")
print("=" * 80)

if len(summary) > 0:
    print(summary.to_string(index=False))
else:
    print("No datasets were found.")