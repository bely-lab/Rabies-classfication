from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

files = sorted(
    p for p in DATA_DIR.glob("*.xlsx")
    if not p.name.startswith("~$")
)

print(f"Found {len(files)} Excel files\n")

for path in files:
    try:
        df = pd.read_excel(path)

        print("=" * 80)
        print(f"FILE: {path.name}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print()

        print("Columns:")
        for i, col in enumerate(df.columns, start=1):
            print(f"{i:3d}. {col}")

    except Exception as e:
        print(f"ERROR reading {path.name}: {e}")