import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/belayneshmossiekndie/Desktop/Haqila/analysis/data")

files = {
    "Rabies": DATA_DIR / "CGPP_Rabies_Animal.xlsx",
    "Anthrax": DATA_DIR / "CGPP_Anthrax_Animal.xlsx",
    "Brucellosis": DATA_DIR / "CGPP_Brucellosis_Animal.xlsx",
}

dfs = {}

for disease, path in files.items():
    df = pd.read_excel(path)

    # Add disease label from the source file
    df["Disease"] = disease

    dfs[disease] = df

    print(f"{disease}: {df.shape[0]} records, {df.shape[1]} columns")


# --------------------------------------------------
# Combine the three datasets
# --------------------------------------------------

data = pd.concat(dfs.values(), ignore_index=True)

print("\nCombined dataset:")
print(data.shape)

# --------------------------------------------------
# Completeness by disease
# --------------------------------------------------

results = []

for disease, df in dfs.items():
    for col in df.columns:
        # Ignore completely blank strings as missing
        non_missing = (
            df[col]
            .replace(r"^\s*$", pd.NA, regex=True)
            .notna()
            .sum()
        )

        results.append({
            "Disease": disease,
            "Variable": col,
            "N": len(df),
            "Non_missing": non_missing,
            "Completeness_%": round(100 * non_missing / len(df), 1)
        })

completeness = pd.DataFrame(results)

# --------------------------------------------------
# Variables that are reasonably complete in ALL 3
# --------------------------------------------------

pivot = completeness.pivot(
    index="Variable",
    columns="Disease",
    values="Completeness_%"
)

# Variables >= 80% complete in every disease
common_usable = pivot[
    (pivot["Rabies"] >= 80) &
    (pivot["Anthrax"] >= 80) &
    (pivot["Brucellosis"] >= 80)
].sort_values(
    ["Rabies", "Anthrax", "Brucellosis"],
    ascending=False
)

print("\nVariables >=80% complete in all three diseases:")
print(common_usable)

print("\nNumber of common variables:", len(common_usable))

# --------------------------------------------------
# Save results
# --------------------------------------------------

completeness.to_csv(
    DATA_DIR / "animal_variable_completeness.csv",
    index=False
)

common_usable.to_csv(
    DATA_DIR / "animal_common_usable_variables.csv"
)

print("\nResults saved.")