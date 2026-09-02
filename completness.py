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


# Show the 34 variables clearly
print("\nCOMMON VARIABLES (>=80% COMPLETE)")
print("=" * 80)

for i, variable in enumerate(common_usable.index, 1):
    print(f"{i:2}. {variable}")

# Show example values for each variable
print("\n\nVARIABLE TYPES AND EXAMPLES")
print("=" * 80)

for variable in common_usable.index:
    if variable == "Disease":
        continue

    print(f"\n{variable}")
    print("  Type:", data[variable].dtype)
    print("  Examples:")

    values = (
        data[variable]
        .dropna()
        .astype(str)
        .value_counts()
        .head(5)
    )

    for value, count in values.items():
        print(f"    {value}  ({count})")# Candidate variable audit

candidate_vars = [
    "area_name/region",
    "area_name/kebele",
    "area_name/village",
    "area_name/_gps_latitude",
    "area_name/_gps_longitude",
    "area_name/_gps_altitude",
    "area_name/_gps_precision",
    "area_name/gps",
    "status_of_the_case/means_of_identification_of_the_case",
    "status_of_the_case/means_of_notification",
    "status_of_the_case/the_case_identified_by_community_volunter_health_development_army",
    "status_of_the_case/name_of_health_facility_case_registered_at",
    "type_of_suspect_case_animal/dose_the_animal_has_owner",
    "type_of_suspect_case_animal/treatment_immunization_status_of_the_case",
    "type_of_suspect_case_animal/type_of_sick_or_died_animal",
    "address_of_contact_person/date_of_the_report",
]

audit = []

for col in candidate_vars:
    if col not in data.columns:
        continue

    s = data[col].replace(r"^\s*$", pd.NA, regex=True)

    audit.append({
        "Variable": col,
        "Completeness_%": round(s.notna().mean() * 100, 1),
        "Unique_values": s.nunique(dropna=True),
        "Examples": " | ".join(
            s.dropna().astype(str).value_counts().head(5).index
        )
    })

audit_df = pd.DataFrame(audit)

print("\nCANDIDATE VARIABLE AUDIT")
print("=" * 100)
print(audit_df.to_string(index=False))

audit_df.to_csv(
    DATA_DIR / "candidate_variable_audit.csv",
    index=False
)