from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

DATA_DIR = Path("/Users/belayneshmossiekndie/Desktop/Haqila/analysis/data")
OUTPUT_DIR = DATA_DIR.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Files to combine
# --------------------------------------------------

files = {
    "Rabies": DATA_DIR / "CGPP_Rabies_Animal.xlsx",
    "Anthrax": DATA_DIR / "CGPP_Anthrax_Animal.xlsx",
    "Brucellosis": DATA_DIR / "CGPP_Brucellosis_Animal.xlsx",
}


# --------------------------------------------------
# 3. Original columns we selected
# --------------------------------------------------

selected_columns = {
    "Region":
        "area_name/region",

    "Latitude":
        "area_name/_gps_latitude",

    "Longitude":
        "area_name/_gps_longitude",

    "Altitude":
        "area_name/_gps_altitude",

    "Animal_Type":
        "type_of_suspect_case_animal/type_of_sick_or_died_animal",

    "Animal_Ownership":
        "type_of_suspect_case_animal/dose_the_animal_has_owner",

    "Immunization_Status":
        "type_of_suspect_case_animal/treatment_immunization_status_of_the_case",

    "Identification_Method":
        "status_of_the_case/means_of_identification_of_the_case",

    "Notification_Method":
        "status_of_the_case/means_of_notification",

    "Community_HDA_Identified":
        "status_of_the_case/the_case_identified_by_community_volunter_health_development_army",

    "Animal_Image":
        "type_of_suspect_case_animal/animal_image",
}


# --------------------------------------------------
# 4. Load and combine
# --------------------------------------------------

datasets = []

for disease, file_path in files.items():

    print(f"\nLoading {disease}: {file_path.name}")

    df = pd.read_excel(file_path)

    print(f"  Original shape: {df.shape}")

    # Check that all required columns exist
    missing = [
        original_name
        for original_name in selected_columns.values()
        if original_name not in df.columns
    ]

    if missing:
        print("  Missing columns:")
        for col in missing:
            print(f"    {col}")
        raise ValueError(f"Required columns missing from {disease} dataset.")

    # Select variables
    clean = df[list(selected_columns.values())].copy()

    # Rename variables
    clean.rename(
        columns={
            original: new
            for new, original in selected_columns.items()
        },
        inplace=True
    )

    # Add disease label
    clean.insert(0, "Disease", disease)

    # Add host
    clean.insert(1, "Host", "Animal")

    datasets.append(clean)


# --------------------------------------------------
# 5. Combine datasets
# --------------------------------------------------

combined = pd.concat(
    datasets,
    ignore_index=True
)


# --------------------------------------------------
# 6. Basic cleaning
# --------------------------------------------------

# Convert blank strings to missing values
combined = combined.replace(r"^\s*$", pd.NA, regex=True)

# Remove completely empty rows
combined = combined.dropna(
    how="all",
    subset=[col for col in combined.columns if col not in ["Disease", "Host"]]
)

# Reset index
combined.reset_index(drop=True, inplace=True)


# --------------------------------------------------
# 7. Print summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("COMBINED CLEAN DATASET")
print("=" * 60)

print(f"Total records: {len(combined)}")
print(f"Total variables: {len(combined.columns)}")

print("\nDisease distribution:")
print(combined["Disease"].value_counts())

print("\nMissing values:")
missing_summary = combined.isna().sum()
missing_summary = missing_summary[missing_summary > 0]

if len(missing_summary) > 0:
    print(missing_summary)
else:
    print("No missing values.")


# --------------------------------------------------
# 8. Save
# --------------------------------------------------

csv_path = OUTPUT_DIR / "CGPP_Animal_Clean_Combined.csv"
xlsx_path = OUTPUT_DIR / "CGPP_Animal_Clean_Combined.xlsx"

combined.to_csv(csv_path, index=False)
combined.to_excel(xlsx_path, index=False)

print("\nSaved:")
print(csv_path)
print(xlsx_path)


# --------------------------------------------------
# 9. Show first rows
# --------------------------------------------------

print("\nFirst 5 rows:")
print(combined.head().to_string(index=False))