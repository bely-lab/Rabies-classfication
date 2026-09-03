from pathlib import Path
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

DATA_DIR = Path("/Users/belayneshmossiekndie/Desktop/Haqila/analysis/data")
OUTPUT_DIR = DATA_DIR.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Input files
# ============================================================

files = {
    "Rabies": DATA_DIR / "CGPP_Rabies_Animal.xlsx",
    "Anthrax": DATA_DIR / "CGPP_Anthrax_Animal.xlsx",
    "Brucellosis": DATA_DIR / "CGPP_Brucellosis_Animal.xlsx",
}


# ============================================================
# 3. Variables to keep
# ============================================================

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
}


# ============================================================
# 4. Find image column automatically
# ============================================================

def find_image_column(columns):

    candidates = []

    for col in columns:
        col_lower = str(col).lower()

        if "image" in col_lower:
            candidates.append(col)

    return candidates


# ============================================================
# 5. Load datasets
# ============================================================

datasets = []

for disease, file_path in files.items():

    print(f"\nLoading {disease}: {file_path.name}")

    df = pd.read_excel(file_path)

    print(f"Original shape: {df.shape}")

    # --------------------------------------------------------
    # Check required tabular columns
    # --------------------------------------------------------

    missing = [
        original_name
        for original_name in selected_columns.values()
        if original_name not in df.columns
    ]

    if missing:
        print("\nMissing columns:")
        for col in missing:
            print(f"  {col}")

        print("\nAvailable columns containing relevant words:")
        for col in df.columns:
            if any(
                word in str(col).lower()
                for word in ["region", "gps", "animal", "owner",
                             "immun", "identif", "notif", "community"]
            ):
                print(f"  {col}")

        raise ValueError(
            f"Required columns missing from {disease} dataset."
        )

    # --------------------------------------------------------
    # Find image column
    # --------------------------------------------------------

    image_candidates = find_image_column(df.columns)

    print("\nImage column candidates:")
    for col in image_candidates:
        print(f"  {col}")

    if len(image_candidates) == 0:
        print("WARNING: No image column found.")
        image_column = None

    elif len(image_candidates) == 1:
        image_column = image_candidates[0]

    else:
        # Prefer a column specifically containing animal + image
        animal_image = [
            col for col in image_candidates
            if "animal" in str(col).lower()
        ]

        if animal_image:
            image_column = animal_image[0]
        else:
            image_column = image_candidates[0]

    if image_column:
        print(f"Using image column: {image_column}")

    # --------------------------------------------------------
    # Select tabular variables
    # --------------------------------------------------------

    clean = df[list(selected_columns.values())].copy()

    # Rename
    clean.rename(
        columns={
            original: new
            for new, original in selected_columns.items()
        },
        inplace=True
    )

    # --------------------------------------------------------
    # Add image column
    # --------------------------------------------------------

    if image_column:
        clean["Animal_Image"] = df[image_column]
    else:
        clean["Animal_Image"] = pd.NA

    # --------------------------------------------------------
    # Add disease label
    # --------------------------------------------------------

    clean.insert(0, "Disease", disease)

    datasets.append(clean)


# ============================================================
# 6. Combine
# ============================================================

combined = pd.concat(
    datasets,
    ignore_index=True
)


# ============================================================
# 7. Basic cleaning
# ============================================================

# Empty strings -> missing
combined = combined.replace(
    r"^\s*$",
    pd.NA,
    regex=True
)

combined.reset_index(drop=True, inplace=True)


# ============================================================
# 8. Summary
# ============================================================

print("\n" + "=" * 60)
print("COMBINED DATASET")
print("=" * 60)

print(f"Rows: {len(combined)}")
print(f"Columns: {len(combined.columns)}")

print("\nDisease distribution:")
print(combined["Disease"].value_counts())

print("\nImage availability:")

image_available = combined["Animal_Image"].notna().sum()
image_missing = combined["Animal_Image"].isna().sum()

print(f"  Image reference available: {image_available}")
print(f"  No image reference:        {image_missing}")

print("\nColumns:")
for i, col in enumerate(combined.columns, 1):
    print(f"{i:2}. {col}")


# ============================================================
# 9. Save
# ============================================================

csv_path = OUTPUT_DIR / "CGPP_Animal_Clean_Combined.csv"
xlsx_path = OUTPUT_DIR / "CGPP_Animal_Clean_Combined.xlsx"

combined.to_csv(csv_path, index=False)
combined.to_excel(xlsx_path, index=False)

print("\nSaved:")
print(csv_path)
print(xlsx_path)


# ============================================================
# 10. Preview
# ============================================================

print("\nFirst 5 rows:")
print(combined.head().to_string(index=False))