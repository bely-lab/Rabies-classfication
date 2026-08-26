from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

files = {
    "Rabies": DATA_DIR / "CGPP_Rabies_Animal.xlsx",
    "Anthrax": DATA_DIR / "CGPP_Anthrax_Animal.xlsx",
    "Brucellosis": DATA_DIR / "CGPP_Brucellosis_Animal.xlsx",
}

datasets = {
    name: pd.read_excel(path)
    for name, path in files.items()
}


# Columns that seem relevant to the animal surveillance record
candidate_columns = [
    "area_name/type_of_suspect_case",
    "area_name/region",
    "area_name/zone_benshangul",
    "area_name/zone_gambella",
    "area_name/zone_oromia",
    "area_name/zone_snnp",
    "area_name/zone_somali",
    "area_name/kebele",
    "area_name/village",

    "status_of_the_case/means_of_identification_of_the_case",
    "status_of_the_case/means_of_notification",
    "status_of_the_case/the_case_identified_by_community_volunter_health_development_army",
    "status_of_the_case/who_identified_the_case",

    "type_of_suspect_case_animal/type_of_sick_or_died_animal",
    "type_of_suspect_case_animal/type_of_sick_or_died_animal_other",
    "type_of_suspect_case_animal/dose_the_animal_has_owner",
    "type_of_suspect_case_animal/treatment_immunization_status_of_the_case",
    "type_of_suspect_case_animal/indicate_the_illness_of_this_surveillance_report_animal",
    "type_of_suspect_case_animal/indicate_the_illness_of_this_surveillance_report_animal_signal",
    "type_of_suspect_case_animal/if_it_is_epidemic_total_number_of_deaths",

    "area_name/_gps_latitude",
    "area_name/_gps_longitude",
]


for name, df in datasets.items():

    print("\n" + "=" * 100)
    print(name)

    for column in candidate_columns:

        if column not in df.columns:
            continue

        print("\n" + "-" * 100)
        print(column)

        print(f"Missing: {df[column].isna().sum()} / {len(df)}")
        print(f"Unique: {df[column].nunique(dropna=True)}")

        values = df[column].dropna().astype(str).value_counts().head(10)

        print("Most common values:")
        print(values.to_string())


        print("\n" + "=" * 100)
print("IMAGE FIELD INSPECTION")

IMAGE_COLUMN = "type_of_suspect_case_animal/image"

for name, df in datasets.items():

    print("\n" + "-" * 100)
    print(name)

    valid = df[IMAGE_COLUMN].dropna()

    print(f"Image field populated: {len(valid)} / {len(df)}")

    print("\nExamples:")
    for value in valid.head(5):
        print(repr(value))