# Candidate variable audit

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