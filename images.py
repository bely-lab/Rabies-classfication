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