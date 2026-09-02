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
        print(f"    {value}  ({count})")