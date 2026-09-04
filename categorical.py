from pathlib import Path
import pandas as pd
from scipy.stats import chi2_contingency


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

DATA_PATH = Path(
    "/Users/belayneshmossiekndie/Desktop/Haqila/analysis/output/CGPP_Animal_Clean_Combined.xlsx"
)

OUTPUT_PATH = DATA_PATH.parent / "categorical_feature_analysis.xlsx"

df = pd.read_excel(DATA_PATH)


# --------------------------------------------------
# 2. Variables to analyse

categorical_vars = [
    "Region",
    "Animal_Type",
    "Animal_Ownership",
    "Immunization_Status",
    "Identification_Method",
    "Notification_Method",
    "Community_HDA_Identified",
]


# --------------------------------------------------
# 3. Cramér's V
# --------------------------------------------------

def cramers_v(table):

    chi2, p, dof, expected = chi2_contingency(table)

    n = table.sum().sum()
    r, k = table.shape

    phi2 = chi2 / n

    # Bias correction
    phi2corr = max(
        0,
        phi2 - ((k - 1) * (r - 1)) / (n - 1)
    )

    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    v = (phi2corr / min(kcorr - 1, rcorr - 1)) ** 0.5

    return chi2, p, v


# 4. Analyse variables


results = []

tables = {}

for variable in categorical_vars:

    print("\n" + "=" * 70)
    print(variable)
    print("=" * 70)

    table = pd.crosstab(
        df[variable],
        df["Disease"]
    )

    print(table)

    chi2, p, v = cramers_v(table)

    results.append({
        "Variable": variable,
        "Chi_square": chi2,
        "p_value": p,
        "Cramers_V": v
    })

    tables[variable] = table


# --------------------------------------------------
# 5. Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Cramers_V",
    ascending=False
)

print("\n" + "=" * 70)
print("FEATURE ASSOCIATION WITH DISEASE")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# --------------------------------------------------
# 6. Save everything
# --------------------------------------------------

with pd.ExcelWriter(
    OUTPUT_PATH,
    engine="openpyxl"
) as writer:

    results_df.to_excel(
        writer,
        sheet_name="Feature_Association",
        index=False
    )

    for variable, table in tables.items():

        table.to_excel(
            writer,
            sheet_name=variable[:31]
        )


print("\nSaved:")
print(OUTPUT_PATH)