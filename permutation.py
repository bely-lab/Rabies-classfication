import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "output/CGPP_Animal_Clean_Combined.csv"
RESULTS_DIR = BASE_DIR / "output/permutation_importance"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

df = pd.read_csv(DATA_FILE)

# Binary target:
# 1 = Rabies
# 0 = Non-Rabies (Anthrax + Brucellosis)

df["Target"] = (df["Disease"] == "Rabies").astype(int)


# ============================================================
# 3. Features
# ============================================================

features = [
    "Region",
    "Latitude",
    "Longitude",
    "Altitude",
    "Animal_Type",
    "Animal_Ownership",
    "Immunization_Status",
    "Identification_Method",
    "Notification_Method",
    "Community_HDA_Identified"
]

X = df[features].copy()
y = df["Target"]


# ============================================================
# 4. Feature groups
# ============================================================

categorical_features = [
    "Region",
    "Animal_Type",
    "Animal_Ownership",
    "Immunization_Status",
    "Identification_Method",
    "Notification_Method",
    "Community_HDA_Identified"
]

numerical_features = [
    "Latitude",
    "Longitude",
    "Altitude"
]


# ============================================================
# 5. Preprocessing
# ============================================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("cat", categorical_pipeline, categorical_features),
    ("num", numerical_pipeline, numerical_features)
])


# ============================================================
# 6. Models
# ============================================================

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbosity=-1
    )
}


# ============================================================
# 7. Cross-validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 8. Permutation importance
# ============================================================

all_results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Running: {model_name}")
    print("=" * 60)

    fold_importances = []
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y), 1
    ):

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Fresh preprocessing/model for each fold
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        # ----------------------------------------------------
        # Baseline performance
        # ----------------------------------------------------

        baseline_pred = pipeline.predict(X_test)

        baseline_ba = balanced_accuracy_score(
            y_test,
            baseline_pred
        )

        fold_scores.append(baseline_ba)

        print(
            f"Fold {fold}: "
            f"Baseline BA = {baseline_ba:.3f}"
        )

        # ----------------------------------------------------
        # Permutation importance
        # ----------------------------------------------------

        rng = np.random.RandomState(42 + fold)

        fold_importance = {}

        for feature in features:

            X_test_permuted = X_test.copy()

            # Randomly shuffle this feature
            X_test_permuted[feature] = rng.permutation(
                X_test_permuted[feature].values
            )

            permuted_pred = pipeline.predict(
                X_test_permuted
            )

            permuted_ba = balanced_accuracy_score(
                y_test,
                permuted_pred
            )

            # How much performance decreased
            importance = baseline_ba - permuted_ba

            fold_importance[feature] = importance

        fold_importances.append(fold_importance)

    # ========================================================
    # 9. Summarize across folds
    # ========================================================

    importance_df = pd.DataFrame(fold_importances)

    summary = pd.DataFrame({
        "Feature": features,
        "Importance_Mean": importance_df.mean().values,
        "Importance_SD": importance_df.std().values
    })

    summary["Model"] = model_name

    summary = summary.sort_values(
        "Importance_Mean",
        ascending=False
    )

    all_results.append(summary)

    # ========================================================
    # 10. Print results
    # ========================================================

    print("\nFeature importance:")
    print(
        summary[
            [
                "Feature",
                "Importance_Mean",
                "Importance_SD"
            ]
        ].round(4).to_string(index=False)
    )

    # ========================================================
    # 11. Save individual model results
    # ========================================================

    safe_name = model_name.lower().replace(" ", "_")

    summary.to_csv(
        RESULTS_DIR / f"{safe_name}_importance.csv",
        index=False
    )

    # ========================================================
    # 12. Plot
    # ========================================================

    plot_data = summary.sort_values(
        "Importance_Mean",
        ascending=True
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        plot_data["Feature"],
        plot_data["Importance_Mean"],
        xerr=plot_data["Importance_SD"]
    )

    plt.axvline(
        0,
        linewidth=0.8
    )

    plt.xlabel(
        "Decrease in Balanced Accuracy after permutation"
    )

    plt.ylabel("Feature")

    plt.title(
        f"{model_name}: Permutation Importance"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / f"{safe_name}_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 13. Combined results
# ============================================================

all_results_df = pd.concat(
    all_results,
    ignore_index=True
)

all_results_df.to_csv(
    RESULTS_DIR / "all_permutation_importance.csv",
    index=False
)


# ============================================================
# 14. Final message
# ============================================================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(f"\nResults saved to:")
print(RESULTS_DIR)