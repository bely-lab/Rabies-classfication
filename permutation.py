import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_scores
from lightgbm import LGBMClassifier


# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv("CGPP_Animal_Clean_Combined.csv")

# Binary target: Rabies vs Non-Rabies
df["Target"] = (df["Disease"] == "Rabies").astype(int)

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
# 2. Preprocessing
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
# 3. Models
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
# 4. Permutation importance using cross-validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

all_results = []

for model_name, model in models.items():

    print(f"\nRunning: {model_name}")

    fold_importances = []
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        # Baseline performance
        baseline_pred = pipeline.predict(X_test)
        baseline_ba = balanced_accuracy_score(
            y_test,
            baseline_pred
        )

        # Permute ORIGINAL FEATURES
        # This keeps one-hot encoded categories together.
        rng = np.random.RandomState(42 + fold)

        fold_importance = {}

        for feature in features:

            X_test_permuted = X_test.copy()

            X_test_permuted[feature] = rng.permutation(
                X_test_permuted[feature].values
            )

            permuted_pred = pipeline.predict(X_test_permuted)

            permuted_ba = balanced_accuracy_score(
                y_test,
                permuted_pred
            )

            # Performance decrease caused by permutation
            importance = baseline_ba - permuted_ba

            fold_importance[feature] = importance

        fold_importances.append(fold_importance)
        fold_scores.append(baseline_ba)

        print(
            f"  Fold {fold}: "
            f"BA = {baseline_ba:.3f}"
        )

    # Convert folds to dataframe
    importance_df = pd.DataFrame(fold_importances)

    summary = pd.DataFrame({
        "Feature": features,
        "Importance_Mean": importance_df.mean().values,
        "Importance_SD": importance_df.std().values
    })

    summary["Model"] = model_name

    all_results.append(summary)


# ============================================================
# 5. Combine results
# ============================================================

importance_results = pd.concat(
    all_results,
    ignore_index=True
)

importance_results = importance_results.sort_values(
    ["Model", "Importance_Mean"],
    ascending=[True, False]
)

print("\n==============================")
print("PERMUTATION IMPORTANCE")
print("==============================")

for model_name in models:

    print(f"\n{model_name}")

    display(
        importance_results[
            importance_results["Model"] == model_name
        ][
            ["Feature", "Importance_Mean", "Importance_SD"]
        ].round(4)
    )


# ============================================================
# 6. Plot
# ============================================================

for model_name in models:

    result = importance_results[
        importance_results["Model"] == model_name
    ].sort_values("Importance_Mean")

    plt.figure(figsize=(9, 6))

    plt.barh(
        result["Feature"],
        result["Importance_Mean"],
        xerr=result["Importance_SD"]
    )

    plt.xlabel(
        "Decrease in Balanced Accuracy after permutation"
    )

    plt.ylabel("Feature")

    plt.title(
        f"{model_name}: Permutation Importance"
    )

    plt.tight_layout()
    plt.show()