import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = (
    "/Users/belayneshmossiekndie/Desktop/Haqila/"
    "analysis/output/CGPP_Animal_Clean_Combined.xlsx"
)

df = pd.read_excel(file_path)

print("Dataset shape:", df.shape)


# ============================================================
# 2. CREATE TARGET
# ============================================================

# 1 = Rabies
# 0 = Anthrax + Brucellosis

df["Target"] = (df["Disease"] == "Rabies").astype(int)

print("\nTarget distribution:")
print(df["Target"].value_counts())

print("\nTarget definition:")
print("1 = Rabies")
print("0 = Non-Rabies (Anthrax + Brucellosis)")


# ============================================================
# 3. SELECT FEATURES
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

features = categorical_features + numerical_features

X = df[features].copy()
y = df["Target"].copy()


# ============================================================
# 4. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ============================================================
# 5. MODELS + HYPERPARAMETER SEARCH
# ============================================================

models = {

    "Logistic Regression": (
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ),
        {
            "model__C": [0.01, 0.1, 1, 10, 100]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", "log2"]
        }
    ),

    "XGBoost": (
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [2, 3, 5],
            "model__learning_rate": [0.03, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        }
    ),

    "LightGBM": (
        LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ),
        {
            "model__n_estimators": [100, 300],
            "model__num_leaves": [7, 15, 31],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [-1, 5, 10]
        }
    ),

    "CatBoost": (
        CatBoostClassifier(
            verbose=False,
            random_seed=42,
            loss_function="Logloss",
            thread_count=-1
        ),
        {
            "model__iterations": [200, 500],
            "model__depth": [4, 6, 8],
            "model__learning_rate": [0.03, 0.1],
            "model__l2_leaf_reg": [3, 10]
        }
    )
}


# ============================================================
# 6. OUTER + INNER CROSS-VALIDATION
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

inner_cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)


# ============================================================
# 7. RUN NESTED CROSS-VALIDATION
# ============================================================

results = []


for model_name, (model, param_grid) in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=inner_cv,
            n_jobs=-1,
            refit=True
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        balanced_accuracy = balanced_accuracy_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        sensitivity = recall_score(
            y_test,
            predictions
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        # Confusion matrix:
        # [[TN, FP],
        #  [FN, TP]]

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).ravel()

        specificity = tn / (tn + fp)

        fold_results.append({
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy,
            "F1": f1,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "ROC-AUC": auc
        })

        print(
            f"Fold {fold}: "
            f"Accuracy={accuracy:.3f}, "
            f"BA={balanced_accuracy:.3f}, "
            f"F1={f1:.3f}, "
            f"Sensitivity={sensitivity:.3f}, "
            f"Specificity={specificity:.3f}, "
            f"AUC={auc:.3f}"
        )

    # --------------------------------------------------------
    # Average across outer folds
    # --------------------------------------------------------

    fold_df = pd.DataFrame(fold_results)

    result = {
        "Model": model_name
    }

    for metric in fold_df.columns:

        result[f"{metric} Mean"] = fold_df[metric].mean()
        result[f"{metric} SD"] = fold_df[metric].std()

    results.append(result)


# ============================================================
# 8. FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Balanced Accuracy Mean",
    ascending=False
)

print("\n\n")
print("=" * 120)
print("FINAL NESTED CROSS-VALIDATION RESULTS")
print("=" * 120)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 9. PRESENTATION TABLE
# ============================================================

presentation_table = results_df[
    [
        "Model",
        "Accuracy Mean",
        "Balanced Accuracy Mean",
        "F1 Mean",
        "Sensitivity Mean",
        "Specificity Mean",
        "ROC-AUC Mean"
    ]
].copy()

print("\n\n")
print("=" * 100)
print("PRESENTATION TABLE")
print("=" * 100)

print(
    presentation_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)