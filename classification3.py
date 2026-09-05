import pandas as pd
import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    make_scorer
)

# Advanced gradient boosting models
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
# 6. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 7. SCORING
# ============================================================

scoring = {
    "balanced_accuracy": make_scorer(
        balanced_accuracy_score
    ),

    "f1": make_scorer(
        f1_score
    ),

    "recall": make_scorer(
        recall_score
    ),

    "roc_auc": "roc_auc"
}


# ============================================================
# 8. NESTED CV
# ============================================================
#
# Outer CV estimates generalization performance.
# Inner CV selects hyperparameters.
#
# This prevents using the same folds both to tune
# and evaluate the model.
# ============================================================

results = []


for model_name, (model, param_grid) in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    outer_scores = {
        "balanced_accuracy": [],
        "f1": [],
        "recall": [],
        "roc_auc": []
    }

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y), start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        # Inner CV for hyperparameter selection
        inner_cv = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=42
        )

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

        outer_scores["balanced_accuracy"].append(
            balanced_accuracy_score(
                y_test,
                predictions
            )
        )

        outer_scores["f1"].append(
            f1_score(
                y_test,
                predictions
            )
        )

        outer_scores["recall"].append(
            recall_score(
                y_test,
                predictions
            )
        )

        outer_scores["roc_auc"].append(
            roc_auc_score(
                y_test,
                probabilities
            )
        )

        print(
            f"Fold {fold}: "
            f"BA={outer_scores['balanced_accuracy'][-1]:.3f}, "
            f"F1={outer_scores['f1'][-1]:.3f}, "
            f"Recall={outer_scores['recall'][-1]:.3f}, "
            f"AUC={outer_scores['roc_auc'][-1]:.3f}"
        )

    results.append({
        "Model": model_name,

        "Balanced Accuracy Mean":
            np.mean(outer_scores["balanced_accuracy"]),

        "Balanced Accuracy SD":
            np.std(outer_scores["balanced_accuracy"]),

        "F1 Mean":
            np.mean(outer_scores["f1"]),

        "F1 SD":
            np.std(outer_scores["f1"]),

        "Recall Mean":
            np.mean(outer_scores["recall"]),

        "Recall SD":
            np.std(outer_scores["recall"]),

        "ROC-AUC Mean":
            np.mean(outer_scores["roc_auc"]),

        "ROC-AUC SD":
            np.std(outer_scores["roc_auc"])
    })


# ============================================================
# 9. FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Balanced Accuracy Mean",
    ascending=False
)

print("\n\n")
print("=" * 90)
print("FINAL NESTED CROSS-VALIDATION RESULTS")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)