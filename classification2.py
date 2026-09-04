import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

file_path = "/Users/belayneshmossiekndie/Desktop/Haqila/analysis/output/CGPP_Animal_Clean_Combined.xlsx"

df = pd.read_excel(file_path)


# --------------------------------------------------
# 2. Create Rabies vs Non-Rabies target
# --------------------------------------------------

df["Target"] = (df["Disease"] == "Rabies").astype(int)

print("Dataset shape:", df.shape)

print("\nTarget distribution:")
print(df["Target"].value_counts())

print("\nTarget meaning:")
print("1 = Rabies")
print("0 = Non-Rabies (Anthrax + Brucellosis)")


# --------------------------------------------------
# 3. Selected variables
# --------------------------------------------------

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

X = df[features]
y = df["Target"]


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 6. Cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "f1": make_scorer(f1_score)
}


# --------------------------------------------------
# 7. Train and evaluate
# --------------------------------------------------

results = []

for model_name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results.append({
        "Model": model_name,
        "Balanced Accuracy Mean": scores["test_balanced_accuracy"].mean(),
        "Balanced Accuracy SD": scores["test_balanced_accuracy"].std(),
        "F1 Mean": scores["test_f1"].mean(),
        "F1 SD": scores["test_f1"].std()
    })


# --------------------------------------------------
# 8. Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\nClassification Results:")
print(results_df.to_string(index=False))