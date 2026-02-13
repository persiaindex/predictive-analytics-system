from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictive_analytics_system.data import load_and_validate_raw_data

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def train_baseline_model(random_state: int = 42) -> float:
    df = load_and_validate_raw_data()

    X = df.drop(columns=["Churn", "customerID"])
    y = (df["Churn"] == "Yes").astype(int)

    categorical_features = X.select_dtypes(include="object").columns
    numerical_features = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=1,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    joblib.dump(pipeline, ARTIFACTS_DIR / "baseline_model.joblib")

    return roc_auc


if __name__ == "__main__":
    score = train_baseline_model()
    print(f"Baseline ROC-AUC: {score:.4f}")
