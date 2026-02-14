from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictive_analytics_system.data import load_and_validate_raw_data

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tenure buckets
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 100],
        labels=["0-12", "12-24", "24-48", "48+"],
        include_lowest=True,
    ).astype("category")

    # Contract stability
    df["contract_stability"] = df["Contract"].map(
        {
            "Month-to-month": "low",
            "One year": "medium",
            "Two year": "high",
        }
    )

    # Charges ratio
    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # Service flags
    df["has_internet"] = df["InternetService"] != "No"
    df["has_phone"] = df["PhoneService"] == "Yes"

    return df


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_baseline_model(config_path: Path) -> float:
    df = load_and_validate_raw_data()
    df = add_engineered_features(df)

    config = load_config(config_path)

    random_state = config["data"]["random_state"]

    model_cfg = config["model"]

    df = load_and_validate_raw_data()

    X = df.drop(columns=["Churn", "customerID"])
    y = (df["Churn"] == "Yes").astype(int)

    categorical_features = X.select_dtypes(include=["object", "string", "category", "bool"]).columns
    numerical_features = X.select_dtypes(exclude=["object", "string", "category", "bool"]).columns

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
        max_iter=model_cfg["max_iter"],
        class_weight=model_cfg["class_weight"],
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
    config_path = Path("configs/baseline.yaml")
    score = train_baseline_model(config_path)
    print(f"Baseline ROC-AUC: {score:.4f}")
