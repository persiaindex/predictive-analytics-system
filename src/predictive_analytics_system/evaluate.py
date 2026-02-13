import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from predictive_analytics_system.data import load_and_validate_raw_data

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "baseline_model.joblib"


def evaluate_model(threshold: float = 0.5) -> None:
    df = load_and_validate_raw_data()

    X = df.drop(columns=["Churn", "customerID"])
    y = (df["Churn"] == "Yes").astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    y_true = y_test
    X = X_test

    pipeline = joblib.load(MODEL_PATH)

    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    print(f"\nROC curve points: {len(fpr)}")
    print(f"PR curve points: {len(precision)}")


def slice_analysis() -> None:
    df = load_and_validate_raw_data()

    X = df.drop(columns=["Churn", "customerID"])
    # y_true = (df["Churn"] == "Yes").astype(int)

    pipeline = joblib.load(MODEL_PATH)
    y_proba = pipeline.predict_proba(X)[:, 1]

    df_eval = df.copy()
    df_eval["churn_proba"] = y_proba
    df_eval["predicted_churn"] = y_proba >= 0.5

    print("\nChurn rate by contract type:")
    print(df_eval.groupby("Contract")["predicted_churn"].mean().sort_values(ascending=False))

    print("\nChurn rate by tenure bucket:")
    df_eval["tenure_bucket"] = pd.cut(df_eval["tenure"], bins=[0, 12, 24, 48, 72])
    print(df_eval.groupby("tenure_bucket")["predicted_churn"].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the churn model.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for converting probabilities to class labels.",
    )
    args = parser.parse_args()

    evaluate_model(threshold=args.threshold)
    print("\nPerforming slice analysis...")
    slice_analysis()
