from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw Telco churn dataset from CSV.

    Parameters
    ----------
    path : Path
        Path to raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at {path}")

    df = pd.read_csv(path)
    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    """
    Perform basic validation checks on raw data.

    Raises
    ------
    ValueError
        If validation fails.
    """
    if df.empty:
        raise ValueError("Dataset is empty")

    required_columns = {"customerID", "Churn"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df["customerID"].isnull().any():
        raise ValueError("customerID contains null values")

    if df["customerID"].duplicated().any():
        raise ValueError("customerID contains duplicate values")


def load_and_validate_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = load_raw_data(path)
    validate_raw_data(df)
    return df
