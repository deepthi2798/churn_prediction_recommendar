"""Data loading + preprocessing for the churn model."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CAT_COLS = ["Geography", "Gender"]
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]
TARGET = "Exited"


def load_and_prepare(csv_path: str, test_size: float = 0.25, random_state: int = 42):
    """Load the raw CSV, encode categoricals, and split train/test.

    Returns: X_train, X_test, y_train, y_test, feature_cols, encoders, raw_test_df
    raw_test_df keeps the original (unencoded) columns for the test rows, so the
    UI can display human-readable values (e.g. "France" instead of 0).
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    df_model = df.copy()
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        encoders[col] = le

    feature_cols = [c for c in df_model.columns if c != TARGET]
    X, y = df_model[feature_cols], df_model[TARGET]

    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        X, y, df.loc[X.index], test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_cols, encoders, raw_test
