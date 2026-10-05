"""Task 5: split and preprocess."""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from . import config


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
):
    """Return X_train, X_test, y_train, y_test.

    Stratified on config.TARGET and seeded with config.SEED.
    """

    X = df.drop(
        columns=[config.TARGET, "mag"],
        errors="ignore",
    )

    y = df[config.TARGET]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config.SEED,
        stratify=y,
    )


def build_preprocessor(
    scaler: str = "robust",
) -> ColumnTransformer:
    """Build numeric and categorical preprocessing pipelines."""

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }

    if scaler not in scalers:
        raise ValueError(
            "scaler must be one of: standard, minmax, robust"
        )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                scalers[scaler],
            ),
        ]
    )

    nominal_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                config.NUMERIC,
            ),
            (
                "nominal",
                nominal_pipeline,
                config.NOMINAL,
            ),
        ],
        remainder="drop",
    )