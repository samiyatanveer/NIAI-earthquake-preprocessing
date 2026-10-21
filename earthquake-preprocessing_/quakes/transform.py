"""Split data and build a train-fitted preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler
from . import config


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Return stratified train/test split; remove target and raw magnitude from X."""
    if config.TARGET not in df.columns:
        raise ValueError(f"Target column '{config.TARGET}' is missing")
    X = df.drop(columns=[config.TARGET, "mag"], errors="ignore")
    y = df[config.TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=config.SEED, stratify=y)


def build_preprocessor(scaler: str = "robust", degree: int = 1) -> ColumnTransformer:
    """Impute numeric/categorical missing values, scale numbers, one-hot categories."""
    scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}
    if scaler not in scalers:
        raise ValueError("scaler must be one of: standard, minmax, robust")
    if degree < 1:
        raise ValueError("degree must be at least 1")
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if degree > 1:
        numeric_steps.append(("polynomial", PolynomialFeatures(degree=degree, include_bias=False)))
    numeric_steps.append(("scaler", scalers[scaler]))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric, config.NUMERIC), ("nominal", categorical, config.NOMINAL)], remainder="drop")
