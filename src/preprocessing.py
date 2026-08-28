"""Reusable preprocessing helpers for the crop-yield pipeline."""
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_dataframe(df, feature_columns, target_column):
    data = df.copy().dropna(subset=feature_columns + [target_column])
    return data.sort_values("year").reset_index(drop=True)


def fit_feature_scaler(train_array):
    """Fit a scaler only on training data to prevent leakage."""
    scaler = StandardScaler()
    scaler.fit(train_array)
    return scaler


def fit_target_scaler(train_targets):
    scaler = StandardScaler()
    scaler.fit(pd.DataFrame(train_targets))
    return scaler


def transform_target(targets, scaler):
    return scaler.transform(pd.DataFrame(targets)).ravel()


def inverse_transform_target(values, scaler):
    return scaler.inverse_transform(pd.DataFrame(values)).ravel()
