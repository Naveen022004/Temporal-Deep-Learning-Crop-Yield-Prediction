import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_dataframe(df, feature_columns, target_column):
    data = df.copy().dropna(subset=feature_columns + [target_column])
    return data.sort_values("year").reset_index(drop=True)


def fit_feature_scaler(train_df, feature_columns):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns])
    return scaler


def transform_features(df, feature_columns, scaler):
    return scaler.transform(df[feature_columns])


def fit_target_scaler(train_targets):
    scaler = StandardScaler()
    scaler.fit(pd.DataFrame(train_targets))
    return scaler
