import numpy as np


def create_temporal_sequences(features, targets, years, sequence_length=5):
    X, y, target_years = [], [], []
    for i in range(sequence_length, len(features)):
        sequence_years = years[i-sequence_length:i]
        if not np.all(np.diff(sequence_years) == 1):
            continue
        X.append(features[i-sequence_length:i])
        y.append(targets[i])
        target_years.append(years[i])
    return np.asarray(X), np.asarray(y), np.asarray(target_years)
