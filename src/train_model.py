import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.linear_model import Ridge


def build_lstm(input_shape):
    model = Sequential([
        LSTM(32, input_shape=input_shape, kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(16, activation="relu", kernel_regularizer=l2(1e-4)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model


def train_lstm(X_train, y_train, X_val=None, y_val=None, epochs=150, batch_size=8):
    model = build_lstm(X_train.shape[1:])
    callbacks = [
        EarlyStopping(monitor="val_loss" if X_val is not None else "loss", patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss" if X_val is not None else "loss", patience=8, factor=0.5),
    ]
    validation_data = (X_val, y_val) if X_val is not None else None
    history = model.fit(X_train, y_train, validation_data=validation_data, epochs=epochs,
                        batch_size=batch_size, shuffle=False, verbose=1, callbacks=callbacks)
    return model, history


def train_ridge_baseline(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    return model


def predict_ridge(model, X):
    return model.predict(X.reshape(len(X), -1))
