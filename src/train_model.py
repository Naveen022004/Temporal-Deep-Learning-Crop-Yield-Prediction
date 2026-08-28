"""Model construction and training utilities.

The defaults intentionally mirror the Colab notebook so refactoring does not
change the experiment configuration or expected outcomes.
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import Ridge


def build_lstm(input_shape):
    return Sequential([
        Input(shape=input_shape),
        LSTM(16, dropout=0.10, recurrent_dropout=0.0, kernel_regularizer=l2(1e-4)),
        Dense(8, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.10),
        Dense(1),
    ])


def compile_lstm(model):
    model.compile(
        optimizer=Adam(learning_rate=3e-4, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )
    return model


def make_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, min_delta=1e-4),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
    ]


def train_lstm(model, X_fit, y_fit, X_val, y_val, epochs=300, batch_size=8):
    return model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=min(batch_size, len(X_fit)),
        verbose=1,
        shuffle=False,
        callbacks=make_callbacks(),
    )


def train_ridge_baseline(X_train, y_train, alpha=10.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    return model


def predict_ridge(model, X):
    return model.predict(X.reshape(len(X), -1))
