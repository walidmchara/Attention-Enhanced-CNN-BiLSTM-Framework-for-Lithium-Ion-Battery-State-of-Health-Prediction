"""Exact CNN-BiLSTM-attention network specification from the source notebooks."""


def build_model(n_features: int = 7):
    from tensorflow.keras.layers import (
        Bidirectional, Conv1D, Dense, Dropout, Input,
        LSTM, MaxPooling1D, Multiply,
    )
    from tensorflow.keras.models import Model

    inputs = Input(shape=(n_features, 1))
    hidden = Conv1D(filters=64, kernel_size=3, activation="relu")(inputs)
    hidden = MaxPooling1D(pool_size=2)(hidden)
    hidden = Dropout(0.3)(hidden)
    hidden = Bidirectional(
        LSTM(100, activation="relu"), name="bilstm"
    )(hidden)
    attention = Dense(200, activation="sigmoid", name="attention_vec")(hidden)
    hidden = Multiply()([hidden, attention])
    outputs = Dense(1, activation="relu")(hidden)
    model = Model(inputs=inputs, outputs=outputs)
    # Accuracy is intentionally retained as an original (poor regression) metric.
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model
