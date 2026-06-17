from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def build_sae(input_dim):
    inp = Input(shape=(input_dim,))
    e1 = Dense(32, activation="relu")(inp)
    e2 = Dense(16, activation="relu")(e1)
    d1 = Dense(32, activation="relu")(e2)
    out = Dense(input_dim, activation="linear")(d1)

    autoencoder = Model(inp, out)
    encoder = Model(inp, e2)

    autoencoder.compile(
        optimizer=Adam(0.001),
        loss="mse"
    )

    return autoencoder, encoder
