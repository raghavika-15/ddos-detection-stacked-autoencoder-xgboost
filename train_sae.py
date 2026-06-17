import numpy as np
from data_loader import load_data
from preprocessing import preprocess
from stacked_autoencoder import build_sae
from tensorflow.keras.callbacks import EarlyStopping

df = load_data()
X_train, X_test, y_train, y_test = preprocess(df)

autoencoder, encoder = build_sae(X_train.shape[1])

early = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

autoencoder.fit(
    X_train, X_train,
    epochs=20,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early],
    verbose=1
)

np.save("X_train_enc.npy", encoder.predict(X_train))
np.save("X_test_enc.npy", encoder.predict(X_test))
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ SAE completed")
