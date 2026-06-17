import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(df):
    # Separate features and label
    X = df.drop("Label", axis=1)
    y = df["Label"]

    # 🔥 CRITICAL FIX: handle inf / nan
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with column median (robust)
    X = X.fillna(X.median())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Scale AFTER cleaning
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.values, y_test.values
