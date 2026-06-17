import numpy as np
import xgboost as xgb


def select_features(X_train, X_test, y_train, y_test, top_k=20):
    """
    Memory-safe feature selection using XGBoost importance
    (GA REMOVED)
    """
    print("🔹 Selecting features using XGBoost importance...")

    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=1
    )

    model.fit(X_train, y_train)

    importances = model.feature_importances_
    selected_features = np.argsort(importances)[-top_k:]

    return selected_features
