import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

X_train = np.load("X_train_enc.npy")
y_train = np.load("y_train.npy")

params = {
    "n_estimators": [150],
    "max_depth": [5, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8]
}

model = xgb.XGBClassifier(
    eval_metric="logloss",
    n_jobs=1
)

grid = GridSearchCV(
    model,
    params,
    scoring="f1",
    cv=3,
    verbose=1
)
xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    n_jobs=1
)

grid.fit(X_train, y_train)
grid.best_estimator_.save_model("ddos_xgb_model.json")

print("✅ XGBoost trained & saved")
