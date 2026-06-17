import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def train_xgboost(X_train, y_train):
    params = {
        "n_estimators": [200, 300],
        "max_depth": [6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    model = xgb.XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False
    )

    grid = GridSearchCV(
        model,
        params,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid.best_estimator_
