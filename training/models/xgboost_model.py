from xgboost import XGBClassifier


NAME = "XGBoost"


def build_model(scale_pos_weight=1.0):
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
