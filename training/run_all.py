import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, load_train_test_split
from models import (
    knn_model,
    logistic_regression_model,
    random_forest_model,
    svm_model,
    xgboost_model,
)


MODEL_MODULES = [
    logistic_regression_model,
    random_forest_model,
    xgboost_model,
    svm_model,
    knn_model,
]

# SVM/KNN scale badly with row count (SVM training and KNN prediction are
# both roughly quadratic in the number of training rows). Training those two
# on a random subsample keeps run_all.py finishing in a reasonable time
# without changing how Logistic Regression/Random Forest/XGBoost are trained.
SUBSAMPLE_MODELS = {"Support Vector Machine", "K-Nearest Neighbors"}
MAX_SUBSAMPLE_ROWS = 8000

BEST_MODEL_PATH = Path(__file__).parent / "models" / "best_model.joblib"


def build_xgboost_with_class_weight(y_train):
    positive = (y_train == 1).sum()
    negative = (y_train == 0).sum()
    scale_pos_weight = negative / max(positive, 1)
    return xgboost_model.build_model(scale_pos_weight=scale_pos_weight)


def build_model_for(module, y_train):
    if module is xgboost_model:
        return build_xgboost_with_class_weight(y_train)
    return module.build_model()


def maybe_subsample(name, X_train, y_train, random_state=42):
    if name not in SUBSAMPLE_MODELS or len(X_train) <= MAX_SUBSAMPLE_ROWS:
        return X_train, y_train

    X_sample = X_train.sample(n=MAX_SUBSAMPLE_ROWS, random_state=random_state)
    y_sample = y_train.loc[X_sample.index]
    return X_sample, y_sample


def train_and_evaluate_all():
    X_train, X_test, y_train, y_test = load_train_test_split()

    results = []

    for module in MODEL_MODULES:
        name = module.NAME
        model = build_model_for(module, y_train)

        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", model),
        ])

        X_fit, y_fit = maybe_subsample(name, X_train, y_train)

        print(f"Training {name} on {len(X_fit):,} rows...")
        pipeline.fit(X_fit, y_fit)

        proba_deceased = pipeline.predict_proba(X_test)[:, 1]
        predicted_class = pipeline.predict(X_test)

        auc = roc_auc_score(y_test, proba_deceased)
        accuracy = accuracy_score(y_test, predicted_class)

        results.append({
            "name": name,
            "pipeline": pipeline,
            "auc": auc,
            "accuracy": accuracy,
        })

    return results, X_test, y_test


def print_comparison(results):
    ranked = sorted(results, key=lambda r: r["auc"], reverse=True)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (ranked by ROC-AUC on held-out test set)")
    print("=" * 70)

    for rank, result in enumerate(ranked, start=1):
        print(
            f"{rank}. {result['name']:<25} "
            f"AUC: {result['auc']:.4f}   "
            f"Accuracy: {result['accuracy']:.4f}"
        )

    return ranked[0]


def predict_survival_percentage(pipeline, person):
    """
    person: dict of feature_name -> value, e.g.
        {"RIDAGEYR": 45, "RIAGENDR": 1, "RIDRETH1": 3,
         "DMDEDUC2": 4, "DMDMARTL": 1, "INDFMPIR": 2.5,
         "DMDCITZN": 1, "DMDHHSIZ": 3}

    Returns a single number: the estimated percent chance of survival.
    """

    row = pd.DataFrame([person])
    proba_deceased = pipeline.predict_proba(row)[0, 1]
    return (1 - proba_deceased) * 100


def main():
    results, X_test, y_test = train_and_evaluate_all()
    best = print_comparison(results)

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], BEST_MODEL_PATH)

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best['name']} (AUC: {best['auc']:.4f})")
    print(f"Saved to: {BEST_MODEL_PATH}")
    print("=" * 70)

    # Demo: show what a single prediction looks like using the best model,
    # on the first person in the held-out test set.
    sample_person = X_test.iloc[[0]]
    survival_percentage = predict_survival_percentage(best["pipeline"], sample_person.iloc[0].to_dict())

    print(f"\nExample prediction for one test-set person:")
    print(sample_person.to_dict(orient="records")[0])
    print(f"\nEstimated survival probability: {survival_percentage:.1f}%")


if __name__ == "__main__":
    main()
