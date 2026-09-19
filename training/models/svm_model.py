from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC


NAME = "Support Vector Machine"


def build_model():
    # SVC only outputs a decision boundary by default, not probabilities.
    # Wrapping it in CalibratedClassifierCV (Platt scaling) adds predict_proba.
    svc = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42,
    )
    return CalibratedClassifierCV(svc, ensemble=False)
