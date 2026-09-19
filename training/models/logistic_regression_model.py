from sklearn.linear_model import LogisticRegression


NAME = "Logistic Regression"


def build_model():
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
