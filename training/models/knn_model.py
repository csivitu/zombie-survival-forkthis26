from sklearn.neighbors import KNeighborsClassifier


NAME = "K-Nearest Neighbors"


def build_model():
    return KNeighborsClassifier(
        n_neighbors=25,
        weights="distance",
        n_jobs=-1,
    )
