from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import load_processed_dataset


TARGET = "MORTSTAT"

# RIDAGEYR/INDFMPIR/DMDHHSIZ are genuinely continuous; the rest are coded
# categories even though NHANES stores them as numbers.
NUMERIC_FEATURES = ["RIDAGEYR", "INDFMPIR", "DMDHHSIZ"]
CATEGORICAL_FEATURES = ["RIAGENDR", "RIDRETH1", "DMDEDUC2", "DMDMARTL", "DMDCITZN"]


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def load_train_test_split(test_size=0.2, random_state=42):
    df = load_processed_dataset()

    features = [
        column for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES
        if column in df.columns
    ]

    X = df[features]
    y = df[TARGET]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
