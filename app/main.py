"""
FastAPI server for the zombie survival model.

Run from the repo root:
    uvicorn app.main:app --reload

Requires a trained model on disk. Produce one first with:
    py training/run_all.py
"""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]

# run_all.py currently writes into training/models/, which is also a Python
# package. Prefer a plain artifacts/ directory, but keep reading the old
# location so an already-trained model still works.
MODEL_PATHS = [
    REPO_ROOT / "artifacts" / "best_model.joblib",
    REPO_ROOT / "training" / "models" / "best_model.joblib",
]

# Must match preprocessing.NUMERIC_FEATURES + CATEGORICAL_FEATURES, in the
# order the ColumnTransformer expects to find them on the DataFrame.
FEATURE_ORDER = [
    "RIDAGEYR",
    "INDFMPIR",
    "DMDHHSIZ",
    "RIAGENDR",
    "RIDRETH1",
    "DMDEDUC2",
    "DMDMARTL",
    "DMDCITZN",
]


class Person(BaseModel):
    RIDAGEYR: float = Field(ge=18, le=85, description="Age in years")
    RIAGENDR: int = Field(ge=1, le=2, description="1 = Male, 2 = Female")
    RIDRETH1: int = Field(ge=1, le=5, description="Race/ethnicity code")
    DMDEDUC2: int = Field(ge=1, le=5, description="Education level")
    DMDMARTL: int = Field(ge=1, le=6, description="Marital status")
    INDFMPIR: float = Field(ge=0, le=5, description="Family income-to-poverty ratio")
    DMDCITZN: int = Field(ge=1, le=2, description="1 = Citizen, 2 = Not a citizen")
    DMDHHSIZ: int = Field(ge=1, le=7, description="People in household")


class Prediction(BaseModel):
    survival_percentage: float
    mortality_percentage: float


app = FastAPI(title="Zombie Survival API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before anything goes public
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_pipeline():
    for path in MODEL_PATHS:
        if path.exists():
            return joblib.load(path)

    searched = ", ".join(str(p) for p in MODEL_PATHS)
    raise FileNotFoundError(
        f"No trained model found. Looked in: {searched}. Run `py training/run_all.py` first."
    )


@app.get("/health")
def health():
    try:
        load_pipeline()
    except FileNotFoundError as exc:
        return {"status": "no_model", "detail": str(exc)}
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(person: Person):
    try:
        pipeline = load_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    row = pd.DataFrame([person.model_dump()])[FEATURE_ORDER]

    proba_deceased = float(pipeline.predict_proba(row)[0, 1])

    return Prediction(
        survival_percentage=round((1 - proba_deceased) * 100, 1),
        mortality_percentage=round(proba_deceased * 100, 1),
    )