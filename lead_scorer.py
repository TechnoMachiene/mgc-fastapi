"""
Lead scoring -- same decisions as Part 3 (untitled7.py):
  - drop lead_id, crm_record_hash (identifiers, not predictive)
  - created_at -> hour_of_day, day_of_week, then the raw column is dropped
  - area: missing -> 'Unknown'
  - budget_pkr_lac, bedrooms, first_response_minutes, agent_experience_years:
    missing -> column median
  - source, city, area, property_type: one-hot encoded
  - LogisticRegression(solver="liblinear", random_state=42)

Looks for leads.csv next to this file (your real ~9,000-row export).
Falls back to data/leads_sample.csv (60 synthetic rows, same columns)
so the page runs before you've added the real file.
"""
import os
from datetime import datetime

import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(__file__)
REAL_CSV = os.path.join(BASE_DIR, "leads.csv")
SAMPLE_CSV = os.path.join(BASE_DIR, "data", "leads_sample.csv")

TARGET = "converted"
ID_COLS = ["lead_id", "crm_record_hash"]
NUMERIC_IMPUTE_COLS = ["budget_pkr_lac", "bedrooms", "first_response_minutes", "agent_experience_years"]
CATEGORICAL_COLS = ["source", "city", "area", "property_type"]

_model = None
_dummy_columns = []      # column order the model was trained on (post one-hot)
_form_fields = []        # columns shown on the web form (excludes ids/target/created_at)
_medians = {}
_using_sample = False


def _engineer(df: pd.DataFrame, fit: bool) -> pd.DataFrame:
    """Same feature engineering as the notebook, applied to either the
    training frame (fit=True, computes medians) or a single new lead
    (fit=False, reuses medians learned at training time)."""
    global _medians
    df = df.copy()

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["hour_of_day"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.dayofweek
    df["area"] = df["area"].fillna("Unknown")

    for col in NUMERIC_IMPUTE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if fit:
            _medians[col] = df[col].median()
        df[col] = df[col].fillna(_medians.get(col))

    df = df.drop(columns=[c for c in ID_COLS + ["created_at"] if c in df.columns])

    # Every remaining non-categorical column should be numeric. Form
    # submissions arrive as strings, so coerce them here; this is a
    # no-op on the already-numeric training data.
    for col in df.columns:
        if col not in CATEGORICAL_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    return df


def _load():
    global _model, _dummy_columns, _form_fields, _using_sample

    csv_path = REAL_CSV if os.path.exists(REAL_CSV) else SAMPLE_CSV
    _using_sample = csv_path == SAMPLE_CSV

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    _form_fields = [c for c in df.columns if c not in ID_COLS + ["created_at", TARGET]]

    processed = _engineer(df, fit=True)
    y = processed.pop(TARGET)

    # Any remaining stray missing values (columns the notebook didn't
    # explicitly impute) -> 0, so a sparse real-world CSV can't crash training.
    processed = processed.fillna(0)

    _model = LogisticRegression(solver="liblinear", random_state=42)
    _model.fit(processed, y)
    _dummy_columns = list(processed.columns)


_load()


def get_fields():
    return _form_fields


def using_sample_data() -> bool:
    return _using_sample


def score_lead(raw: dict) -> dict:
    row = {col: raw.get(col) for col in _form_fields}
    row["created_at"] = datetime.now().isoformat()  # not asked on the form; only the derived hour/day matter
    df = pd.DataFrame([row])

    processed = _engineer(df, fit=False)
    processed = processed.reindex(columns=_dummy_columns, fill_value=0)
    processed = processed.fillna(0)

    try:
        proba = float(_model.predict_proba(processed)[0][1])
    except Exception as e:
        return {"error": f"Could not score lead: {e}"}
    return {"probability": round(proba, 3)}
