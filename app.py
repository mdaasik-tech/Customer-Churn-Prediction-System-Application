"""
Customer Churn Prediction System
================================
Machine Learning based churn prediction using XGBoost, built with Streamlit + Plotly.

The ML workflow is taken directly from the project notebook
(Customer_churn_prediction_Using_XGBoost.ipynb):

    Target            : churn (Yes / No)  -> LabelEncoder
    Identifier        : customer_id       -> dropped
    Numerical         : StandardScaler
    Categorical       : OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Model             : XGBClassifier(objective="binary:logistic", eval_metric="logloss")
    Split             : test_size=0.20, random_state=42, stratified
    Tuning            : RandomizedSearchCV(n_iter=100, cv=5, scoring="accuracy")
    Risk levels       : High >= 0.70, Medium >= 0.40, otherwise Low

Author : Mohammed Aasik
"""

import hashlib
import json
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# =============================================================================
# CONSTANTS  (all values below come from the original notebook)
# =============================================================================

APP_TITLE = "Customer Churn Prediction System"
AUTHOR_NAME = "Mohammed Aasik"
LINKEDIN_URL = "https://www.linkedin.com/in/mohammed-aasik-aspiring-machine-learning-engineer-257787433/"
GITHUB_URL = "https://github.com/mdaasik-tech"

DEFAULT_DATASET_PATH = Path(__file__).parent / "customer_churn_prediction_dataset.csv"
DEFAULT_TARGET = "churn"
DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ITER = 100
CV_FOLDS = 5
SCORING = "accuracy"
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40

# Hyper-parameter search space (identical to the notebook)
PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 200, 300, 400, 500, 700, 1000],
    "model__max_depth": [2, 3, 4, 5, 6, 7, 8, 10],
    "model__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
    "model__min_child_weight": [1, 2, 3, 5, 7, 10],
    "model__gamma": [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__reg_alpha": [0, 0.001, 0.01, 0.1, 1],
    "model__reg_lambda": [0.1, 0.5, 1, 2, 5, 10],
}

TARGET_NAME_HINTS = ["churn", "exited", "attrition", "left", "target", "label"]
POSITIVE_LABEL_HINTS = {"yes", "true", "1", "1.0", "churn", "churned", "left", "y", "t", "exited"}
BOOLEAN_WORDS = {"yes", "no", "true", "false", "y", "n", "t", "f"}
ID_NAME_PATTERN = re.compile(r"(^|[_\s\-])(id|uuid|guid|roll_?no|registration_?no)$", re.IGNORECASE)
ID_CAMEL_PATTERN = re.compile(r"[a-z0-9]ID$")
DATE_VALUE_PATTERN = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
MIN_ROWS = 50
MIN_CLASS_COUNT = 10
MAX_CATEGORY_LEVELS = 50

RISK_COLORS = {"Low Risk": "#2e9e5b", "Medium Risk": "#e0a526", "High Risk": "#d64545"}


# =============================================================================
# SMALL HELPERS
# =============================================================================


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def is_bool(series: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(series)


def is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def assign_risk(probability: float, high: float, medium: float) -> str:
    """Risk level logic from the notebook (thresholds are adjustable in the sidebar)."""
    if probability >= high:
        return "High Risk"
    if probability >= medium:
        return "Medium Risk"
    return "Low Risk"


def create_download_file(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes for st.download_button()."""
    return df.to_csv(index=False).encode("utf-8")


def dataframe_key(df: pd.DataFrame) -> str:
    """Short content hash used for cache keys and widget keys."""
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    cols = "|".join(map(str, df.columns)).encode("utf-8")
    return hashlib.md5(hashed + cols).hexdigest()[:12]


def show_error(message: str, exc: Exception | None = None) -> None:
    """Friendly error for users; traceback only inside an optional debug expander."""
    st.error(message)
    if exc is not None:
        with st.expander("🛠️ Developer details (optional)"):
            st.code("".join(traceback.format_exception(exc)))


# =============================================================================
# 1. DATA LOADING
# =============================================================================


@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    """Load the default churn dataset that ships with the project."""
    return pd.read_csv(DEFAULT_DATASET_PATH)


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_bytes: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV (raises pandas errors for empty / invalid files)."""
    import io

    return pd.read_csv(io.BytesIO(file_bytes))


# =============================================================================
# 2. CLEANING & VALIDATION
# =============================================================================


@st.cache_data(show_spinner=False)
def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Safe, target-independent cleaning:
    strip names/text, inf -> NaN, numeric strings -> numbers, date strings -> datetime,
    drop fully empty rows and exact duplicate rows. Nothing is imputed here.
    """
    df = raw.copy()
    report = {"rows_before": len(df), "numeric_converted": [], "date_converted": [], "inf_replaced": 0}

    df.columns = df.columns.astype(str).str.strip()  # strip column names

    for col in df.columns:
        s = df[col]
        if is_numeric(s):
            n_inf = int(np.isinf(s.astype(float)).sum())
            if n_inf:
                report["inf_replaced"] += n_inf
                df[col] = s.replace([np.inf, -np.inf], np.nan)
            continue
        if is_bool(s) or is_datetime(s):
            continue

        # Text-like column: strip whitespace, blank -> NaN
        text = s.astype("string").str.strip()
        text = text.replace("", pd.NA)
        non_null = text.dropna()
        if non_null.empty:
            df[col] = text.astype(object)
            continue

        # Numeric strings such as "85" or "1,200.5" (mixed-type columns are kept as text)
        converted = pd.to_numeric(non_null.str.replace(",", "", regex=False), errors="coerce")
        if converted.notna().mean() >= 0.90:
            full = pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce")
            df[col] = full.replace([np.inf, -np.inf], np.nan)
            report["numeric_converted"].append(col)
            continue

        # Date-like strings (only if values really look like dates)
        sample = non_null.head(200)
        looks_like_date = sample.str.match(DATE_VALUE_PATTERN).mean() >= 0.90
        if looks_like_date:
            parsed = pd.to_datetime(text, errors="coerce", format="mixed")
            if parsed.notna().sum() / max(len(non_null), 1) >= 0.90:
                df[col] = parsed
                report["date_converted"].append(col)
                continue

        df[col] = text.astype(object)

    before = len(df)
    df = df.dropna(how="all")
    report["empty_rows_removed"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)
    report["rows_after"] = len(df)
    return df, report


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return blocking problems (empty list = dataset can be used)."""
    problems = []
    if df is None or df.empty:
        return ["⚠️ Uploaded dataset is empty."]
    if df.shape[1] < 3:
        problems.append("⚠️ Dataset needs at least a target column and two usable feature columns.")
    if len(df) < MIN_ROWS:
        problems.append(f"⚠️ Dataset has only {len(df)} rows. At least {MIN_ROWS} rows are needed.")
    return problems


# =============================================================================
# 3. TARGET + FEATURE DETECTION
# =============================================================================


def detect_target_column(df: pd.DataFrame) -> tuple[str | None, list[str]]:
    """
    Returns (target, options).
    - target is set when exactly one clear candidate exists (notebook target 'churn' first).
    - otherwise the user must choose from `options` (binary columns).
    """
    lowered = {c.lower(): c for c in df.columns}
    if DEFAULT_TARGET in lowered:
        return lowered[DEFAULT_TARGET], []

    candidates = [c for c in df.columns if c.lower() in TARGET_NAME_HINTS]
    if len(candidates) == 1:
        return candidates[0], []

    binary_cols = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 2]
    return None, candidates if len(candidates) > 1 else binary_cols


def guess_positive_label(values: list[str]) -> str:
    """Pick which target value means 'churned'. Falls back to the last sorted value."""
    for v in values:
        if v.strip().lower() in POSITIVE_LABEL_HINTS:
            return v
    return sorted(values)[-1]


def detect_id_columns(df: pd.DataFrame, target: str | None = None) -> list[str]:
    """Identifier columns: name-based (customer_id, ID, uuid...) or unique text per row."""
    ids = []
    for col in df.columns:
        if col == target:
            continue
        if ID_NAME_PATTERN.search(col) or ID_CAMEL_PATTERN.search(col):
            ids.append(col)
            continue
        s = df[col]
        if not (is_numeric(s) or is_bool(s) or is_datetime(s)):
            non_null = s.dropna()
            if len(non_null) > 20 and non_null.nunique() == len(non_null):
                ids.append(col)
    return ids


def detect_feature_types(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Classify every column. 'Group' says how the ML pipeline treats it:
    numerical | categorical | date | id | target | excluded
    """
    id_cols = set(detect_id_columns(df, target))
    rows = []
    for col in df.columns:
        s = df[col]
        nunique = int(s.nunique(dropna=True))
        if col == target:
            ftype, group = "Target", "target"
        elif col in id_cols:
            ftype, group = "Identifier", "id"
        elif nunique <= 1:
            ftype, group = "Constant (unusable)", "excluded"
        elif is_datetime(s):
            ftype, group = "Date", "date"
        elif is_bool(s):
            ftype, group = "Boolean", "categorical"
        elif is_numeric(s):
            if set(s.dropna().unique()) <= {0, 1}:
                ftype, group = "Boolean (0/1)", "numerical"  # same as notebook: senior_citizen
            else:
                ftype, group = "Numerical", "numerical"
        else:
            lowered = {str(v).strip().lower() for v in s.dropna().unique()}
            if lowered <= BOOLEAN_WORDS:
                ftype, group = "Boolean (text)", "categorical"  # one-hot, same as notebook
            elif nunique > MAX_CATEGORY_LEVELS:
                ftype, group = "High-cardinality text", "excluded"
            else:
                ftype, group = "Categorical", "categorical"
        rows.append({"Column": col, "Type": ftype, "Group": group, "Unique": nunique})
    return pd.DataFrame(rows)


# =============================================================================
# 4. FEATURE PREPARATION
# =============================================================================


def to_model_frame(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    date_cols: list[str],
) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    """
    Build the model input frame from original columns.
    Used for BOTH training and the prediction form, so both stay consistent.
    Returns (frame, numerical_features, categorical_features, origin_map).
    """
    out = pd.DataFrame(index=df.index)
    origin: dict[str, str] = {}

    for col in numeric_cols:
        out[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        origin[col] = col

    for col in date_cols:  # raw datetimes are never fed to XGBoost
        parsed = pd.to_datetime(df[col], errors="coerce")
        for part, values in {
            "year": parsed.dt.year,
            "month": parsed.dt.month,
            "day": parsed.dt.day,
            "dayofweek": parsed.dt.dayofweek,
        }.items():
            name = f"{col}_{part}"
            out[name] = values.astype(float)
            origin[name] = col

    numerical_final = list(out.columns)

    for col in categorical_cols:
        out[col] = df[col].astype("string").fillna("Missing").str.strip().astype(object)
        origin[col] = col

    return out, numerical_final, list(categorical_cols), origin


def build_model(numerical: list[str], categorical: list[str], random_state: int) -> Pipeline:
    """Preprocessing + XGBClassifier pipeline, same structure as the notebook."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numerical),
            (
                "cat",
                Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                categorical,
            ),
        ]
    )
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", xgb)])


def tune_model(base: Pipeline, X_train, y_train, n_iter: int, random_state: int) -> RandomizedSearchCV:
    """RandomizedSearchCV with the notebook's search space, CV and scoring."""
    min_class = int(np.bincount(y_train).min())
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring=SCORING,
        cv=min(CV_FOLDS, min_class),
        random_state=random_state,
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(y_true, y_pred, y_prob) -> dict:
    """Classification metrics only (this is a classification project)."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def feature_importance_tables(model: Pipeline, numerical: list[str], categorical: list[str], origin: dict):
    """
    Map XGBoost importances back to readable names.
    One-hot columns are mapped using the fitted encoder's categories (not string guessing),
    so encoded features are never attached to the wrong original column.
    """
    pre = model.named_steps["preprocessor"]
    importances = model.named_steps["model"].feature_importances_

    names, parents = [], []
    for col in numerical:
        names.append(col)
        parents.append(origin.get(col, col))
    if categorical:
        encoder = pre.named_transformers_["cat"].named_steps["onehot"]
        for col, cats in zip(categorical, encoder.categories_):
            for cat in cats:
                names.append(f"{col} = {cat}")
                parents.append(col)

    if len(names) != len(importances):  # safe fallback
        names = list(pre.get_feature_names_out())
        parents = [n.split("__", 1)[-1] for n in names]

    detail = pd.DataFrame({"Feature": names, "Original Column": parents, "Importance": importances})
    detail = detail.sort_values("Importance", ascending=False).reset_index(drop=True)
    by_column = (
        detail.groupby("Original Column", as_index=False)["Importance"].sum().sort_values("Importance", ascending=False)
    )
    return detail, by_column.reset_index(drop=True)


def collect_form_info(
    d: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], date_cols: list[str]
) -> list[dict]:
    """Describe each ORIGINAL feature so the prediction form can be built dynamically."""
    info = []
    for col in numeric_cols:
        s = pd.to_numeric(d[col], errors="coerce").dropna()
        if set(s.unique()) <= {0, 1}:
            info.append({"col": col, "kind": "bool01", "default": int(s.mode().iloc[0]) if len(s) else 0})
        else:
            is_int = bool((s % 1 == 0).all())
            info.append(
                {
                    "col": col,
                    "kind": "number",
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "default": float(s.median()),
                    "is_int": is_int,
                }
            )
    for col in categorical_cols:
        if is_bool(d[col]):
            info.append({"col": col, "kind": "bool", "default": bool(d[col].mode().iloc[0])})
        else:
            options = sorted(d[col].dropna().astype(str).str.strip().unique().tolist())
            counts = d[col].astype(str).str.strip().value_counts()
            info.append({"col": col, "kind": "select", "options": options, "default": counts.index[0]})
    for col in date_cols:
        s = pd.to_datetime(d[col], errors="coerce").dropna()
        info.append(
            {"col": col, "kind": "date", "min": s.min().date(), "max": s.max().date(), "default": s.median().date()}
        )
    return info


# =============================================================================
# 5. TRAINING  (cached so reruns do not retrain)
# =============================================================================


@st.cache_resource(show_spinner=False)
def train_cached(_df: pd.DataFrame, df_key: str, cfg_json: str) -> dict:
    """Train base + tuned XGBoost with the notebook workflow. Cached by data + settings."""
    cfg = json.loads(cfg_json)
    target, positive = cfg["target"], cfg["positive"]
    numeric_cols, categorical_cols, date_cols = cfg["numeric"], cfg["categorical"], cfg["date"]
    random_state = cfg["random_state"]

    d = _df.dropna(subset=[target]).copy()
    y_text = d[target].astype(str).str.strip()
    labels = sorted(y_text.unique())
    if len(labels) != 2:
        raise ValueError(f"Target '{target}' must have exactly 2 classes, found {len(labels)}.")
    if int(y_text.value_counts().min()) < MIN_CLASS_COUNT:
        raise ValueError(f"Each target class needs at least {MIN_CLASS_COUNT} rows.")

    # Target encoding with LabelEncoder (notebook). Then make sure 1 = churned class.
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(y_text)
    positive_code = list(label_encoder.classes_).index(positive)
    y = (encoded == positive_code).astype(int)
    negative = [c for c in label_encoder.classes_ if c != positive][0]

    X, numerical, categorical, origin = to_model_frame(d, numeric_cols, categorical_cols, date_cols)

    # Split FIRST, then fit preprocessing on training data only (inside the Pipeline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=random_state, stratify=y
    )

    base = build_model(numerical, categorical, random_state)
    base.fit(X_train, y_train)
    base_pred = base.predict(X_test)
    base_prob = base.predict_proba(X_test)[:, 1]

    search = tune_model(build_model(numerical, categorical, random_state), X_train, y_train, cfg["n_iter"], random_state)
    best = search.best_estimator_
    pred = best.predict(X_test)
    prob = best.predict_proba(X_test)[:, 1]

    fpr_b, tpr_b, _ = roc_curve(y_test, base_prob)
    fpr_t, tpr_t, _ = roc_curve(y_test, prob)
    prec, rec, _ = precision_recall_curve(y_test, prob)
    detail, by_column = feature_importance_tables(best, numerical, categorical, origin)

    return {
        "cfg": cfg,
        "positive": positive,
        "negative": negative,
        "label_encoder_classes": list(label_encoder.classes_),
        "best_model": best,
        "search": search,
        "X": X,
        "X_test": X_test,
        "y_test": np.asarray(y_test),
        "y_all": pd.Series(y, index=X.index),
        "pred": pred,
        "prob": prob,
        "base_metrics": evaluate_model(y_test, base_pred, base_prob),
        "tuned_metrics": evaluate_model(y_test, pred, prob),
        "report": pd.DataFrame(
            classification_report(y_test, pred, target_names=[negative, positive], output_dict=True, zero_division=0)
        ).T,
        "cm": confusion_matrix(y_test, pred),
        "roc": {"base": (fpr_b, tpr_b), "tuned": (fpr_t, tpr_t)},
        "pr": (rec, prec, auc(rec, prec)),
        "importance": detail,
        "importance_by_column": by_column,
        "numerical": numerical,
        "categorical": categorical,
        "form_info": collect_form_info(d, numeric_cols, categorical_cols, date_cols),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cv_used": search.cv,
    }


def generate_predictions(res: dict, source: pd.DataFrame, id_cols: list[str], high: float, medium: float) -> pd.DataFrame:
    """Build the test-set prediction table (risk level uses current sidebar thresholds)."""
    X_test = res["X_test"]
    pos, neg = res["positive"], res["negative"]
    actual = np.where(res["y_test"] == 1, pos, neg)
    predicted = np.where(res["pred"] == 1, pos, neg)
    out = pd.DataFrame(index=X_test.index)
    for col in id_cols:
        out[col] = source.loc[X_test.index, col]
    out["Actual"] = actual
    out["Predicted"] = predicted
    out["Churn_Probability"] = res["prob"]
    out["Risk_Level"] = [assign_risk(p, high, medium) for p in res["prob"]]
    out["Correct"] = out["Actual"] == out["Predicted"]
    out["Prediction_Error"] = np.abs(res["y_test"] - res["prob"])
    return pd.concat([out, X_test], axis=1)


# =============================================================================
# 6. UI HELPERS
# =============================================================================


def plot(fig: go.Figure) -> None:
    """Plotly chart that follows Streamlit's light/dark theme."""
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


def unavailable() -> None:
    st.info("ℹ️ This visualization is not available for the current dataset.")


def create_prediction_form(res: dict, key_prefix: str):
    """Dynamic prediction form generated from the model's own features."""
    values: dict = {}
    infos = res["form_info"]
    with st.form(f"predict_form_{key_prefix}"):
        cols = st.columns(3)
        for i, info in enumerate(infos):
            col, kind = info["col"], info["kind"]
            label = col.replace("_", " ").title()
            key = f"{key_prefix}_{col}"
            with cols[i % 3]:
                if kind == "number":
                    lo = info["min"]
                    non_negative = lo >= 0
                    if info["is_int"]:
                        values[col] = st.number_input(
                            label, value=int(info["default"]), step=1, min_value=0 if non_negative else None, key=key
                        )
                    else:
                        values[col] = st.number_input(
                            label,
                            value=round(info["default"], 2),
                            step=0.01,
                            min_value=0.0 if non_negative else None,
                            key=key,
                        )
                elif kind == "bool01":
                    values[col] = int(st.checkbox(label, value=bool(info["default"]), key=key))
                elif kind == "bool":
                    values[col] = st.checkbox(label, value=info["default"], key=key)
                elif kind == "select":
                    options = info["options"]
                    values[col] = st.selectbox(label, options, index=options.index(info["default"]), key=key)
                elif kind == "date":
                    values[col] = st.date_input(
                        label, value=info["default"], min_value=None, max_value=None, key=key
                    )
        submitted = st.form_submit_button("🎯 Predict Customer Churn", type="primary")
    return submitted, values


def validate_prediction_input(res: dict, values: dict) -> list[str]:
    """Warnings for values outside the range the model saw during training."""
    warnings = []
    for info in res["form_info"]:
        if info["kind"] == "number":
            v = values[info["col"]]
            if v < info["min"] or v > info["max"]:
                warnings.append(
                    f"`{info['col']}` = {v} is outside the training range "
                    f"({info['min']:.2f} to {info['max']:.2f}); the prediction may be less reliable."
                )
    return warnings


def risk_gauge(probability: float, high: float, medium: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4c6ef5"},
                "steps": [
                    {"range": [0, medium * 100], "color": "rgba(46,158,91,0.35)"},
                    {"range": [medium * 100, high * 100], "color": "rgba(224,165,38,0.35)"},
                    {"range": [high * 100, 100], "color": "rgba(214,69,69,0.35)"},
                ],
            },
        )
    )
    fig.update_layout(height=280)
    return fig


def build_evaluation_table(res: dict) -> pd.DataFrame:
    """Everything about the model evaluation as one downloadable table."""
    rows = []
    for name, value in res["base_metrics"].items():
        rows.append(("Base model", name, round(value, 4)))
    for name, value in res["tuned_metrics"].items():
        rows.append(("Tuned model", name, round(value, 4)))
    rows.append(("Tuning", f"Best CV {SCORING}", round(res["search"].best_score_, 4)))
    for k, v in res["search"].best_params_.items():
        rows.append(("Best parameters", k.replace("model__", ""), v))
    return pd.DataFrame(rows, columns=["Section", "Item", "Value"])


# =============================================================================
# 7. TABS
# =============================================================================


def render_overview(df, types, target, res, preds):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Features Used", len(res["cfg"]["numeric"]) + len(res["cfg"]["categorical"]) + len(res["cfg"]["date"]))
    c3.metric("Numerical Features", len(res["cfg"]["numeric"]))
    c4.metric("Categorical Features", len(res["cfg"]["categorical"]))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Target", target)
    c6.metric("Model", "XGBClassifier")
    c7.metric("Accuracy", f"{res['tuned_metrics']['Accuracy']:.3f}")
    c8.metric("ROC-AUC", f"{res['tuned_metrics']['ROC-AUC']:.3f}")

    with st.expander("🏆 Best XGBoost parameters"):
        show_best_params(res)

    left, right = st.columns(2)
    with left:
        counts = df[target].astype(str).value_counts().reset_index()
        counts.columns = [target, "Customers"]
        plot(px.pie(counts, names=target, values="Customers", hole=0.45, title="Churn Distribution"))
    with right:
        risk = preds["Risk_Level"].value_counts().reindex(list(RISK_COLORS)).fillna(0).reset_index()
        risk.columns = ["Risk Level", "Customers"]
        plot(
            px.bar(
                risk, x="Risk Level", y="Customers", color="Risk Level",
                color_discrete_map=RISK_COLORS, title="Test Set Risk Distribution",
            )
        )

    left, right = st.columns(2)
    num_cols = res["cfg"]["numeric"]
    cat_cols = res["cfg"]["categorical"]
    with left:
        if num_cols:
            col = st.selectbox("Numerical feature", num_cols, key="ov_num")
            plot(
                px.histogram(
                    df, x=col, color=df[target].astype(str), barmode="overlay", opacity=0.65,
                    nbins=30, title=f"{col} by {target}",
                )
            )
        else:
            unavailable()
    with right:
        if cat_cols:
            col = st.selectbox("Categorical feature", cat_cols, key="ov_cat")
            rate = (
                df.assign(_flag=(df[target].astype(str) == res["positive"]).astype(int) * 100)
                .groupby(col)["_flag"].mean().sort_values(ascending=False).reset_index()
            )
            plot(px.bar(rate, x=col, y="_flag", labels={"_flag": "Churn Rate (%)"}, title=f"Churn Rate by {col}"))
        else:
            unavailable()

    numeric_df = df[[c for c in num_cols if c in df.columns]].copy()
    numeric_df[f"{target}_flag"] = (df[target].astype(str) == res["positive"]).astype(int)
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        plot(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Heatmap"))
    else:
        unavailable()


def show_best_params(res):
    params = res["search"].best_params_
    table = pd.DataFrame({"Parameter": [k.replace("model__", "") for k in params], "Best Value": list(params.values())})
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(f"Best cross-validation {SCORING}: **{res['search'].best_score_:.4f}** ({res['cv_used']}-fold CV)")


def render_dataset(df, raw, report, types, id_cols, res):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    c4.metric("Duplicates removed", report["duplicates_removed"])

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Data Types")
        st.dataframe(
            pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.astype(str).values}),
            hide_index=True, width="stretch",
        )
    with right:
        st.subheader("Missing Values")
        miss = df.isna().sum()
        st.dataframe(
            pd.DataFrame({"Column": miss.index, "Missing Count": miss.values, "Missing %": (miss.values / len(df) * 100).round(2)}),
            hide_index=True, width="stretch",
        )

    st.subheader("Feature Classification")
    used = set(res["cfg"]["numeric"] + res["cfg"]["categorical"] + res["cfg"]["date"])
    table = types.copy()
    table["Used for Model"] = table["Column"].map(lambda c: "✅ Yes" if c in used else "—")
    st.dataframe(table.drop(columns=["Group"]), hide_index=True, width="stretch")

    with st.expander("Cleaning report"):
        st.write(
            {
                "Rows before cleaning": report["rows_before"],
                "Rows after cleaning": report["rows_after"],
                "Empty rows removed": report["empty_rows_removed"],
                "Duplicate rows removed": report["duplicates_removed"],
                "Infinite values replaced with NaN": report["inf_replaced"],
                "Numeric-string columns converted": report["numeric_converted"],
                "Date columns converted": report["date_converted"],
            }
        )

    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        st.subheader("Numerical Summary")
        st.dataframe(num_df.describe().T, width="stretch")
    cat_df = df.select_dtypes(exclude=["number", "datetime"])
    if not cat_df.empty:
        st.subheader("Categorical Summary")
        st.dataframe(cat_df.describe().T, width="stretch")

    st.download_button("📥 Download Processed Dataset", create_download_file(df), "processed_dataset.csv", "text/csv")


def render_performance(res):
    tuned, base = res["tuned_metrics"], res["base_metrics"]
    cols = st.columns(5)
    for c, name in zip(cols, tuned):
        c.metric(name, f"{tuned[name]:.3f}", f"{tuned[name] - base[name]:+.3f} vs base")
    st.caption(f"Test set: {res['n_test']} customers · Best CV {SCORING}: {res['search'].best_score_:.4f}")

    comparison = pd.DataFrame({"Metric": list(tuned), "Base Model": list(base.values()), "Tuned Model": list(tuned.values())})
    left, right = st.columns(2)
    with left:
        melted = comparison.melt("Metric", var_name="Model", value_name="Score")
        fig = px.bar(melted, x="Metric", y="Score", color="Model", barmode="group", title="Base vs Tuned XGBoost")
        fig.update_yaxes(range=[0, 1])
        plot(fig)
    with right:
        labels = [res["negative"], res["positive"]]
        fig = px.imshow(
            res["cm"], x=[f"Predicted {l}" for l in labels], y=[f"Actual {l}" for l in labels],
            text_auto="d", color_continuous_scale="Blues", title="Confusion Matrix (Tuned)",
        )
        plot(fig)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=res["roc"]["base"][0], y=res["roc"]["base"][1], name=f"Base AUC = {base['ROC-AUC']:.3f}")
        fig.add_scatter(x=res["roc"]["tuned"][0], y=res["roc"]["tuned"][1], name=f"Tuned AUC = {tuned['ROC-AUC']:.3f}")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random guess")
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        plot(fig)
    with right:
        rec, prec, pr_auc = res["pr"]
        fig = go.Figure(go.Scatter(x=rec, y=prec, name=f"PR-AUC = {pr_auc:.3f}"))
        fig.update_layout(title=f"Precision-Recall Curve (PR-AUC = {pr_auc:.3f})", xaxis_title="Recall", yaxis_title="Precision")
        plot(fig)

    fig = px.histogram(x=res["prob"], nbins=20, labels={"x": "Churn Probability"}, title="Churn Probability Distribution (test set)")
    plot(fig)

    st.subheader("Classification Report")
    st.dataframe(res["report"].round(3), width="stretch")

    st.subheader("Base vs Tuned")
    st.dataframe(comparison.round(4), hide_index=True, width="stretch")

    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **Accuracy**: out of all customers, how many were predicted correctly.\n"
            "- **Precision**: when the model says *will churn*, how often it is right.\n"
            "- **Recall**: out of customers who really churned, how many the model caught.\n"
            "- **F1 Score**: one number balancing precision and recall.\n"
            "- **ROC-AUC**: how well the model ranks churners above non-churners (0.5 = random, 1.0 = perfect)."
        )

    if tuned["ROC-AUC"] < 0.70:
        st.info(
            f"ℹ️ ROC-AUC is {tuned['ROC-AUC']:.3f}, so the model has only modest predictive power on this dataset. "
            "Tuning cannot create signal that the features do not contain; richer features usually help more than more tuning."
        )

    st.download_button(
        "📥 Download Model Evaluation", create_download_file(build_evaluation_table(res)), "model_evaluation.csv", "text/csv"
    )


def render_predictions(res, preds):
    st.caption("Predictions on the held-out test set (customers the model never saw during training).")
    c1, c2, c3 = st.columns(3)
    risk_filter = c1.multiselect("Risk level", list(RISK_COLORS), default=list(RISK_COLORS))
    only_wrong = c2.checkbox("Only wrong predictions", value=False)
    sort_by = c3.selectbox("Sort by", ["Churn_Probability", "Prediction_Error"])

    view = preds[preds["Risk_Level"].isin(risk_filter)]
    if only_wrong:
        view = view[~view["Correct"]]
    view = view.sort_values(sort_by, ascending=False)

    st.dataframe(
        view,
        width="stretch",
        column_config={
            "Churn_Probability": st.column_config.ProgressColumn("Churn_Probability", min_value=0.0, max_value=1.0, format="%.3f"),
            "Prediction_Error": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(f"Showing {len(view)} of {len(preds)} customers. Prediction_Error = |actual (0/1) − predicted probability|.")
    st.download_button("📥 Download Predictions CSV", create_download_file(preds.reset_index(drop=True)), "predictions.csv", "text/csv")


def render_predictor(res, high, medium, key_prefix):
    st.subheader("Predict churn for a single customer")
    st.caption("The form is generated from the features the trained model actually uses.")
    submitted, values = create_prediction_form(res, key_prefix)
    if not submitted:
        return

    try:
        for msg in validate_prediction_input(res, values):
            st.warning(f"⚠️ {msg}")

        cfg = res["cfg"]
        row = pd.DataFrame([values])
        for col in cfg["date"]:
            row[col] = pd.to_datetime(row[col])
        frame, _, _, _ = to_model_frame(row, cfg["numeric"], cfg["categorical"], cfg["date"])
        expected = list(res["X"].columns)
        if set(frame.columns) != set(expected):
            raise ValueError("Input features do not match the training features.")
        frame = frame[expected]  # guarantee training column order

        prob = float(res["best_model"].predict_proba(frame)[0, 1])
        label = res["positive"] if res["best_model"].predict(frame)[0] == 1 else res["negative"]
        risk = assign_risk(prob, high, medium)

        with st.container(border=True):
            left, right = st.columns([1, 1])
            with left:
                st.markdown("#### 📊 Predicted Churn Probability")
                st.metric("Probability", f"{prob * 100:.1f}%")
                st.markdown("#### 🎯 Final Prediction")
                if label == res["positive"]:
                    st.error(f"**{cfg['target']} = {label}**  ·  {risk}")
                else:
                    st.success(f"**{cfg['target']} = {label}**  ·  {risk}")
                st.write(f"**Risk thresholds:** High ≥ {high:.2f} · Medium ≥ {medium:.2f}")
            with right:
                plot(risk_gauge(prob, high, medium))
        st.caption("This is a statistical estimate from a trained model, not a guarantee of customer behaviour.")
    except Exception as exc:  # noqa: BLE001
        show_error("⚠️ Could not generate a prediction. Please check the input values.", exc)


def render_feature_analysis(df, res, target):
    detail, by_column = res["importance"], res["importance_by_column"]
    st.subheader("🔍 Feature Importance")
    st.caption("Importance shows how useful a feature was to the model for prediction. It does not prove cause and effect.")

    top_n = st.slider("Number of features to show", 5, max(5, min(30, len(detail))), min(15, len(detail)))
    top = detail.head(top_n).iloc[::-1]
    plot(px.bar(top, x="Importance", y="Feature", orientation="h", title=f"Top {top_n} Encoded Features"))

    left, right = st.columns([3, 2])
    with left:
        plot(
            px.bar(
                by_column.iloc[::-1], x="Importance", y="Original Column", orientation="h",
                title="Importance Grouped by Original Column",
            )
        )
    with right:
        st.dataframe(detail.round(4), hide_index=True, width="stretch", height=380)

    st.subheader("Feature vs Churn")
    num_cols = res["cfg"]["numeric"]
    if num_cols:
        col = st.selectbox("Numerical feature (box plot)", num_cols, key="fa_num")
        plot(px.box(df, x=df[target].astype(str), y=col, color=df[target].astype(str), title=f"{col} vs {target}"))
    else:
        unavailable()

    st.download_button("📥 Download Feature Importance", create_download_file(detail), "feature_importance.csv", "text/csv")


def render_model_info(res):
    cfg = res["cfg"]
    st.subheader("What is XGBoost?")
    st.markdown(
        "XGBoost (Extreme Gradient Boosting) builds many small decision trees **one after another**. "
        "Each new tree focuses on fixing the mistakes of the trees before it. Think of a team of students "
        "where every new student studies only the questions the earlier students got wrong."
    )
    st.subheader("Why XGBoost here?")
    st.markdown(
        "Customer behaviour is rarely a straight line. XGBoost can learn non-linear patterns and feature "
        "interactions (for example, *month-to-month contract* combined with *low tenure*) without manual feature engineering."
    )

    st.subheader("ML Workflow")
    st.code(
        "Dataset\n ↓\nData Cleaning\n ↓\nFeature Selection\n ↓\nTrain/Test Split (stratified)\n ↓\n"
        "Preprocessing (fit on training data only)\n   • StandardScaler → numerical features\n   • OneHotEncoder  → categorical features\n ↓\n"
        "RandomizedSearchCV (hyperparameter tuning)\n ↓\nBest XGBClassifier\n ↓\nChurn probability\n ↓\nRisk level (Low / Medium / High)",
        language="text",
    )

    st.subheader("Configuration used")
    st.dataframe(
        pd.DataFrame(
            [
                ("Target", f"{cfg['target']} (churned class = '{res['positive']}')"),
                ("Target encoding", "LabelEncoder"),
                ("Numerical features", ", ".join(res["numerical"]) or "—"),
                ("Categorical features", ", ".join(res["categorical"]) or "—"),
                ("Numerical preprocessing", "StandardScaler"),
                ("Categorical preprocessing", "OneHotEncoder (handle_unknown='ignore')"),
                ("Model", "XGBClassifier (objective=binary:logistic, eval_metric=logloss)"),
                ("Train / test split", f"{res['n_train']} / {res['n_test']} (test_size={cfg['test_size']}, random_state={cfg['random_state']}, stratified)"),
                ("Tuning", f"RandomizedSearchCV (n_iter={cfg['n_iter']}, cv={res['cv_used']}, scoring='{SCORING}')"),
            ],
            columns=["Item", "Value"],
        ),
        hide_index=True, width="stretch",
    )

    st.subheader("🏆 Best XGBoost Parameters")
    show_best_params(res)

    st.subheader("Key terms")
    st.markdown(
        "- **Boosting**: building trees in sequence, each correcting the previous ones.\n"
        "- **Learning rate**: how big a step each new tree takes (small = slower but safer).\n"
        "- **Max depth**: how many questions a single tree may ask.\n"
        "- **Number of estimators**: how many trees are built.\n"
        "- **LabelEncoder**: turns the target words (No/Yes) into 0/1.\n"
        "- **OneHotEncoder**: turns each category into its own 0/1 column.\n"
        "- **StandardScaler**: puts numbers on a similar scale (mean 0, spread 1).\n"
        "- **Hyperparameter tuning**: trying many settings and keeping the best by cross-validation.\n"
        "- **Risk level**: High ≥ 0.70, Medium ≥ 0.40, otherwise Low (adjustable in the sidebar)."
    )


def render_author_footer():
    st.divider()
    st.markdown(
        f"**Author:** {AUTHOR_NAME}  ·  [LinkedIn]({LINKEDIN_URL})  ·  [GitHub]({GITHUB_URL})"
    )


# =============================================================================
# 8. MAIN APP
# =============================================================================


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📉", layout="wide")
    st.title("📉 Customer Churn Prediction System")
    st.caption("Machine Learning Based Customer Churn Prediction using XGBoost")

    # ---------------- Sidebar: dataset ----------------
    with st.sidebar:
        st.header("📂 Dataset")
        source = st.radio("Data source", ["Default dataset", "Upload CSV"], label_visibility="collapsed")
        uploaded = None
        if source == "Upload CSV":
            uploaded = st.file_uploader("Upload Customer Churn Dataset", type=["csv"])

    try:
        if source == "Upload CSV":
            if uploaded is None:
                st.info("📂 Upload a CSV in the sidebar, or switch back to the default dataset.")
                render_author_footer()
                st.stop()
            raw = load_uploaded_dataset(uploaded.getvalue())
        else:
            if not DEFAULT_DATASET_PATH.exists():
                show_error("⚠️ Default dataset `customer_churn_prediction_dataset.csv` was not found next to app.py.")
                st.stop()
            raw = load_default_dataset()
    except pd.errors.EmptyDataError:
        show_error("⚠️ Uploaded dataset is empty.")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        show_error("⚠️ Unable to read the uploaded file. Please upload a valid CSV.", exc)
        st.stop()

    problems = validate_dataset(raw)
    if problems:
        for p in problems:
            st.error(p)
        st.stop()

    try:
        df, report = clean_dataset(raw)
        problems = validate_dataset(df)
        if problems:
            for p in problems:
                st.error(p)
            st.stop()

        # ---------------- Target selection ----------------
        target, options = detect_target_column(df)
        with st.sidebar:
            if target is None:
                if not options:
                    st.error("⚠️ Target column could not be identified.")
                    st.stop()
                st.warning("Multiple possible targets found.")
                target = st.selectbox("🎯 Select Target Column", options)
            else:
                st.success(f"🎯 Target: **{target}**")

        df = df.dropna(subset=[target]).copy()
        labels = sorted(df[target].astype(str).str.strip().unique())
        if len(labels) != 2:
            st.error(
                f"⚠️ Dataset is incompatible with the current ML pipeline: target '{target}' must have exactly "
                f"2 classes (found {len(labels)})."
            )
            st.stop()
        if int(df[target].astype(str).str.strip().value_counts().min()) < MIN_CLASS_COUNT:
            st.error(f"⚠️ Each target class needs at least {MIN_CLASS_COUNT} rows.")
            st.stop()

        with st.sidebar:
            positive = st.selectbox("Churned class (positive)", labels, index=labels.index(guess_positive_label(labels)))

        types = detect_feature_types(df, target)
        id_cols = types.loc[types["Group"] == "id", "Column"].tolist()
        num_opts = types.loc[types["Group"] == "numerical", "Column"].tolist()
        cat_opts = types.loc[types["Group"] == "categorical", "Column"].tolist()
        date_opts = types.loc[types["Group"] == "date", "Column"].tolist()
        if not (num_opts or cat_opts or date_opts):
            st.error("⚠️ Dataset does not contain enough usable features.")
            st.stop()

        sig = dataframe_key(df)

        # ---------------- Sidebar: settings ----------------
        with st.sidebar:
            st.header("🎯 Prediction Settings")
            high = st.slider("High-risk threshold", 0.50, 0.95, HIGH_RISK_THRESHOLD, 0.01)
            medium = st.slider("Medium-risk threshold", 0.10, 0.60, MEDIUM_RISK_THRESHOLD, 0.01)
            if medium >= high:
                st.warning("Medium threshold must be lower than the high threshold.")
                st.stop()

            with st.form(f"settings_{sig}"):
                st.header("⚙️ Model Settings")
                test_size = st.slider("Test size", 0.10, 0.40, DEFAULT_TEST_SIZE, 0.05)
                random_state = int(st.number_input("Random state", 0, 10_000, DEFAULT_RANDOM_STATE, 1))
                n_iter = int(
                    st.number_input(
                        "Search iterations (RandomizedSearchCV)", 5, 300, DEFAULT_N_ITER, 5,
                        help="The notebook uses 100. Lower values train faster.",
                    )
                )
                st.header("🧩 Feature Selection")
                sel_num = st.multiselect("Numerical Features", num_opts, default=num_opts)
                sel_cat = st.multiselect("Categorical Features", cat_opts, default=cat_opts)
                sel_date = st.multiselect("Date Features (expanded to year/month/day/weekday)", date_opts, default=[]) if date_opts else []
                if id_cols:
                    st.caption("Identifier (excluded): " + ", ".join(id_cols))
                st.caption(f"Target (never a feature): {target}")
                submitted = st.form_submit_button("🚀 Train / Run Model", type="primary", width="stretch")

            st.divider()
            st.markdown(f"**Author:** {AUTHOR_NAME}")
            st.markdown(f"[LinkedIn]({LINKEDIN_URL}) · [GitHub]({GITHUB_URL})")

        if not (sel_num or sel_cat or sel_date):
            st.error("⚠️ Select at least one feature in the sidebar.")
            st.stop()

        cfg = {
            "target": target, "positive": positive,
            "numeric": sel_num, "categorical": sel_cat, "date": sel_date,
            "test_size": test_size, "random_state": random_state, "n_iter": n_iter,
        }

        # ---------------- Train (cached) ----------------
        with st.spinner(
            f"Training XGBoost and searching {n_iter} parameter combinations "
            f"(first run only; results are cached) ..."
        ):
            res = train_cached(df, sig, json.dumps(cfg, sort_keys=True))
        if submitted:
            st.toast("Model is ready ✅")

        preds = generate_predictions(res, df, id_cols, high, medium)

    except st.errors.StreamlitAPIException:
        raise
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ in ("StopException", "RerunException"):
            raise
        show_error(
            "⚠️ Unable to process the dataset. Please verify that it contains a binary target and compatible features.",
            exc,
        )
        st.stop()

    # ---------------- Top metric cards ----------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TOTAL CUSTOMERS", f"{len(df):,}")
    m2.metric("FEATURES USED", len(sel_num) + len(sel_cat) + len(sel_date))
    m3.metric("ACCURACY", f"{res['tuned_metrics']['Accuracy']:.3f}")
    m4.metric("ROC-AUC", f"{res['tuned_metrics']['ROC-AUC']:.3f}")

    tabs = st.tabs(
        [
            "📊 Overview", "📁 Dataset", "📈 Model Performance", "🎯 Predictions",
            "🧑‍💼 Customer Predictor", "🔍 Feature Analysis", "⚙️ Model Information",
        ]
    )
    renderers = [
        lambda: render_overview(df, types, target, res, preds),
        lambda: render_dataset(df, raw, report, types, id_cols, res),
        lambda: render_performance(res),
        lambda: render_predictions(res, preds),
        lambda: render_predictor(res, high, medium, sig),
        lambda: render_feature_analysis(df, res, target),
        lambda: render_model_info(res),
    ]
    for tab, render in zip(tabs, renderers):
        with tab:
            try:
                render()
            except Exception as exc:  # noqa: BLE001
                show_error("⚠️ This section could not be displayed for the current dataset.", exc)

    render_author_footer()


if __name__ == "__main__":
    main()
