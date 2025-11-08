# app.py
# TBI ARDS risk predictor with stacking (level-2 LogisticRegressionCV)
# - Loads meta model: best_model.pkl (your meta_lr_cv.pkl renamed)
# - Loads level-1 models: model_*.pkl from repo root
# - No uploads needed
# - Clinical inputs: VP, MV (categorical 0/1) + numeric labs/vitals
# - Explainer: exact linear logit contributions using LR coefficients
#   (no SHAP Explainer needed; robust and fast)

from pathlib import Path
import json
import traceback

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from catboost import Pool, CatBoostClassifier

# =============================
# Basic settings
# =============================
st.set_page_config(page_title="TBI ARDS Risk Predictor", layout="wide")
st.title("🧠 TBI ARDS Risk Predictor (Stacking)")

# Show key versions for easy debugging
try:
    import sklearn, lightgbm, xgboost, catboost
    st.caption(
        f"env: sklearn {sklearn.__version__} | lightgbm {lightgbm.__version__} | "
        f"xgboost {xgboost.__version__} | catboost {catboost.__version__}"
    )
except Exception:
    pass

ROOT = Path(".")
META_PATH = ROOT / "best_model.pkl"   # put your meta_lr_cv.pkl renamed to best_model.pkl in repo root

# =============================
# Clinical feature schema (your training set)
# =============================
CLIN_SPECS = {
    # categorical 0/1
    "VP": {"type": "cat", "labels": {0: "No vasopressor", 1: "Vasopressor used"}, "default": 0},
    "MV": {"type": "cat", "labels": {0: "No mechanical ventilation", 1: "Mechanical ventilation"}, "default": 0},

    # numeric: min, max, default, step
    "APACHEII":   {"type": "num", "min": 0.0,  "max": 71.0,  "default": 18.0,  "step": 1.0},
    "SOFA":       {"type": "num", "min": 0.0,  "max": 24.0,  "default": 5.0,   "step": 1.0},
    "GCS":        {"type": "num", "min": 3.0,  "max": 15.0,  "default": 12.0,  "step": 1.0},
    "CCI":        {"type": "num", "min": 0.0,  "max": 25.0,  "default": 2.0,   "step": 1.0},
    "MAP":        {"type": "num", "min": 30.0, "max": 200.0, "default": 75.0,  "step": 1.0},   # mmHg
    "Temp":       {"type": "num", "min": 30.0, "max": 43.0,  "default": 37.8,  "step": 0.1},   # °C
    "RR":         {"type": "num", "min": 5.0,  "max": 60.0,  "default": 20.0,  "step": 1.0},   # /min
    "Calcium":    {"type": "num", "min": 0.5,  "max": 3.0,   "default": 2.2,   "step": 0.1},   # mmol/L
    "Sodium":     {"type": "num", "min": 110., "max": 170.,  "default": 138.,  "step": 1.0},   # mmol/L
    "Glucose":    {"type": "num", "min": 2.0,  "max": 33.0,  "default": 8.5,   "step": 0.1},   # mmol/L
    "Creatinine": {"type": "num", "min": 20.0, "max": 1200., "default": 90.0,  "step": 5.0},   # µmol/L
    "HB":         {"type": "num", "min": 40.0, "max": 200.0, "default": 120.0, "step": 1.0},   # g/L
}
CLIN_ORDER = list(CLIN_SPECS.keys())

DISPLAY_NAME = {
    "VP": "Vasopressor use",
    "MV": "Mechanical ventilation",
    "APACHEII": "APACHE-II (score)",
    "SOFA": "SOFA (score)",
    "GCS": "GCS (score)",
    "CCI": "Charlson comorbidity index",
    "MAP": "Mean arterial pressure (mmHg)",
    "Temp": "Temperature (°C)",
    "RR": "Respiratory rate (/min)",
    "Calcium": "Calcium (mmol/L)",
    "Sodium": "Sodium (mmol/L)",
    "Glucose": "Glucose (mmol/L)",
    "Creatinine": "Creatinine (µmol/L)",
    "HB": "Hemoglobin (g/L)",
}

# =============================
# Sidebar: inputs
# =============================
st.sidebar.header("Input parameters")
inputs = {}
for feat in CLIN_ORDER:
    spec = CLIN_SPECS[feat]
    label = DISPLAY_NAME.get(feat, feat)
    if spec["type"] == "cat":
        labels = list(spec["labels"].values())
        keys = list(spec["labels"].keys())
        default_idx = keys.index(spec.get("default", keys[0]))
        sel = st.sidebar.selectbox(label, labels, index=default_idx)
        inv = {v: k for k, v in spec["labels"].items()}
        inputs[feat] = inv[sel]
    else:
        inputs[feat] = st.sidebar.number_input(
            label,
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=float(spec["default"]),
            step=float(spec["step"]),
            format="%.3f" if spec["step"] < 1 else "%.0f",
        )

go = st.sidebar.button("🔮 Predict", type="primary")

# Build 1-row DF in canonical order
def build_clin_df(d: dict) -> pd.DataFrame:
    row = {k: d[k] for k in CLIN_ORDER}
    X = pd.DataFrame([row], columns=CLIN_ORDER)
    # category cast as int for 0/1
    for c in ["VP", "MV"]:
        if c in X.columns:
            X[c] = X[c].astype(int)
    return X

# =============================
# Load models
# =============================
@st.cache_resource
def load_meta_model():
    if not META_PATH.exists():
        st.error("Meta model file 'best_model.pkl' not found in repository root.")
        st.stop()
    return joblib.load(META_PATH)

@st.cache_resource
def load_level1_models():
    """
    Scan repo root for model_*.pkl and return {name: model}.
    Accepts both ascii and unicode names after 'model_'.
    """
    models = {}
    for fp in ROOT.glob("model_*.pkl"):
        name = fp.stem[len("model_") :]  # take the part after model_
        try:
            m = joblib.load(fp)
            models[name] = m
        except Exception as e:
            st.warning(f"Failed to load {fp.name}: {e}")
    return models

meta_model = load_meta_model()
level1_models = load_level1_models()

# Expected level-1 feature names for meta
exp_meta_feats = list(getattr(meta_model, "feature_names_in_", []))
if not exp_meta_feats:
    # Fallback to a common set if feature_names_in_ is missing
    exp_meta_feats = ["ada","cat","dt","gbm","knn","lgb","logistic","mlp","rf","svm","xgb"]

st.caption("Level-2 expects level-1 features: " + ", ".join(exp_meta_feats))

# =============================
# Utilities for aligning inputs to model
# =============================
def _get_feat_in(m):
    feat = getattr(m, "feature_names_in_", None)
    if feat is None and isinstance(m, Pipeline):
        est = m.named_steps.get("clf", None)
        feat = getattr(est, "feature_names_in_", None)
    return list(feat) if feat is not None else None

def _needs_string_cats(pipe: Pipeline):
    cat_cols_need_str = set()
    if not isinstance(pipe, Pipeline):
        return cat_cols_need_str
    try:
        for _, step in pipe.named_steps.items():
            if isinstance(step, ColumnTransformer):
                for tr_name, trf, cols in step.transformers_:
                    from sklearn.preprocessing import OneHotEncoder
                    if isinstance(trf, OneHotEncoder) and trf.categories_ is not None:
                        if isinstance(cols, (list, tuple)):
                            for j, col in enumerate(cols):
                                if j < len(trf.categories_):
                                    cats = trf.categories_[j]
                                    if any(isinstance(v, str) for v in cats):
                                        cat_cols_need_str.add(col)
    except Exception:
        pass
    return cat_cols_need_str

def align_to_model_features(m, X_in: pd.DataFrame, cast_cat="auto") -> pd.DataFrame:
    X = X_in.copy()
    feat_in = _get_feat_in(m)

    # case-insensitive rename to match training names exactly
    if feat_in is not None:
        lower_to_train = {c.lower(): c for c in feat_in}
        ren = {}
        for c in X.columns:
            tgt = lower_to_train.get(c.lower())
            if tgt and tgt != c:
                ren[c] = tgt
        if ren:
            X = X.rename(columns=ren)

    # guess category casting
    clf = m.named_steps.get("clf", None) if isinstance(m, Pipeline) else m
    is_catboost = isinstance(clf, CatBoostClassifier)
    cat_need_str = _needs_string_cats(m) if isinstance(m, Pipeline) else set()

    if cast_cat == "auto":
        to_str = set()
        if is_catboost:
            to_str |= set(X.columns)  # safest for CatBoost
        to_str |= cat_need_str
        for c in to_str:
            if c in X.columns:
                X[c] = X[c].astype("object").astype(str)
    elif cast_cat == "str":
        for c in X.columns:
            X[c] = X[c].astype("object").astype(str)
    elif cast_cat == "int":
        for c in X.columns:
            if pd.api.types.is_object_dtype(X[c]):
                try:
                    X[c] = X[c].astype("Int64").astype(int)
                except Exception:
                    pass

    # add missing columns with zeros and reorder
    if feat_in is not None:
        for c in feat_in:
            if c not in X.columns:
                X[c] = 0
        X = X.reindex(columns=feat_in)

    # numeric cast where possible
    for c in X.columns:
        if not pd.api.types.is_object_dtype(X[c]):
            try:
                X[c] = pd.to_numeric(X[c], errors="ignore")
            except Exception:
                pass
    return X

def run_level1_strict(m, X_raw: pd.DataFrame, name: str):
    """
    Try predict_proba first, then decision_function, finally predict.
    Returns (float_value, mode)
    """
    # try int casting first, then str casting
    errors = []
    for mode in ("int", "str"):
        try:
            X = align_to_model_features(m, X_raw, cast_cat=mode)
            # CatBoost branch
            if isinstance(m, Pipeline) and isinstance(m.named_steps.get("clf", None), CatBoostClassifier):
                pool = Pool(X, cat_features=[i for i, col in enumerate(X.columns) if pd.api.types.is_object_dtype(X[col])])
                p = m.named_steps["clf"].predict_proba(pool)[:, 1]
                return float(p[0]), "proba"
            # Normal sklearn-like
            if hasattr(m, "predict_proba"):
                p = np.asarray(m.predict_proba(X))
                if p.ndim == 2 and p.shape[1] >= 2:
                    return float(p[0, 1]), "proba"
            if hasattr(m, "decision_function"):
                z = float(np.ravel(m.decision_function(X))[0])
                p = 1.0 / (1.0 + np.exp(-z))
                return float(p), "decision"
            y = float(np.ravel(m.predict(X))[0])
            return y, "value"
        except Exception as e:
            errors.append((mode, f"{type(e).__name__}: {e}", list(X_raw.columns)))
            continue
    msg = [f"Level-1 model '{name}' failed in both modes:"]
    for m0, err, cols in errors:
        msg.append(f"  - mode={m0}, error={err}, cols={cols}")
    raise RuntimeError("\n".join(msg))

def build_meta_input(level1_dict: dict, X_clin: pd.DataFrame, feature_names: list[str]):
    """
    Returns (X_meta 1xK, modes_used dict)
    """
    rec = {}
    modes = {}
    # ensure all required level-1 models exist
    missing = [f for f in feature_names if f not in level1_dict]
    if missing:
        raise FileNotFoundError(f"Missing level-1 model files for: {missing}")
    for name in feature_names:
        p, mode = run_level1_strict(level1_dict[name], X_clin, name)
        rec[name] = p
        modes[name] = mode
    X_meta = pd.DataFrame([[rec[c] for c in feature_names]], columns=feature_names)
    return X_meta, modes

# =============================
# Prediction and explanation
# =============================
if go:
    # 1) build clinical row
    X_clin_raw = build_clin_df(inputs)

    # 2) run all level-1 models in meta expected order
    try:
        X_meta, l1_modes = build_meta_input(level1_models, X_clin_raw, exp_meta_feats)
    except Exception as e:
        st.error(f"Level-1 failed: {e}")
        st.stop()

    st.subheader("Level-1 outputs (probabilities fed into meta model)")
    st.dataframe(X_meta, use_container_width=True)

    # 3) meta probability
    try:
        prob = float(meta_model.predict_proba(X_meta)[0, 1])
    except Exception:
        # try decision_function
        if hasattr(meta_model, "decision_function"):
            z = float(meta_model.decision_function(X_meta)[0])
            prob = 1.0 / (1.0 + np.exp(-z))
        else:
            # last fallback
            prob = float(meta_model.predict(X_meta)[0])

    st.subheader("Prediction")
    st.metric("Predicted ARDS probability", f"{prob:.1%}")

    # 4) linear logit contributions for meta LR (exact and fast)
    try:
        feat_names = list(X_meta.columns)
        coef = np.ravel(meta_model.coef_)            # shape (K,)
        intercept = float(np.ravel(meta_model.intercept_)[0])

        x = X_meta.iloc[0].values.astype(float)      # shape (K,)
        # baseline: if you have OOF means saved as JSON, you can load and use them here
        baseline = np.full_like(x, 0.5, dtype=float)

        base_logit = intercept + float(np.dot(coef, baseline))
        contrib = coef * (x - baseline)              # per-feature logit deltas

        order = np.argsort(np.abs(contrib))[::-1]
        contrib_sorted = contrib[order]
        feat_sorted = [feat_names[i] for i in order]
        x_sorted = x[order]

        expl = shap.Explanation(
            values=contrib_sorted,
            base_values=base_logit,
            data=x_sorted,
            feature_names=feat_sorted
        )

        st.subheader("Level-2 explanation (logit contributions)")
        fig = plt.figure(figsize=(10, 4))
        shap.plots.waterfall(expl, show=False)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.caption("Note: converting logit contributions to probability can be approximated by multiplying each bar by p·(1-p).")

    except Exception as e:
        st.info(f"Linear contribution plot unavailable: {e}")

else:
    st.info("Set parameters on the left and click Predict.")
