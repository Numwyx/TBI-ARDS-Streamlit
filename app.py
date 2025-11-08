# app.py — TBI-ARDS Risk Predictor (Full Stacking, level-1 PKLs in repo root)

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="TBI-ARDS Stacking Predictor", layout="wide")
st.title("🧠 TBI-ARDS Stacking Predictor")

# Files
META_PATH = Path("best_model.pkl")   # level-2 meta model
ROOT = Path(".")                     # level-1 PKLs live in repository root

# Level-1 feature names expected by the meta model
META_FEATURES = ["ada","cat","dt","gbm","knn","lgb","logistic","mlp","rf","svm","xgb"]

# Clinical inputs for the UI
FEATURE_SPECS = {
    "VP":         {"type": "cat", "labels": {0: "No vasopressor", 1: "Vasopressor used"}, "default": 0},
    "MV":         {"type": "cat", "labels": {0: "No mechanical ventilation", 1: "Mechanical ventilation"}, "default": 0},
    "APACHEII":   {"type": "num", "min": 0.0,  "max": 71.0,  "default": 18.0, "step": 1.0},
    "SOFA":       {"type": "num", "min": 0.0,  "max": 24.0,  "default": 6.0,  "step": 1.0},
    "GCS":        {"type": "num", "min": 3.0,  "max": 15.0,  "default": 10.0, "step": 1.0},
    "CCI":        {"type": "num", "min": 0.0,  "max": 20.0,  "default": 2.0,  "step": 1.0},
    "MAP":        {"type": "num", "min": 30.0, "max": 200.0, "default": 80.0, "step": 1.0},
    "Temp":       {"type": "num", "min": 30.0, "max": 43.0,  "default": 37.5, "step": 0.1},
    "RR":         {"type": "num", "min": 5.0,  "max": 60.0,  "default": 20.0, "step": 1.0},
    "Calcium":    {"type": "num", "min": 0.5,  "max": 3.0,   "default": 2.2,  "step": 0.1},
    "Sodium":     {"type": "num", "min": 110.0,"max": 170.0, "default": 138.0,"step": 1.0},
    "Glucose":    {"type": "num", "min": 2.0,  "max": 33.0,  "default": 8.5,  "step": 0.1},
    "Creatinine": {"type": "num", "min": 20.0, "max": 1200.0,"default": 90.0, "step": 5.0},
    "HB":         {"type": "num", "min": 40.0, "max": 200.0, "default": 120.0,"step": 1.0},
}
FEATURE_ORDER = list(FEATURE_SPECS.keys())

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
    "Calcium": "Serum calcium (mmol/L)",
    "Sodium": "Serum sodium (mmol/L)",
    "Glucose": "Blood glucose (mmol/L)",
    "Creatinine": "Creatinine (µmol/L)",
    "HB": "Hemoglobin (g/L)",
}

# ---------- load models ----------
@st.cache_resource
def load_meta():
    if not META_PATH.exists():
        st.error("Missing meta model: best_model.pkl")
        st.stop()
    return joblib.load(META_PATH)

@st.cache_resource
def load_level1_models_from_root():
    models = {}
    missing = []
    for name in META_FEATURES:
        fp = ROOT / f"model_{name}.pkl"
        if fp.exists():
            try:
                models[name] = joblib.load(fp)
            except Exception as e:
                st.warning(f"[WARN] failed to load {fp.name}: {e}")
        else:
            missing.append(name)
    return models, missing

meta = load_meta()
level1_models, missing = load_level1_models_from_root()
if missing:
    st.error("Missing level-1 model files in repository root:\n" +
             "\n".join([f"model_{n}.pkl" for n in missing]))
    st.stop()

# ---------- helpers ----------
def expected_features(model):
    feat_in = getattr(model, "feature_names_in_", None)
    if feat_in is None and hasattr(model, "named_steps"):
        try:
            feat_in = getattr(model.named_steps.get("clf", None), "feature_names_in_", None)
        except Exception:
            feat_in = None
    return list(feat_in) if feat_in is not None else None

def build_clinical_df(d):
    return pd.DataFrame([{k: d[k] for k in FEATURE_ORDER}], columns=FEATURE_ORDER)

def cast_categories(df, mode="int"):
    Z = df.copy()
    if mode == "int":
        for c in ["VP","MV"]:
            if c in Z.columns:
                try: Z[c] = Z[c].astype(int)
                except Exception: pass
    else:
        for c in ["VP","MV"]:
            if c in Z.columns:
                Z[c] = Z[c].astype("Int64").astype(str)
    return Z

def align_to_model_schema(model, X, defaults=None):
    feat_in = expected_features(model)
    if feat_in is None:
        return X.copy()
    X2 = X.copy()
    lower_to_train = {c.lower(): c for c in feat_in}
    rename_map = {}
    for col in X2.columns:
        tgt = lower_to_train.get(col.lower())
        if tgt and tgt != col:
            rename_map[col] = tgt
    if rename_map:
        X2 = X2.rename(columns=rename_map)
    # add missing with defaults
    miss = [c for c in feat_in if c not in X2.columns]
    if miss:
        for c in miss:
            if defaults and c in defaults:
                X2[c] = defaults[c]
            else:
                # fall back to zero if no better default
                X2[c] = 0.0
    return X2.reindex(columns=feat_in)

def predict_proba_or_value(model, X):
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        if p.ndim == 2 and p.shape[1] >= 2:
            return float(p[0,1]), "prob"
    if hasattr(model, "decision_function"):
        z = float(np.ravel(model.decision_function(X))[0])
        p = 1.0/(1.0+np.exp(-z))
        return float(p), "prob"
    yhat = float(np.ravel(model.predict(X))[0])
    return yhat, "value"

def run_level1(model1, X_clin):
    # try int then str automatically
    Xi = cast_categories(X_clin, "int")
    Xi = align_to_model_schema(model1, Xi)
    try:
        p, _ = predict_proba_or_value(model1, Xi)
        return p
    except Exception:
        Xs = cast_categories(X_clin, "str")
        Xs = align_to_model_schema(model1, Xs)
        p, _ = predict_proba_or_value(model1, Xs)
        return p

def build_meta_input(level1_dict, X_clin):
    rec = {}
    for name in META_FEATURES:
        rec[name] = run_level1(level1_dict[name], X_clin)
    return pd.DataFrame([[rec[c] for c in META_FEATURES]], columns=META_FEATURES)

# ---------- sidebar ----------
st.sidebar.header("Patient Parameters")
inputs = {}
for feat in FEATURE_ORDER:
    spec = FEATURE_SPECS[feat]
    label = DISPLAY_NAME.get(feat, feat)
    if spec["type"] == "cat":
        labels = list(spec["labels"].values())
        idx = list(spec["labels"].keys()).index(spec["default"])
        sel = st.sidebar.selectbox(label, labels, index=idx)
        inputs[feat] = {v:k for k,v in spec["labels"].items()}[sel]
    else:
        inputs[feat] = st.sidebar.number_input(
            label,
            min_value=float(spec["min"]), max_value=float(spec["max"]),
            value=float(spec["default"]), step=float(spec["step"])
        )
go = st.sidebar.button("🔮 Predict ARDS Risk", type="primary")

# ---------- main ----------
if go:
    X_clin_raw = build_clinical_df(inputs)
    X_clin = cast_categories(X_clin_raw, "int")

    # 1) run all level-1 models on clinical inputs
    X_meta = build_meta_input(level1_models, X_clin)

    # 2) align to meta expected order if present
    exp_meta = expected_features(meta)
    if exp_meta is not None:
        for c in exp_meta:
            if c not in X_meta.columns:
                X_meta[c] = 0.0
        X_meta = X_meta.reindex(columns=exp_meta)
    else:
        X_meta = X_meta.reindex(columns=META_FEATURES)

    # 3) predict with meta
    score, kind = predict_proba_or_value(meta, X_meta)

    st.subheader("Prediction")
    if kind == "prob":
        st.metric("Predicted ARDS probability", f"{score:.1%}")
    else:
        st.metric("Predicted output", f"{score:.4f}")

    st.caption("Level-1 model probabilities used as meta features")
    st.dataframe(X_meta, use_container_width=True)

    st.divider()
    st.subheader("SHAP Explanation on Meta Model")

    shap_values = None
    base = None
    try:
        expl = shap.Explainer(meta)
        sv = expl(X_meta)
        base = float(np.ravel(sv.base_values)[0])
        shap_values = np.ravel(sv.values)
    except Exception as e:
        st.info(f"SHAP not available: {e}")

    if shap_values is not None:
        try:
            shap.plots.waterfall(
                shap.Explanation(values=shap_values, base_values=base,
                                 data=X_meta.iloc[0].values,
                                 feature_names=list(X_meta.columns)),
                show=False
            )
            plt.tight_layout()
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception:
            try:
                shap.force_plot(base, shap_values, X_meta.iloc[0].values,
                                feature_names=list(X_meta.columns), matplotlib=True, show=False)
                plt.tight_layout()
                st.pyplot(plt.gcf(), clear_figure=True)
            except Exception as e:
                st.info(f"Could not render SHAP plot: {e}")

    with st.expander("Debug"):
        st.write("Meta expects:", exp_meta if exp_meta is not None else META_FEATURES)
        st.write("Loaded level-1:", sorted(level1_models.keys()))
        st.write("Clinical columns:", list(X_clin_raw.columns))
else:
    st.info("Set patient parameters on the left and click Predict ARDS Risk.")
